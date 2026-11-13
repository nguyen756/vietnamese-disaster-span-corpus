"""Synthetic correctness checks only: no dataset training and no optimizer steps."""
from io import BytesIO
from itertools import product
from pathlib import Path
import math
import random
import sys
import numpy as np
import pytest
import torch
from transformers import RobertaConfig, RobertaModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import collate
from model import NERModel
from preprocessing import align_characters
from evaluation import independently_check, aggregate
from train_crf_pilot import score


def batch():
    return collate([
        {'units': [(0, 1, 'a'), (2, 3, 'b')], 'tags': ['B-X', 'I-X'],
         'chunks': [{'input_ids': [0, 4, 5, 2]}], 'pooling_pairs': [[0, 1, 0, 1.], [0, 2, 1, 1.]]},
        {'units': [(0, 1, 'c')], 'tags': ['O'], 'chunks': [{'input_ids': [0, 6, 2]}],
         'pooling_pairs': [[0, 1, 0, 1.]]}], {'O': 0, 'B-X': 1, 'I-X': 2})


def model(architecture):
    torch.manual_seed(42)
    config = RobertaConfig(vocab_size=20, hidden_size=16, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=24, max_position_embeddings=20,
                          hidden_dropout_prob=0., attention_probs_dropout_prob=0., pad_token_id=1)
    return NERModel(RobertaModel(config, add_pooling_layer=False), ['O', 'B-X', 'I-X'], architecture, hidden=4, dropout=0.).eval()


@pytest.mark.parametrize('architecture', ['phobert', 'bilstm', 'bilstm_crf'])
def test_three_heads_forward_mask_loss_and_weight_roundtrip(architecture):
    candidate = model(architecture)
    b = batch()
    with torch.inference_mode():
        emissions = candidate(b)
        assert emissions.shape == (2, 2, 3)
        assert torch.isfinite(candidate.loss(emissions, b))
        predictions = candidate.tag_sequences(emissions, b)
        assert list(map(len, predictions)) == [2, 1]
        stream = BytesIO()
        torch.save(candidate.state_dict(), stream)
        stream.seek(0)
        restored = model(architecture)
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        assert torch.equal(emissions, restored(b))
        assert predictions == restored.tag_sequences(restored(b), b)


def test_crf_masked_likelihood_and_viterbi_match_exhaustive_sequences():
    candidate = model('bilstm_crf')
    b = batch()
    emissions = torch.tensor([[[.2, -.1, .4], [.5, -.4, .1]], [[-.2, .8, .3], [100., 200., 300.]]])
    crf = candidate.crf
    def sequence_score(scores, tags):
        value = crf.start_transitions[tags[0]] + crf.end_transitions[tags[-1]]
        for i, tag in enumerate(tags):
            value = value + scores[i, tag]
            if i:
                value = value + crf.transitions[tags[i-1], tag]
        return value
    numerator = 0.
    expected_decoding = []
    for row, length, truth in [(0, 2, [1, 2]), (1, 1, [0])]:
        sequences = list(product(range(3), repeat=length))
        values = torch.stack([sequence_score(emissions[row], sequence) for sequence in sequences])
        numerator = numerator + torch.logsumexp(values, 0) - sequence_score(emissions[row], truth)
        expected_decoding.append(list(sequences[int(values.argmax())]))
    assert torch.allclose(candidate.loss(emissions, b), numerator / 3, atol=1e-6)
    assert candidate.tag_sequences(emissions, b) == expected_decoding


def test_alignment_preserves_original_offsets_for_tone_changes_and_underscores():
    for original, segmented in [('Hoà Bình', 'Hòa_Bình'), ('a_b', 'a _ b'), ('“ngập…”', '" ngập ... "')]:
        mapping, edits = align_characters(original, segmented)
        assert set().union(*mapping) == {i for i, c in enumerate(original) if not c.isspace()}
    with pytest.raises(ValueError):
        align_characters('Hoà Bình', 'completely unrelated text')


def test_bilstm_and_bilstm_crf_start_with_identical_emission_weights():
    a, b = model('bilstm'), model('bilstm_crf')
    with torch.inference_mode():
        assert torch.equal(a(batch()), b(batch()))


def test_unrepresentable_gold_reference_is_not_dropped_and_std_is_sample():
    rows = [{'text': 'abcdef', 'label': [[1, 3, 'X']]}]
    predictions = [[[0, 6, 'X']]]
    result = score(rows, predictions, ['X'])
    assert independently_check(rows, predictions, ['X'], result)
    assert result['micro']['tp'] == 0 and result['micro']['fn'] == result['micro']['fp'] == 1
    assert aggregate([.5, .6, .7])['sample_std'] == pytest.approx(.1)


def test_preprocessing_is_label_independent_and_pooling_covers_every_original_unit():
    from tokenizers import Tokenizer
    from common import PACKAGE
    from preprocessing import prepare_text
    class FixedSegmenter:
        def word_segment(self, text):
            return ['Hoà_Bình ngập 2 m .']
    tokenizer = Tokenizer.from_file(str(PACKAGE / 'assets/phobert-base/tokenizer.json'))
    row = {'text': 'Hòa Bình ngập 2m.', 'source_row': 1}
    a = prepare_text(row, FixedSegmenter(), tokenizer)
    b = prepare_text(dict(row, label=[[0, 999, 'INVALID']]), FixedSegmenter(), tokenizer)
    assert a == b
    assert a['normalization_changed']
    weights = {}
    for _, _, unit, weight in a['pooling_pairs']:
        weights[unit] = weights.get(unit, 0.) + weight
    assert set(weights) == set(range(len(a['units'])))
    assert all(value == pytest.approx(1.) for value in weights.values())


def test_atomic_checkpoint_links_and_history_roundtrip_without_training(tmp_path):
    from training import atomic_checkpoint, link_checkpoint, write_history
    checkpoint = tmp_path / 'checkpoints/epoch_01.pt'
    checkpoint.parent.mkdir()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor([1.]))])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    history = [{'epoch': 1, 'mean_training_loss': .5, 'training_seconds': 1., 'evaluation_seconds': .1,
                'encoder_lr': .001, 'head_lr': .002,
                'development': {'micro': {'precision': .4, 'recall': .6, 'f1': .48}, 'macro_f1': .45}}]
    state = {'completed_epochs': 1, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
             'python_rng': random.getstate(), 'history': history}
    atomic_checkpoint(checkpoint, state)
    link_checkpoint(tmp_path, 'last_checkpoint.pt', checkpoint)
    restored = torch.load(tmp_path / 'last_checkpoint.pt', weights_only=False)
    assert restored['history'] == history
    assert torch.equal(restored['torch_rng'], state['torch_rng'])
    assert restored['scheduler'] == state['scheduler']
    assert restored['optimizer'] == state['optimizer']
    write_history(tmp_path, history)
    assert (tmp_path / 'history.csv').read_text().splitlines()[1].startswith('1,0.5,')
    assert not list(checkpoint.parent.glob('*.tmp'))


def test_best_only_replaces_on_strict_improvement_and_never_writes_epoch_files(tmp_path):
    from training import update_best_checkpoint
    history = []
    serialized = []
    for epoch, f1 in enumerate([.5, .4, .5, .6, .55], 1):
        history.append({'epoch': epoch, 'development': {'micro': {'f1': f1}}})
        def state():
            serialized.append(epoch)
            return {'completed_epochs': epoch, 'history': list(history), 'model': {'weight': torch.tensor([epoch])}}
        updated = update_best_checkpoint(tmp_path, history, state)
        assert updated == (epoch in (1, 4))
        value = torch.load(tmp_path / 'best_checkpoint.pt', weights_only=False)
        assert value['completed_epochs'] == (1 if epoch < 4 else 4)
    assert serialized == [1, 4]  # expensive state construction is also skipped for worse/tied scores
    assert [p.name for p in tmp_path.glob('*.pt') if not p.is_symlink()] == ['best_checkpoint.pt']
    assert not (tmp_path / 'checkpoints').exists()
    assert not (tmp_path / 'last_checkpoint.pt').exists()
    assert (tmp_path / 'best_development_checkpoint.pt').resolve() == tmp_path / 'best_checkpoint.pt'


def test_failed_best_save_preserves_the_previous_checkpoint(tmp_path, monkeypatch):
    import training
    history = [{'epoch': 1, 'development': {'micro': {'f1': .5}}}]
    training.update_best_checkpoint(tmp_path, history, lambda: {'completed_epochs': 1})
    original_bytes = (tmp_path / 'best_checkpoint.pt').read_bytes()
    history.append({'epoch': 2, 'development': {'micro': {'f1': .6}}})
    def fail_save(value, path):
        path.write_bytes(b'incomplete temporary checkpoint')
        raise OSError('simulated write failure')
    monkeypatch.setattr(training.torch, 'save', fail_save)
    with pytest.raises(OSError):
        training.update_best_checkpoint(tmp_path, history, lambda: {'completed_epochs': 2})
    assert (tmp_path / 'best_checkpoint.pt').read_bytes() == original_bytes
    assert torch.load(tmp_path / 'best_checkpoint.pt', weights_only=False)['completed_epochs'] == 1


def test_resume_from_best_archives_replayed_epoch_evidence(tmp_path):
    from common import save_json
    from training import archive_resume_rollback
    save_json(tmp_path / 'history.json', [{'epoch': i} for i in range(1, 7)])
    for epoch in range(1, 7):
        save_json(tmp_path / f'metrics_epoch_{epoch:02d}.json', {'epoch': epoch})
        (tmp_path / f'development_epoch_{epoch:02d}.jsonl').write_text('{}\n')
    archive = archive_resume_rollback(tmp_path, 4)
    assert archive is not None
    assert (archive / 'history.json').exists()
    assert [p.name for p in sorted(archive.glob('metrics_epoch_*.json'))] == ['metrics_epoch_05.json', 'metrics_epoch_06.json']
    assert (tmp_path / 'metrics_epoch_04.json').exists()
    assert not (tmp_path / 'metrics_epoch_05.json').exists()
    assert (archive / 'development_epoch_06.jsonl').read_text() == '{}\n'


def test_legacy_and_new_run_specs_are_accepted_but_numerical_changes_are_rejected(monkeypatch):
    import common
    legacy = common.read_json(common.EXP / 'runs/phobert_seed42/experiment_config.json')
    assert common.validate_run_spec(legacy) == 'all_epochs'
    current = {**legacy, **common.protocol(), 'code_sha256': common.source_hashes()}
    assert common.validate_run_spec(current) == 'best_only'
    bad = {**current, 'encoder_lr': .01}
    with pytest.raises(AssertionError):
        common.validate_run_spec(bad)
    actual = common.protocol()
    monkeypatch.setattr(common, 'protocol', lambda: {**actual, 'encoder_lr': .01})
    with pytest.raises(AssertionError, match='only permits checkpoint-storage changes'):
        common.compatible_plans()


def test_retention_inventory_supports_one_best_and_twenty_legacy_checkpoints(tmp_path, monkeypatch):
    import common
    monkeypatch.setattr(common, 'validate_run_spec', lambda spec: spec['policy'])
    (tmp_path / 'best_checkpoint.pt').write_bytes(b'checkpoint')
    result = {'configuration': {'policy': 'best_only'}, 'retained_checkpoint_epoch': 4,
              'best_development_epoch': 4, 'best_checkpoint_sha256': common.sha(tmp_path / 'best_checkpoint.pt')}
    policy, paths = common.retained_checkpoints(tmp_path, result)
    assert policy == 'best_only' and len(paths) == 1
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    for epoch in range(1, 21):
        (checkpoint_dir / f'epoch_{epoch:02d}.pt').write_bytes(b'legacy')
    policy, paths = common.retained_checkpoints(tmp_path, {'configuration': {'policy': 'all_epochs'}})
    assert policy == 'all_epochs' and len(paths) == 20
    with pytest.raises(AssertionError, match='unexpectedly retained epoch checkpoints'):
        common.retained_checkpoints(tmp_path, result)


def test_mixed_device_reporting_records_the_actual_hardware_change():
    from evaluation import training_device_accounting
    models = {'phobert_seed42': (None, {'architecture': 'phobert', 'device': 'cpu'}, None),
              'bilstm_seed42': (None, {'architecture': 'bilstm', 'device': 'cuda'}, None)}
    record = training_device_accounting(models)
    assert record['mixed_training_devices']
    assert record['per_model']['bilstm_seed42'] == 'cuda'
    assert record['per_architecture'] == {'phobert': ['cpu'], 'bilstm': ['cuda']}


def test_pinned_best_only_cpu_runs_remain_verifiable_after_reporting_revision():
    import common
    previous = common.EXP / 'revisions/best_only_v2'
    spec = {**common.read_json(previous / 'protocol.json'), 'versions': common.versions(),
            'cache_manifest_sha256': common.sha(common.EXP / 'cache/manifest.json'),
            'code_sha256': {p.name: common.sha(p) for p in sorted(previous.glob('*.py'))}}
    assert common.validate_run_spec(spec) == 'best_only'
