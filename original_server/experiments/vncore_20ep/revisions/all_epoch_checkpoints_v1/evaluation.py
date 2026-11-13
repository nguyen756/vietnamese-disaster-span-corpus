"""Final epoch-20 evaluation, enabled only after all nine runs complete."""
from collections import Counter
import csv
import fcntl
import json
from pathlib import Path
import statistics
import time
import zipfile
from common import (EXP, PACKAGE, ARCHITECTURES, SEEDS, protocol, run_path, read_json, save_json,
                    sha, read_jsonl, write_jsonl, score, error_analysis, load_cache, segmenter,
                    log_event, source_hashes, versions)


def independently_check(rows, predictions, labels, expected):
    counts = {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in labels}
    assert len(rows) == len(predictions)
    for row, predicted in zip(rows, predictions):
        previous = 0
        for start, end, label in predicted:
            assert type(start) is int and type(end) is int and previous <= start < end <= len(row['text'])
            assert label in labels
            previous = end
        truth, found = set(map(tuple, row['label'])), set(map(tuple, predicted))
        assert len(found) == len(predicted)
        for key, spans in [('tp', truth & found), ('fp', found - truth), ('fn', truth - found)]:
            for _, _, label in spans:
                counts[label][key] += 1
    for label in labels:
        assert all(expected['per_label'][label][k] == counts[label][k] for k in ['tp', 'fp', 'fn'])
    assert all(expected['micro'][key] == sum(c[key] for c in counts.values()) for key in ['tp', 'fp', 'fn'])
    return True


def aggregate(values):
    assert len(values) == 3
    return {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values),
            'std_ddof': 1, 'seeds': list(SEEDS), 'values': values}


def evaluate_all():
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from preprocessing import prepare_text
    from training import build_model, predict
    target = EXP / 'evaluation'
    target.mkdir(exist_ok=True)
    with (target / 'evaluation.lock').open('w') as process_lock:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if all((target / name).exists() for name in ['summary.json', 'RESULTS.md', 'metrics.csv', 'verification.json']):
            raise FileExistsError('Final evaluation already complete; use the existing report, not a new test-driven experiment')
        manifest, _, prepared_dev = load_cache()
        models, lock = {}, {'protocol_sha256': sha(EXP / 'protocol.json'),
                            'cache_manifest_sha256': sha(EXP / 'cache/manifest.json'),
                            'gold_sha256': sha(PACKAGE / 'final_paper_kit/reserved_test/gold_test_final.jsonl'),
                            'code_sha256': source_hashes(), 'primary_checkpoint_epoch': 20, 'models': {}}
        assert lock['gold_sha256'] == protocol()['gold_sha256']
        devices = set()
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                run = run_path(architecture, seed)
                result = read_json(run / 'results.json')
                spec = read_json(run / 'experiment_config.json')
                assert result['completed_epochs'] == result['primary_checkpoint_epoch'] == 20
                assert result['architecture'] == architecture and result['seed'] == seed
                assert result['configuration'] == spec
                assert spec['code_sha256'] == source_hashes() and spec['versions'] == versions()
                assert spec['cache_manifest_sha256'] == lock['cache_manifest_sha256']
                assert all(spec[k] == v for k, v in protocol().items())
                assert [h['epoch'] for h in result['history']] == list(range(1, 21))
                assert not result['gold_test_used_during_training']
                assert all((run / f'checkpoints/epoch_{epoch:02d}.pt').is_file() for epoch in range(1, 21))
                digest = sha(run / 'model_epoch20.safetensors')
                assert digest == result['primary_model_sha256']
                devices.add(spec['device'])
                models[run.name] = (run, spec, result)
                lock['models'][run.name] = digest
        assert len(models) == 9 and len(devices) == 1, 'All nine runs must use the same training device'
        path = target / 'model_lock.json'
        if path.exists():
            assert read_json(path) == lock, 'Evaluation model/input lock changed'
        else:
            save_json(path, lock)
        # Gold reference contents enter this process only AFTER all nine models are locked.
        gold = [dict(row, source_row=f'gold_{i:04d}') for i, row in enumerate(
            read_jsonl(PACKAGE / 'final_paper_kit/reserved_test/gold_test_final.jsonl'), 1)]
        dev = read_jsonl(PACKAGE / 'data/validation.jsonl')
        assert len(gold) == 1000 and len(dev) == 803
        tokenizer = Tokenizer.from_file(str(PACKAGE / 'assets/phobert-base/tokenizer.json'))
        vncore = segmenter()
        prepared_gold = [prepare_text({'text': r['text'], 'source_row': r['source_row']}, vncore, tokenizer) for r in gold]
        write_jsonl(target / 'prepared_gold_text_only.jsonl', prepared_gold)
        torch.set_num_threads(protocol()['threads'])
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)
        labels = read_json(PACKAGE / 'reference/annotation_schema.json')['labels']
        reports, checks = {}, {}
        (target / 'predictions').mkdir(exist_ok=True)
        (target / 'errors').mkdir(exist_ok=True)
        for name, (run, spec, training) in models.items():
            started = time.perf_counter()
            log_event(target, 'evaluating', model=name, checkpoint_epoch=20)
            model = build_model(spec['architecture'], spec['bio_labels'], pretrained=False)
            model.load_state_dict(load_file(str(run / 'model_epoch20.safetensors')), strict=True)
            # Match each run's fixed training device to reproduce development predictions exactly.
            device = torch.device(spec['device'])
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA-trained runs require CUDA for exact checkpoint replay')
            model.to(device)
            dev_pred = predict(model, prepared_dev, spec['bio_labels'], device, protocol()['batch_size'])
            saved = read_jsonl(run / 'development_epoch_20.jsonl')
            assert len(saved) == 803
            for row, previous, predicted in zip(dev, saved, dev_pred):
                assert all(row[k] == previous[k] for k in ['source_row', 'document_id', 'text', 'label'])
                assert predicted == previous['predicted_label'], f'Checkpoint replay mismatch: {name}, {row["source_row"]}'
            dev_metrics = score(dev, dev_pred, labels)
            assert dev_metrics == training['development_epoch20']
            gold_pred = predict(model, prepared_gold, spec['bio_labels'], device, protocol()['batch_size'])
            gold_metrics = score(gold, gold_pred, labels)
            independently_check(dev, dev_pred, labels, dev_metrics)
            independently_check(gold, gold_pred, labels, gold_metrics)
            del model
            categories = {}
            for split, rows, predictions in [('development', dev, dev_pred), ('gold_test', gold, gold_pred)]:
                write_jsonl(target / 'predictions' / f'{name}_{split}.jsonl',
                            [dict(row, predicted_label=p) for row, p in zip(rows, predictions)])
                errors, category = error_analysis(rows, predictions)
                write_jsonl(target / 'errors' / f'{name}_{split}.jsonl', errors)
                categories[split] = dict(category)
            reports[name] = {'architecture': spec['architecture'], 'seed': spec['seed'], 'checkpoint_epoch': 20,
                             'model_sha256': lock['models'][name], 'development': dev_metrics, 'gold_test': gold_metrics,
                             'best_development_epoch': training['best_development_epoch'],
                             'best_development_f1': training['best_development']['micro']['f1'],
                             'error_categories': categories, 'evaluation_seconds': time.perf_counter() - started}
            checks[name] = {'checkpoint_replay_rows': 803, 'replay_exact': True,
                            'independent_counts_match': True, 'gold_rows_scored': 1000,
                            'gold_reference_spans_retained': sum(len(r['label']) for r in gold)}
            save_json(target / 'model_metrics.json', reports)
        summaries = {}
        for architecture in ARCHITECTURES:
            summaries[architecture] = {}
            for split in ['development', 'gold_test']:
                summaries[architecture][split] = {
                    key: aggregate([reports[f'{architecture}_seed{seed}'][split]['micro'][key] for seed in SEEDS])
                    for key in ['precision', 'recall', 'f1']}
                summaries[architecture][split]['macro_f1'] = aggregate([
                    reports[f'{architecture}_seed{seed}'][split]['macro_f1'] for seed in SEEDS])
                summaries[architecture][split]['per_label_f1'] = {
                    label: aggregate([reports[f'{architecture}_seed{seed}'][split]['per_label'][label]['f1'] for seed in SEEDS])
                    for label in labels}
        for name, (run, _, _) in models.items():
            assert sha(run / 'model_epoch20.safetensors') == lock['models'][name]
        assert sha(PACKAGE / 'final_paper_kit/reserved_test/gold_test_final.jsonl') == lock['gold_sha256']
        assert sha(EXP / 'cache/manifest.json') == lock['cache_manifest_sha256']
        save_json(target / 'verification.json', {'passed': True, 'models': checks, 'model_and_gold_hashes_locked': True,
                                                'all_nine_runs_completed_twenty_epochs': True,
                                                'test_based_checkpoint_selection': False})
        save_json(target / 'summary.json', {'protocol': protocol(), 'models': reports,
                                            'mean_and_sample_std': summaries,
                                            'gold_test_previously_used': True, 'primary_checkpoint_epoch': 20})
        with (target / 'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            fields = ['model', 'architecture', 'seed', 'split', 'label', 'precision', 'recall', 'f1', 'support', 'tp', 'fp', 'fn']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for name, report in reports.items():
                for split in ['development', 'gold_test']:
                    for label, values in [('micro', report[split]['micro']), *report[split]['per_label'].items()]:
                        writer.writerow({'model': name, 'architecture': report['architecture'], 'seed': report['seed'],
                                         'split': split, 'label': label,
                                         **{k: values[k] for k in ['precision', 'recall', 'f1', 'support', 'tp', 'fp', 'fn']}})
        lines = ['# VnCoreNLP: 9 runs × 20 full epochs', '', 'Primary checkpoint: epoch 20 for every run; no early stopping.', '',
                 '| Architecture | Seed | Dev F1 (epoch 20) | Gold F1 (epoch 20) | Best dev epoch |', '|---|---:|---:|---:|---:|']
        for name, report in reports.items():
            lines.append(f"| {report['architecture']} | {report['seed']} | {100*report['development']['micro']['f1']:.2f}% | {100*report['gold_test']['micro']['f1']:.2f}% | {report['best_development_epoch']} |")
        lines += ['', '| Architecture | Dev F1 mean ± sample SD | Gold F1 mean ± sample SD |', '|---|---:|---:|']
        for architecture, splits in summaries.items():
            values = [f"{100*splits[s]['f1']['mean']:.2f}% ± {100*splits[s]['f1']['sample_std']:.2f} pp" for s in ['development', 'gold_test']]
            lines.append(f'| {architecture} | {values[0]} | {values[1]} |')
        lines += ['', 'Gold Test was evaluated in an earlier experiment. This is a follow-up experiment on a previously used test set.',
                  'Most training/development labels remain AI-generated; no references, exclusions, or split assignments were changed.',
                  'All epoch checkpoints remain on the server; primary model files and all metrics/predictions/errors/logs are packaged separately.']
        (target / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        log_event(target, 'evaluation_complete', models=9, gold_rows=1000)


def package_results():
    target = EXP / 'evaluation'
    assert read_json(target / 'verification.json')['passed']
    summary = read_json(target / 'summary.json')
    archive = EXP / 'vncore_20ep_results.zip'
    if archive.exists():
        raise FileExistsError(f'Archive already exists: {archive}')
    files = {}
    def add(path, name):
        assert name not in files
        files[name] = path
    for path in target.rglob('*'):
        if path.is_file() and not path.name.endswith('.lock') and path.name != 'RETURN_MANIFEST.json':
            add(path, 'evaluation/' + path.relative_to(target).as_posix())
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            run = run_path(architecture, seed)
            assert len(list((run / 'checkpoints').glob('epoch_*.pt'))) == 20
            assert sha(run / 'model_epoch20.safetensors') == summary['models'][run.name]['model_sha256']
            for path in run.rglob('*'):
                if path.is_file() and path.suffix not in ('.pt', '.lock', '.tmp'):
                    add(path, 'runs/' + run.name + '/' + path.relative_to(run).as_posix())
    for path in EXP.glob('*'):
        if path.is_file() and path.suffix in ('.py', '.json', '.sh', '.md', '.txt'):
            add(path, 'experiment/' + path.name)
    for path in (EXP / 'configs').glob('*.json'):
        add(path, 'experiment/configs/' + path.name)
    for path in (EXP / 'cache').glob('*'):
        if path.is_file():
            add(path, 'preprocessing/' + path.name)
    for name in ['java_download.json', 'vncorenlp_assets.json']:
        add(EXP / 'tools' / name, 'tools/' + name)
    for path in (EXP / 'tools/vncorenlp').rglob('*'):
        if path.is_file():
            add(path, 'tools/vncorenlp/' + path.relative_to(EXP / 'tools/vncorenlp').as_posix())
    for name in ['train.jsonl', 'validation.jsonl', 'split_manifest.json', 'excluded_from_training.jsonl']:
        add(PACKAGE / 'data' / name, 'inputs/' + name)
    add(PACKAGE / 'final_paper_kit/reserved_test/gold_test_final.jsonl', 'inputs/gold_test_final.jsonl')
    for path in (PACKAGE / 'scripts').glob('*.py'):
        add(path, 'original_training_code/' + path.name)
    manifest = {'files': {name: sha(path) for name, path in files.items()},
                'epoch_checkpoints_retained_on_server': 180, 'primary_epoch': 20,
                'primary_model_files_included': 9}
    save_json(target / 'RETURN_MANIFEST.json', manifest)
    # The manifest may exist from a previous packaging attempt; include it exactly once.
    files['evaluation/RETURN_MANIFEST.json'] = target / 'RETURN_MANIFEST.json'
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, path in sorted(files.items()):
            z.write(path, 'vncore_20ep_results/' + name)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
    temporary.replace(archive)
    (EXP / 'vncore_20ep_results.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n')
    log_event(None, 'packaged', archive=str(archive), files=len(files))
