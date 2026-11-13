"""VnCoreNLP word segmentation with explicit projection to original Unicode offsets.

No labels enter segmentation or character alignment. VnCoreNLP's output text is
fed to PhoBERT unchanged. Original text/annotations are never rewritten. Tone,
punctuation and literal-underscore differences are audited, never hidden behind
an alternative word segmenter. Unexpected substantial edits fail explicitly.
"""
from collections import Counter
from difflib import SequenceMatcher
import math
import unicodedata
from common import (EXP, PACKAGE, sha, save_json, read_json, read_jsonl, write_jsonl,
                    segmenter, versions, tokenize, encode, decode, protocol, log_event)
from phobert_ner_data import chunk_ids, official_piece_ids, pooling_pairs

PUNCTUATION = {'“': '"', '”': '"', '‘': "'", '’': "'", '–': '-', '—': '-', '…': '...'}


def character_stream(text, segmented=False):
    chars, positions = [], []
    for index, char in enumerate(text):
        if char.isspace() or char == '_':
            continue
        normalized = PUNCTUATION.get(char, char).casefold()
        normalized = ''.join(c for c in unicodedata.normalize('NFD', normalized)
                             if unicodedata.category(c) != 'Mn')
        for value in normalized:
            chars.append(value)
            positions.append(index)
    return ''.join(chars), positions


def align_characters(original, segmented):
    source, source_indices = character_stream(original)
    target, target_indices = character_stream(segmented, segmented=True)
    assert source and target, 'Empty source or VnCoreNLP output'
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    if matcher.ratio() < 0.95:
        raise ValueError(f'VnCoreNLP changed too much text for safe alignment: {original!r} => {segmented!r}')
    mapping = [set() for _ in segmented]
    edits = ([{'operation': 'literal_source_underscore_projection', 'positions': [i for i, c in enumerate(original) if c == '_']}] if '_' in original else [])
    for operation, a, b, c, d in matcher.get_opcodes():
        if operation == 'equal':
            for si, ti in zip(range(a, b), range(c, d)):
                mapping[target_indices[ti]].add(source_indices[si])
        else:
            edits.append({'operation': operation, 'source': source[a:b], 'target': target[c:d]})
            if c < d:
                # Retain every original character even for many-to-one changes.
                for k in range(c, d):
                    left = a + math.floor((k - c) * (b - a) / (d - c))
                    right = a + math.ceil((k - c + 1) * (b - a) / (d - c))
                    mapping[target_indices[k]].update(source_indices[left:right])
            elif a < b:
                neighbor = target_indices[min(c, len(target_indices) - 1)]
                mapping[neighbor].update(source_indices[a:b])
    covered = set().union(*mapping)
    expected = {i for i, char in enumerate(original) if not char.isspace()}
    # Standalone Unicode combining marks belong to the preceding source character.
    for position in sorted(expected - covered):
        nearest = min((i for i, values in enumerate(mapping) if values),
                      key=lambda i: min(abs(position - v) for v in mapping[i]))
        mapping[nearest].add(position)
    assert set().union(*mapping) >= expected
    return mapping, edits


def prepare_text(row, model, tokenizer):
    # Only the original text and identifier are read here, even if a caller has labels.
    text = row['text']
    segmented = ' '.join(model.word_segment(text))
    mapping, edits = align_characters(text, segmented)
    raw_source = ''.join(c for c in text if not c.isspace() and c != '_')
    raw_segmented = ''.join(c for c in segmented if not c.isspace() and c != '_')
    normalization_changed = raw_source != raw_segmented
    encoded = tokenizer.encode(segmented, add_special_tokens=False)
    offsets = []
    for first, last in encoded.offsets:
        positions = set().union(*mapping[first:last]) if first < last else set()
        offsets.append((min(positions), max(positions) + 1) if positions else None)
    ids = official_piece_ids(encoded.ids)
    chunks = chunk_ids(ids)
    units = tokenize(text)
    pairs = pooling_pairs(units, offsets, chunks)
    assert {p[2] for p in pairs} == set(range(len(units)))
    return {'source_row': row['source_row'], 'units': units, 'tags': [],
            'segmented_text': segmented, 'segmenter': 'VnCoreNLP-1.2/wseg',
            'alignment_edits': edits, 'normalization_changed': normalization_changed, 'piece_count': len(ids), 'chunks': chunks,
            'marker_only_pieces': sum(value is None for value in offsets),
            'converted_merge_entries_mapped_to_unk': sum(a != b for a, b in zip(encoded.ids, ids)),
            'pooling_pairs': pairs}


def prepare_training_cache():
    from tokenizers import Tokenizer
    from transformers import PhobertTokenizer
    cache = EXP / 'cache'
    if (cache / 'manifest.json').exists():
        from common import load_cache
        manifest, train, dev = load_cache()
        log_event(None, 'cache_verified', train_rows=len(train), development_rows=len(dev))
        return
    cache.mkdir(exist_ok=True)
    model = segmenter()
    base = PACKAGE / 'assets/phobert-base'
    tokenizer = Tokenizer.from_file(str(base / 'tokenizer.json'))
    official = PhobertTokenizer(vocab_file=str(base / 'vocab.txt'), merges_file=str(base / 'bpe.codes'))
    audits = []
    stats = {}
    for split, filename, expected in [('train', 'train.jsonl', 9686), ('development', 'validation.jsonl', 803)]:
        rows = read_jsonl(PACKAGE / 'data' / filename)
        assert len(rows) == expected
        prepared = []
        for index, row in enumerate(rows, 1):
            item = prepare_text({'text': row['text'], 'source_row': row['source_row']}, model, tokenizer)
            units, tags, bad = encode(row)
            assert not bad and [list(u) for u in units] == [list(u) for u in item['units']]
            assert decode(units, tags) == row['label']
            item['tags'] = tags
            assert official.encode(item['segmented_text'], add_special_tokens=False) == official_piece_ids(
                tokenizer.encode(item['segmented_text'], add_special_tokens=False).ids)
            prepared.append(item)
            if item['alignment_edits'] or item['normalization_changed']:
                audits.append({'split': split, 'source_row': row['source_row'], 'text': row['text'],
                               'segmented_text': item['segmented_text'], 'normalization_changed': item['normalization_changed'], 'edits': item['alignment_edits']})
            if index % 1000 == 0:
                log_event(None, 'preprocessing', split=split, rows=index, total=expected)
        write_jsonl(cache / f'{split}.jsonl', prepared)
        stats[split] = {'rows': len(rows), 'multi_chunk_rows': sum(len(p['chunks']) > 1 for p in prepared),
                        'alignment_edit_rows': sum(bool(p['alignment_edits']) for p in prepared),
                        'normalization_changed_rows': sum(p['normalization_changed'] for p in prepared),
                        'max_encoder_pieces': max(p['piece_count'] for p in prepared)}
    write_jsonl(cache / 'alignment_audit.jsonl', audits)
    save_json(cache / 'statistics.json', stats)
    assets = read_json(EXP / 'tools/vncorenlp_assets.json')
    inputs = read_json(PACKAGE / 'data/cache_spec.json')['files']
    inputs = {name: value for name, value in inputs.items() if name != 'data/prepared_inputs.jsonl'}
    manifest = {'protocol_sha256': sha(EXP / 'protocol.json'),
                'preprocessing_code_sha256': sha(EXP / 'preprocessing.py'),
                'versions': {n: versions()[n] for n in ['tokenizers', 'py-vncorenlp', 'pyjnius']},
                'inputs': inputs, 'assets': {name: info['sha256'] for name, info in assets['files'].items()},
                'vncorenlp_repository_commit': assets['repository_commit'],
                'files': {p.name: sha(p) for p in sorted(cache.iterdir()) if p.is_file() and p.name != 'manifest.json'},
                'gold_test_preprocessed': False, 'alternative_segmenter_used': False, 'statistics': stats}
    save_json(cache / 'manifest.json', manifest)
    log_event(None, 'cache_prepared', statistics=stats)
