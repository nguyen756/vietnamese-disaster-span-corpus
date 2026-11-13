"""Original numerical functions; imports isolated for portability. PRIVATE."""
from difflib import SequenceMatcher
import math,unicodedata
from .source_units import tokenize
from .pooling import chunk_ids,official_piece_ids,pooling_pairs

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

