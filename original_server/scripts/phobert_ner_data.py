"""PhoBERT word-segmented inputs aligned back to untouched source-token units.

The model receives text-only segmentation. Labels never determine tokenization,
chunk boundaries, pooling or inference. A sequence head predicts BIO labels on
the same original Unicode units as the CRF, including syllables within a merged
Vietnamese word, so segmentation cannot silently alter the annotation schema.
"""
from collections import Counter
import re

from train_crf_pilot import encode, tokenize


def segmented_view(text, words):
    """Map segmented visible characters to source positions; detect rewriting."""
    chars, mapping, cursor = [], [], 0
    for word in words:
        if chars:
            chars.append(" ")
            mapping.append(None)
        components = re.findall(r"\S+", word)
        for i, component in enumerate(components):
            if i:
                chars.append("_")
                mapping.append(None)
            for char in component:
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                if cursor >= len(text) or text[cursor] != char:
                    raise ValueError("Segmenter changed source characters")
                chars.append(char)
                mapping.append(cursor)
                cursor += 1
    if text[cursor:].strip():
        raise ValueError("Segmenter omitted source characters")
    return "".join(chars), mapping


def piece_offsets(offsets, mapping):
    result = []
    for start, end in offsets:
        positions = [position for position in mapping[start:end] if position is not None]
        # An inserted segmentation underscore can itself become a BPE piece.
        # It carries encoder context, but is not an original source character.
        result.append((min(positions), max(positions) + 1) if positions else None)
    return result


def chunk_ids(ids, maximum=254, stride=192):
    assert 0 < stride <= maximum
    result = []
    for first in range(0, len(ids), stride):
        last = min(first + maximum, len(ids))
        result.append({"first_piece": first, "input_ids": [0] + ids[first:last] + [2]})
        if last == len(ids):
            break
    return result


def official_piece_ids(ids, vocabulary_size=64001, unknown_id=3):
    """Converted BPE assets may append merge-only entries beyond model vocab.

    Official PhobertTokenizer maps those absent vocabulary entries to <unk>.
    Preserve their offsets and apply the same ID mapping; never resize the
    pretrained embedding table to fabricate vectors for conversion artifacts.
    """
    return [token_id if token_id < vocabulary_size else unknown_id for token_id in ids]


def pooling_pairs(units, offsets, chunks):
    """Sparse weighted mean of overlapping subwords for each source unit."""
    result, weights = [], Counter()
    for ci, chunk in enumerate(chunks):
        first = chunk["first_piece"]
        count = len(chunk["input_ids"]) - 2
        for pi in range(first, first + count):
            if offsets[pi] is None:
                continue
            start, end = offsets[pi]
            for ui, (a, b, _) in enumerate(units):
                overlap = min(end, b) - max(start, a)
                if overlap > 0:
                    result.append([ci, pi - first + 1, ui, overlap])
                    weights[ui] += overlap
    if len(weights) != len(units):
        raise ValueError("Source units are missing encoder representations")
    return [[ci, pi, ui, weight / weights[ui]] for ci, pi, ui, weight in result]


def prepare(row, segmenter, tokenizer):
    units, tags, unaligned = encode(row) if "label" in row else (tokenize(row["text"]), [], [])
    if unaligned:
        raise ValueError(f"Source labels not representable: {unaligned}")
    fallback = False
    try:
        view, mapping = segmented_view(row["text"], segmenter(row["text"]))
    except ValueError:
        # Do not normalize or rewrite the original text to conceal a mismatch.
        fallback = True
        view, mapping = segmented_view(row["text"], [unit[2] for unit in units])
    encoded = tokenizer.encode(view, add_special_tokens=False)
    offsets = piece_offsets(encoded.offsets, mapping)
    ids = official_piece_ids(encoded.ids)
    chunks = chunk_ids(ids)
    return {"source_row": row.get("source_row"), "units": units, "tags": tags,
            "segmented_text": view, "segmentation_fallback": fallback,
            "marker_only_pieces": sum(offset is None for offset in offsets),
            "converted_merge_entries_mapped_to_unk": sum(a != b for a, b in zip(encoded.ids, ids)),
            "piece_count": len(encoded.ids), "chunks": chunks,
            "pooling_pairs": pooling_pairs(units, offsets, chunks)}
