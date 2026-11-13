"""Original numerical functions; imports isolated for portability. PRIVATE."""
from collections import Counter

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

