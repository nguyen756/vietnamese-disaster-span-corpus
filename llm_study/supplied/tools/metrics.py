"""Strict sets and exact one-to-one maximum-total-intersection matching.

References and predictions must be flat nonoverlapping spans. Positive-overlap
matches cannot cross in this setting, so ordered dynamic programming gives the
same global optimum as unrestricted bipartite assignment. Tuple optimization
maximizes total intersecting characters first, positive pair count second.
"""
from collections import Counter


def overlap_objective(predicted, reference):
    p, g = sorted(predicted), sorted(reference)
    for spans in [p, g]:
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), 'Flat nonoverlap required'
    table = [[(0, 0) for _ in range(len(g) + 1)] for _ in range(len(p) + 1)]
    for i, a in enumerate(p, 1):
        for j, b in enumerate(g, 1):
            candidates = [table[i - 1][j], table[i][j - 1]]
            intersection = min(a[1], b[1]) - max(a[0], b[0])
            if intersection > 0 and a[2] == b[2]:
                value, count = table[i - 1][j - 1]
                candidates.append((value + intersection, count + 1))
            table[i][j] = max(candidates)
    return table[-1][-1]


def score(reference, predicted, labels, mode='strict'):
    assert len(reference) == len(predicted)
    counts = {label: Counter() for label in labels}
    for row, found in zip(reference, predicted):
        truth = row['label']
        for label in labels:
            p, g = [s for s in found if s[2] == label], [s for s in truth if s[2] == label]
            if mode == 'strict':
                tp = len(set(map(tuple, p)) & set(map(tuple, g)))
            else:
                _, tp = overlap_objective(p, g)
            counts[label].update(tp=tp, fp=len(p) - tp, fn=len(g) - tp)

    def ratios(count):
        tp, fp, fn = [count[k] for k in ['tp', 'fp', 'fn']]
        return {'tp': tp, 'fp': fp, 'fn': fn, 'support': tp + fn,
                'precision': tp / (tp + fp) if tp + fp else 0,
                'recall': tp / (tp + fn) if tp + fn else 0,
                'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
    return {'micro': ratios(sum(counts.values(), Counter())),
            'per_label': {label: ratios(v) for label, v in counts.items()}}


def self_check():
    # One high-overlap pair must beat two low-overlap pairs.
    assert overlap_objective([[0, 1, 'A'], [1, 102, 'A']], [[0, 100, 'A'], [101, 102, 'A']]) == (99, 1)
    # Equal total intersection prefers more positive pairs.
    assert overlap_objective([[0, 1, 'A'], [1, 5, 'A']], [[0, 3, 'A'], [4, 5, 'A']]) == (2, 2)
    import random
    rng = random.Random(20261111)
    for _ in range(100):
        a, b = [], []
        for dest in [a, b]:
            cuts = sorted(rng.sample(range(30), 6))
            dest.extend([[cuts[i], cuts[i + 1], 'A'] for i in range(0, 6, 2)])
        def brute(i, used):
            if i == len(a):
                return (0, 0)
            values = [brute(i + 1, used)]
            for j, y in enumerate(b):
                length = min(a[i][1], y[1]) - max(a[i][0], y[0])
                if j not in used and length > 0:
                    x, n = brute(i + 1, used | {j})
                    values.append((length + x, 1 + n))
            return max(values)
        assert overlap_objective(a, b) == brute(0, set())
    return {'passed': True, 'cardinality_first_counterexample_checked': True,
            'tie_pair_count_checked': True, 'exhaustive_assignment_comparisons': 100}
