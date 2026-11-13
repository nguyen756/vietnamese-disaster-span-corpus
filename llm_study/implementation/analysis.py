"""E1/E2/E4/E5/E6 scoring and reproducible document-cluster uncertainty."""
from collections import Counter
import random
from common import LABELS, RUNS, digest
from metrics import score as flat_score


def assignment_objective(predicted, reference):
    """Rectangular Hungarian, with dummy columns; works for nonflat oracle spans."""
    p, g = sorted(set(map(tuple, predicted))), sorted(set(map(tuple, reference)))
    if not p or not g:
        return (0, 0)
    scale = min(len(p), len(g)) + 1
    weights = [[max(0, min(a[1], b[1]) - max(a[0], b[0])) if a[2] == b[2] else 0
                for b in g] for a in p]
    costs = [[-(w * scale + 1) if w else 0 for w in row] + [0] * len(p) for row in weights]
    n, m = len(p), len(g) + len(p)
    u, v, matched, way = [0]*(n+1), [0]*(m+1), [0]*(m+1), [0]*(m+1)
    for i in range(1, n+1):
        matched[0], j0 = i, 0
        minimum, used = [float('inf')]*(m+1), [False]*(m+1)
        while True:
            used[j0] = True
            i0, delta, j1 = matched[j0], float('inf'), 0
            for j in range(1, m+1):
                if not used[j]:
                    current = costs[i0-1][j-1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m+1):
                if used[j]:
                    u[matched[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if not matched[j0]:
                break
        while j0:
            j1 = way[j0]
            matched[j0], j0 = matched[j1], j1
    intersections = [weights[matched[j]-1][j-1] for j in range(1, len(g)+1) if matched[j]]
    return sum(intersections), sum(w > 0 for w in intersections)


def ratios(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn, 'support': tp+fn,
            'precision': tp/(tp+fp) if tp+fp else 0,
            'recall': tp/(tp+fn) if tp+fn else 0,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}


def general_score(reference, predicted, mode='strict'):
    if len(reference) != len(predicted):
        raise ValueError('Row count mismatch')
    counts = {label: [0, 0, 0] for label in LABELS}
    for row, found in zip(reference, predicted):
        for label in LABELS:
            p = set(tuple(s) for s in found if s[2] == label)
            g = set(tuple(s) for s in row['label'] if s[2] == label)
            tp = len(p & g) if mode == 'strict' else assignment_objective(p, g)[1]
            for i, v in enumerate([tp, len(p)-tp, len(g)-tp]):
                counts[label][i] += v
    return {'micro': ratios(*[sum(c[i] for c in counts.values()) for i in range(3)]),
            'per_label': {k: ratios(*v) for k, v in counts.items()}}


def paired_cluster_bootstrap(reference, predictions, draws=1000, seed=20261112):
    """Precompute row counts; identical document draws for every run and metric."""
    if not reference:
        raise ValueError('Empty reference')
    documents = sorted({r['document_id'] for r in reference})
    groups = {d: [i for i, r in enumerate(reference) if r['document_id'] == d] for d in documents}
    rng = random.Random(seed)
    indices = [[rng.randrange(len(documents)) for _ in documents] for _ in range(draws)]
    names = ['micro', *LABELS]
    output, distributions = {}, {}
    for run, predicted in predictions.items():
        output[run], distributions[run] = {}, {}
        for mode in ['strict', 'overlap']:
            per_doc = []
            for doc in documents:
                ix = groups[doc]
                result = flat_score([reference[i] for i in ix], [predicted[i] for i in ix], LABELS, mode)
                per_doc.append([[(result['micro'] if n == 'micro' else result['per_label'][n])[k]
                                for k in ['tp', 'fp', 'fn']] for n in names])
            values = {n: {k: [] for k in ['precision', 'recall', 'f1']} for n in names}
            for sample in indices:
                for j, name in enumerate(names):
                    metric = ratios(*[sum(per_doc[i][j][k] for i in sample) for k in range(3)])
                    for k in values[name]:
                        values[name][k].append(metric[k])
            def interval(values):
                values = sorted(values)
                return [values[int((len(values)-1)*0.025)], values[int((len(values)-1)*0.975)]]
            output[run][mode] = {n: {k: interval(v) for k, v in metrics.items()} for n, metrics in values.items()}
            distributions[run][mode] = values
    return {'method': 'percentile document-cluster bootstrap; all rows of sampled documents with multiplicity',
            'draws': draws, 'seed': seed, 'documents': documents, 'draw_indices': indices,
            'draw_indices_sha256': digest(indices), 'intervals_95': output,
            'distributions': distributions,
            'zero_support': 'precision/recall/F1 = 0 when denominator is zero; per label too',
            'limits': 'Document IDs need not be independent events; historical URL overlap remains.'}


def oracle(reference, primary, routed):
    output, collisions, nonflat = [], 0, 0
    for i, (row, spans) in enumerate(zip(reference, primary)):
        replacements = []
        for span in spans:
            key = (i, *span)
            if key not in routed:
                replacements.append(tuple(span))
                continue
            candidates = [(min(span[1], g[1])-max(span[0], g[0]), tuple(g)) for g in row['label']]
            candidates = [(n, g) for n, g in candidates if n > 0]
            if candidates:
                replacements.append(min(candidates, key=lambda v: (-v[0], v[1]))[1])
        dedup = sorted(set(replacements))
        collisions += len(replacements)-len(dedup)
        nonflat += any(a[1] > b[0] for a, b in zip(dedup, dedup[1:]))
        output.append([list(s) for s in dedup])
    return output, {'identical_replacement_collisions': collisions, 'nonflat_rows': nonflat}


def gates(reference, compiled, token_counts=None):
    if set(compiled) != set(RUNS) or any(len(compiled[r]) != len(reference) for r in RUNS):
        raise ValueError('Full fixed K=4 grid required')
    if any(row.get('outcome') == 'not_run' for data in compiled.values() for row in data):
        raise ValueError('Unfinished grid cannot be simulated')
    primary = [r['label'] for r in compiled['M1-P2']]
    pool = [(i, *s) for i, spans in enumerate(primary) for s in spans]
    votes = Counter()
    for run in RUNS:
        for i, row in enumerate(compiled[run]):
            votes.update((i, *s) for s in set(map(tuple, row['label'])))
    reject_rows = {i for i, r in enumerate(compiled['M1-P2']) if r['structural_rejection']}
    routed = {'G0': set(), 'G1': {s for s in pool if s[0] in reject_rows},
              **{f'G2({t})': {s for s in pool if votes[s] <= t} for t in [1, 2, 3]},
              'route_all': set(pool)}
    def measure(route):
        predictions, metadata = oracle(reference, primary, route)
        sentences = sorted({s[0] for s in route})
        tokens = None
        if token_counts is not None and all(token_counts.get(reference[i]['id']) is not None for i in sentences):
            tokens = sum(token_counts[reference[i]['id']] for i in sentences)
        return {'routed_spans': len(route), 'budget': len(route)/len(pool) if pool else None,
                'routed_sentences': len(sentences), 'routed_tokens': tokens,
                'routing_keys': sorted(route), 'strict': general_score(reference, predictions),
                'overlap': general_score(reference, predictions, 'overlap'), **metadata}
    result = {name: measure(route) for name, route in routed.items()}
    for t in [1, 2, 3]:
        rng, draws = random.Random(20261112+t), []
        count = len(routed[f'G2({t})'])
        for index in range(100):
            selected = set(rng.sample(pool, count))
            measured = measure(selected)
            draws.append({'draw': index, **measured})
        result[f'R(G2({t}))'] = {'rng_seed': 20261112+t, 'draws': draws,
            'mean_micro': {mode: {metric: sum(d[mode]['micro'][metric] for d in draws)/100
                                         for metric in ['precision', 'recall', 'f1']}
                           for mode in ['strict', 'overlap']},
            'mean_per_label': {mode: {label:{metric:sum(d[mode]['per_label'][label][metric] for d in draws)/100
                                           for metric in ['precision','recall','f1']} for label in LABELS}
                               for mode in ['strict','overlap']}}
    support = sum(len(r['label']) for r in reference)
    missed = sum(not any(min(g[1], p[1])-max(g[0], p[0]) > 0 for p in found)
                 for row, found in zip(reference, primary) for g in row['label'])
    return {'oracle_simulation_only': True, 'K': 4, 'primary_spans': len(pool), 'gates': result,
            'G1_rejected_rows_with_empty_primary': sorted(i for i in reject_rows if not primary[i]),
            'reference_spans_uncovered': missed,
            'coverage_recall_upper_bound': (support-missed)/support if support else None,
            'actual_route_all_recall': result['route_all']['strict']['micro']['recall'],
            'failures_by_run': {run: dict(Counter(r['outcome'] for r in data)) for run, data in compiled.items()},
            'token_count_basis': 'explicit separately measured tokenizer counts, otherwise null',
            'tie_policy': 'maximum positive intersection of ANY category, ascending (start,end,category)',
            'no_observed_human_correction_or_time': True}
