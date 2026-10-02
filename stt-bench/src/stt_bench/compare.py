"""Secondary paired comparison; never hide provider-specific coverage."""
import json
import numpy as np

from .data import write_json
from .measurement import DEADLINES_MS


def paired_difference(left, right, index, iterations=2000, seed=42):
    groups = {}
    for a, b in zip(left, right):
        group = a.get('dependency_group')
        if not group or group != b.get('dependency_group'):
            return dict(status='unavailable_missing_or_different_dependency_groups')
        w1, w2 = a['deadlines'][index]['word_errors'], b['deadlines'][index]['word_errors']
        if w1['reference_words'] != w2['reference_words']:
            raise ValueError('Paired reference word counts differ')
        totals = groups.setdefault(group, [0, 0, 0])
        totals[0] += sum(w1[k] for k in ('substitutions', 'insertions', 'deletions'))
        totals[1] += sum(w2[k] for k in ('substitutions', 'insertions', 'deletions'))
        totals[2] += w1['reference_words']
    if len(groups) < 2 or any(v[2] == 0 for v in groups.values()):
        return dict(status='unavailable_insufficient_groups')
    values = np.array(list(groups.values()))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        a, b, n = values[rng.integers(len(values), size=len(values))].sum(axis=0)
        differences.append((a-b)/n)
    a, b, n = values.sum(axis=0)
    low, high = np.percentile(differences, [2.5, 97.5])
    return dict(status='available', left_minus_right_wer=float((a-b)/n), lower=float(low), upper=float(high),
                confidence=.95, groups=len(groups), iterations=iterations, seed=seed,
                ordering_resolved=bool(low > 0 or high < 0))


def compare_reports(left_path, right_path, out):
    a, b = (json.loads(p.read_text()) for p in (left_path, right_path))
    if a.get('measurement_version') != b.get('measurement_version') or a.get('measurement_version') not in (3, 4):
        raise ValueError('Paired comparison requires matching v3 or v4 measurement reports')
    keys = ('manifest_sha256', 'normalization', 'scoring_source_hashes', 'review')
    keys += ('shared_measurement_contract',) if a['measurement_version'] == 4 else ('capture_protocol',)
    for key in keys:
        if a.get(key) != b.get(key):
            raise ValueError(f'Paired reports must use identical {key}')
    if not all(a.get('capture_protocol', {}).get('source_hashes', {}).values()):
        raise ValueError('Paired comparison requires recorded capture protocol source hashes')
    if a['mode'] != 'live' or b['mode'] != 'live':
        raise ValueError('Dry runs cannot be provider comparisons')
    right = {r['clip_id']: r for r in b['clips']}
    if set(right) != {r['clip_id'] for r in a['clips']}:
        raise ValueError('Paired reports require identical clip IDs')
    rows = []
    for condition in sorted({r['condition'] for r in a['clips']}):
        all_left = [r for r in a['clips'] if r['condition'] == condition]
        for i, deadline in enumerate(DEADLINES_MS):
            def eligible(r):
                d = r['deadlines'][i]
                return d['word_errors'] is not None and d['pacing_valid']
            common = [r for r in all_left if eligible(r) and eligible(right[r['clip_id']])]
            rows.append(dict(condition=condition, deadline_ms=deadline, planned_clips=len(all_left),
                             left_eligible=sum(eligible(r) for r in all_left),
                             right_eligible=sum(eligible(right[r['clip_id']]) for r in all_left),
                             common_clips=len(common), paired=paired_difference(common, [right[r['clip_id']] for r in common], i)))
    result = dict(left={'identity': a['run_identity'], 'run_started_at': a['run_started_at'], 'configuration': a['run_configuration']},
                  right={'identity': b['run_identity'], 'run_started_at': b['run_started_at'], 'configuration': b['run_configuration']}, comparisons=rows,
                  note='Secondary common-input comparison only. Intervals are unadjusted pairwise 95% intervals, not a multi-provider ranking. Exclusions and reference quality still apply.')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'comparison.json', result)
    return result
