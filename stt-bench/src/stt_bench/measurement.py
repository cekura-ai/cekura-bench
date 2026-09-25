"""First-request deadline observations and reproducible grouped uncertainty."""
from collections import Counter

import numpy as np

from .providers import transcript_at
from .entities import aggregate_values, value_errors
from .score import aggregate_wer, word_errors

DEADLINES_MS = (0, 250, 500, 1000)


def deadline_observations(events, reference, entities, attempt, mode, config=None):
    t0 = next((e['time_seconds'] for e in events if e['kind'] == 'speech_end'), None)
    last = max((e['time_seconds'] for e in events), default=-1)
    errors = [e['time_seconds'] for e in events if
        (e['kind'] == 'error' and e.get('error_type') != 'Interrupted') or
        (isinstance(e.get('message'), dict) and e['message'].get('type') == 'Error')]
    interrupted = [e['time_seconds'] for e in events if e.get('error_type') == 'Interrupted']
    closed = [e['time_seconds'] for e in events if
              (isinstance(e.get('message'), dict) and e['message'].get('type') == 'Metadata') or
              e['kind'] == 'provider_terminal']
    observations = []
    for deadline in DEADLINES_MS:
        state = dict(text='', final_text='', partial_text='', provisional=False, reconstruction_status='supported')
        at = t0 + deadline / 1000 if t0 is not None else None
        if mode != 'live':
            status = 'dry_run'
        elif attempt is None:
            status = 'not_attempted'
        elif interrupted and (at is None or min(interrupted) <= at):
            status = 'locally_interrupted'
        elif not attempt['model_verified'] and not errors:
            status = 'model_unverified'
        elif at is None:
            status = 'failed_before_speech_end' if errors else 'missing_speech_end'
        else:
            state = transcript_at(events, at, config)
            if state['reconstruction_status'] != 'supported':
                status = state['reconstruction_status']
            elif last < at and not errors and not closed:
                status = 'observation_window_incomplete'
            elif not attempt['model_verified'] and state['text']:
                status = 'model_unverified'
            elif any(t <= at for t in errors):
                status = 'request_failed'
            else:
                status = 'text_available' if state['text'] else 'no_text_yet'
        usable = status in ('failed_before_speech_end', 'request_failed', 'text_available', 'no_text_yet')
        observations.append(dict(deadline_ms=deadline, **state, status=status,
                                 pacing_valid=bool(attempt and attempt['pacing']['valid']),
                                 word_errors=word_errors(reference, state['text']) if usable else None,
                                 entity_values=value_errors(reference, state['text'], entities) if usable else None))
    return observations


def grouped_interval(rows, words, *, iterations=2000, seed=42):
    """Ratio-of-totals bootstrap; dependency_group is assigned by dataset owners."""
    if not rows or any(not r.get('dependency_group') for r in rows):
        return dict(status='unavailable_missing_dependency_groups', confidence=0.95)
    groups = {}
    for row in rows:
        w = words(row)
        if w is None:
            return dict(status='unavailable_missing_observations', confidence=0.95)
        group = groups.setdefault(row['dependency_group'], [0, 0])
        group[0] += w['substitutions'] + w['insertions'] + w['deletions']
        group[1] += w['reference_words']
    if len(groups) < 2 or any(n == 0 for _, n in groups.values()):
        return dict(status='unavailable_insufficient_groups', groups=len(groups), confidence=0.95)
    values = np.array(list(groups.values()))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        totals = values[rng.integers(len(values), size=len(values))].sum(axis=0)
        samples.append(totals[0] / totals[1])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return dict(status='available', confidence=0.95, lower=float(lo), upper=float(hi),
                groups=len(groups), iterations=iterations, seed=seed,
                method='dependency-group bootstrap of corpus error/reference-word totals')


def summarize_deadlines(rows):
    summaries = []
    for index, deadline in enumerate(DEADLINES_MS):
        usable = [r for r in rows if r['deadlines'][index]['word_errors'] is not None]
        controlled = [r for r in usable if r['deadlines'][index]['pacing_valid']]
        texts = [r['deadlines'][index] for r in usable]
        summaries.append(dict(deadline_ms=deadline, planned_clips=len(rows), measured_clips=len(usable),
                              missing_clips=len(rows)-len(usable),
                              pacing_invalid_clips=sum(not s['pacing_valid'] for s in texts),
                              provisional_clips=sum(s['provisional'] for s in texts),
                              status_counts=dict(Counter(r['deadlines'][index]['status'] for r in rows)),
                              **aggregate_wer([s['word_errors'] for s in texts]),
                              entity_values=aggregate_values([s['entity_values'] for s in texts]),
                              controlled_subset=dict(clips=len(controlled), **aggregate_wer([r['deadlines'][index]['word_errors'] for r in controlled])),
                              confidence_interval=grouped_interval(usable, lambda r: r['deadlines'][index]['word_errors']),
                              headline_eligible=bool(rows and len(usable) == len(rows) and len(controlled) == len(rows)
                                                     and all(r.get('human_review_verified') for r in rows))))
    return summaries
