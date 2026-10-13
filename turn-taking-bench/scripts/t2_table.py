"""Per-model, per-condition Track 2 table from t2-run / t2-rescore / t2-combine summary dirs.

Usage: t2_table.py --gate-status TEXT DIR [DIR ...] > table.md
Each DIR holds t2-summary.json (one model, one caller condition; voices pooled by
template when combined). Runs from before caller conditions existed are clean.
When a clean summary and a degraded one exist for the same model and the same
caller voices, a second table gives the degraded-minus-clean deltas (point
estimates; robustness is reported as the delta from clean).
"""
import argparse
import json
from pathlib import Path

ORDER = ["clean", "telephone", "noise-20db", "noise-10db", "reverb", "dropouts"]


def f(v, p=2):
    return 'n/a' if v is None else f'{v:.{p}f}'


def ci(v, flip=False):
    if not v:
        return ''
    lo, hi = (1 - v[1], 1 - v[0]) if flip else v
    return f' [{lo:.2f}, {hi:.2f}]'


def d(a, b, p=2):
    return 'n/a' if a is None or b is None else f'{a - b:+.{p}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate-status', required=True)
    ap.add_argument('dirs', nargs='+', type=Path)
    a = ap.parse_args()
    summaries = []
    for dr in a.dirs:
        s = json.loads((dr / 't2-summary.json').read_text())
        s.setdefault('condition', 'clean')
        s['_voices'] = ','.join(s.get('caller_voices') or []) or '?'
        s['_model'] = s.get('model') or s.get('provider')
        summaries.append(s)
    summaries.sort(key=lambda s: (s['_model'], s['_voices'], ORDER.index(s['condition']) if s['condition'] in ORDER else 99))
    print(f"**Track 2 (experimental; separate from Track 1; no rank). Gate status: {a.gate_status}**\n")
    print('Interruption = barge-in yield; excess = matched yield - no-stimulus base rate at the same offsets. '
          'Backchannel = continue rate (1 - stop rate) vs the control continue rate. Caller condition = Track 1 '
          'service transform applied to the caller stream only (model output as received). 95% CIs: '
          'whole-conversation (template-cluster when voices pooled) percentile bootstrap, 2,000 draws, seed 7.\n')
    print('| model | caller condition | voices | yield rate | excess yield | stop ms p50 / p90 | backchannel continue | control continue | scored/injected/planned (barge) | (backchannel) | controls scored | top exclusions |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for s in summaries:
        i, b = s['interruption'], s['backchannel']
        lat = i['stop_ms']
        cont = None if b['stop_rate'] is None else 1 - b['stop_rate']
        base_cont = None if b['control_base_stop_rate'] is None else 1 - b['control_base_stop_rate']
        p50ci = ' [%.0f, %.0f]' % tuple(lat['p50_ci95']) if lat.get('p50_ci95') else ''
        p90ci = ' [%.0f, %.0f]' % tuple(lat['p90_ci95']) if lat.get('p90_ci95') else ''
        voids = {**i['void_reasons'], **{('bc:' + k): v for k, v in b['void_reasons'].items()}}
        row = (s['_model'], s['condition'], s['_voices'],
               f"{f(i['stop_rate'])}{ci(i['ci95']['stop_rate'])}",
               f"{f(i['excess_stop_rate'])}{ci(i['ci95']['excess_stop_rate'])}",
               f"{f(lat['p50'], 0)}{p50ci} / {f(lat['p90'], 0)}{p90ci} (n={lat['n']})",
               f"{f(cont)}{ci(b['ci95']['stop_rate'], flip=True)}",
               f"{f(base_cont)}{ci(b['ci95']['control_base_stop_rate'], flip=True)}",
               f"{i['n']}/{i['injected']}/{i['cells']}", f"{b['n']}/{b['injected']}/{b['cells']}",
               f"{s['controls']['scored']}/{s['controls']['cells']}",
               '; '.join(f'{k} {v}' for k, v in sorted(voids.items(), key=lambda kv: -kv[1])[:4]))
        print('| ' + ' | '.join(row) + ' |')
    clean = {(s['_model'], s['_voices']): s for s in summaries if s['condition'] == 'clean'}
    deltas = [(s, clean[(s['_model'], s['_voices'])]) for s in summaries
              if s['condition'] != 'clean' and (s['_model'], s['_voices']) in clean]
    if deltas:
        print('\nDelta from clean (same model, same caller voices; degraded minus clean, point estimates):\n')
        print('| model | caller condition | voices | yield rate | excess yield | stop ms p50 | backchannel continue | scored barge (cond / clean) |')
        print('|---|---|---|---|---|---|---|---|')
        for s, c in deltas:
            i, ci_ = s['interruption'], c['interruption']
            b, cb = s['backchannel'], c['backchannel']
            cont = lambda x: None if x['stop_rate'] is None else 1 - x['stop_rate']
            print('| ' + ' | '.join((s['_model'], s['condition'], s['_voices'], d(i['stop_rate'], ci_['stop_rate']),
                                     d(i['excess_stop_rate'], ci_['excess_stop_rate']),
                                     d(i['stop_ms']['p50'], ci_['stop_ms']['p50'], 0), d(cont(b), cont(cb)),
                                     f"{i['n']} / {ci_['n']}")) + ' |')


if __name__ == '__main__':
    main()
