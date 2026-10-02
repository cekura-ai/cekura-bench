"""Render the captured pilot comparison without any network calls."""
import argparse
import csv
import json
from pathlib import Path

from finalization_pilot import TAP_CLIP, summarize


def number(value, scale=1, suffix=''):
    return 'unavailable' if value is None else f'{value*scale:.2f}{suffix}'


def render(root):
    data = json.loads((root / 'results.json').read_text())
    recomputed = summarize(data['observations'])
    if recomputed != data:
        raise ValueError('Pilot summary differs from its observations')
    lines = ['# Finalization settings pilot results', '',
             'Diagnostic comparison on Vercel. This report does not change the leaderboard.', '',
             'Both settings received the same fixed public clips. Scores below use only clips valid under both settings.', '',
             '## Paired results', '',
             '| Model | Paired clips | Reference words | WER: old → new | Last final text, median: old → new | Trailing inserted words: old → new |',
             '| --- | ---: | ---: | --- | --- | --- |']
    csv_rows = []
    for model, record in data['models'].items():
        old, new = (record['variants'][v] for v in ('baseline', 'candidate'))
        lines.append(f"| {model} | {record['paired_clips']} | {old['paired_wer']['reference_words']} | "
                     f"{number(old['paired_wer']['wer'],100,'%')} → {number(new['paired_wer']['wer'],100,'%')} | "
                     f"{number(old['last_final_text']['p50_ms'],suffix=' ms')} → {number(new['last_final_text']['p50_ms'],suffix=' ms')} | "
                     f"{old['trailing_insertions']} → {new['trailing_insertions']} |")
        for variant, metrics in record['variants'].items():
            csv_rows.append(dict(model=model, variant=variant, paired_clips=record['paired_clips'],
                **{k: metrics[k] for k in ('planned','attempted','usable','failed','not_run','trailing_insertions')},
                **metrics['paired_wer'], last_final_text_p50_ms=metrics['last_final_text']['p50_ms'],
                last_final_text_p90_ms=metrics['last_final_text']['p90_ms'],
                last_final_text_n=metrics['last_final_text']['n']))
    lines += ['', '## Reliability', '',
              '| Model | Setting | Planned | Attempted | Usable | Failed | Not run | Failure reasons |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for model, record in data['models'].items():
        for variant, m in record['variants'].items():
            lines.append(f"| {model} | {variant} | {m['planned']} | {m['attempted']} | {m['usable']} | "
                         f"{m['failed']} | {m['not_run']} | {m['failure_reasons']} |")
    random_only = summarize([r for r in data['observations'] if r['clip_id'] != TAP_CLIP])
    lines += ['', '## Sensitivity: exclude the deliberately selected repetition example', '',
              '| Model | Paired clips | WER: old → new |', '| --- | ---: | --- |']
    for model, record in random_only['models'].items():
        old, new = (record['variants'][v]['paired_wer']['wer'] for v in ('baseline','candidate'))
        lines.append(f"| {model} | {record['paired_clips']} | {number(old,100,'%')} → {number(new,100,'%')} |")
    lines += ['', '## Interpretation limits', '',
              '- A small sample cannot establish a general ranking or guarantee that an intermittent failure is fixed.',
              '- One clip was deliberately chosen because it failed historically. The sensitivity table excludes it.',
              '- Speechmatics changes both max_delay and the speech-end signal. This experiment tests their combined effect.',
              '- AssemblyAI uses the default streaming profile in this pilot. The dashboard uses min_latency with an English language bias; this pilot does not validate that published profile.',
              '- Original shared-process capture and isolated-process continuation are recorded separately in capture_phase. Interrupted sessions remain failures; they were not retried.',
              '- Latency is the last final text received relative to speech end. Later extra text extends it; negative values mean text finalized before speech end.',
              '- Timings are paired on Vercel. They are not directly compared with historical dashboard timings.',
              '- Trailing insertions are alignment results. They are not a verified hallucination rate.',
              '- All raw text and timestamps remain in the evidence archive. No silence-tail subtraction is applied.', '']
    (root / 'comparison.md').write_text('\n'.join(lines))
    with (root / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    render(args.out)
