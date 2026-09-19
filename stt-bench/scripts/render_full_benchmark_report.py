#!/usr/bin/env python3
"""Render already-replayed full-run results. Never submits audio or changes scores."""
import argparse
from datetime import datetime
import html
import json
from pathlib import Path


def percent(value):
    return 'Unavailable' if value is None else f'{value:.2%}'


def milliseconds(value):
    return 'Unavailable' if value is None else f'{value:,.0f} ms'


def elapsed(start, end):
    if not start or not end:
        return 'Unavailable'
    minutes = (datetime.fromisoformat(end.replace('Z', '+00:00')) -
               datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds() / 60
    return f'{minutes:,.1f} minutes'


def sections(report):
    execution = report.get('execution', {})
    status = execution.get('status', 'partial')
    yield 'text', f"Status: {status}. Generated {report['generated_at']}."
    if status != 'complete':
        yield 'text', 'PARTIAL RESULTS: this is not a completed benchmark. Failed and unattempted items are retained in the coverage tables.'
    yield 'text', ('Each model has the same 1,000 public clips and eight complete private recordings. '
                   'These measurements were collected with concurrent workers. Word error rate (WER) is '
                   'total substitutions, insertions and deletions divided by total reference words. '
                   'WER covers valid completed items only; the coverage columns show what is missing.')
    yield 'text', (f"Elapsed runtime: {elapsed(execution.get('startedAt'), execution.get('finishedAt') or report['generated_at'])}. "
                   f"All compute confirmed stopped: {'yes' if execution.get('all_compute_stopped') else 'no'}. "
                   'Collection archives are checksum-checked and scores are replayed locally before inclusion.')
    for model, data in report['models'].items():
        yield 'heading', model
        live = execution.get('models', {}).get(model, {})
        yield 'text', (f"Provider block: {live.get('blocked') or 'none'}. Private portion blocked: {bool(live.get('privateBlocked'))}. "
                       f"Dispatch ceiling: {live.get('ceiling', 'unavailable')}; peak overlapping captures: {data['peak_capture_overlap']}. "
                       'Capture overlap includes connecting and finishing requests; it is not proof that the provider accepted that many streams.')
        if model.startswith('assemblyai'):
            yield 'text', 'AssemblyAI configuration: universal-3-5-pro, mode=min_latency, language_codes=[en]. New stream starts are globally limited to five per minute with a 62-second safety window. Existing streams may overlap; the worker ceiling is separate from the session-start rate. No prompts, keyterms, agent context, or silence overrides. Original audio is sent in 60 ms packets (80/100 ms end packets as needed), preserving every sample. Private word timing uses the actual delivery of the packet containing the word end. Packet-aware pacing gates apply; wire timing differs from 20 ms providers.'
        attempts = [a for item in data['items'] for a in item['attempts']]
        starts = [a['started_at'] for a in attempts if a.get('started_at')]
        ends = [a['finished_at'] for a in attempts if a.get('finished_at')]
        if starts and ends:
            yield 'text', f"Collected attempts span {elapsed(min(starts), max(ends))}, from {min(starts)} to {max(ends)}. This excludes setup before the first request and collection after the last request."
        headers = ['Dataset', 'First-pass usable', 'Recovered', 'Final usable / planned', 'Failed', 'Unattempted', 'First-pass WER', 'Final WER']
        rows = []
        for cohort, group in data['groups'].items():
            rows.append([cohort, group['first_attempt_successful'], group['recovered'],
                         f"{group['successful']} / {group['planned']}", group['failed'], group['unattempted'],
                         percent(group['first_attempt_valid_wer']['wer']), percent(group['final_wer']['wer'])])
        yield 'table', (headers, rows)
        group = data['groups']['combined']
        yield 'text', (f"Attempts: {group['attempts']}; failed attempts: {group['failed_attempts']}; pacing-invalid attempts: {group['pacing_failures']}. "
                       f"Submitted audio: {group['submitted_seconds'] / 60:,.2f} minutes in this variant; "
                       f"{data.get('all_variants_submitted_seconds', group['submitted_seconds']) / 60:,.2f} minutes including prior transport validation. "
                       f"Failure classes: {json.dumps(group['failure_classes'], sort_keys=True)}.")
        no_audio = sum(not a['pacing']['valid'] and a['sent_audio_seconds'] == 0 for a in attempts)
        with_audio = sum(not a['pacing']['valid'] and a['sent_audio_seconds'] > 0 for a in attempts)
        yield 'text', f"Pacing detail: {no_audio} invalid attempts sent no audio, so transmission pacing was unavailable; {with_audio} invalid attempts sent some audio. Per-attempt gate reasons remain in the JSON."
        yield 'subheading', 'Public transcript accuracy at fixed deadlines — first attempts only'
        yield 'text', 'Accuracy below means 1 − WER and can be negative when insertions exceed reference words. Available interim text is included. Recovery attempts never replace these observations.'
        rows = []
        for deadline in data['groups']['public']['deadlines']:
            wer = deadline['wer']
            rows.append(['Speech end' if deadline['deadline_ms'] == 0 else f"+{deadline['deadline_ms']} ms",
                         percent(wer), percent(None if wer is None else 1 - wer),
                         f"{deadline['measured_clips']} / {deadline['planned_clips']}",
                         deadline['missing_clips'], deadline['pacing_invalid_clips'], deadline['provisional_clips']])
        yield 'table', (['Deadline', 'WER', '1 − WER', 'Measured / planned', 'Missing', 'Pacing invalid', 'Provisional text'], rows)
        yield 'subheading', 'Private word-finalization delay — first attempts only'
        private = data['groups']['private']['word_finalization']
        measured = private['n']
        reference = private['reference_words']
        yield 'text', ('Delay is the receipt time of a correctly finalized word minus the actual delivery time of its reference word-end audio frame. '
                       'Only unambiguous word matches from valid first attempts are measured. '
                       f"Measured words: {measured:,} / {reference:,} reference words in attempted recordings "
                       f"({percent(measured / reference if reference else None)}). "
                       'Unattempted recordings do not enter this word denominator; their count is shown above.')
        yield 'table', (['Median', '90th percentile', '95th percentile'], [[milliseconds(private.get('p50_ms')), milliseconds(private.get('p90_ms')), milliseconds(private.get('p95_ms'))]])
        yield 'text', f"Private word exclusions: {json.dumps(private['exclusions'], sort_keys=True)}."
        yield 'subheading', 'Stream completion after the reference audio boundary — first attempts only'
        yield 'text', 'Public timing starts at the benchmark speech-end boundary. Private timing starts at the end of the original recording, before the added terminal silence. The original private recordings include quiet tails after their final spoken words. These values measure stream completion with the configured finalization behavior.'
        rows = []
        for cohort in ('public', 'private'):
            timing = data['groups'][cohort]['completion_first_attempt']
            rows.append([cohort, timing['n'], milliseconds(timing['p50_ms']), milliseconds(timing['p90_ms'])])
        yield 'table', (['Dataset', 'Measured items', 'Median', '90th percentile'], rows)
        if live.get('reductions'):
            yield 'text', 'Recorded concurrency reductions and account-limit reconciliation:'
            yield 'table', (['Time (UTC)', 'From', 'To', 'Reason'], [[r['at'], r.get('from', '—'), r['to'], r['reason']] for r in live['reductions']])
        if model in report.get('private_variants', {}):
            yield 'text', ('Gradium private mode: consecutive provider sessions of up to 270 seconds. Every original source frame is sent once in order. '
                           'Sessions reset recognition context and add silence tails and reconnect gaps. Pacing gates apply inside each session; gaps are recorded separately. '
                           'This private latency mode is not directly interchangeable with an uninterrupted session. Prior transport variants and their failed attempts remain in results.json.')
    yield 'heading', 'Evidence and interpretation'
    yield 'text', ('The original run plan, dataset hashes, code amendments, assignments, command identities and stopped-compute receipts remain alongside this report. '
                   'All item records distinguish successful, failed and unattempted work. Recovery can improve final accuracy and coverage but cannot replace first-attempt latency or deadline accuracy. '
                   'Percentiles are computed from individual observations, not averaged across workers. Supplied reference transcripts and word times were not independently checked by listening.')


def render(report, root):
    md = ['# Full concurrent STT benchmark', '']
    body = ['<h1>Full concurrent STT benchmark</h1>', '<nav><a href="results.json">Complete scores and item records</a> · <a href="plan.json">Frozen plan</a> · <a href="controller.json">Execution evidence</a> · <a href="RESULTS.md">Markdown report</a></nav>']
    if (root / 'RUN_NOTES.md').exists():
        md += ['[Execution notes, preserved failures, and runtime amendments](RUN_NOTES.md)', '']
        body.append('<p><a href="RUN_NOTES.md">Execution notes, preserved failures, and runtime amendments</a></p>')
    for kind, value in sections(report):
        if kind == 'table':
            headers, rows = value
            md += ['| ' + ' | '.join(map(str, headers)) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
            md += ['| ' + ' | '.join(str(cell).replace('|', '\\|') for cell in row) + ' |' for row in rows]
            body.append('<div class="table"><table><thead><tr>' + ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(cell))}</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>')
        else:
            prefix, tag = {'heading': ('## ', 'h2'), 'subheading': ('### ', 'h3'), 'text': ('', 'p')}[kind]
            md.append(prefix + value)
            body.append(f'<{tag}>{html.escape(value)}</{tag}>')
        md.append('')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full concurrent STT benchmark</title><style>body{font:16px system-ui;max-width:1180px;margin:40px auto;padding:0 24px;color:#202522;background:#fafaf7}h1{font-size:34px}h2{margin-top:56px;border-top:2px solid #ccd3cc;padding-top:24px}h3{margin-top:30px}p{line-height:1.65;max-width:100ch}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}th{background:#eef1eb}a{color:#206244}nav{line-height:2}</style><body>' + ''.join(body) + '</body></html>'
    (root / 'RESULTS.md').write_text('\n'.join(md))
    (root / 'index.html').write_text(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    render(json.loads((args.root / 'results.json').read_text()), args.root)
