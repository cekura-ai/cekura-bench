"""Build an offline comparison from saved final summaries. Never calls providers.

Run from any directory: python3 scripts/build_benchmark_html.py
Only the HTML output is written; source results and benchmark code stay untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3fe50170d520c951957b86996ef082a6ab87b394"
MANIFEST = ROOT / "datasets/pipecat-stt-benchmark" / REVISION / "full/manifest.json"
DEADLINES = (0, 250, 500, 1000)
COUNTS = ("substitutions", "insertions", "deletions", "reference_words")
# Explicit run selection avoids old smoke results, failed runs, and moving status files.
MODELS = (
    ("deepgram-nova-2", "Nova-2", "Deepgram", "#176455", "vocera-deepgram-nova-2-20260912-unformatted"),
    ("deepgram-nova-3", "Nova-3", "Deepgram", "#168274", "vocera-deepgram-nova-3-20260912"),
    ("deepgram-flux-en", "Flux English", "Deepgram", "#526a25", "vocera-deepgram-flux-en-20260912"),
    ("deepgram-flux-multilingual", "Flux Multilingual", "Deepgram", "#8b6713", "vocera-deepgram-flux-multilingual-20260912"),
    ("openai-gpt-realtime-whisper", "GPT Realtime Whisper", "OpenAI", "#334f90", "vocera-openai-gpt-realtime-whisper-20260912"),
    ("openai-gpt-4o-transcribe", "GPT-4o Transcribe", "OpenAI", "#586dba", "vocera-openai-gpt-4o-transcribe-20260912"),
    ("openai-gpt-4o-mini-transcribe", "GPT-4o Mini Transcribe", "OpenAI", "#76559c", "vocera-openai-gpt-4o-mini-transcribe-20260912"),
    ("gemini-3.5-transcribe-live", "Gemini 3.5", "Google", "#ae4c24", "vocera-gemini-3-5-transcribe-live-20260912"),
    ("elevenlabs-scribe-v2-realtime", "Scribe v2 Realtime", "ElevenLabs", "#a64467", "vocera-elevenlabs-scribe-v2-realtime-20260912"),
    ("cartesia-ink-2", "Ink 2", "Cartesia", "#5b6570", "vocera-cartesia-ink-2-20260912"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aggregate(errors):
    """Corpus WER: sum the counts, then divide. Empty data is unavailable."""
    totals = {key: sum(row[key] for row in errors) for key in COUNTS}
    totals["wer"] = (sum(totals[k] for k in COUNTS[:3]) / totals["reference_words"]
                     if totals["reference_words"] else None)
    return totals


def percentiles(values):
    values = sorted(v for v in values if v is not None)
    def at(q):
        if not values:
            return None
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {"n": len(values), "p50_ms": at(.5), "p90_ms": at(.9)}


def assert_metrics(actual, expected, context):
    for key, value in actual.items():
        other = expected.get(key)
        equal = (value is None and other is None) or (
            isinstance(value, (int, float)) and isinstance(other, (int, float))
            and math.isfinite(other) and math.isclose(value, other, rel_tol=1e-10, abs_tol=1e-8))
        require(equal, f"{context}: {key} differs from saved summary ({value!r} vs {other!r})")


def accuracy(rows):
    good = [row for row in rows if row["accuracy_usable"]]
    return {**aggregate([row["word_errors"] for row in good]), "n": len(good)}


def deadlines(rows):
    result = []
    for index, ms in enumerate(DEADLINES):
        observations = [row["deadlines"][index] for row in rows]
        measured = [d for d in observations if d["word_errors"] is not None]
        result.append({"deadline_ms": ms, "measured_clips": len(measured),
                       "planned_clips": len(rows), "missing_clips": len(rows) - len(measured),
                       "pacing_invalid_clips": sum(not d["pacing_valid"] for d in measured),
                       **aggregate([d["word_errors"] for d in measured])})
    return result


def validate_summaries(summaries, manifest, manifest_hash):
    require(len(summaries) == len(MODELS), "Exactly 10 non-Speechmatics summaries are required")
    expected = [r["clip_id"] for r in manifest["clips"]]
    require(len(expected) == 1000 and len(set(expected)) == 1000, "Expected 1,000 unique frozen clips")
    require(manifest["dataset_id"] == "pipecat-stt-benchmark" and manifest["source_revision"] == REVISION,
            "Unexpected dataset or revision")
    baseline = summaries[0]
    for spec, summary in zip(MODELS, summaries):
        model = spec[0]
        require(summary["model"] == model, f"Wrong model for {model}; Speechmatics is excluded")
        require(summary["status"] == "complete" and summary["full_coverage"] is True,
                f"{model}: run is incomplete")
        require(summary["planned_clips"] == summary["completed_clips"] == 1000,
                f"{model}: expected all 1,000 clips")
        require(summary["schema_version"] == summary["measurement_version"] == 4,
                f"{model}: expected measurement version 4")
        identity = summary["identity"]
        require(identity["model"] == model and identity["dataset"] == "pipecat-stt-benchmark"
                and identity["full_manifest_sha256"] == manifest_hash, f"{model}: frozen input identity differs")
        for key in ("normalization", "shared_measurement_contract"):
            require(bool(summary[key]) and summary[key] == baseline[key], f"{model}: {key} differs")
        require(summary["shared_measurement_contract"]["reference_manifest"] == manifest_hash,
                f"{model}: measurement reference differs")
        require(summary["shared_measurement_contract"]["deadlines_ms"] == list(DEADLINES),
                f"{model}: unexpected deadlines")
        rows = summary["clips"]
        require([r["clip_id"] for r in rows] == expected, f"{model}: missing, duplicate, or reordered clips")
        for row in rows:
            require([d["deadline_ms"] for d in row["deadlines"]] == list(DEADLINES),
                    f"{model}: missing or reordered deadline observations")
            for counts in [row.get("word_errors")] + [d["word_errors"] for d in row["deadlines"]]:
                if counts is not None:
                    require(all(isinstance(counts.get(k), int) and counts[k] >= 0 for k in COUNTS),
                            f"{model}: invalid word-error counts")
            require(not row["accuracy_usable"] or row.get("word_errors") is not None,
                    f"{model}: usable clip has no word counts")
        acc = accuracy(rows)
        assert_metrics({k: acc[k] for k in (*COUNTS, "wer")}, summary["eventual_word_errors"], model)
        require(acc["n"] == summary["scored_clips"], f"{model}: usable count differs")
        require(len(summary["deadlines"]) == 4, f"{model}: incomplete deadline summary")
        for actual, saved in zip(deadlines(rows), summary["deadlines"]):
            assert_metrics(actual, saved, f"{model} deadline {actual['deadline_ms']}")
        good = [r for r in rows if r["accuracy_usable"]]
        for timing in ("finalize", "completion"):
            assert_metrics(percentiles([r[f"{timing}_latency_ms"] for r in good]),
                           summary[f"{timing}_latency"], f"{model} {timing}")
        attempts = [a for r in rows for a in r["attempts"]]
        require(sum(not a["valid"] for a in attempts) == summary["failures"], f"{model}: failure count differs")
        require(sum(a["attempt"] > 1 for a in attempts) == summary["retries"], f"{model}: retry count differs")
        require(summary["smoke_cost_included"] is False, f"{model}: smoke cost must remain separate")
        cost = summary["estimated_cost_usd"]
        require(cost is None or (isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0),
                f"{model}: invalid estimated cost")
    # Validate shared references, not just matching clip identifiers.
    for index in range(1000):
        reference = baseline["clips"][index]["reference"]
        require(all(s["clips"][index]["reference"] == reference for s in summaries), "Reference text mismatch")


def build_data(summaries, manifest, manifest_hash, sources):
    validate_summaries(summaries, manifest, manifest_hash)
    ids = [row["clip_id"] for row in manifest["clips"]]
    common_final = set(ids)
    common_deadline = set(ids)
    for summary in summaries:
        common_final &= {r["clip_id"] for r in summary["clips"] if r["accuracy_usable"]}
        common_deadline &= {r["clip_id"] for r in summary["clips"]
                            if all(d["word_errors"] is not None and d["pacing_valid"] for d in r["deadlines"])}
    models = []
    for spec, summary, source in zip(MODELS, summaries, sources):
        model, label, provider, color, run_id = spec
        rows = summary["clips"]
        attempts = [a for r in rows for a in r["attempts"]]
        models.append({"id": model, "label": label, "provider": provider, "color": color,
            "run_id": run_id, "source": source, "completed": summary["completed_clips"],
            "usable": summary["scored_clips"], "final": accuracy(rows), "deadlines": deadlines(rows),
            "shared_final": accuracy([r for r in rows if r["clip_id"] in common_final]),
            "shared_deadlines": deadlines([r for r in rows if r["clip_id"] in common_deadline]),
            "finalize": summary["finalize_latency"], "completion": summary["completion_latency"],
            "failures": summary["failures"], "retry_attempts": summary["retries"],
            "retried_clips": sum(any(a["attempt"] > 1 for a in r["attempts"]) for r in rows),
            "attempts": len(attempts), "cost": summary["estimated_cost_usd"],
            "audio_minutes": sum(a["sent_audio_seconds"] for a in attempts) / 60,
            "exclusions": dict(sorted(Counter(reason for r in rows for reason in r["exclusion_reasons"]).items())),
            "failure_categories": {
                "Completion timeout": sum(bool(a["completion_timed_out"]) for a in attempts),
                "Transport failure": sum(bool(a["transport_failed"]) for a in attempts),
                "Invalid audio pacing": sum(not a["pacing"]["valid"] for a in attempts)},
            "provider_contract": summary["provider_contract"],
            "model_versions": sorted({v for r in rows for v in r["model_versions"]}),
            "model_verification": sorted({r.get("model_verification_basis", "Not recorded") for r in rows}),
        })
    return {"title": "Streaming speech benchmarks", "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": manifest["dataset_id"], "revision": manifest["source_revision"],
            "manifest_sha256": manifest_hash, "planned_clips": 1000, "measurement_version": 4,
            "normalization": summaries[0]["normalization"],
            "shared_measurement_contract": summaries[0]["shared_measurement_contract"],
            "excluded_models": ["speechmatics-standard", "speechmatics-enhanced"],
            "shared_final_ids": [i for i in ids if i in common_final],
            "shared_deadline_ids": [i for i in ids if i in common_deadline], "models": models}


def load_private(path):
    """Embed only aggregate metrics; never copy private transcripts or audio."""
    raw = path.read_bytes()
    saved = json.loads(raw)
    require(saved['mode'] == 'private-longform-v1' and saved['complete'],
            'Private comparison must be a terminal private-longform-v1 snapshot')
    rows = saved['recordings']
    require(len(rows) == saved['planned_results'] == 112, 'Private comparison requires 112 entries')
    require(dict(Counter(r['status'] for r in rows)) == saved['status_counts'],
            'Private status counts differ')
    require(len(saved['models']) == len({m['model'] for m in saved['models']}) == 14,
            'Private comparison requires 14 unique models')
    labels = {m[0]: (m[1], m[3]) for m in MODELS}
    labels.update({'speechmatics-standard': ('Speechmatics Standard', '#827045'),
                   'speechmatics-enhanced': ('Speechmatics Enhanced', '#a08250'),
                   'google-chirp-2': ('Chirp 2', '#a55435'),
                   'google-chirp-3': ('Chirp 3', '#ba7835')})
    keys = ('model', 'status', 'scored', 'planned', 'word_errors', 'latency_median_ms',
            'latency_p95_ms', 'latency_words', 'latency_coverage_percent',
            'failed_benchmark_attempts', 'benchmark_retries', 'failed_smoke_attempts',
            'completion_delay', 'cost_estimate_benchmark_usd', 'cost_estimate_smoke_usd')
    models = []
    for m in saved['models']:
        own = [r for r in rows if r['model'] == m['model']]
        require(len(own) == len({r['clip_id'] for r in own}) == m['planned'] == 8,
                f"{m['model']}: private recording coverage differs")
        require(sum(r['status'] == 'scored' for r in own) == m['scored'] and m['archive_verified'],
                f"{m['model']}: private scoring or archive verification differs")
        errors = m['word_errors']
        assert_metrics({'wer': aggregate([errors])['wer']}, errors, m['model'])
        require((errors['wer'] is None) == (m['scored'] == 0), 'Missing private score must stay unavailable')
        model = {key: m[key] for key in keys}
        model['word_errors'] = {key: errors[key] for key in (*COUNTS, 'wer')}
        model['label'], model['color'] = labels[m['model']]
        models.append(model)
    return {'models': models, 'status_counts': saved['status_counts'],
            'planned_results': saved['planned_results'], 'source': str(path),
            'sha256': hashlib.sha256(raw).hexdigest(), 'dataset': saved['mode']}


def render(data, template):
    # Omit these limited-coverage entries from the page, without changing scores.
    excluded = {'google-chirp-3', 'sarvam-saaras-v3-realtime', 'soniox-stt-rt-v5'}
    data = deepcopy(data)
    if 'models' in data:
        data['models'] = [m for m in data['models'] if m.get('id') not in excluded]
    for clip in data.get('clip_review', {}).get('clips', []):
        clip['results'] = {key: value for key, value in clip['results'].items()
                           if key not in excluded}
    # JSON is data, not executable markup, even if a source string contains </script>.
    encoded = json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    require(template.count("__BENCHMARK_DATA__") == 1, "Template must contain exactly one data placeholder")
    review_ui = (Path(__file__).with_name('benchmark_clip_review.html').read_text()
                 if data.get('clip_review') else '')
    return template.replace("__BENCHMARK_DATA__", encoded).replace('__CLIP_REVIEW_UI__', review_ui)


def refresh_private(data, directory):
    """Adopt completed, verified recoveries; keep earlier first-attempt timing."""
    data = deepcopy(data)
    data['recovery_sources'], data['pending_recoveries'] = [], []
    original_root = Path(data['source']).parents[1]
    for model in data['models']:
        folder = directory / model['model']
        controller_path = folder / 'controller.json'
        if not controller_path.exists():
            continue
        controller = json.loads(controller_path.read_text())
        if controller['status'] != 'complete':
            data['pending_recoveries'].append(model['label'])
            continue
        require(controller.get('archiveVerified') and controller.get('computeStopped'),
                f"{model['model']}: recovery archive is not verified")
        path = folder / 'summary.json'
        raw = path.read_bytes()
        summary = json.loads(raw)
        original = json.loads((original_root / model['model'] / 'summary.json').read_text())
        require(summary['status'] == 'complete' and summary['model'] == model['model'],
                'Recovery model or terminal status differs')
        require(summary['identity']['manifest'] == original['identity']['manifest'],
                'Recovery must use the original frozen private manifest')
        require(summary['identity']['config'] == original['identity']['config'],
                'Recovery model settings differ')
        archive_path = folder / 'artifacts.tar.gz'
        with archive_path.open('rb') as stream:
            archive_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        require(archive_hash == (folder / 'artifacts.sha256').read_text().split()[0],
                'Recovery archive checksum differs')
        with tarfile.open(archive_path) as archive:
            for row in [summary.get('smoke'), *summary['recordings']]:
                if row is None:
                    continue
                for attempt in row['attempts']:
                    with archive.extractfile(attempt['raw'].removeprefix('output/')) as stream:
                        require(hashlib.file_digest(stream, 'sha256').hexdigest() == attempt['raw_sha256'],
                                'Recovery raw attempt checksum differs')
        assert_metrics(aggregate([r['selected']['word_errors'] for r in summary['recordings']
                                  if r['selected']]), summary['word_errors'], 'Recovery WER')
        records = {r['clip_id']: deepcopy(r) for r in original['recordings']}
        for new in summary['recordings']:
            old = records.get(new['clip_id'])
            if old and old['selected']:
                continue
            if old:
                old['attempts'].extend(new['attempts'])
                old['selected'] = new['selected']
                # Keep the latency from the original first attempt, even if invalid.
            else:
                records[new['clip_id']] = deepcopy(new)
        require(len(records) == model['planned'] == 8, 'Recovery recording coverage differs')
        good = [r['selected'] for r in records.values() if r['selected']]
        attempts = [a for r in records.values() for a in r['attempts']]
        delays = sorted(w['delay_ms'] for r in records.values() for w in r['first_attempt_latency']['words'])
        def timing(values):
            values = sorted(values)
            def at(q):
                if not values:
                    return None
                i = (len(values) - 1) * q
                return values[math.floor(i)] + (values[math.ceil(i)] - values[math.floor(i)]) * (i % 1)
            return {'count': len(values), 'median_ms': at(.5), 'p95_ms': at(.95)}
        def cost(items):
            costs = [a['estimated_cost_usd'] for a in items if a['submitted_seconds'] > 0]
            return sum(costs) if all(c is not None for c in costs) else None
        smoke = (summary.get('smoke') or {}).get('attempts', [])
        previous_smoke = (original.get('smoke') or {}).get('attempts', [])
        old_scored = model['scored']
        old_failed = len(original['recordings']) - old_scored
        old_not_run = model['planned'] - len(original['recordings'])
        stats = timing(delays)
        model.update(status='complete', scored=len(good), word_errors=aggregate([a['word_errors'] for a in good]),
            latency_median_ms=stats['median_ms'], latency_p95_ms=stats['p95_ms'], latency_words=len(delays),
            latency_coverage_percent=100 * len(delays) / sum(r['first_attempt_latency']['reference_words'] for r in records.values()),
            failed_benchmark_attempts=sum(not a['valid'] for a in attempts),
            benchmark_retries=sum(max(0, len(r['attempts']) - 1) for r in records.values()),
            failed_smoke_attempts=sum(not a['valid'] for a in smoke + previous_smoke),
            completion_delay=timing([a['completion_delay_ms'] for a in good if a['completion_delay_ms'] is not None]),
            cost_estimate_benchmark_usd=cost(attempts), cost_estimate_smoke_usd=cost(smoke + previous_smoke))
        data['status_counts']['scored'] += len(good) - old_scored
        data['status_counts']['failed'] += len(records) - len(good) - old_failed
        data['status_counts']['not_run'] -= old_not_run
        data['recovery_sources'].append({'model': model['label'], 'path': str(path),
                                         'sha256': hashlib.sha256(raw).hexdigest()})
    return data


def load_short_tests(root):
    """Copy only aggregate fields from the two saved diagnostic comparisons."""
    result = []
    for title, filename, key in [
        ('Short trials · 10 clips per cohort', 'trial-short-20260914/comparison.json', 'rows'),
        ('Gradium and Reson8 · 31 public clips per provider', 'limited-gradium-reson8-20260914/comparison.json', 'results'),
    ]:
        path = root / filename
        raw = path.read_bytes()
        saved = json.loads(raw)
        rows = []
        for r in saved[key]:
            require(0 <= r['usable'] <= r['attempted'] <= r['planned'], 'Short-test coverage differs')
            require((r['wer'] is None) == (r['usable'] == 0), 'Unavailable short-test WER must remain missing')
            if 'word_counts' in r:
                assert_metrics({'wer': aggregate([r['word_counts']])['wer']}, r, 'Short-test WER')
            row = {k: r[k] for k in ('model', 'cohort', 'planned', 'attempted', 'usable', 'wer')}
            row['deadlines'] = [{k: d[k] for k in ('deadline_ms', 'wer', 'measured_clips')} for d in r['deadlines']]
            blocked = any(s['model'] == r['model'] and s['status'] == 'stopped_on_provider_error'
                          for s in saved.get('states', []))
            if blocked:
                # Empty-text penalties from a rejected request are not model accuracy.
                for deadline in row['deadlines']:
                    deadline['wer'] = None
                    deadline['measured_clips'] = 0
            row['final_text'] = r.get('final_text_latency', r.get('final_text_delay'))
            row['completion'] = r.get('completion_latency', r.get('completion'))
            row['retries'] = r.get('retries', r.get('retry_count'))
            rows.append(row)
        result.append({'title': title, 'rows': rows, 'source': str(path),
                       'sha256': hashlib.sha256(raw).hexdigest()})
    return result


def main():
    from unified_benchmark import build
    parser = argparse.ArgumentParser(description="Build the unified English benchmark from saved evidence; no provider calls.")
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/benchmark-dashboard/index.html")
    parser.add_argument('--with-clip-review', action='store_true', help='Include public transcripts, diffs, and lossless listening audio')
    parser.add_argument('--include-private-review', action='store_true', help='Also package the eight private recordings and transcripts')
    parser.add_argument('--share-zip', action='store_true', help='Package the review HTML and audio into one ZIP')
    args = parser.parse_args()
    try:
        data = build(args.reports_root)
        from offline_rescore import inworld_evidence, write_correction
        correction = data.pop('_rescore_audit')
        correction['inworld_evidence'] = inworld_evidence(args.reports_root)
        write_correction(args.reports_root, data, correction)
        if args.include_private_review or args.share_zip:
            args.with_clip_review = True
        if args.with_clip_review:
            from benchmark_clip_review import build_review
            data['clip_review'] = build_review(data, args.reports_root, ROOT, args.out.parent,
                                             include_private=args.include_private_review)
        page = render(data, Path(__file__).with_name("benchmark_dashboard.html").read_text())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(page, encoding="utf-8")
        print(f"Created {args.out.resolve()} ({len(page.encode()):,} bytes)")
        print(f"{len(data['models'])} models; {sum(m['rankable'] for m in data['models'])} ranked on {data['common_public']['clips']} common public clips; zero provider calls.")
        if args.share_zip:
            from benchmark_clip_review import share_zip
            bundle = share_zip(args.out, data['clip_review'])
            print(f'Share {bundle.resolve()} ({bundle.stat().st_size / 1_000_000:.1f} MB); extract and open index.html.')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Cannot build benchmark report: {exc}\n")


if __name__ == "__main__":
    main()
