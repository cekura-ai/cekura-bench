# Production Voice Agent Quality Benchmark - Cisco focus

This snapshot adds **Cisco / Mia, +1 984 314 9680**, tested on **24 November
2026**, to the unchanged [November 12 reference panel](../2026-11-12/README.md).
It contains **17 providers, 32 production phone numbers and 1,440 selected
calls**. Cisco contributes one number and 45 calls: the same 15 scenarios with
three repetitions each.

Open [report.html](report.html) for the website-style comparison: composite
versus median latency, the ranked provider table, all thirteen metric charts,
definitions, weights and linked calls. Cisco is highlighted in the scatter
plot, table and every metric chart; its summary and call coverage are expanded.
The reference providers retain their November 12 scores. Cisco uses repaired
judges with the same metric meanings and export scope; comparison positions are
directional because the providers were evaluated at different times and judge
revisions.

## Cisco results

- Composite quality: **93.95/100** (94.0 displayed).
- Median qualifying turn latency: **3,760 ms**, across **77 events**.
- Mean interruption stop time: **506.67 ms**, across **six eligible events**.
- Speech integrity: **45/45 passing**; unnecessary repetition: **5/5**.
- Remaining concerns: two abandoned explanations after “uh-huh”; two premature
  responses that cut off the caller's second question; one missed distress
  acknowledgment; one clarification of the wrong point. Linked examples are
  in the report's Cisco summary.

The calls reached Mia, the Cisco Webex assistant. These are observations about
the tested access point and scripted conversation scenarios. They do not
estimate everyday customer-call failure frequency or Cisco-wide behavior.

## Selection and scoring

There were **47 executions**, of which 45 are selected. Run 4780955 was
replaced by 4782023 because an unscripted caller rescue contaminated the
acknowledgment test. Run 4780959 was replaced by 4782488 because the original
ended without an assessable reply and never exercised the planned interruption.
The JSON retains both replacement mappings.

The metric IDs, weights, scenario-balanced and number-balanced aggregation,
0-5 and binary normalization, N/A handling, and 300-3,000 ms stop-time anchors
are identical to the reference snapshot. Expected Outcome is diagnostic and
excluded from the composite. It still falsely passes the wrong-point
clarification; Response Relevance correctly penalizes that behavior.

Latency uses the median of individual qualifying events. Navigation,
responses to caller interruptions, idle check-ins and responses after the
terminal caller goodbye are excluded, matching the reference export. This
corrects the earlier standalone Cisco PDF's latency figure, which included
closing replies. The quality composite is unaffected.

Voice tone and clarity uses the same isolated-AI NISQA preprocessing and
checkpoint as the reference snapshot:
`7ec4cf937514dd3f8860b21e66fabd8ca87a168572675ef8d979c4c4ad2e805c`.
Only the model scores are committed, not audio or signed recording URLs.

## Files and validation

- [results.json](results.json): the established snapshot schema, all selected
  run IDs, metric scores, timing events, provider aggregates and report hash.
- [cisco-calls.json](cisco-calls.json): the 45 Cisco input records, including
  public deep links to each call.
- [build_results.py](build_results.py): adds Cisco and recomputes its aggregates
  from those records while preserving every reference provider.
- [render.py](render.py): the existing website-style report renderer with Cisco
  highlighting; [logo.svg](logo.svg) is the official Cekura asset.
- [validate.py](validate.py): reconciles every selected call, scenario count,
  metric aggregate, composite, evidence URL and report hash, and checks that
  the original sixteen providers are unchanged.

From the repository root:

```sh
python3 results/production-voice-quality/2026-11-24/build_results.py
python3 results/production-voice-quality/2026-11-24/validate.py
python3 -m pytest tests/test_production_quality_results.py tests/test_cisco_quality_results.py -q
```

This is a follow-up to [PR #60](https://github.com/cekura-ai/cekura-bench/pull/60)
under [CEK-8446](https://linear.app/cekura/issue/CEK-8446). It adds a versioned
data/report snapshot; the live website still needs to ingest that snapshot.
