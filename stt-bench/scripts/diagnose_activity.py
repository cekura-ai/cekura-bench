"""Matched process-activity comparison; timer-only, no provider or socket calls."""
import argparse
import asyncio
from contextlib import nullcontext
import json
from pathlib import Path
import time

from stt_bench.macos_timer import MacOSDeadlineTimer


async def trial(activity, seconds):
    from stt_bench.macos_activity import LatencyActivity
    count = round(seconds / .020) + 50
    cpu, wall = time.process_time(), time.perf_counter()
    with LatencyActivity() if activity else nullcontext():
        async with MacOSDeadlineTimer() as timer:
            start = time.perf_counter()
            due = start + .020
            rows = []
            for i in range(count):
                ideal = start + (i + 1) * .020
                await timer.wait_until(due)
                sent = time.perf_counter()
                rows.append(dict(index=i, sent=sent, scheduled=due, ideal=ideal))
                due = max(ideal + .020, sent + .019)
    gaps = [(b['sent']-a['sent'])*1000 for a,b in zip(rows, rows[1:])]
    ratio = (rows[-1]['sent']-rows[0]['sent']) / ((count-1)*.020)
    return dict(activity=activity, valid=min(gaps)>=18 and max(gaps)<=40 and .98<=ratio<=1.02,
                interval_ms_max=max(gaps), interval_ms_min=min(gaps), actual_over_ideal=ratio,
                cpu_percent=100*(time.process_time()-cpu)/(time.perf_counter()-wall), frames=rows)


async def main(out, seconds, repeats):
    out.mkdir(parents=True, exist_ok=False)
    for source in (Path(__file__), Path('src/stt_bench/macos_timer.py'), Path('src/stt_bench/macos_activity.py')):
        (out/source.name).write_text(source.read_text())
    results = []
    for repeat in range(repeats):
        for activity in ([False, True] if repeat%2==0 else [True, False]):
            row = dict(repeat=repeat, **await trial(activity, seconds))
            results.append(row)
            (out/'results.json').write_text(json.dumps(results))
            print(json.dumps({k:v for k,v in row.items() if k!='frames'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(args.out, args.seconds, args.repeats))
