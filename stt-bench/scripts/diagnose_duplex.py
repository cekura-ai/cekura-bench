"""Independent receiver A/B: cooperative wait size versus disk logger."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from stt_bench import diagnostics, timing
from stt_bench.data import write_json
from stt_bench.streaming import EventLog


class MemoryLog:
    def __init__(self, path):
        self.path = path
        self.origin = time.perf_counter()
        self.events = []

    def now(self):
        return time.perf_counter() - self.origin

    def emit(self, kind, *, at=None, **fields):
        self.events.append(dict(kind=kind, time_seconds=self.now() if at is None else at, **fields))

    def close(self):
        self.path.write_text(''.join(json.dumps(e) + '\n' for e in self.events))


async def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    original_identity = diagnostics.pacing_identity
    rows = []
    for repeat in range(args.repeats):
        choices = [(wait, logger) for wait in (.002, .001) for logger in ('disk', 'memory')]
        if repeat % 2:
            choices.reverse()
        for wait, logger in choices:
            timing.COARSE_WAIT_SECONDS = wait
            diagnostics.EventLog = EventLog if logger == 'disk' else MemoryLog
            # Experimental overrides must never qualify as production preflights.
            diagnostics.pacing_identity = lambda: dict(**original_identity(),
                experiment_overrides=dict(coarse_wait_seconds=wait, logger=logger))
            path = args.out / f'{repeat}-{wait}-{logger}'
            cpu_start, wall_start = time.process_time(), time.perf_counter()
            report = await diagnostics.local_probe(path, seconds=args.seconds, repeats=1)
            cpu, wall = time.process_time()-cpu_start, time.perf_counter()-wall_start
            row = dict(repeat=repeat, wait=wait, logger=logger,
                       cpu_percent=cpu/wall*100, **report['trials'][0])
            rows.append(row)
            write_json(args.out / 'results.json', dict(mode='duplex_diagnosis', rows=rows))
            print(f"{repeat} wait={wait} logger={logger} valid={row['valid']} send={row.get('interval_ms_max')} receive={row.get('receiver_gap_ms_max')} reasons={row['gate_reasons']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--repeats', type=int, default=2)
    asyncio.run(main(parser.parse_args()))
