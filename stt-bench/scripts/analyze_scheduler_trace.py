"""Align saved scheduler/syscall exports with the stopped diagnostic process.

Uses clock bounds around matched kevent calls, not the rounded/cropped trace
wall-clock header. XML reference IDs are resolved before selecting rows.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from stt_bench.data import sha256, write_json


def rows(path):
    root = ET.parse(path).getroot()
    ids = {e.get('id'): e for e in root.iter() if e.get('id')}
    def resolve(element):
        return ids[element.get('ref')] if element.get('ref') else element
    columns = [e.findtext('mnemonic') for e in root.find('.//schema').findall('col')]
    return [{name: resolve(value) for name, value in zip(columns, row)} for row in root.iter('row')]


def main():
    directory = Path('reports')
    stop_path = directory / 'latency-root-cause-scheduler-1s/diagnostic-stop.json'
    stop = json.loads(stop_path.read_text())
    marker = 'Main Thread ' + hex(stop['native_thread_id'])
    syscall_path = directory / 'latency-root-cause-syscalls.xml'
    calls = [r for r in rows(syscall_path) if marker in r['thread'].get('fmt', '')
             and r['call'].get('fmt') == 'kevent']
    measured = max(stop['selects'], key=lambda s: s['end']-s['start'])
    call = min(calls, key=lambda r: abs(int(r['duration'].text)/1e9-(measured['end']-measured['start'])))
    def bounds(observed, native):
        start = int(native['start'].text)/1e9
        end = start + int(native['duration'].text)/1e9
        return observed['start']-start, observed['end']-end
    low, high = bounds(measured, call)
    assert 0 <= high-low < .0001, 'No close unique native wait match'
    matches = []
    for observed in stop['selects']:
        candidates = [bounds(observed, native) for native in calls]
        candidates = [(a,b) for a,b in candidates if a <= b and max(low,a) <= min(high,b)]
        if len(candidates) == 1:
            a,b = candidates[0]
            low,high = max(low,a),min(high,b)
            matches.append(dict(measured_wait=observed, origin_lower=a, origin_upper=b))
    assert len(matches) >= 3 and high >= low
    timer = stop['timers'][-1]
    state_path = directory / 'latency-root-cause-thread-state-1s.xml'
    intervals = []
    for r in rows(state_path):
        if marker not in r['thread'].get('fmt', ''):
            continue
        start = int(r['start'].text)/1e9
        duration = int(r['duration'].text)/1e9
        if start+duration < timer['deadline']-high or start > timer['resumed']-low:
            continue
        intervals.append(dict(start=start, duration_ms=duration*1000, state=r['state'].get('fmt'),
                              priority=r['priority'].get('fmt'), note=r['note'].get('fmt')))
    runnable = max((r for r in intervals if r['state'] == 'Runnable'), key=lambda r:r['duration_ms'])
    cpu_path = directory / 'latency-root-cause-cpu-state.xml'
    totals, memory = {}, {}
    a,b = runnable['start'], runnable['start']+runnable['duration_ms']/1000
    for r in rows(cpu_path):
        start = int(r['start'].text)/1e9
        overlap = max(0, min(b,start+int(r['duration'].text)/1e9)-max(a,start))*1000
        if not overlap or r['state'].get('fmt') != 'Running':
            continue
        name = r['thread'].get('fmt', 'unknown').split()[0]
        totals[name] = totals.get(name, 0) + overlap
        if name.startswith('VM_'):
            key = name + ' priority ' + r['priority'].get('fmt', 'unknown')
            memory[key] = memory.get(key, 0) + overlap
    result = dict(pid=stop['pid'], native_thread_id=stop['native_thread_id'],
                  clock_origin_bounds=[low,high], clock_uncertainty_us=(high-low)*1e6,
                  matched_wait_count=len(matches), matched_waits=matches,
                  timer=timer, intervals=intervals, longest_runnable_interval=runnable,
                  runnable_wakeup_after_deadline_ms_bounds=[1000*(a+low-timer['deadline']),
                                                            1000*(a+high-timer['deadline'])],
                  memory_workers_running_core_ms=memory,
                  recorded_running_core_ms=sum(totals.values()),
                  evidence=[dict(path=str(p),sha256=sha256(p)) for p in
                            (stop_path,syscall_path,state_path,cpu_path)],
                  limitation='Profiler adds load. Runnable interval is directly measured; concurrent memory-management work is not proof it caused every unprofiled stall. CPU coverage is incomplete; no claim of full CPU saturation.')
    write_json(directory/'latency-scheduler-verified.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('matched_waits','intervals','evidence')}, indent=2))


if __name__ == '__main__':
    main()
