# Human Bench: main task completion (Cekura custom-code metric, binary, no LLM).
#
# Built from s2s-human-bench/src/s2s_human_bench/tasks.py by `s2s-human-bench task-metric`; change
# tasks.py and rebuild, never this copy. Reads the scenario's main task from the test profile field
# eval_target_task (JSON: requirements, must_not_run, then_undo, undo_tools) and decides from the
# call's tool calls whether the agent completed it.
# TRUE: done. FALSE: not done, with the reason. None: no task in the profile, or no verdict.

import json
import re
import unicodedata
from datetime import date
_ABBREV = {'street': 'st', 'avenue': 'ave', 'apartment': 'apt', 'road': 'rd', 'drive': 'dr', 'court': 'ct', 'suite': 'ste', 'boulevard': 'blvd', 'lane': 'ln', 'place': 'pl'}
_MONTHS = {m: i for i, m in enumerate('january february march april may june july august september october november december'.split(), 1)}
_STOP = {'the', 'a', 'an', 'of', 'on', 'at', 'in', 'and', 'to', 'for', 'by'}
_NUMBER_WORDS = {w: str(i) for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve'.split())}
_ON_FILE = 'on file'
_ANY = '*'
_UNSET = {'', 'none', 'null', 'n/a', 'unknown'}
_WEEKDAYS = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')

def tool_calls(transcript):
    """(start time, tool name, arguments) for every tool call the agent made, in time order."""
    calls = []
    for i, turn in enumerate(transcript):
        data = turn.get('data') or {}
        if turn.get('role') != 'Function Call' or not data.get('name'):
            continue
        args = data.get('arguments')
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        calls.append((float(turn.get('start_time') or i), data['name'], args if isinstance(args, dict) else {}))
    return sorted(calls, key=lambda c: c[0])

def _plain(value):
    text = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower()
    words = re.sub('[^a-z0-9]+', ' ', text).split()
    return ' '.join((_ABBREV.get(w, _NUMBER_WORDS.get(w, w)) for w in words))

def _digits(value):
    return re.sub('\\D', '', str(value))

def _dates(text, year=None):
    found = {f'{y}-{int(m):02d}-{int(d):02d}' for y, m, d in re.findall('(\\d{4})[-/](\\d{1,2})[-/](\\d{1,2})', text)}
    for month, day, yr in re.findall('([a-z]+)\\s+(\\d{1,2})(?:st|nd|rd|th)?(?:,?\\s+(\\d{4}))?', text.lower()):
        if month in _MONTHS and (yr or year):
            found.add(f'{yr or year}-{_MONTHS[month]:02d}-{int(day):02d}')
    return found

def _times(text):
    found = set()
    for h1, m1, h2, m2, ampm in re.findall('(?<![\\d-])(\\d{1,2})(?::(\\d{2}))?\\s*(?:-|–|to)\\s*(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm|a\\.m\\.|p\\.m\\.)', text.lower()):
        for h, m in ((h1, m1), (h2, m2)):
            found.add(f"{int(h) % 12 + (12 if ampm.startswith('p') else 0):02d}:{m or '00'}")
    for h, m, ampm in re.findall('(?<![\\d-])(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm|a\\.m\\.|p\\.m\\.)', text.lower()):
        hour = int(h) % 12 + (12 if ampm.startswith('p') else 0)
        found.add(f"{hour:02d}:{m or '00'}")
    for h, m in re.findall('(?<![\\d])(\\d{1,2}):(\\d{2})(?!\\s*(?:am|pm|a\\.m|p\\.m))', text.lower()):
        found.add(f'{int(h):02d}:{m}')
    return found

def _number(value):
    text = str(value).replace(',', '').replace('$', '').strip()
    return float(text) if re.fullmatch('-?\\d+(\\.\\d+)?', text) else None

def _unset(value):
    return value is None or value in ([], {}) or (isinstance(value, str) and value.strip().lower() in _UNSET)

def value_matches(expected, actual):
    """Whether the agent's argument ``actual`` gives the value ``expected``."""
    if _unset(actual):
        return _ON_FILE in _plain(expected)
    if isinstance(expected, list):
        pool = actual if isinstance(actual, list) else [actual]
        return all((any((value_matches(e, a) for a in pool)) or any((_plain(e) in _plain(a) for a in pool)) for e in expected))
    exp = str(expected).strip()
    if exp.lower() in ('true', 'false'):
        return str(actual).strip().lower() in (('true', 'yes', '1') if exp.lower() == 'true' else ('false', 'no', '0', 'none'))
    num = _number(exp)
    if num is not None and len(_digits(exp)) < 7:
        got = _number(actual)
        return got is not None and abs(got - num) < 0.005
    years = [int(y) for y in re.findall('\\b(20\\d\\d)\\b', exp)]
    dates = _dates(exp)
    if dates:
        got = str(actual)
        times = _times(exp)
        got_dates = _dates(got, years[0] if years else None)
        if not dates <= got_dates:
            named = {w for w in _WEEKDAYS if w in got.lower()}
            wanted = {date.fromisoformat(d).strftime('%A').lower() for d in dates}
            if got_dates or (named and named != wanted) or (not named and (not (times and min(times) in _times(got)))):
                return False
        return not times or min(times) in _times(got)
    times = _times(exp)
    if times and (not re.search('[a-z]{4,}', exp.lower().replace('am', '').replace('pm', ''))):
        return min(times) in _times(str(actual))
    digits, alnum = (_digits(exp), re.sub('[^A-Za-z0-9]', '', exp))
    if len(digits) >= 7 and len(digits) >= 0.6 * len(alnum):
        return _digits(actual)[-10:] == digits[-10:]
    if alnum and ' ' not in exp and any((c.isdigit() for c in alnum)):
        got = re.sub('[^A-Za-z0-9]', '', str(actual)).upper()
        return got == alnum.upper() or (len(alnum) >= 4 and alnum.upper() in got)
    want, got = (_plain(exp).split(), set(_plain(actual).split()))
    words = [w for w in want if w not in _STOP]
    if not words:
        return True
    numbers = [w for w in words if w.isdigit()]
    return all((n in got for n in numbers)) and sum((w in got for w in words)) >= 0.7 * len(words)

def _leaves(value):
    """The scalar values inside nested arguments (a list of plain values counts as one value)."""
    if isinstance(value, dict):
        return [leaf for v in value.values() for leaf in _leaves(v)]
    if isinstance(value, list) and any((isinstance(v, (dict, list)) for v in value)):
        return [leaf for v in value for leaf in _leaves(v)]
    return [value]

def row_matches(row, args):
    """Every value of the row is in the call. A value is looked for under its own argument name;
    when the call has no such argument (it was renamed in the deployed tool definition), under
    any argument the row does not name. Optional (``?``) values only count when passed."""
    names = {key.lstrip('?') for key in row}
    others = [leaf for k, v in args.items() if k not in names for leaf in _leaves(v)]

    def found(key, value):
        given = args.get(key.lstrip('?'))
        if _unset(given):
            if key.startswith('?'):
                return True
            return value != _ANY and (value_matches(value, None) or any((value_matches(value, leaf) for leaf in others)))
        return value == _ANY or any((value_matches(value, leaf) for leaf in [given, *_leaves(given)]))
    return all((value is None or found(key, value) for key, value in row.items()))

def _separate(options, used=frozenset()):
    """Whether each requirement can be met by a different call (two rides need two bookings)."""
    return not options or any((_separate(options[1:], used | {i}) for i in options[0] if i not in used))

def task_verdict(spec, transcript):
    """(passed, short reason) for one call against its scenario's task."""
    calls = tool_calls(transcript)
    names = {name for _, name, _ in calls}
    for tool in spec.get('must_not_run', []):
        if tool in names:
            return (False, f'{tool} ran after the caller took it back')
    options = []
    for requirement in spec['requirements']:
        tools = [alt['tool'] for alt in requirement['any_of']]
        hits = sorted({i for alt in requirement['any_of'] for i, (_, name, args) in enumerate(calls) if name == alt['tool'] and any((row_matches(row, args) for row in alt['accepted']))})
        if not hits:
            tried = [name for _, name, _ in calls if name in tools]
            return (False, f"{tools[0]} {('called with other values' if tried else 'not called')}")
        options.append(hits)
    if not _separate(options):
        return (False, f'{calls[options[-1][0]][1]} called once where {len(options)} separate calls were needed')
    first_action = min((calls[i][0] for hits in options for i in hits))
    last_match = {}
    for hits in options:
        for i in hits:
            t, name, _ = calls[i]
            last_match[name] = max(last_match.get(name, t), t)
    for undo in spec.get('then_undo', []):
        if not any((name == undo and t > first_action for t, name, _ in calls)):
            return (False, f'{undo} not called after the caller took it back')
    if not spec.get('then_undo'):
        for tool, last in last_match.items():
            for undo in spec.get('undo_tools', {}).get(tool, []):
                if any((name == undo and t > last for t, name, _ in calls)):
                    return (False, f'{tool} undone by {undo}')
    return (True, 'done')


def _task_spec(data):
    profile = data.get("test_profile") or {}
    nested = (profile.get("information") or {}).get("main_agent_variables") or profile.get("main_agent_variables") or {}
    raw = profile.get("eval_target_task") or nested.get("eval_target_task")
    if not raw:
        return None
    return json.loads(raw, strict=False) if isinstance(raw, str) else raw


def _transcript(data):
    raw = data.get("cekura_transcript_json")
    if raw is None:
        raw = data.get("transcript_json")
    if isinstance(raw, str):
        raw = json.loads(raw, strict=False)
    if not isinstance(raw, list):
        raise ValueError("transcript unavailable")
    return raw


try:
    _spec = _task_spec(data)
    if _spec is None:
        _result, _explanation = None, "No main task in this test profile (not reached)."
    else:
        _done, _reason = task_verdict(_spec, _transcript(data))
        _result = _done
        _explanation = "Main task done: every required tool call was made with the caller's values." if _done else f"Main task not done: {_reason}."
except Exception as error:
    _result = None
    _explanation = "Evaluation error (no verdict): " + type(error).__name__ + ": " + str(error)
