"""Website row builder on a synthetic run; no network or provider calls."""
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from site_row import ci95, conversation, counts, site_row

MODEL = 'example-live'


def write_full(root, clips):
    root.mkdir()
    plan = json.dumps({'items': [dict(clip_id=c, reference=ref) for c, _, ref, _ in clips]}).encode()
    with tarfile.open(root / 'input.tar.gz', 'w:gz') as t:
        info = tarfile.TarInfo('full-input/plan.json')
        info.size = len(plan)
        t.addfile(info, io.BytesIO(plan))
    items = [dict(clip_id=c, cohort=cohort, selected_attempt=2,
                  attempts=[dict(attempt=1, transcript='ignored'), dict(attempt=2, transcript=hyp)])
             for c, cohort, _, hyp in clips]
    (root / 'results.json').write_text(json.dumps({'models': {MODEL: {'items': items}}}))


def write_turns(root, clips):
    report = root / 'report-20260101T000000'
    (report / MODEL).mkdir(parents=True)
    (report / MODEL / 'results.json').write_text(json.dumps({'clips': clips}))
    timing = dict(counts=dict(attempted=len(clips)),
                  ttfs=dict(n=2, p50_ms=180.4, p90_ms=600.6, p95_ms=757.0),
                  ttft=dict(n=2, p50_ms=2039.2, p90_ms=16126.0, p95_ms=26321.5))
    (report / 'comparison.json').write_text(json.dumps({'models': {MODEL: timing}}))


def test_ellipsis_reads_as_a_space():
    assert counts('well so then', 'well…so then') == dict(S=0, I=0, D=0, N=3)


def test_empty_reference_counts_every_word_as_an_insertion():
    assert counts('', 'hello there') == dict(S=0, I=2, D=0, N=0)


def test_speakers_share_their_conversation():
    assert conversation('private/conversation-03-A') == conversation('conversation-03-B') == 'conversation-03'


def test_interval_is_seeded():
    groups = [[dict(S=s, I=0, D=0, N=10)] for s in (0, 1, 2, 3)]
    assert ci95(groups) == ci95(groups)
    low, high = ci95(groups)
    assert 0 <= low <= 15 <= high <= 30


def test_site_row(tmp_path):
    write_full(tmp_path / 'full', [
        ('pipecat-b', 'public', 'the cat sat', 'the cat sat'),
        ('pipecat-a', 'public', 'a dog ran home', 'a dog ran'),
        ('conversation-01-A', 'private', 'yes please', 'yes'),
        ('conversation-01-B', 'private', 'thank you', 'thank you'),
        ('conversation-02-A', 'private', 'see you later', 'see you'),
    ])
    write_turns(tmp_path / 'turns', [
        dict(clip_id='t1', reference='good morning', transcript='good morning', valid=True),
        dict(clip_id='t2', reference='please call me back', transcript='please call back', valid=True),
        dict(clip_id='t3', reference='never scored', transcript=None, valid=False),
    ])
    out = site_row(tmp_path / 'full', tmp_path / 'turns', MODEL)
    row = out['row']
    assert row['publicErrors'] == dict(deletions=1, insertions=0, referenceWords=7, substitutions=0)
    assert row['privateErrors'] == dict(deletions=2, insertions=0, referenceWords=7, substitutions=0)
    assert row['publicWer'] == 100 / 7 and row['privateWer'] == 200 / 7 and row['overallWer'] == 300 / 14
    assert row['turnWer'] == 100 / 6 and row['referenceWords'] == 6
    assert (row['attempted'], row['valid'], row['failed']) == (3, 2, 1)
    assert (row['overallPlanned'], row['overallIncluded'], row['publicUsable'], row['privateUsable']) == (5, 5, 2, 3)
    assert (row['ttfsP50'], row['ttftP50'], row['ttfsN'], row['ttftN']) == (180, 2039, 2, 2)
    assert out['percentiles'] == {'ttfs': {'50': 180, '90': 601, '95': 757},
                                  'ttft': {'50': 2039, '90': 16126, '95': 26322}}
