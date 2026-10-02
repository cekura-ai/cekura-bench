"""Pair identity safeguards: never deduplicate actual repeated speech."""
from stt_bench.elevenlabs_segments import PairedFinals, replay


def event(text, *, timestamp=False, session=0, time=1):
    return dict(kind='provider_message', session_index=session, time_seconds=time,
        message=dict(message_type='committed_transcript_with_timestamps' if timestamp else 'committed_transcript', text=text))


def test_formatting_annotation_counts_once_and_keeps_original_receipt_time():
    result = replay([event('Hello, world.'), event('hello world',timestamp=True,time=2)])
    assert result['original'] == 'Hello, world. hello world'
    assert result['transcript'] == 'Hello, world.'
    assert result['snapshots'] == [dict(time_seconds=1,text='Hello, world.')]
    assert len(result['duplicates']) == 1


def test_repeated_plain_speech_is_never_dropped():
    result = replay([event('yes'),event('YES',timestamp=True),event('yes'),event('YES',timestamp=True)])
    assert result['transcript'] == 'yes yes'


def test_different_words_are_preserved_and_flagged():
    result = replay([event('one'),event('1',timestamp=True)])
    assert result['transcript'] == 'one 1'
    assert len(result['conflicts']) == 1
    assert not result['duplicates']


def test_annotation_cannot_consume_another_session_or_another_annotation():
    p = PairedFinals()
    for e in [event('yes'),event('yes',timestamp=True,session=1),
              event('YES',timestamp=True),event('YES',timestamp=True)]:
        p.feed(e)
    assert p.text == 'yes YES yes'
    assert len(p.duplicates) == 1


def test_audio_clock_is_preserved():
    result = replay([dict(kind='audio_sent', index=0, time_seconds=.25),event('hello')])
    assert result['frames'] == {0:.25}
