import json
from pathlib import Path
from stt_bench.provider_protocol import Protocol
from stt_bench.turn_metrics import transcript_timeline

BASE=json.loads(Path('config/profiles/private-turns-v1/elevenlabs-scribe-v2-realtime.json').read_text())
CONFIG={**BASE,'transcript_reconstruction':'elevenlabs-committed-segments-v2'}
def plain(text):return dict(message_type='committed_transcript',text=text)
def timed(text,start,end):return dict(message_type='committed_transcript_with_timestamps',text=text,words=[dict(start=start,end=end)])

def test_auto_commits_append_and_timestamp_annotations_do_not_duplicate():
 p=Protocol(CONFIG);p.feed(plain('Oh yeah.'));p.feed(timed('Oh yeah.',0,35));p.feed(dict(message_type='partial_transcript',text='They are'))
 assert p.snapshot()['text']=='Oh yeah. They are'
 p.requested=True;p.feed(plain('They are good.'));p.feed(timed('They are good.',35.5,41))
 assert p.snapshot()['text']=='Oh yeah. They are good.'
 assert p.snapshot()['reconstruction_status']=='supported' and p.ack
 assert not p.snapshot()['partial_text']

def test_timestamp_capitalization_and_duplicates_do_not_shift_final_text():
 messages=[plain('You know.'),timed('you know.',0,1),timed('you know.',0,1)]
 events=[dict(kind='provider_message',time_seconds=i+1,message=m) for i,m in enumerate(messages)]
 timeline=transcript_timeline(events,CONFIG)
 assert [s['final_text'] for s in timeline]==['You know.']*3
 assert all(s['reconstruction_status']=='supported' for s in timeline)

def test_distinct_repeated_words_survive_and_changed_annotation_is_flagged():
 p=Protocol(CONFIG)
 for start in (0,2):p.feed(plain('yes'));p.feed(timed('yes',start,start+1))
 assert p.snapshot()['final_text']=='yes yes'
 p.feed(timed('no',2,3));assert p.unsupported

def test_historical_reconstruction_remains_unchanged():
 p=Protocol(BASE);p.feed(plain('one'));p.feed(plain('two'));assert p.unsupported

def test_speechmatics_silence_ranges_do_not_conflict_with_later_words():
 c=json.loads(Path('config/profiles/private-turns-v1/speechmatics-enhanced.json').read_text())
 def event(text,start,end):return dict(message='AddTranscript',metadata=dict(transcript=text,start_time=start,end_time=end))
 old=Protocol(c);new=Protocol({**c,'transcript_reconstruction':'speechmatics-empty-silence-ranges-v2'})
 for p in (old,new):
  p.feed(event('',31.44,33.24));p.feed(event('Oh ',33.16,34.52))
 assert old.unsupported and not new.unsupported
 assert new.snapshot()['text']=='Oh'
 new.feed(event('Another',34,35));assert new.unsupported

def test_v3_late_duplicate_partial_does_not_reopen_committed_segment():
 p=Protocol({**CONFIG,'transcript_reconstruction':'elevenlabs-committed-segments-v3'})
 p.feed(plain('The whole completed sentence.'))
 p.feed(dict(message_type='partial_transcript',text='The whole completed sentence.'))
 p.feed(timed('The whole completed sentence.',0,4))
 assert not p.snapshot()['partial_text']
 p.feed(dict(message_type='partial_transcript',text='A new unfinished sentence'))
 assert p.snapshot()['partial_text']=='A new unfinished sentence'
