"""Reconstruct long-form ElevenLabs text without counting annotation events twice."""
from .score import normalize_words

VERSION = 'elevenlabs-private-paired-finals-v1'


def same_words(left, right):
    # Both checks matter: normalization alone can equate genuinely different
    # spellings/numbers. Only case, punctuation and spacing may differ here.
    letters = lambda text: ''.join(c for c in text.casefold() if c.isalnum())
    return letters(left) == letters(right) and normalize_words(left) == normalize_words(right)


class PairedFinals:
    """Pair only a timestamp annotation with its preceding plain commit.

    Repeated plain commits and word-changing annotations remain in the output.
    Session identity prevents a timestamp from consuming another session's text.
    """
    def __init__(self, *, historical=False):
        self.historical = historical
        self.pending = {}
        self.segments = {}
        self.duplicates = []
        self.conflicts = []

    @property
    def text(self):
        return ' '.join(' '.join(self.segments[s]) for s in sorted(self.segments)).strip()

    def feed(self, event):
        if event['kind'] != 'provider_message':
            return
        message = event['message']
        kind = message.get('message_type')
        if kind not in ('committed_transcript', 'committed_transcript_with_timestamps'):
            return
        session = event.get('session_index', 0)
        text = message.get('text', '')
        previous = self.pending.get(session)
        annotation = kind == 'committed_transcript_with_timestamps'
        duplicate = annotation and previous is not None and (
            previous == text if self.historical else same_words(previous, text))
        if duplicate:
            if previous != text:
                self.duplicates.append(dict(session_index=session,
                    received_seconds=event['time_seconds'], words=len(normalize_words(text).split())))
        else:
            self.segments.setdefault(session, []).append(text)
            if annotation and previous is not None:
                self.conflicts.append(dict(session_index=session, received_seconds=event['time_seconds']))
        self.pending[session] = None if annotation else text


def replay(events):
    old, corrected = PairedFinals(historical=True), PairedFinals()
    snapshots, frames = [], {}
    for event in events:
        if event['kind'] == 'audio_sent':
            frames[event['index']] = event['time_seconds']
        if event['kind'] != 'provider_message':
            continue
        before = corrected.text
        old.feed(event)
        corrected.feed(event)
        if corrected.text != before:
            snapshots.append(dict(time_seconds=event['time_seconds'], text=corrected.text))
    return dict(original=old.text, transcript=corrected.text, snapshots=snapshots,
                frames=frames, duplicates=corrected.duplicates, conflicts=corrected.conflicts)
