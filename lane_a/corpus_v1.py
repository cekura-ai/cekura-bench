"""Lane A corpus v1 -- the caller script, authored here.

Domain is the appointments agent whose tool contract is already public in
``agent-definitions/appointments``, so the scenarios, the tools and the scoring
all refer to the same world. Every record named here is the synthetic data from
that contract; no production content appears in this repository at any point.

Voices are chosen to vary gender and accent because voice-sensitive endpointing
is itself a finding, and a single-voice corpus hides it completely. Results are
stratified by voice rather than averaged over it.
"""

from __future__ import annotations

from lane_a.clips import ClipSpec, Corpus, Voice

VOICES = (
    Voice(label="f-us", vendor="elevenlabs", voice_id="EXAVITQu4vr4xnSDxMaL"),
    Voice(label="m-us", vendor="elevenlabs", voice_id="CwhRBWXzGAHq8TQ4Fs17"),
    Voice(label="f-gb", vendor="elevenlabs", voice_id="Xb7hH8MSUJpSbSDYk0k2"),
)

CLIPS = (
    # Opening turns -- the stimulus for plain response latency.
    ClipSpec("open.book", "Hi, I'd like to book an appointment with Doctor Lee, please."),
    ClipSpec("open.reschedule", "Hello, I need to move my appointment to a different day."),

    # Endpointing ladder. Rendered as two halves; the probe inserts the pause,
    # so one render serves the whole 400-to-2000 ms ladder and the two halves are
    # bit-identical across every rung.
    ClipSpec("phone.part1", "My phone number is four one five,", note="ladder first half"),
    ClipSpec("phone.part2", "five five five, zero one two three.", note="ladder second half"),

    # Content-free interaction tokens. Public data is appropriate here later:
    # prosody carries the meaning and the words carry none.
    ClipSpec("token.backchannel", "Mm hmm."),
    ClipSpec("token.hesitation", "Um, let me think."),

    # Barge-in and correction -- closed-loop, anchored on the agent's own onset.
    ClipSpec("bargein.stop", "Actually, hold on a second."),
    ClipSpec("correct.midanswer", "Sorry, I meant Tuesday, not Thursday."),

    # Task turns against the public mock-tool contract. The caller is the record
    # in that contract with no upcoming appointment, so the booking path is
    # unambiguous: a patient who already has one sends any sensible agent down a
    # reschedule branch instead, and the expected trace would be arguable.
    ClipSpec("task.identify", "This is James Carter, my number is two zero two, five five five, zero one eight eight."),
    ClipSpec("task.reason", "It's for an annual checkup."),
    ClipSpec("task.date", "July the eighth would work. A morning slot, if you have one."),
    ClipSpec("task.confirm", "Yes, that works. Please go ahead and book it."),
    ClipSpec("task.cancel", "I'd like to cancel my appointment. The confirmation number is C W five zero zero one."),
)


def build(root: str = "corpus/lane-a", version: str = "0.1.0") -> Corpus:
    return Corpus(root, version).add(*CLIPS).add_voice(*VOICES)
