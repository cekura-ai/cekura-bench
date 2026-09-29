"""The service bench corpus v1 -- the caller script, authored here.

Domain is the appointments agent whose tool contract is already public in
``agent-definitions/appointments``, plus the front-door intake agent in
``agent-definitions/medicare``, so the scenarios, the tools and the scoring all
refer to the same world. Every record named here is the synthetic data from
those contracts; no production content appears in this repository at any point.

Voices are chosen to vary gender and accent because voice-sensitive endpointing
is itself a finding, and a single-voice corpus hides it completely. Results are
stratified by voice rather than averaged over it.

Clip ids are grouped by role. ``open.*`` are first turns; ``identify.*`` carry a
phone number and name; ``task.*`` are replies inside a task; ``token.*`` are
content-free; ``mc.*`` belong to the intake contract. A scenario names its clips
by id, so a line can be re-rendered or re-recorded without touching a scenario.
"""

from __future__ import annotations

from service.clips import ClipSpec, Corpus, Voice

VOICES = (
    Voice(label="f-us", vendor="elevenlabs", voice_id="EXAVITQu4vr4xnSDxMaL"),
    Voice(label="m-us", vendor="elevenlabs", voice_id="CwhRBWXzGAHq8TQ4Fs17"),
    Voice(label="f-gb", vendor="elevenlabs", voice_id="Xb7hH8MSUJpSbSDYk0k2"),
)

CLIPS = (
    # ── opening turns: the stimulus for response latency, and each task's first line
    ClipSpec("open.book", "Hi, I'd like to book an appointment with Doctor Lee, please."),
    ClipSpec("open.book.patel", "Hello, I'd like to book an appointment with Doctor Patel."),
    ClipSpec("open.reschedule", "Hello, I need to move my appointment to a different day."),
    ClipSpec("open.cancel", "Hi there. I need to cancel one of my appointments."),
    ClipSpec("open.confirm", "Hi, I just want to check when my next appointment is."),
    ClipSpec("open.question", "Hi, what should I bring with me to my first appointment?"),
    ClipSpec("open.emergency", "I'm having chest pain right now and I can't catch my breath."),
    ClipSpec("open.digits", "My account number is seven four two, nine one six, three three eight, zero five."),

    # ── endpointing ladders. Two halves each; the probe inserts the pause, so
    # one render serves every rung and the halves are bit-identical across rungs.
    ClipSpec("phone.part1", "My phone number is four one five,", note="ladder first half"),
    ClipSpec("phone.part2", "five five five, zero one two three.", note="ladder second half"),
    ClipSpec("date.part1", "I could come in on the eighth of July,", note="second ladder first half"),
    ClipSpec("date.part2", "or the ninth, if that's easier for you.", note="second ladder second half"),

    # ── content-free interaction tokens. Prosody carries the meaning, words carry none.
    ClipSpec("token.backchannel", "Mm hmm."),
    ClipSpec("token.hesitation", "Um, let me think."),

    # ── barge-in and correction, anchored on the agent's own onset
    ClipSpec("bargein.stop", "Actually, hold on a second."),
    ClipSpec("correct.midanswer", "Sorry, I meant Tuesday, not Thursday."),

    # ── identity turns. Each is one record of the published appointments contract.
    ClipSpec("task.identify", "This is James Carter, my number is two zero two, five five five, zero one eight eight."),
    ClipSpec("identify.alt", "It's James Carter. Two oh two, triple five, oh one eight eight."),
    ClipSpec("identify.maria", "This is Maria Gomez. My number is four one five, five five five, zero one two three."),
    ClipSpec("identify.wei", "Wei Chen. The number is six one seven, five five five, nine two one zero."),
    ClipSpec("identify.robert", "Robert Lane, and my phone number is three one zero, five five five, zero one four seven."),
    ClipSpec("identify.unknown", "This is Dana Brooks, my number is nine nine nine, five five five, zero zero zero zero."),
    ClipSpec("identify.fail", "Sam Reyes. My number is five zero zero, five five five, zero nine one one."),

    # ── task replies against the appointments contract
    ClipSpec("task.reason", "It's for an annual checkup."),
    ClipSpec("task.date", "July the eighth would work. A morning slot, if you have one."),
    ClipSpec("task.date.anytime", "July the eighth, any time that day is fine."),
    ClipSpec("task.date.july9", "How about July the ninth, in the morning if possible?"),
    ClipSpec("task.date.july10", "July the tenth, please. Any time."),
    ClipSpec("task.date.range", "Sometime the week of July sixth, whatever you have."),
    ClipSpec("task.date.correct", "July the ninth. Sorry, no, I mean the eighth. July the eighth."),
    ClipSpec("task.choose", "The nine o'clock, please."),
    ClipSpec("task.which", "The one on July sixth with Doctor Patel."),
    ClipSpec("task.which.consult", "The consultation on July seventh."),
    ClipSpec("task.dob", "My date of birth is March second, nineteen ninety."),
    ClipSpec("task.confirm", "Yes, that works. Please go ahead and book it."),
    ClipSpec("task.yes", "Yes, that's right."),
    ClipSpec("task.cancel", "I'd like to cancel my appointment. The confirmation number is C W five zero zero one."),
    ClipSpec("task.distraction", "Before that, should I stop taking my blood pressure medication before the visit?"),
    ClipSpec("task.done", "No, that's everything. Thank you."),

    # ── intake contract (medicare): permissions, qualification, routing
    ClipSpec("mc.open.shop", "Hi, I'd like to look at the Medicare Advantage plans available in my area."),
    ClipSpec("mc.open.review", "Hello, I'd like someone to review the Medicare plan I'm on."),
    ClipSpec("mc.open.claim", "Hi, I'm calling about a claim on my plan that was denied."),
    ClipSpec("mc.self", "I'm calling for myself."),
    ClipSpec("mc.interest.ma", "I'm interested in a Medicare Advantage plan."),
    ClipSpec("mc.parts.yes", "Yes, I have both Part A and Part B."),
    ClipSpec("mc.consent.yes", "Yes, that's fine. You can contact me and share my details with the agent."),
    ClipSpec("mc.consent.no", "No, I don't want to be contacted, and don't share my information."),
    ClipSpec("mc.name.phone.maria", "My name is Maria Gomez, and my callback number is four one five, five five five, zero one nine nine."),
    ClipSpec("mc.name.charles", "My name is Charles Brown."),
    ClipSpec("mc.location.ca", "I'm in California, ZIP code nine four one zero five."),
    ClipSpec("mc.coverage.original", "I have Original Medicare right now."),
    ClipSpec("mc.intent.new", "I'm shopping for a new plan."),
    ClipSpec("mc.age.on", "I'm sixty-eight, I've been on Medicare for a few years."),
    ClipSpec("mc.claim.type", "It's a claim that was denied."),
)


def build(root: str = "corpus/service", version: str = "0.2.0") -> Corpus:
    return Corpus(root, version).add(*CLIPS).add_voice(*VOICES)
