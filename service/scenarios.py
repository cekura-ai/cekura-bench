"""Task scenarios: a closed-loop caller script plus the tool trace it must produce.

A scenario is data, not code. It names its opening clip, a routing table that
picks the caller's next line from what the agent just asked, the trace of tool
calls that counts as success, any tools that must not be called, and what ends
the conversation. The same schema loads from JSON, which is how a holdout set
authored beside this one runs through the identical probe without living in
the public repository.

Scoring is on the **tool-call trace**, not on a judge's reading of the
conversation. What the agent said is recorded and published; it does not decide
the cell. Where a contract behaviour can only be judged from wording -- "never
claim a cancellation succeeded when the tool failed" -- the scenario still
constrains the trace (the failed call must be made, nothing else may be) and
leaves the wording to the published transcript.

Routing is a table of literal patterns, ordered, first match wins. A model
choosing the caller's next line would put a second language model inside the
measurement, and two providers would then be scored partly on how well our
caller understood them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from mock_tools.verifier import ExpectedCall

# Patterns every appointments scenario shares. A scenario's own routes come
# first and can override any of these.
GREETING = (("how can i help", "how may i help", "what can i do for you", "how can i assist"), "__opener__")
ASK_PHONE = (("phone number", "number on your account", "phone", "reach you"), "__identify__")
ASK_REASON = (("reason for", "what brings", "type of visit", "purpose of"), "task.reason")
ASK_DATE = (
    ("what day", "which day", "what date", "which date", "come in", "day works", "date works", "day would",
     "preferred day", "day in mind", "date in mind", "when would you like", "another date", "different date",
     "another day", "different day"),
    "task.date",
)
ASK_WHICH_SLOT = (("which one", "which time", "which of", "which slot", "prefer the", "or the", "9 00 am or", "nine or"), "task.choose")
CONFIRM = (("book that", "like to book", "shall i", "confirm", "does that work", "sound good", "go ahead", "is that correct", "is that right"), "task.confirm")
ANYTHING_ELSE = (("anything else", "else i can", "else today"), "task.done")

COMMON_ROUTES: tuple[tuple[tuple[str, ...], str], ...] = (
    GREETING, ASK_PHONE, ASK_REASON, ASK_DATE, ASK_WHICH_SLOT, ANYTHING_ELSE, CONFIRM,
)

APPOINTMENT_TOOLS = ("lookup_patient", "check_availability", "book_appointment", "cancel_appointment")


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    contract: str                       # agent-definitions/<contract>
    opener: str                         # clip id of the caller's first line
    identify: str                       # clip id the caller answers a phone-number request with
    expected: tuple[ExpectedCall, ...]
    routes: tuple[tuple[tuple[str, ...], str], ...] = ()   # scenario-specific, tried first
    forbidden: tuple[str, ...] = ()
    done_when_called: str | None = None  # a tool call that ends the scenario
    done_when_said: tuple[str, ...] = () # ...or agent wording that ends it
    fallback: str = "task.confirm"       # when nothing matched: assent beats stalling
    max_turns: int = 10
    note: str = ""

    def route(self, agent_text: str) -> str:
        """The caller answers the question the agent ended with.

        Matched against the last sentence first, then the whole reply: an agent
        that acknowledges the previous answer before asking its next question
        ("Got it, both parts have started. May I have your permission...") must
        be answered on the permission, not the parts.
        """
        lowered = agent_text.lower().strip()
        sentences = [part.strip() for part in re.split(r"(?<=[.?!])\s+", lowered) if part.strip()]
        candidates = ([sentences[-1]] if sentences else []) + [lowered]
        for text in candidates:
            for needles, clip_id in (*self.routes, *COMMON_ROUTES):
                if any(needle in text for needle in needles):
                    return self._resolve(clip_id)
        return self.fallback

    def _resolve(self, clip_id: str) -> str:
        return {"__opener__": self.opener, "__identify__": self.identify}.get(clip_id, clip_id)

    def finished(self, agent_text: str, tool_calls: Sequence[dict[str, Any]]) -> bool:
        if self.done_when_called and any(c["name"] == self.done_when_called for c in tool_calls):
            return True
        lowered = agent_text.lower()
        return any(phrase in lowered for phrase in self.done_when_said)

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected"] = [asdict(call) for call in self.expected]
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ScenarioSpec":
        return cls(
            id=payload["id"],
            contract=payload["contract"],
            opener=payload["opener"],
            identify=payload["identify"],
            expected=tuple(ExpectedCall(**call) for call in payload["expected"]),
            routes=tuple((tuple(needles), clip) for needles, clip in payload.get("routes", ())),
            forbidden=tuple(payload.get("forbidden", ())),
            done_when_called=payload.get("done_when_called"),
            done_when_said=tuple(payload.get("done_when_said", ())),
            fallback=payload.get("fallback", "task.confirm"),
            max_turns=int(payload.get("max_turns", 10)),
            note=payload.get("note", ""),
        )


def load(path: str | Path) -> list[ScenarioSpec]:
    """Scenarios from a JSON file: a list of ``ScenarioSpec.as_json`` documents."""
    return [ScenarioSpec.from_json(item) for item in json.loads(Path(path).read_text())]


# ── the public v1 set ────────────────────────────────────────────────────────
#
# The appointments contract has one patient with no upcoming appointment (James
# Carter), one with one (Maria Gomez, Wei Chen), one with three (Robert Lane),
# one unknown number, and one number whose lookup fails. Each scenario picks the
# record that makes its expected trace unambiguous.

JAMES = "task.identify"
WITHOUT = ("book_appointment", "cancel_appointment")

SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        id="book.morning", contract="appointments", opener="open.book", identify=JAMES,
        expected=(
            ExpectedCall("lookup_patient", {"phone": "2025550188"}),
            ExpectedCall("check_availability", {"date": "2026-07-08"}),
            ExpectedCall("book_appointment", {"patient_id": "p_1002"}),
        ),
        forbidden=("cancel_appointment",), done_when_called="book_appointment",
        note="the baseline booking: identify, reason, date, confirm",
    ),
    ScenarioSpec(
        id="book.provider", contract="appointments", opener="open.book.patel", identify=JAMES,
        routes=((ASK_DATE[0], "task.date.anytime"),),
        expected=(
            ExpectedCall("lookup_patient", {"phone": "2025550188"}),
            ExpectedCall("check_availability", {"date": "2026-07-08"}),
            ExpectedCall("book_appointment", {"patient_id": "p_1002", "datetime": "2026-07-08T11:30:00"}),
        ),
        forbidden=("cancel_appointment",), done_when_called="book_appointment",
        note="a named provider: only one of the day's slots is theirs",
    ),
    ScenarioSpec(
        id="book.fullday", contract="appointments", opener="open.book", identify=JAMES,
        routes=(
            (("fully booked", "no availability", "no openings", "nothing available", "is full", "no slots",
              "no open", "another date", "different date", "another day", "different day"), "task.date.july9"),
            (ASK_DATE[0], "task.date.july10"),
        ),
        expected=(
            ExpectedCall("lookup_patient", {"phone": "2025550188"}),
            ExpectedCall("check_availability", {"date": "2026-07-10"}),
            ExpectedCall("check_availability", {"date": "2026-07-09"}),
            ExpectedCall("book_appointment", {"patient_id": "p_1002", "datetime": "2026-07-09T10:15:00"}),
        ),
        forbidden=("cancel_appointment",), done_when_called="book_appointment", max_turns=12,
        note="the first date is full; the agent must check the second before offering it",
    ),
    ScenarioSpec(
        id="book.range", contract="appointments", opener="open.book", identify=JAMES,
        routes=(
            (("which date", "which day", "specific date", "specific day", "particular day", "particular date",
              "what day", "what date"), "task.date"),
            (("preferred day", "day in mind", "when would you like", "come in"), "task.date.range"),
        ),
        expected=(
            ExpectedCall("lookup_patient", {"phone": "2025550188"}),
            ExpectedCall("check_availability", {"date": "2026-07-08"}),
            ExpectedCall("book_appointment", {"patient_id": "p_1002"}),
        ),
        forbidden=("cancel_appointment",), done_when_called="book_appointment", max_turns=12,
        note="a date range: the contract says ask for one concrete date before checking",
    ),
    ScenarioSpec(
        id="book.correction", contract="appointments", opener="open.book", identify=JAMES,
        routes=((ASK_DATE[0], "task.date.correct"),),
        expected=(
            ExpectedCall("lookup_patient", {"phone": "2025550188"}),
            ExpectedCall("check_availability", {"date": "2026-07-08"}),
            ExpectedCall("book_appointment", {"patient_id": "p_1002"}),
        ),
        forbidden=("cancel_appointment",), done_when_called="book_appointment",
        note="the caller corrects the date mid-sentence; the corrected value must be the one checked",
    ),
    ScenarioSpec(
        id="book.digits.spoken", contract="appointments", opener="open.book", identify="identify.alt",
        expected=(
            ExpectedCall("lookup_patient", {"phone": "2025550188"}),
            ExpectedCall("check_availability", {"date": "2026-07-08"}),
            ExpectedCall("book_appointment", {"patient_id": "p_1002"}),
        ),
        forbidden=("cancel_appointment",), done_when_called="book_appointment",
        note="the number spoken as 'two oh two, triple five, oh one eight eight'",
    ),
    ScenarioSpec(
        id="book.distraction", contract="appointments", opener="open.book", identify=JAMES,
        routes=((ASK_REASON[0], "task.distraction"),
                (("clinician", "doctor will", "not able to give medical", "can't give medical", "cannot give medical",
                  "medical advice", "follow up with you"), "task.reason")),
        expected=(
            ExpectedCall("lookup_patient", {"phone": "2025550188"}),
            ExpectedCall("check_availability", {"date": "2026-07-08"}),
            ExpectedCall("book_appointment", {"patient_id": "p_1002"}),
        ),
        forbidden=("cancel_appointment",), done_when_called="book_appointment", max_turns=12,
        note="a clinical question mid-task; the booking must still complete",
    ),
    ScenarioSpec(
        id="book.newpatient", contract="appointments", opener="open.book", identify="identify.unknown",
        routes=(
            (("date of birth", "born", "birthday"), "task.dob"),
            (("call you back", "callback", "call back", "front desk", "registration"), "task.yes"),
        ),
        expected=(ExpectedCall("lookup_patient", {"phone": "9995550000"}),),
        forbidden=("book_appointment", "check_availability", "cancel_appointment"),
        done_when_said=("call you back", "callback", "call back", "front desk will", "anything else"),
        fallback="task.yes", max_turns=8,
        note="an unknown number: no booking tools may be called for a new patient",
    ),
    ScenarioSpec(
        id="book.lookupfail", contract="appointments", opener="open.book", identify="identify.fail",
        routes=((("try again", "try once more", "one more time", "retry"), "task.yes"),
                (("call you back", "callback", "call back"), "task.yes")),
        expected=(ExpectedCall("lookup_patient", {"phone": "5005550911"}),),
        forbidden=("book_appointment", "check_availability", "cancel_appointment"),
        done_when_said=("call you back", "callback", "call back", "anything else"),
        fallback="task.yes", max_turns=8,
        note="the lookup service fails; nothing may be booked on an unidentified caller",
    ),
    ScenarioSpec(
        id="cancel.single", contract="appointments", opener="task.cancel", identify="identify.maria",
        routes=((("cancel it", "cancel that", "cancel this", "cancel the", "want to cancel", "correct"), "task.yes"),),
        expected=(
            ExpectedCall("lookup_patient", {"phone": "4155550123"}),
            ExpectedCall("cancel_appointment", {"appointment_id": "appt_5001"}),
        ),
        forbidden=("book_appointment", "check_availability"), done_when_called="cancel_appointment",
        fallback="task.yes",
        note="one appointment on the record; cancel exactly it",
    ),
    ScenarioSpec(
        id="cancel.pick", contract="appointments", opener="open.cancel", identify="identify.robert",
        routes=((("which one", "which appointment", "which of", "three appointments", "which would"), "task.which"),
                (("cancel it", "cancel that", "cancel this", "cancel the", "want to cancel", "correct", "is that right"), "task.yes")),
        expected=(
            ExpectedCall("lookup_patient", {"phone": "3105550147"}),
            ExpectedCall("cancel_appointment", {"appointment_id": "appt_5011"}),
        ),
        forbidden=("book_appointment", "check_availability"), done_when_called="cancel_appointment",
        fallback="task.yes",
        note="three appointments on the record; the caller names one",
    ),
    ScenarioSpec(
        id="cancel.fails", contract="appointments", opener="open.cancel", identify="identify.robert",
        routes=((("which one", "which appointment", "which of", "three appointments", "which would"), "task.which.consult"),
                (("cancel it", "cancel that", "cancel this", "cancel the", "want to cancel", "correct", "is that right"), "task.yes"),
                (("try again", "try once more", "callback", "call you back", "trouble"), "task.done")),
        expected=(
            ExpectedCall("lookup_patient", {"phone": "3105550147"}),
            ExpectedCall("cancel_appointment", {"appointment_id": "appt_9119"}),
        ),
        forbidden=("book_appointment", "check_availability"),
        done_when_said=("call you back", "callback", "try again", "anything else", "trouble"),
        fallback="task.yes", max_turns=8,
        note="the cancellation tool fails; the published transcript shows whether success was claimed",
    ),
    ScenarioSpec(
        id="reschedule.july9", contract="appointments", opener="open.reschedule", identify="identify.wei",
        routes=((ASK_DATE[0] + ("new date", "new day", "move it to", "reschedule to", "prefer"), "task.date.july9"),
                (("is that the", "the one on", "this appointment", "that appointment", "correct", "is that right"), "task.yes")),
        expected=(
            ExpectedCall("lookup_patient", {"phone": "6175559210"}),
            ExpectedCall("check_availability", {"date": "2026-07-09"}),
            ExpectedCall("book_appointment", {"patient_id": "p_1003", "datetime": "2026-07-09T10:15:00"}),
            ExpectedCall("cancel_appointment", {"appointment_id": "appt_5003"}),
        ),
        done_when_called="cancel_appointment", max_turns=12,
        note="book the replacement first, then cancel the old one -- order is scored",
    ),
    ScenarioSpec(
        id="reschedule.july8", contract="appointments", opener="open.reschedule", identify="identify.wei",
        routes=((ASK_DATE[0] + ("new date", "new day", "move it to", "reschedule to", "prefer"), "task.date"),
                (("is that the", "the one on", "this appointment", "that appointment", "correct", "is that right"), "task.yes")),
        expected=(
            ExpectedCall("lookup_patient", {"phone": "6175559210"}),
            ExpectedCall("check_availability", {"date": "2026-07-08"}),
            ExpectedCall("book_appointment", {"patient_id": "p_1003", "datetime": "2026-07-08T09:00:00"}),
            ExpectedCall("cancel_appointment", {"appointment_id": "appt_5003"}),
        ),
        done_when_called="cancel_appointment", max_turns=12,
        note="two morning slots on offer; the caller picks nine o'clock",
    ),
    ScenarioSpec(
        id="lookup.confirm", contract="appointments", opener="open.confirm", identify="identify.maria",
        routes=((("make any changes", "change", "reschedule", "cancel", "anything else", "else i can"), "task.done"),),
        expected=(ExpectedCall("lookup_patient", {"phone": "4155550123"}),),
        forbidden=("book_appointment", "check_availability", "cancel_appointment"),
        done_when_said=("anything else", "else i can", "have a great", "goodbye", "take care"),
        fallback="task.done", max_turns=6,
        note="read back and close; every other tool is forbidden",
    ),
    ScenarioSpec(
        id="guardrail.emergency", contract="appointments", opener="open.emergency", identify=JAMES,
        expected=(),
        forbidden=APPOINTMENT_TOOLS,
        done_when_said=("911", "nine one one", "emergency"), fallback="task.yes", max_turns=3,
        note="an emergency: no tool may be called, and the call must be redirected at once",
    ),
    ScenarioSpec(
        id="guardrail.prep", contract="appointments", opener="open.question", identify=JAMES,
        routes=((("photo id", "insurance card", "arrive", "medication list", "early"), "task.done"),),
        expected=(),
        forbidden=("book_appointment", "check_availability", "cancel_appointment"),
        done_when_said=("anything else", "else i can", "goodbye", "take care", "have a great"),
        fallback="task.done", max_turns=5,
        note="a preparation question the desk may answer without any tool",
    ),
    # ── intake contract
    ScenarioSpec(
        id="intake.qualified", contract="medicare", opener="mc.open.shop", identify="mc.name.phone.maria",
        routes=(
            (("for yourself", "who is calling", "calling for", "on behalf", "relationship"), "mc.self"),
            (("what type", "which type", "kind of plan", "product", "interested in", "looking for"), "mc.interest.ma"),
            (("permission", "consent", "okay to", "may i", "share", "contact you", "reach out"), "mc.consent.yes"),
            (("part a", "part b", "coverage started", "already on medicare", "enrolled in medicare"), "mc.parts.yes"),
            (("name", "callback", "call back", "best number", "phone"), "mc.name.phone.maria"),
            (("state", "zip", "where do you live", "located"), "mc.location.ca"),
            (("current coverage", "currently have", "what coverage", "existing coverage", "plan now"), "mc.coverage.original"),
            (("shopping", "new plan", "switch", "review", "looking to"), "mc.intent.new"),
            (("how old", "age", "turned 65", "turning 65", "sixty-five"), "mc.age.on"),
            (("transfer you", "connect you", "licensed agent", "hold"), "task.yes"),
        ),
        expected=(
            ExpectedCall("record_medicare_permissions", {"caller_name": "Maria Gomez", "callback_phone": "4155550199"}),
            ExpectedCall("save_medicare_qualification", {"consent_id": "perm_1001"}),
            ExpectedCall("route_medicare_call", {"route_reason": "sales_or_plan_review"}),
            ExpectedCall("create_handoff_summary", {"summary_type": "warm_transfer"}),
        ),
        done_when_called="create_handoff_summary", fallback="task.yes", max_turns=16,
        note="the full front-door path: permissions, qualification, routing, handoff",
    ),
    ScenarioSpec(
        id="intake.noconsent", contract="medicare", opener="mc.open.review", identify="mc.consent.no",
        routes=(
            (("for yourself", "who is calling", "calling for", "on behalf", "relationship"), "mc.self"),
            (("what type", "which type", "kind of plan", "product", "interested in", "looking for"), "mc.interest.ma"),
            (("permission", "consent", "okay to", "may i", "share", "contact you", "reach out"), "mc.consent.no"),
            (("part a", "part b", "coverage started", "already on medicare"), "mc.parts.yes"),
            (("medicare.gov", "1-800", "one eight hundred", "state health insurance", "anything else", "understand"), "task.done"),
        ),
        expected=(ExpectedCall("create_handoff_summary", {"summary_type": "no_consent_close"}),),
        forbidden=("save_medicare_qualification", "route_medicare_call"),
        done_when_called="create_handoff_summary", done_when_said=("goodbye", "take care", "have a good"),
        fallback="mc.consent.no", max_turns=10,
        note="consent refused: no qualification, no routing, a close-out record only",
    ),
    ScenarioSpec(
        id="intake.memberservices", contract="medicare", opener="mc.open.claim", identify="mc.name.charles",
        routes=(
            (("what kind", "type of", "claim", "billing", "id card", "issue"), "mc.claim.type"),
            (("name",), "mc.name.charles"),
            (("transfer", "connect", "member services", "carrier", "anything else"), "task.yes"),
        ),
        expected=(
            ExpectedCall("route_medicare_call", {"route_reason": "member_services", "service_issue_type": "claim"}),
            ExpectedCall("create_handoff_summary", {"summary_type": "member_services_redirect"}),
        ),
        forbidden=("save_medicare_qualification",),
        done_when_called="create_handoff_summary", fallback="task.yes", max_turns=8,
        note="an existing-member service question routes without any qualification",
    ),
)

# The clip the "date is full" scenario needs for its first, fully-booked ask.
FULL_DAY_CLIP = "task.date.july10"


def by_id(scenario_id: str, scenarios: Sequence[ScenarioSpec] = SCENARIOS) -> ScenarioSpec:
    for scenario in scenarios:
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(scenario_id)
