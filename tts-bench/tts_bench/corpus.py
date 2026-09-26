"""The TTS corpus: what a voice agent actually has to say, cohorted by what makes it hard.

Every item carries the text an agent would send and a ``spoken_reference``: the
same content written the way it should sound. Round-trip scoring compares the
transcript of the synthesised audio with both and keeps the better match, so a
service is never punished for reading ``$347.89`` correctly, and never excused
for reading it wrong.

Production-shaped, invented values. Nothing here came from a call.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

CORPUS_VERSION = "0.1.0"

# cohort -> what the cohort tests
COHORTS = {
    "prose": "ordinary agent sentences; the baseline every other cohort is read against",
    "currency": "amounts with cents, thousands separators and percentages",
    "datetime": "dates, clock times, durations and date ranges",
    "phone": "telephone numbers with grouping, extensions and repeated digits",
    "alnum": "confirmation codes and identifiers mixing letters and digits",
    "spelled": "sequences that must be spelled letter by letter",
    "contact": "email addresses, URLs and street addresses",
    "names": "personal and place names whose pronunciation is not spelling-derived",
    "repair": "agent-side self-correction and hesitation written into the text",
    "units": "abbreviations, units and titles that expand in speech",
    "questions": "questions whose meaning depends on rising intonation",
    "long": "multi-sentence turns of fifty words or more",
}


@dataclass(frozen=True)
class Item:
    id: str
    cohort: str
    text: str
    spoken_reference: str
    note: str = ""

    @property
    def words(self) -> int:
        return len(self.text.split())


def _i(id: str, cohort: str, text: str, spoken: str, note: str = "") -> Item:
    assert cohort in COHORTS, cohort
    return Item(id, cohort, text, spoken, note)


ITEMS: tuple[Item, ...] = (
    # ── prose ─────────────────────────────────────────────────────────────
    _i("prose.greet", "prose", "Thanks for calling Cedar Valley Family Practice, this is Riley. How can I help you today?",
       "thanks for calling cedar valley family practice this is riley how can i help you today"),
    _i("prose.hold", "prose", "Sure, give me just a moment while I pull that up for you.",
       "sure give me just a moment while i pull that up for you"),
    _i("prose.confirm", "prose", "All set. I've booked that for you and you'll get a text confirmation shortly.",
       "all set i've booked that for you and you'll get a text confirmation shortly"),
    _i("prose.apology", "prose", "I'm sorry about that. Let me see what I can do to make it right.",
       "i'm sorry about that let me see what i can do to make it right"),
    _i("prose.transfer", "prose", "I'll transfer you to a member of our billing team who can look into that further.",
       "i'll transfer you to a member of our billing team who can look into that further"),
    _i("prose.anything", "prose", "Is there anything else I can help you with before we wrap up?",
       "is there anything else i can help you with before we wrap up"),
    _i("prose.closing", "prose", "Perfect. Thanks again for calling, and have a great rest of your day.",
       "perfect thanks again for calling and have a great rest of your day"),
    _i("prose.instructions", "prose", "Please arrive ten minutes early and bring your insurance card and a photo ID.",
       "please arrive ten minutes early and bring your insurance card and a photo i d"),
    # ── currency ──────────────────────────────────────────────────────────
    _i("currency.balance", "currency", "Your current balance is $1,347.09.",
       "your current balance is one thousand three hundred forty seven dollars and nine cents"),
    _i("currency.copay", "currency", "The copay for that visit is $35, and the remaining $212.50 goes toward your deductible.",
       "the copay for that visit is thirty five dollars and the remaining two hundred twelve dollars and fifty cents goes toward your deductible"),
    _i("currency.refund", "currency", "A refund of $89.99 was issued on the card ending in 4417.",
       "a refund of eighty nine dollars and ninety nine cents was issued on the card ending in four four one seven"),
    _i("currency.percent", "currency", "That plan is 20% off this month, which saves you $359.88 over the year.",
       "that plan is twenty percent off this month which saves you three hundred fifty nine dollars and eighty eight cents over the year"),
    _i("currency.large", "currency", "The quote came to $12,480.00 before tax, or $13,478.40 including it.",
       "the quote came to twelve thousand four hundred eighty dollars before tax or thirteen thousand four hundred seventy eight dollars and forty cents including it"),
    _i("currency.small", "currency", "There's a $0.75 processing fee on each transaction.",
       "there's a seventy five cent processing fee on each transaction"),
    # ── datetime ──────────────────────────────────────────────────────────
    _i("datetime.appt", "datetime", "Your appointment is on Tuesday, March 5th at 10:30 AM.",
       "your appointment is on tuesday march fifth at ten thirty a m"),
    _i("datetime.options", "datetime", "I can offer 11:15 AM on Wednesday or 2:45 PM on Thursday.",
       "i can offer eleven fifteen a m on wednesday or two forty five p m on thursday"),
    _i("datetime.range", "datetime", "The technician will arrive between 8:00 and 12:00 on July 9th.",
       "the technician will arrive between eight and twelve on july ninth"),
    _i("datetime.iso", "datetime", "The order was placed on 2026-03-14 and shipped two days later.",
       "the order was placed on march fourteenth twenty twenty six and shipped two days later"),
    _i("datetime.duration", "datetime", "The procedure takes about 45 minutes, and you'll be with us for roughly 1.5 hours in total.",
       "the procedure takes about forty five minutes and you'll be with us for roughly one and a half hours in total"),
    _i("datetime.year", "datetime", "The policy renews on the 1st of January, 2027.",
       "the policy renews on the first of january twenty twenty seven"),
    # ── phone ─────────────────────────────────────────────────────────────
    _i("phone.callback", "phone", "You can reach us at 555-013-7742.",
       "you can reach us at five five five zero one three seven seven four two"),
    _i("phone.readback", "phone", "I have your number as (555) 020-8816. Is that right?",
       "i have your number as five five five zero two zero eight eight one six is that right"),
    _i("phone.extension", "phone", "Call 555-014-3300, extension 2041, and ask for scheduling.",
       "call five five five zero one four three three zero zero extension two zero four one and ask for scheduling"),
    _i("phone.repeated", "phone", "The pharmacy's number is 555-011-1000.",
       "the pharmacy's number is five five five zero one one one zero zero zero"),
    _i("phone.tollfree", "phone", "For claims, dial 1-800-555-0199.",
       "for claims dial one eight hundred five five five zero one nine nine"),
    # ── alnum ─────────────────────────────────────────────────────────────
    _i("alnum.confirmation", "alnum", "Your confirmation code is K4Q72.",
       "your confirmation code is k four q seven two"),
    _i("alnum.order", "alnum", "Order ORD-458291 shipped this morning.",
       "order o r d four five eight two nine one shipped this morning"),
    _i("alnum.tracking", "alnum", "The tracking number is 1Z 7A4 X29 0341 8867.",
       "the tracking number is one z seven a four x two nine zero three four one eight eight six seven"),
    _i("alnum.member", "alnum", "I see member ID CW5001 on the account.",
       "i see member i d c w five zero zero one on the account"),
    _i("alnum.confusable", "alnum", "The code is B0D1-I8O0. That's B as in bravo, zero, D as in delta, one.",
       "the code is b zero d one i eight o zero that's b as in bravo zero d as in delta one",
       "letters and digits that look alike"),
    _i("alnum.ticket", "alnum", "Your ticket number is INC0091827, and it's been assigned to the network team.",
       "your ticket number is i n c zero zero nine one eight two seven and it's been assigned to the network team"),
    # ── spelled ───────────────────────────────────────────────────────────
    _i("spelled.surname", "spelled", "That's Smythe, spelled S-M-Y-T-H-E.",
       "that's smythe spelled s m y t h e"),
    _i("spelled.street", "spelled", "Elmhurst, that's E-L-M-H-U-R-S-T.",
       "elmhurst that's e l m h u r s t"),
    _i("spelled.email_local", "spelled", "The username is jdoe, J-D-O-E, all lowercase.",
       "the username is j doe j d o e all lowercase"),
    _i("spelled.plate", "spelled", "License plate 7 H K J 2 2 9.",
       "license plate seven h k j two two nine"),
    # ── contact ───────────────────────────────────────────────────────────
    _i("contact.email", "contact", "Send it to billing.support@example.com.",
       "send it to billing dot support at example dot com"),
    _i("contact.url", "contact", "You can manage it online at example.com/account/settings.",
       "you can manage it online at example dot com slash account slash settings"),
    _i("contact.address", "contact", "We're at 1480 N. Maple Ave., Suite 210, Springfield.",
       "we're at fourteen eighty north maple avenue suite two ten springfield"),
    _i("contact.zip", "contact", "The ZIP code on file is 62704.",
       "the zip code on file is six two seven zero four"),
    _i("contact.email_digits", "contact", "I have m.rivera88@example.org, is that still current?",
       "i have m dot rivera eighty eight at example dot org is that still current"),
    # ── names ─────────────────────────────────────────────────────────────
    _i("names.nguyen", "names", "I have Dr. Nguyen available on Monday.", "i have doctor nguyen available on monday"),
    _i("names.siobhan", "names", "Siobhan Ó Briain called about the invoice.", "siobhan o'brien called about the invoice"),
    _i("names.joaquin", "names", "The appointment is with Joaquín Ibarra.", "the appointment is with joaquin ibarra"),
    _i("names.ngozi", "names", "Please hold for Ngozi Okonkwo.", "please hold for ngozi okonkwo"),
    _i("names.place", "names", "The clinic is in Worcester, near the Leicester exit.", "the clinic is in worcester near the leicester exit"),
    # ── repair ────────────────────────────────────────────────────────────
    _i("repair.actually", "repair", "That's on the 4th... actually, no, the 5th. Sorry about that.",
       "that's on the fourth actually no the fifth sorry about that"),
    _i("repair.letme", "repair", "It looks like, um, let me check that again for you.",
       "it looks like um let me check that again for you"),
    _i("repair.restart", "repair", "So the total is... one second. The total is $48.20.",
       "so the total is one second the total is forty eight dollars and twenty cents"),
    _i("repair.misspoke", "repair", "I said Tuesday, I meant Thursday. Thursday at 3.",
       "i said tuesday i meant thursday thursday at three"),
    # ── units ─────────────────────────────────────────────────────────────
    _i("units.dose", "units", "Take 500 mg twice a day, with food.", "take five hundred milligrams twice a day with food"),
    _i("units.eta", "units", "The ETA is 3 PM ET, and the driver will text when they're 10 min away.",
       "the e t a is three p m eastern and the driver will text when they're ten minutes away"),
    _i("units.titles", "units", "Dr. Patel and Mr. St. John will both be at the 2 o'clock.",
       "doctor patel and mister saint john will both be at the two o'clock"),
    _i("units.measure", "units", "It's about 2.5 km from the station, roughly a 30-minute walk.",
       "it's about two and a half kilometers from the station roughly a thirty minute walk"),
    # ── questions ─────────────────────────────────────────────────────────
    _i("questions.confirm", "questions", "You'd like to cancel the appointment on the 12th, is that right?",
       "you'd like to cancel the appointment on the twelfth is that right"),
    _i("questions.which", "questions", "Would the morning or the afternoon work better for you?",
       "would the morning or the afternoon work better for you"),
    _i("questions.clarify", "questions", "Sorry, did you say fifteen or fifty?", "sorry did you say fifteen or fifty"),
    _i("questions.open", "questions", "And what's the best number to reach you on?", "and what's the best number to reach you on"),
    # ── long ──────────────────────────────────────────────────────────────
    _i("long.prep", "long",
       "Before the appointment, please don't eat or drink anything after midnight, except small sips of water with your "
       "regular medication. Bring a list of everything you take, including over-the-counter items. If you use a blood "
       "thinner, call us today so the doctor can advise you.",
       "before the appointment please don't eat or drink anything after midnight except small sips of water with your "
       "regular medication bring a list of everything you take including over the counter items if you use a blood "
       "thinner call us today so the doctor can advise you"),
    _i("long.policy", "long",
       "Our cancellation policy asks for 24 hours' notice. If you cancel with less notice than that, there's a $25 fee, "
       "which we waive for emergencies. You can cancel by phone, through the patient portal, or by replying to the "
       "reminder text.",
       "our cancellation policy asks for twenty four hours notice if you cancel with less notice than that there's a "
       "twenty five dollar fee which we waive for emergencies you can cancel by phone through the patient portal or by "
       "replying to the reminder text"),
    _i("long.summary", "long",
       "So to summarize: you're booked with Dr. Nguyen on Thursday, March 12th at 2:45 PM, your copay will be $35, and "
       "we have your number as 555-013-7742. I'll send the confirmation to m.rivera88@example.org.",
       "so to summarize you're booked with doctor nguyen on thursday march twelfth at two forty five p m your copay will "
       "be thirty five dollars and we have your number as five five five zero one three seven seven four two i'll send "
       "the confirmation to m dot rivera eighty eight at example dot org"),
    _i("long.troubleshoot", "long",
       "Let's try a reset. Unplug the router, wait a full 30 seconds, and plug it back in. The lights will blink for "
       "about two minutes. Once the front light is solid green, try connecting again and let me know what you see.",
       "let's try a reset unplug the router wait a full thirty seconds and plug it back in the lights will blink for "
       "about two minutes once the front light is solid green try connecting again and let me know what you see"),
)

# Fixed items the probes that do not care about content reach for.
SENTINEL_ITEM = "prose.greet"
LONG_ITEM = "long.prep"


def by_id() -> dict[str, Item]:
    return {item.id: item for item in ITEMS}


def by_cohort(items: Iterable[Item] = ITEMS) -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {}
    for item in items:
        out.setdefault(item.cohort, []).append(item)
    return out


def as_json(items: Iterable[Item] = ITEMS, version: str = CORPUS_VERSION) -> dict[str, Any]:
    return {"version": version, "cohorts": COHORTS, "items": [asdict(i) for i in items]}


def load(path: str | Path) -> tuple[str, list[Item]]:
    """A corpus from a file in the same shape, for a holdout or an external set."""
    payload = json.loads(Path(path).read_text())
    items = [Item(**{k: v for k, v in raw.items() if k in Item.__dataclass_fields__}) for raw in payload["items"]]
    return str(payload.get("version", "external")), items
