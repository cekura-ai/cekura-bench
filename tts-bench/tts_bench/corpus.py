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

CORPUS_VERSION = "0.2.0"

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
    "paragraph": "agent turns near 500 characters: throughput on the prompt length other boards time, and stability over a long utterance",
    "heteronym": "words spelled alike and said differently, where only the context decides",
    "abbrev": "abbreviations, acronyms and initialisms that expand, stay letters, or become a word",
    "symbols": "symbols, codes, web addresses and keypad instructions read as a caller needs them",
    "terms": "standalone domain terms (medications, procedures) that have one right pronunciation",
    "numbers": "ordinals, fractions, decimals, ranges, temperatures, roman numerals and years",
}

# The four pronunciation categories a span is filed under.
CATEGORIES = {
    "term": "a standalone word or name with one accepted pronunciation",
    "context": "a reading the surrounding words decide (heteronyms, St., Dr.)",
    "sequence": "a string that must come out exactly (codes, emails, web addresses, keypad keys)",
    "shorthand": "written shorthand that expands (numbers, dates, units, currency, symbols)",
}


@dataclass(frozen=True)
class Span:
    """One hard part of an item, scored on its own.

    ``accept`` lists the readings that count as right, written as words; a
    span passes when one of them appears in the transcript after the scoring
    normaliser. A span whose right reading a transcript cannot show (a
    heteronym is spelled the same either way) is ``check="audio"``: it is
    labelled for a listener or an audio judge, and no transcript scores it.
    """

    text: str
    category: str
    accept: tuple[str, ...] = ()
    check: str = "transcript"          # or "audio"
    note: str = ""


@dataclass(frozen=True)
class Item:
    id: str
    cohort: str
    text: str
    spoken_reference: str
    note: str = ""
    spans: tuple[Span, ...] = ()

    @property
    def words(self) -> int:
        return len(self.text.split())


def _i(id: str, cohort: str, text: str, spoken: str, note: str = "", spans: tuple[Span, ...] = ()) -> Item:
    assert cohort in COHORTS, cohort
    for span in spans:
        assert span.category in CATEGORIES and span.text in text, (id, span.text)
        assert span.check == "audio" or span.accept, (id, span.text)
    return Item(id, cohort, text, spoken, note, spans)


def _s(text: str, category: str, *accept: str, check: str = "transcript", note: str = "") -> Span:
    return Span(text, category, tuple(accept), check, note)


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
    # ── paragraph ─────────────────────────────────────────────────────────
    _i("paragraph.benefits", "paragraph",
       "Here's how your plan works for this visit. Because it's an in-network specialist, you'll pay a $40 copay at "
       "check-in, and your deductible doesn't apply. If the doctor orders lab work, that's covered at 80 percent after "
       "the deductible, and you've met $620 of your $1,500 deductible so far this year. Imaging like an MRI needs prior "
       "authorization, which usually takes three to five business days, so we'd submit that request as soon as it's "
       "ordered. Do you want me to email you a summary of all this?",
       "here's how your plan works for this visit because it's an in network specialist you'll pay a forty dollar copay "
       "at check in and your deductible doesn't apply if the doctor orders lab work that's covered at eighty percent "
       "after the deductible and you've met six hundred twenty dollars of your one thousand five hundred dollar "
       "deductible so far this year imaging like an m r i needs prior authorization which usually takes three to five "
       "business days so we'd submit that request as soon as it's ordered do you want me to email you a summary of all "
       "this"),
    _i("paragraph.delivery", "paragraph",
       "I can see the order now. It shipped yesterday from our Memphis warehouse and it's currently in transit with the "
       "carrier. The latest scan shows it left the regional hub at 6:40 this morning, and the estimated delivery is "
       "Friday between 9 AM and 1 PM. Someone will need to sign for it because the total is over $500. If nobody's "
       "home, the driver will leave a notice and try again the next business day, or you can pick it up from the local "
       "depot after 4 PM with a photo ID.",
       "i can see the order now it shipped yesterday from our memphis warehouse and it's currently in transit with the "
       "carrier the latest scan shows it left the regional hub at six forty this morning and the estimated delivery is "
       "friday between nine a m and one p m someone will need to sign for it because the total is over five hundred "
       "dollars if nobody's home the driver will leave a notice and try again the next business day or you can pick it "
       "up from the local depot after four p m with a photo i d"),
    _i("paragraph.discharge", "paragraph",
       "Before you go, let me walk you through the discharge instructions. Keep the dressing dry for the first 48 hours, "
       "then you can shower normally. Take the ibuprofen every six hours as needed for pain, but no more than four "
       "doses in a day, and always with food. If you notice a fever above 101 degrees, redness spreading from the "
       "incision, or bleeding that won't stop, call the after-hours line right away. Your follow-up visit is booked for "
       "Tuesday the 18th at 10:15, and the nurse will remove the stitches then.",
       "before you go let me walk you through the discharge instructions keep the dressing dry for the first forty eight "
       "hours then you can shower normally take the ibuprofen every six hours as needed for pain but no more than four "
       "doses in a day and always with food if you notice a fever above one hundred one degrees redness spreading from "
       "the incision or bleeding that won't stop call the after hours line right away your follow up visit is booked "
       "for tuesday the eighteenth at ten fifteen and the nurse will remove the stitches then"),
    _i("paragraph.account", "paragraph",
       "Thanks, I've verified your identity. I can see two charges from September 3rd that you don't recognize, one for "
       "$89.99 and one for $12.50, both from an online merchant. I've frozen the card ending in 4417 so no new charges "
       "can go through, and I've opened a dispute for both amounts. You'll see a temporary credit within two business "
       "days while we investigate. A replacement card will arrive in five to seven days, and you'll get a text when it "
       "ships. Is there anything else on the account you'd like me to check?",
       "thanks i've verified your identity i can see two charges from september third that you don't recognize one for "
       "eighty nine dollars and ninety nine cents and one for twelve dollars and fifty cents both from an online "
       "merchant i've frozen the card ending in four four one seven so no new charges can go through and i've opened a "
       "dispute for both amounts you'll see a temporary credit within two business days while we investigate a "
       "replacement card will arrive in five to seven days and you'll get a text when it ships is there anything else "
       "on the account you'd like me to check"),
    # ── heteronym ─────────────────────────────────────────────────────────
    _i("heteronym.read", "heteronym", "Please read the notice. I read it to you yesterday, but it's worth another look.",
       "please read the notice i read it to you yesterday but it's worth another look",
       spans=(_s("read the notice", "context", check="audio", note="present tense, reed"),
              _s("I read it", "context", check="audio", note="past tense, red"))),
    _i("heteronym.lead", "heteronym", "The lead nurse asked whether the old pipes still have lead in them.",
       "the lead nurse asked whether the old pipes still have lead in them",
       spans=(_s("lead nurse", "context", check="audio", note="leed"), _s("have lead", "context", check="audio", note="led, the metal"))),
    _i("heteronym.record", "heteronym", "We record every call, so there will be a record of what we agreed.",
       "we record every call so there will be a record of what we agreed",
       spans=(_s("We record", "context", check="audio", note="verb, stress on the second syllable"),
              _s("a record", "context", check="audio", note="noun, stress on the first syllable"))),
    _i("heteronym.close", "heteronym", "We're close to closing, so please close the form and submit it now.",
       "we're close to closing so please close the form and submit it now",
       spans=(_s("close to closing", "context", check="audio", note="near, soft s"), _s("close the form", "context", check="audio", note="shut, z sound"))),
    _i("heteronym.live", "heteronym", "Do you live in the county? The live chat can confirm your coverage area.",
       "do you live in the county the live chat can confirm your coverage area",
       spans=(_s("live in", "context", check="audio", note="liv"), _s("live chat", "context", check="audio", note="lyve"))),
    # ── abbrev ────────────────────────────────────────────────────────────
    _i("abbrev.street", "abbrev", "Our office on Main St. is right next to St. Mary's Hospital.",
       "our office on main street is right next to saint mary's hospital",
       spans=(_s("Main St.", "context", "main street"), _s("St. Mary's", "context", "saint mary's", "saint marys"))),
    _i("abbrev.drive", "abbrev", "Dr. Okafor's new clinic is on Lakeview Dr., past the second light.",
       "doctor okafor's new clinic is on lakeview drive past the second light",
       spans=(_s("Dr. Okafor's", "context", "doctor okafor's", "doctor okafors"), _s("Lakeview Dr.", "context", "lakeview drive"))),
    _i("abbrev.shorthand", "abbrev", "The appt. is approx. 45 min., incl. check-in and vitals.",
       "the appointment is approximately forty five minutes including check in and vitals",
       spans=(_s("appt.", "shorthand", "appointment"), _s("approx.", "shorthand", "approximately"),
              _s("45 min.", "shorthand", "45 minutes"), _s("incl.", "shorthand", "including"))),
    _i("abbrev.acronyms", "abbrev", "Please send the HIPAA form ASAP, and check the FAQ before you call back.",
       "please send the hipaa form asap and check the f a q before you call back",
       spans=(_s("HIPAA", "term", check="audio", note="said as a word, hip-uh"),
              _s("ASAP", "shorthand", check="audio", note="letters or a-sap; both accepted"),
              _s("FAQ", "shorthand", check="audio", note="letters or fak; both accepted"))),
    _i("abbrev.initialisms", "abbrev", "Your PCP sent the MRI results to the ER at 7 a.m.",
       "your p c p sent the m r i results to the e r at seven a m",
       spans=(_s("PCP", "sequence", "pcp"), _s("MRI", "sequence", "mri"), _s("ER", "sequence", "er"),
              _s("7 a.m.", "shorthand", "7 am", "7 a m"))),
    # ── symbols ───────────────────────────────────────────────────────────
    _i("symbols.promo", "symbols", "Use code SAVE20 for 20% off orders over $50 & free shipping.",
       "use code save twenty for twenty percent off orders over fifty dollars and free shipping",
       spans=(_s("SAVE20", "sequence", "save 20", "save20"), _s("20%", "shorthand", "20 percent", "20%"),
              _s("$50", "shorthand", "50 dollars", "$50"), _s("&", "shorthand", "and"))),
    _i("symbols.url", "symbols", "Go to example.com/support/billing and choose Contact Us.",
       "go to example dot com slash support slash billing and choose contact us",
       spans=(_s("example.com/support/billing", "sequence", "example dot com slash support slash billing",
                 "example.com/support/billing", "example.com slash support slash billing"),)),
    _i("symbols.keypad", "symbols", "Press # and then 2, or press * to hear the menu again.",
       "press pound and then two or press star to hear the menu again",
       spans=(_s("#", "sequence", "pound", "hash", "number sign"), _s("*", "sequence", "star", "asterisk"))),
    _i("symbols.email", "symbols", "You can write to help@north-clinic.org or call 1-800-555-0199.",
       "you can write to help at north dash clinic dot org or call one eight hundred five five five zero one nine nine",
       spans=(_s("help@north-clinic.org", "sequence", "help at north dash clinic dot org", "help@north-clinic.org",
                 "help at north-clinic.org"),
              _s("1-800-555-0199", "sequence", "1 800 555 0199", "18005550199", "1-800-555-0199"))),
    # ── terms ─────────────────────────────────────────────────────────────
    _i("terms.thyroid", "terms", "Your prescriptions for levothyroxine and metformin are ready for pickup.",
       "your prescriptions for levothyroxine and metformin are ready for pickup",
       spans=(_s("levothyroxine", "term", "levothyroxine"), _s("metformin", "term", "metformin"))),
    _i("terms.fever", "terms", "For the fever, the doctor suggests acetaminophen rather than ibuprofen.",
       "for the fever the doctor suggests acetaminophen rather than ibuprofen",
       spans=(_s("acetaminophen", "term", "acetaminophen"), _s("ibuprofen", "term", "ibuprofen"))),
    _i("terms.referral", "terms", "The referral is to gastroenterology for a colonoscopy consult.",
       "the referral is to gastroenterology for a colonoscopy consult",
       spans=(_s("gastroenterology", "term", "gastroenterology"), _s("colonoscopy", "term", "colonoscopy"))),
    _i("terms.allergy", "terms", "Your chart lists an allergy to amoxicillin and a history of anaphylaxis.",
       "your chart lists an allergy to amoxicillin and a history of anaphylaxis",
       spans=(_s("amoxicillin", "term", "amoxicillin"), _s("anaphylaxis", "term", "anaphylaxis"))),
    # ── numbers ───────────────────────────────────────────────────────────
    _i("numbers.ordinal", "numbers", "It's the 3rd door on the left, about ¾ of the way down the hall.",
       "it's the third door on the left about three quarters of the way down the hall",
       spans=(_s("3rd", "shorthand", "3rd", "third"), _s("¾", "shorthand", "3 quarters", "three quarters", "3/4"))),
    _i("numbers.temperature", "numbers", "Store the vaccine between 36°F and 46°F, never below -4°F.",
       "store the vaccine between thirty six degrees fahrenheit and forty six degrees fahrenheit never below minus four "
       "degrees fahrenheit",
       spans=(_s("36°F", "shorthand", "36 degrees fahrenheit", "36 degrees"),
              _s("-4°F", "shorthand", "minus 4", "negative 4"))),
    _i("numbers.roman", "numbers", "Chapter IV covers Henry VIII and the Act of 1534.",
       "chapter four covers henry the eighth and the act of fifteen thirty four",
       spans=(_s("Chapter IV", "context", "chapter 4", "chapter four"),
              _s("Henry VIII", "context", "henry the 8th", "henry viii", "henry the eighth"),
              _s("1534", "shorthand", "1534"))),
    _i("numbers.decimal", "numbers", "Membership grew by 1.2 million, or 3.75%, over the year.",
       "membership grew by one point two million or three point seven five percent over the year",
       spans=(_s("1.2 million", "shorthand", "1.2 million", "1 point 2 million"),
              _s("3.75%", "shorthand", "3.75%", "3.75 percent", "3 point 75 percent"))),
    _i("numbers.year", "numbers", "In 2019 we served 2,019 families, and by 2025 about 3,400.",
       "in twenty nineteen we served two thousand nineteen families and by twenty twenty five about three thousand four "
       "hundred",
       spans=(_s("In 2019", "context", "in 2019"), _s("2,019 families", "context", "2019 families", "2,019 families"),
              _s("3,400", "shorthand", "3400", "3,400"))),
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
    return {"version": version, "cohorts": COHORTS, "categories": CATEGORIES, "items": [asdict(i) for i in items]}


def load(path: str | Path) -> tuple[str, list[Item]]:
    """A corpus from a file in the same shape, for a holdout or an external set."""
    payload = json.loads(Path(path).read_text())
    items = []
    for raw in payload["items"]:
        fields = {k: v for k, v in raw.items() if k in Item.__dataclass_fields__}
        fields["spans"] = tuple(Span(**{**span, "accept": tuple(span.get("accept", ()))}) for span in raw.get("spans", ()))
        items.append(Item(**fields))
    return str(payload.get("version", "external")), items
