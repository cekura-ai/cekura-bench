"""Round-trip scoring: either rendering of the text is correct, identifiers compare by their parts, digits are checked apart."""

from dataclasses import asdict

import pytest

from tts_bench import corpus
from tts_bench.score import _with_current_readings, score_spans, score_transcript


def test_written_and_spoken_renderings_both_score_clean():
    text, spoken = "Your current balance is $1,347.09.", "your current balance is one thousand three hundred forty seven dollars and nine cents"
    assert score_transcript(text, spoken, "your current balance is one thousand three hundred forty seven dollars and nine cents")["best"]["wer"] == 0.0
    assert score_transcript(text, spoken, "Your current balance is $1,347.09")["best"]["wer"] == 0.0


def test_identifiers_agree_however_they_were_split_and_digits_are_checked_apart():
    text, spoken = "I see member ID CW5001 on the account.", "i see member i d c w five zero zero one on the account"
    result = score_transcript(text, spoken, "I see member ID C W 5001 on the account")
    assert result["best"]["wer"] == 0.0 and result["digits_match"]
    assert "idcw" not in result["best"]["normalized_hypothesis"]
    wrong = score_transcript(text, spoken, "I see member ID C W 5007 on the account")
    assert wrong["digits_expected"] == "5001" and wrong["digits_heard"] == "5007" and not wrong["digits_match"]


def test_a_real_word_is_never_glued_to_a_spelled_letter():
    result = score_transcript("Your confirmation code is K4Q72.", "your confirmation code is k four q seven two", "Your confirmation code is K4 Q72.")
    assert result["best"]["wer"] == 0.0
    assert "isk" not in result["best"]["normalized_reference"]


def test_email_spoken_form_matches_a_spoken_hypothesis():
    result = score_transcript("Send it to billing.support@example.com.", "send it to billing dot support at example dot com",
                              "Send it to billing dot support at example dot com.")
    assert result["best"]["wer"] == 0.0 and result["best_reference"] == "spoken_reference"


def test_digits_come_from_the_written_text_only():
    no_digits = score_transcript("Siobhan Ó Briain called about the invoice.", "siobhan o'brien called about the invoice", "Siobhan O'Brien called about the invoice.")
    assert no_digits["digits_expected"] == "" and no_digits["digits_match"] is False
    phone = score_transcript("You can reach us at 555-013-7742.", "you can reach us at five five five zero one three seven seven four two",
                             "You can reach us at five five five zero one three seven seven four two.")
    assert phone["digits_expected"] == "5550137742" and phone["digits_match"], phone["digits_heard"]
    time = score_transcript("Your appointment is on Tuesday, March 5th at 10:30 AM.", "your appointment is on tuesday march fifth at ten thirty a m",
                            "Your appointment is on Tuesday, March fifth at ten thirty AM.")
    assert time["digits_expected"] == "51030" and time["digits_match"], time["digits_heard"]


def test_amount_zeros_a_reader_would_not_say_are_not_expected_digits():
    large = score_transcript("The quote came to $12,480.00 before tax.", "the quote came to twelve thousand four hundred eighty dollars before tax",
                             "The quote came to $12,480 before tax.")
    assert large["digits_expected"] == "12480" and large["digits_match"]
    small = score_transcript("There's a $0.75 processing fee.", "there's a seventy five cent processing fee", "There's a 75 cent processing fee.")
    assert small["digits_expected"] == "75" and small["digits_match"]


def _passes(item_id: str, hypothesis: str) -> dict[str, bool]:
    spans = [asdict(span) for span in corpus.by_id()[item_id].spans]
    return {r["text"]: r["pass"] for r in score_spans(spans, hypothesis) if r["check"] == "transcript"}


@pytest.mark.parametrize("item_id, hypothesis, span", [
    ("tech.keys", "Press CTRL plus SHIFT plus T to reopen the tab you just closed, then sign in again.", "Ctrl+Shift+T"),
    ("tech.keys", "press control plus shift plus t to reopen the tab you just closed then sign in again", "Ctrl+Shift+T"),
    ("dense.shipment", "Your package left our Reno hub at 6.05 a.m. and should arrive Thursday before 8 p.m.", "6:05 AM"),
    ("tech.alert", "CPU usage peaked at 94% at 2.15 a.m. and was back under 60% by 3 o'clock.", "2:15 AM"),
    ("numbers.apostrophe", "The class of 09 meets on September 9th this year in the third floor lounge.", "'09"),
    ("numbers.apostrophe", "the class of o9 meets on september ninth this year in the third floor lounge", "'09"),
    ("numbers.temperature", "Store the vaccine between 36°F and 46°F, never below minus 4°F.", "36°F"),
    ("numbers.temperature", "Store the vaccine between 36°F and 46°F, never below minus 4°F.", "-4°F"),
    ("worded.time", "The driver scanned it at 3.47, so that's inside the 2-4 window.", "three forty-seven"),
    ("worded.time", "The driver scanned it at 3.47, so that's inside the 2-4 window.", "two-to-four"),
    ("worded.duration", "Refunds usually take 48-72 hours to show on your statement.", "forty-eight to seventy-two hours"),
    ("tech.version", "you'll need version seven dot two dot one or later", "7.2.1"),
    ("dense.labs", "and your LDL is 118 mg per decilitre, so no change to the metformin for now.", "118 mg/dL"),
    ("numbers.height", "The patient is 5ft 11in and weighs 182lbs, so the dose is one teaspoon twice daily.", "5'11\""),
])
def test_a_correct_reading_passes_however_the_transcriber_wrote_it(item_id, hypothesis, span):
    assert _passes(item_id, hypothesis)[span]


@pytest.mark.parametrize("item_id, hypothesis, span", [
    ("numbers.apostrophe", "the class of nine meets on september ninth this year in the third floor lounge", "'09"),
    ("numbers.temperature", "Store the vaccine between 36°F and 46°F, never below 4°F.", "-4°F"),
    ("dense.renewal", "Your plan renews on 11-02 at $64.99 a month. Cancel before 10-31 to avoid the charge.", "11/02"),
    ("dense.labs", "your ldl is one hundred and eighteen mgdl so no change to the metformin for now", "118 mg/dL"),
    ("tech.keys", "press control plus shift plus two to reopen the tab you just closed", "Ctrl+Shift+T"),
])
def test_a_misreading_still_fails(item_id, hypothesis, span):
    assert not _passes(item_id, hypothesis)[span]


def test_a_rescore_adds_readings_accepted_since_the_run_and_keeps_changed_spans_as_recorded():
    recorded = [{"text": "Ctrl+Shift+T", "category": "sequence", "accept": ["control shift t"]},
                {"text": "Ctrl+Alt+T", "category": "sequence", "accept": ["control alt t"]}]
    keys, renamed = _with_current_readings("tech.keys", recorded)
    assert keys["accept"][0] == "control shift t" and "control plus shift plus t" in keys["accept"]
    assert renamed["accept"] == ["control alt t"]
