"""Round-trip scoring: either rendering of the text is correct, identifiers compare by their parts, digits are checked apart."""

from dataclasses import asdict

import pytest

from tts_bench import corpus
from tts_bench.score import _with_current_readings, hard_part_verdicts, name_positions, score_spans, score_transcript, word_verdict


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
    assert result["best"]["wer"] == 0.0
    written = score_transcript("Send it to billing.support@example.com.", "send it to billing dot support at example dot com",
                               "Send it to billing.support at example.com.")
    assert written["best"]["wer"] == 0.0


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


TRANSCRIBER_WRITINGS = [
    ("tech.alert", "CPU usage peaked at 94% at 2.15am and was back under 60% by 3 o'clock."),
    ("paragraph.delivery", None),
    ("numbers.decimal", None),
    ("currency.small", "There's a 75 cent processing fee on each transaction."),
    ("datetime.options", "I can offer 8.50 a.m. on Monday or 3.20 p.m. on Friday."),
    ("contact.email_digits", "I have m.rivera88 at example.org. Is that still current?"),
    ("contact.url", "You can manage it online at example.com slash account slash settings."),
    ("symbols.url", "Go to example.com slash support slash billing and choose contact us."),
    ("symbols.email", "You can write to help at north-clinic.org or call 1-800-555-0199."),
    ("tech.keys", "press ctrl plus shift plus t to reopen the tab you just closed then sign in again"),
    ("phone.readback", "I have your number as 555-020-8816, is that right?"),
    ("worded.time", "The driver scanned it at 347, so that's inside the 2-4 window."),
    ("numbers.temperature", "Store the vaccine between 36°F and 46°F, never below minus 4°F."),
    ("numbers.height", "The patient is 5'11'' and weighs 182 pounds, so the dose is one teaspoon twice daily."),
    ("datetime.range", "the technician will arrive between eight o'clock and twelve o'clock on july ninth"),
    ("numbers.roman", "Chapter four covers Henry VIII and the Act of 1534."),
    ("numbers.apostrophe", "The class of 09 meets on September 9th this year in the third floor lounge."),
    ("currency.copay", "The co-pay for that visit is $35, and the remaining $212.50 goes toward your deductible."),
]


@pytest.mark.parametrize("item_id, hypothesis", TRANSCRIBER_WRITINGS)
def test_a_correct_reading_costs_no_words_however_the_transcriber_wrote_it(item_id, hypothesis):
    item = corpus.by_id()[item_id]
    if hypothesis is None:     # the text as written, with a transcriber's dotted time in place of the colon
        hypothesis = item.text.replace(":", ".") if item_id == "paragraph.delivery" else item.text.replace("%", " percent")
    result = score_transcript(item.text, item.spoken_reference, hypothesis, readings=item.readings)
    assert result["best"]["errors"] == 0, result["best"]


@pytest.mark.parametrize("item_id, hypothesis", [
    ("units.eta", "The ETA is 3 p.m. Eastern time, and the driver will text when they're 10 minutes away."),
    ("prose.greet", "Thanks for calling Cedar Valley Family Practice. This is Rylee. How can I help you today?"),
])
def test_an_accepted_reading_costs_no_words(item_id, hypothesis):
    item = corpus.by_id()[item_id]
    assert score_transcript(item.text, item.spoken_reference, hypothesis, readings=item.readings)["best"]["errors"] == 0
    assert score_transcript(item.text, item.spoken_reference, hypothesis)["best"]["errors"] > 0


@pytest.mark.parametrize("item_id, hypothesis", [
    ("tech.alert", "CPU usage peaked at 94% at 2.50am and was back under 60% by 3 o'clock."),
    ("contact.url", "You can manage it online at example.com slash account slash setting."),
    ("units.eta", "The ETA is 3 p.m. Central, and the driver will text when they're 10 minutes away."),
])
def test_a_wrong_reading_still_costs_words(item_id, hypothesis):
    item = corpus.by_id()[item_id]
    assert score_transcript(item.text, item.spoken_reference, hypothesis, readings=item.readings)["best"]["errors"] > 0


def _heard(passes):
    return {"model": "m", "best": {"wer": 0.0}, "spans": [{"text": "x", "category": c, "check": "transcript", "pass": p} for c, p in passes]}


def test_a_term_counts_only_if_every_transcriber_heard_it_right():
    verdicts = hard_part_verdicts({"deepgram": _heard([("term", True)]), "openai-whisper": _heard([("term", False)])})
    assert verdicts[0]["pass"] is False


def test_other_hard_parts_go_to_the_majority_and_wait_for_the_third_when_two_split():
    two = {"deepgram": _heard([("sequence", False), ("shorthand", True)]), "openai-whisper": _heard([("sequence", True), ("shorthand", True)])}
    assert [v["pass"] for v in hard_part_verdicts(two)] == [None, True]
    three = {**two, "zoom": _heard([("sequence", True), ("shorthand", False)])}
    assert [v["pass"] for v in hard_part_verdicts(three)] == [True, True]


def _transcribed(item_id, *hypotheses):
    item = corpus.by_id()[item_id]
    return {f"t{n}": score_transcript(item.text, item.spoken_reference, h, readings=item.readings) for n, h in enumerate(hypotheses)}


def test_a_word_only_one_transcriber_missed_does_not_count_once_a_third_has_heard_it():
    clipped = "for calling cedar valley family practice this is riley how can i help you today"
    right = "Thanks for calling Cedar Valley Family Practice, this is Riley. How can I help you today?"
    assert word_verdict(_transcribed("prose.greet", clipped, right))["errors"] is None
    assert word_verdict(_transcribed("prose.greet", clipped, right, right))["errors"] == 0


def test_a_word_most_transcribers_missed_still_counts():
    short = "thanks for calling cedar valley family practice this is riley how can i help you"
    right = "Thanks for calling Cedar Valley Family Practice, this is Riley. How can I help you today?"
    assert word_verdict(_transcribed("prose.greet", short, short, right))["errors"] == 1
    assert word_verdict(_transcribed("prose.greet", short, short))["errors"] == 1


def _names(item_id):
    item = corpus.by_id()[item_id]
    return name_positions(item.spoken_reference, item.names)


def test_a_name_counts_wrong_only_when_every_transcriber_missed_it():
    right, garbled = "Please hold for Ngozi Okonkwo.", "please hold for nagosio okonkwo"
    assert word_verdict(_transcribed("names.ngozi", garbled, garbled, right), _names("names.ngozi"))["errors"] == 0
    assert word_verdict(_transcribed("names.ngozi", garbled, garbled, garbled), _names("names.ngozi"))["errors"] == 1
    # two transcribers that both missed a name wait for the third
    assert word_verdict(_transcribed("names.ngozi", garbled, garbled), _names("names.ngozi"))["errors"] is None


def test_an_ordinary_word_in_a_name_sentence_still_goes_to_the_majority():
    dropped, right = "hold for ngozi okonkwo", "Please hold for Ngozi Okonkwo."
    assert word_verdict(_transcribed("names.ngozi", dropped, dropped, right), _names("names.ngozi"))["errors"] == 1
    assert word_verdict(_transcribed("names.ngozi", dropped, dropped), _names("names.ngozi"))["errors"] == 1


def test_name_positions_follow_the_normaliser():
    assert sorted(_names("names.siobhan")) == [0, 1, 2]      # siobhan o'brien -> siobhan 0 brien
    assert sorted(_names("names.place")) == [4, 7]
