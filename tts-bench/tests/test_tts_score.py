"""Round-trip scoring: either rendering of the text is correct, identifiers compare by their parts, digits are checked apart."""

from tts_bench.score import score_transcript


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
