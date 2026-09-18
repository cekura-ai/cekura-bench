"""Comparing transcripts without a judge: formatting must not decide the score."""

from __future__ import annotations

from lane_a.transcript import compare_transcripts, normalize


class TestNormalisation:
    def test_spoken_digits_and_written_digits_agree(self):
        assert normalize("two oh two, five five five, zero one eight eight") == normalize("202-555-0188")

    def test_triple_and_double_expand(self):
        assert "".join(normalize("triple five")) == "555"
        assert "".join(normalize("double zero seven")) == "007"


class TestScoring:
    def test_the_right_digits_in_a_different_format_pass(self):
        scored = compare_transcripts(
            "This is James Carter, my number is two zero two, five five five, zero one eight eight.",
            "This is James Carter. My number is (202) 555-0188.",
        )
        assert scored["digits_match"] and scored["digits_expected"] == "2025550188"

    def test_one_wrong_digit_fails_digits_but_not_words(self):
        scored = compare_transcripts("my number is two zero two", "my number is two one two")
        assert not scored["digits_match"]
        assert scored["wer"] > 0

    def test_a_missing_transcript_scores_as_all_errors(self):
        scored = compare_transcripts("hello there", "")
        assert scored["wer"] == 1.0 and not scored["digits_match"]


class TestCodes:
    def test_spelled_letters_and_a_written_code_agree_on_the_digits(self):
        scored = compare_transcripts("The confirmation number is C W five zero zero one.", "The confirmation number is CW5001.")
        assert scored["digits_match"] and scored["digits_heard"] == "5001"
