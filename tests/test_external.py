"""Grading the validation set, which must not be wrong in our favour or against it.

These run offline: the loader needs the network, the grader does not, and the
grader is where a mistake would be invisible. A grader that scored every wrong
answer in a category as correct would make a broken audio path look fine, which
is the one thing this validation exists to catch.
"""

from __future__ import annotations

from lane_a.external import Question, extract_answer, grade, sample


def question(category: str, answer: str, qid: int = 1) -> Question:
    return Question(id=qid, category=category, official_answer=answer, file_name=f"data/question_{qid}.mp3")


class TestGrading:
    def test_invalid_is_not_read_as_valid(self):
        """``valid`` is a substring of ``invalid``; a naive match scores every miss correct."""
        wants_invalid = question("formal_fallacies", "invalid")
        assert grade("That argument is invalid.", wants_invalid)
        assert not grade("The argument is valid.", wants_invalid)

        wants_valid = question("formal_fallacies", "valid")
        assert grade("Valid.", wants_valid)
        assert not grade("Invalid, because the premises do not entail it.", wants_valid)

    def test_spoken_numbers_count_the_same_as_digits(self):
        counting = question("object_counting", "8")
        assert grade("You have eight vegetables.", counting)
        assert grade("8", counting)
        assert not grade("You have seven.", counting)

    def test_yes_no_answers(self):
        yes = question("navigate", "Yes")
        assert grade("Yes, you return to the starting point.", yes)
        assert not grade("No, you end up somewhere else.", yes)

    def test_an_answer_that_is_not_there_is_wrong_not_correct(self):
        """A reply with no answer in it must never grade as a pass."""
        assert extract_answer("I'm not sure I follow the question.", "navigate") is None
        assert not grade("I'm not sure I follow the question.", question("navigate", "Yes"))
        assert not grade("", question("object_counting", "3"))

    def test_the_first_answer_token_wins(self):
        """Models asked for one word sometimes add a restatement; take what they led with."""
        assert extract_answer("No. The path does not return.", "web_of_lies") == "no"


class TestSampling:
    def test_the_draw_is_stratified_and_repeatable(self):
        pool = [question(c, "Yes", qid=i) for c in ("navigate", "web_of_lies") for i in range(20)]
        first = sample(pool, per_category=5, seed=3)
        second = sample(pool, per_category=5, seed=3)
        assert [q.id for q in first] == [q.id for q in second], "a seeded draw must repeat"
        assert len(first) == 10
        assert len({q.category for q in first}) == 2


class TestUsageDetail:
    def test_the_audio_and_text_split_survives_into_the_record(self):
        """A total with no split cannot be turned into a cost.

        Audio and text tokens are priced differently and the provider reports the
        breakdown one level down, so summing only the top level loses the number
        the cost column is computed from.
        """
        from lane_a.metrics import _flatten_usage

        flat = _flatten_usage(
            {
                "total_tokens": 102,
                "input_tokens": 89,
                "input_token_details": {"text_tokens": 41, "audio_tokens": 48},
                "output_token_details": {"audio_tokens": 30, "reasoning_tokens": 13},
            }
        )
        assert flat["input_token_details.audio_tokens"] == 48
        assert flat["output_token_details.audio_tokens"] == 30
        assert flat["total_tokens"] == 102
