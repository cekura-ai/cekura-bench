# Authored dataset: taxonomy and splits

Every template is one spec file (schema v2, `src/turn_bench/author/spec.py`). A template expands into one conversation per voice pairing, and each conversation into one variant per audio condition. Labels are planted by the script and verified against Sesame TurnBench's gold builder and scorer (`turn-bench author check`).

## Splits

Templates are assigned to a split per template, stratified by family; every pairing and condition of a template stays in its split. Only the public (dev) templates are in this repository. The leaderboard (test) and holdout templates are kept privately; their file checksums are published with the public dataset.

| family | public (this repo) | leaderboard (private) | holdout (private) |
|---|---|---|---|
| Mid-turn pauses (should wait) | 6 | 9 | 4 |
| Long turns (should wait) | 1 | 1 | 0 |
| Turn endings (should reply) | 1 | 2 | 1 |
| Cut-ins (should stop) | 5 | 9 | 4 |
| "Mm-hmm"s (should keep talking) | 2 | 4 | 1 |
| **total** | **15** | **25** | **10** |

## Public templates

| id | family | subtype | what it tests |
|---|---|---|---|
| NB-01 | pauses | clause_pause | Clause pauses of 0.8, 1.2, 1.8 and 3 s between complete clauses |
| NB-04 | pauses | digits_phone | A phone number read in three, three and four digits with pauses between groups |
| NB-09 | pauses | address | A street address and zip code read with pauses between the parts |
| NB-12 | pauses | slow_speaker | A slow speaker (speed 0.75) with long clauses and mid-turn pauses |
| NB-13 | pauses | fast_talker | A fast talker (speed 1.15 to 1.2) with brief mid-turn pauses |
| NB-19 | turn_ends | plain | Ordinary question and statement ends with no pauses, backchannels or barge-ins |
| NB-21 | barge_in | early | Early barge-ins 0.5 to 2 s into each agent reply |
| NB-23 | barge_in | late | Late barge-ins at least 10 s into long agent explanations |
| NB-27 | barge_in | two_word_stop | Two-word stops that take the floor: hold on, wait wait, no stop, one sec |
| NB-29 | backchannels | two_word | Two-word acknowledgements during long agent explanations: uh-huh yeah, okay sure, right right |
| NB-34 | backchannels | single_word | Three backchannels in each of three long replies: sure, got it, yep, true, okay, mm |
| TB-04 | pauses | – | Slow speaker (speed 0.8) with mid-turn pauses |
| TB-06 | long_turns | – | Self-correction mid-turn: a question abandoned and replaced |
| TB-09 | barge_in | – | Barge-in 1.5 s into a long agent answer, then a redirect |
| TB-15 | barge_in | – | Late barge-in on the agent's goodbye |

## Label rules

- A `<break>` is labelled at its detected length, never its scripted one; a line whose breaks do not come out as detectable pauses fails the build.
- Pauses under 0.6 s, filler-led pauses, rushed digit groups and every pause of 3 s or more are spliced as exact silence rather than left to the TTS.
- A cut-in or "mm-hmm" must land inside the agent's speech; one placed after the agent has finished is a build error, not a label.
- Unscripted natural gaps stay unlabelled, as in Sesame's annotation.
