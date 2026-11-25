from stt_bench.score import NORMALIZATION, SCORING_VERSION, normalize_words, word_errors


def errors(reference, hypothesis):
    counts = word_errors(reference, hypothesis)
    return counts['substitutions'], counts['insertions'], counts['deletions']


def test_every_spelling_of_a_backchannel_is_dropped():
    for text in ('Uh-huh.', 'uh huh', 'Uhhuh', 'Mm-hmm.', 'mm hmm', 'Mmhmm', 'Mhm.', 'Mhmm', 'Mm-hm', 'Hmm.', 'Hm.',
                 'Mmmm', 'Uh‑huh', 'Mm–hmm'):
        assert normalize_words(text) == '', text


def test_a_backchannel_costs_nothing_whatever_its_spelling():
    for spelling in ('Uh-huh', 'Mm-hmm', 'Mhm', 'Mhmm', 'Mm-hm', 'Hm'):
        assert errors('Nice.', f'{spelling}. Nice.') == (0, 0, 0), spelling
        assert errors(f'{spelling}. Nice.', 'Nice.') == (0, 0, 0), spelling
    assert normalize_words('Yeah, uh-huh, sure.') == 'yeah sure'


def test_uh_uh_means_no_and_stays_one_word():
    assert normalize_words('Uh-uh.') == 'uhuh'
    assert errors('Uh-uh.', 'Uh-uh.') == (0, 0, 0)
    assert errors('Uh-uh.', 'Uh-huh.') == (0, 0, 1)
    assert normalize_words('uh-uh-uh') == ''


def test_words_that_only_look_like_backchannels_are_kept():
    assert normalize_words('Huh?') == 'huh'
    assert normalize_words('Hummus') == 'hummus'
    assert normalize_words('humming hmmm') == 'humming'
    assert normalize_words("I'm hmm not sure") == 'i am not sure'
    assert normalize_words("I'm mm-hmm") == 'i am'


def test_the_scoring_version_names_the_rule():
    assert SCORING_VERSION == NORMALIZATION['version'] == 'english-wer-v3-backchannels'
    assert 'backchannel' in NORMALIZATION['preprocess']
