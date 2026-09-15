"""Scoring correction and sample comparability; no network or provider calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from offline_rescore import rescore_records, write_correction
from unified_benchmark import rank_common_public, finalization_contract, aggregate
from stt_bench.score import (NORMALIZER, aligned_word_errors, normalize_words,
                            rescore_saved_word_errors, strip_punctuation_tokens, word_errors)


@pytest.mark.parametrize('ending', ['...', '…', '..', '....', ' ... … — ! ?'])
def test_ellipsis_is_not_a_word_error(ending):
    assert word_errors('anymore so', 'anymore so' + ending)['insertions'] == 0
    assert word_errors('anymore so' + ending, 'anymore so')['deletions'] == 0


@pytest.mark.parametrize('ref,hyp,s,i,d,n', [
    ('...', '…', 0, 0, 0, 0), ('', '', 0, 0, 0, 0),
    ('...', 'extra words', 0, 2, 0, 0), ('hello...', '', 0, 0, 1, 1),
    ('hello...', 'hello friend...', 0, 1, 0, 1),
    ('hello', 'goodbye...', 1, 0, 0, 1),
])
def test_empty_and_genuine_errors(ref, hyp, s, i, d, n):
    result = word_errors(ref, hyp)
    assert [result[k] for k in ('substitutions', 'insertions', 'deletions', 'reference_words')] == [s, i, d, n]


@pytest.mark.parametrize('text', ['3.14', '$20.50', '£20', '50%', 'twenty dollars', "don't", 'a-b'])
def test_existing_number_currency_and_word_normalization_is_preserved(text):
    assert normalize_words(text) == NORMALIZER(text).strip()


def test_cleanup_does_not_strip_inside_tokens_or_remove_currency_symbols():
    assert strip_punctuation_tokens("3.14 $20.50 £20 50% don't a-b . .. … —") == "3.14 $20.50 £20 50% don't a-b"


def legacy(ref, hyp):
    return aligned_word_errors(NORMALIZER(ref), NORMALIZER(hyp))


def test_realign_saved_text_and_reject_corrupt_original_counts():
    old = legacy('hello so', 'hello so...')
    assert old['insertions'] == 1
    assert rescore_saved_word_errors(old)['insertions'] == 0
    old['insertions'] = 0
    with pytest.raises(ValueError, match='Saved word counts differ'):
        rescore_saved_word_errors(old)


def test_rescore_preserves_failed_attempts_selected_recovery_and_timestamps():
    first = dict(attempt=1, valid=False, failure_class='pacing_invalid', transcript='hi...',
                 t0_seconds=2, final_transcript_received_seconds=3.03,
                 word_errors=legacy('hi', 'hi...'),
                 deadlines=[dict(deadline_ms=250, text='hi...', pacing_valid=False,
                                 word_errors=legacy('hi', 'hi...'))])
    second = dict(attempt=2, valid=True, transcript='hi...', raw_sha256='selected-hash', word_errors=legacy('hi', 'hi...'))
    records = {'a': [dict(id='clip', cohort='pipecat', selected_attempt=2, counts=second['word_errors'],
                          first=first, attempts=[first, second], words=None)]}
    snapshot = deepcopy(records)
    result, audit = rescore_records(records)
    assert records == snapshot
    row = result['a'][0]
    assert row['counts']['insertions'] == 0
    assert row['first']['deadlines'][0]['word_errors']['insertions'] == 0
    assert row['first']['valid'] is False
    assert row['selected_attempt'] == 2
    assert row['first']['final_transcript_received_seconds'] == 3.03
    assert row['attempts'][1]['transcript'] == 'hi...'
    assert audit['original_evidence_sha256'] == audit['corrected_evidence_sha256']


def test_common_public_ranking_excludes_trials_and_nonpublic_rows():
    def row(cid, errors=0, cohort='pipecat'):
        return dict(id=cid, cohort=cohort, counts=dict(substitutions=errors, insertions=0, deletions=0, reference_words=10))
    records = {'a': [row('shared'), row('only-a', 8), row('private', 9, 'private')],
               'b': [row('shared', 1)], 'trial': [row('only-trial')]}
    models = [dict(id=k, rankable=k != 'trial', cohorts={'pipecat': {'wer': .2}, 'private': {'wer': .3}}) for k in records]
    common = rank_common_public(models, records)
    assert common['clip_ids'] == ['shared']
    assert common['reference_words'] == 10
    assert [(m['rank'], m['headline']['wer']) for m in models] == [(1, 0), (2, .1), (None, None)]
    assert models[0]['private_minus_public_pp'] == pytest.approx(10)


def test_empty_intersection_does_not_invent_a_rank():
    models = [dict(id='a', rankable=True, cohorts={'pipecat': {'wer': None}, 'private': {'wer': None}})]
    assert rank_common_public(models, {'a': []})['clips'] == 0
    assert models[0]['rank'] is None
    assert models[0]['headline']['wer'] is None


def test_inworld_and_gradium_sent_signals_but_speechmatics_and_assembly_did_not():
    for model in ['inworld-stt-1', 'gradium-default']:
        assert finalization_contract(model)['group'] == 'signal_at_speech_end'
    for model in ['speechmatics-standard', 'speechmatics-enhanced', 'assemblyai-universal-3-5-pro-min-latency']:
        assert finalization_contract(model)['group'] == 'stream_end'


def test_correction_refuses_changed_sources_without_writing(tmp_path):
    raw = b'{"evidence": true}'
    source = tmp_path / 'original.json'
    source.write_bytes(raw)
    data = dict(sources=[dict(path='reports/original.json', sha256=hashlib.sha256(raw).hexdigest())])
    write_correction(tmp_path, data, {'checked': True})
    assert source.read_bytes() == raw
    assert json.loads((tmp_path / 'offline-rescore-v2/results.json').read_text()) == data
    source.write_text('changed')
    with pytest.raises(ValueError, match='Source changed'):
        write_correction(tmp_path, data, {})
