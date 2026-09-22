from scripts.consolidate_turn_results import recovery_summary
from stt_bench.score import word_errors


def test_recovery_stays_separate_and_only_fills_failed_turns():
    timing={'ttft_ms':400,'ttfs_ms':200}
    original=[{'clip_id':'a','reference':'hello world','valid':False,'word_errors':None,'turn_timing':{}},
              {'clip_id':'b','reference':'yes','valid':True,'word_errors':word_errors('yes','no'),'turn_timing':timing}]
    later=[{'clip_id':'a','transcript':'hello world','valid':True,'turn_timing':timing}]
    result=recovery_summary(original,later)
    assert result['additional_turns_recovered']==1
    assert result['valid_after_recovery']==2
    assert result['after_recovery_accuracy']['wer']==1/3
    assert result['recovery_subset_ttft']['p50_ms']==400
    assert result['after_recovery_ttfs']['n']==2
    assert original[0]['valid'] is False
    failed=recovery_summary(original,[dict(later[0],valid=False)])
    assert failed['failed']==1 and failed['additional_turns_recovered']==0
    assert failed['recovery_subset_ttft']['n']==0
