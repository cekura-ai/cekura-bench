"""Regression: stopped public summary must not hide successful private evidence."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from assemblyai_standard_evidence import load,MODEL,SOURCE
from unified_benchmark import full_record,reduce_model,rank_frozen_combined
from offline_rescore import rescore_records
import json


def test_saved_standard_profile_recovers_private_and_partial_public_without_ranking():
    reports=Path('reports')
    if not (reports/SOURCE).exists():pytest.skip('Saved private benchmark evidence is local only')
    saved=load(reports)
    records,_=rescore_records({MODEL:[full_record(r) for r in saved['models'][MODEL]['items']]})
    model=reduce_model(MODEL,records[MODEL],dict(pipecat=1000,fleurs=180,private=8),False,[SOURCE])
    rank_frozen_combined([model],records,json.loads(Path('config/rankings/combined-public-private-v1.json').read_text()))
    assert model['cohorts']['private']['usable']==8
    assert model['cohorts']['private']['wer']==pytest.approx(527/12555)
    assert model['cohorts']['pipecat']['attempted']==10
    assert model['cohorts']['pipecat']['usable']==9
    assert model['combined']['wer']==pytest.approx(528/12732)
    assert model['rank'] is None
    assert model['ranking_score']['wer'] is None
    assert len(records[MODEL])==18  # No smoke clips counted as main-run observations.


def test_changed_archive_is_rejected_before_import(tmp_path):
    folder=tmp_path/'assemblyai-full-20260914/private/wire60';folder.mkdir(parents=True)
    (folder/'evidence.tar.gz').write_bytes(b'changed')
    (folder/'evidence.sha256').write_text('wrong  evidence.tar.gz\n')
    with pytest.raises(ValueError,match='checksum changed'):load(tmp_path)
