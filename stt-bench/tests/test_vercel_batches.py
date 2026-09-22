import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('vercel_batches', Path(__file__).parents[1] / 'scripts/vercel_batches.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_partition_preserves_every_clip_exactly_once_in_order_and_reserves_retries():
    clips = [dict(clip_id=str(i), submitted_seconds=1 + i % 70) for i in range(1000)]
    batches = module.partition(clips)
    assert [c for batch in batches for c in batch] == clips
    assert all(module.allowance(batch) <= 900 for batch in batches)
    assert all(module.allowance(batch) >= sum(2 * c['submitted_seconds'] for c in batch) for batch in batches)


def test_refuses_a_clip_that_cannot_fit():
    with pytest.raises(ValueError, match='exceeds'):
        module.partition([dict(submitted_seconds=1000)])


def test_manifest_preserves_audio_references_and_reference_text_without_changing_parent():
    clips = [dict(clip_id=str(i), audio=f'audio/{i}.wav', reference=f'reference {i}') for i in range(3)]
    parent = dict(clips=clips, subset='full', source_revision='revision', selection={'count': 3})
    child = module.batch_manifest(parent, clips[:2], 'parent-hash', 0)
    assert parent['subset'] == 'full' and len(parent['clips']) == 3
    assert child['clips'] == clips[:2]
    assert child['source_revision'] == parent['source_revision']
    assert child['batch']['parent_manifest_sha256'] == 'parent-hash'
    assert child['selection']['count'] == 2
