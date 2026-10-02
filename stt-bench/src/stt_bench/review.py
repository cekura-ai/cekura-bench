"""Hash-bound human review inventory. Never infer listening review from text review."""
import hashlib
import json

from .data import sha256, write_json


def reference_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def export_review(manifest_path, out):
    manifest = json.loads(manifest_path.read_text())
    records = []
    for clip in manifest['clips']:
        records.append(dict(clip_id=clip['clip_id'], audio_sha256=clip['audio_sha256'],
                            reference_sha256=reference_hash(clip['reference']), reference=clip['reference'],
                            audio_path=str((manifest_path.parent / clip['audio']).resolve()),
                            entities=clip.get('entities'), reviewed_by=None, reviewed_at=None,
                            reference_listened_verified=False, boundary_listened_verified=False,
                            entities_listened_verified=False, dependency_group=None,
                            conversation_id=clip.get('conversation_id'), speaker_id=clip.get('speaker_id'),
                            source_channel=clip.get('source_channel'), source_sample_rate=clip.get('source_sample_rate'),
                            recording_condition=clip['condition'], turn_start_seconds=clip.get('turn_start_seconds'),
                            turn_end_seconds=clip.get('turn_end_seconds'), notes=''))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'review.json', dict(schema_version=1, manifest_sha256=sha256(manifest_path),
                                       transcription_policy=None, grouping_policy=None, clips=records))
    return out / 'review.json'


def load_review(manifest_path, review_path=None):
    manifest = json.loads(manifest_path.read_text())
    if review_path is None:
        return {}, dict(status='not_independently_verified', verified_clips=0, planned_clips=len(manifest['clips']))
    review = json.loads(review_path.read_text())
    if review.get('manifest_sha256') != sha256(manifest_path):
        raise ValueError('Review belongs to a different frozen manifest')
    records = {r['clip_id']: r for r in review['clips']}
    if len(records) != len(review['clips']) or set(records) != {c['clip_id'] for c in manifest['clips']}:
        raise ValueError('Review must cover every frozen clip exactly once')
    statuses = {}
    for clip in manifest['clips']:
        r = records[clip['clip_id']]
        if (r.get('audio_sha256') != clip['audio_sha256'] or r.get('reference_sha256') != reference_hash(clip['reference'])
                or r.get('reference') != clip['reference'] or r.get('entities') != clip.get('entities')):
            raise ValueError('Reviewed content changed; correct and freeze a new dataset before reviewing')
        verified = bool(r.get('reviewed_by') and r.get('reviewed_at') and review.get('transcription_policy')
                        and all(r.get(k) is True for k in ('reference_listened_verified', 'boundary_listened_verified'))
                        and (not clip.get('entities') or r.get('entities_listened_verified') is True))
        group = r.get('dependency_group')
        if group is not None and (not isinstance(group, str) or not group.strip() or not review.get('grouping_policy')):
            raise ValueError('Dependency groups require nonempty IDs and an explicit grouping policy')
        statuses[clip['clip_id']] = dict(human_review_verified=verified, dependency_group=group)
    count = sum(s['human_review_verified'] for s in statuses.values())
    return statuses, dict(status='verified' if count == len(statuses) else 'review_incomplete',
                          verified_clips=count, planned_clips=len(statuses), review_sha256=sha256(review_path),
                          transcription_policy=review.get('transcription_policy'), grouping_policy=review.get('grouping_policy'))
