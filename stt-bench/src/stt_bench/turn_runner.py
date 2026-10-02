"""Validated-turn profiles, explicit run budgets, and the shared clip runner."""
import json
import hashlib
from pathlib import Path

from .data import sha256, write_json
from .providers import validate, is_nova
from .run import run
from .turn_metrics import MEASUREMENT
from .turns import require, verify_turn_manifest


def controlled_profile(base):
    config = dict(base)
    config['measurement_profile'] = MEASUREMENT
    config['profile_version'] = 'private-turns-v1'
    if is_nova(config):
        config.update(finalization='manual_at_speech_end', completion_basis='close_stream_metadata')
    elif config['provider']=='assemblyai':
        config.update(force_endpoint=True, finalization='manual_at_speech_end', finalize_ack_supported=False)
    elif config['provider']=='speechmatics' and config['model']!='linden-1':
        config.update(force_end_of_utterance=True, finalization='manual_at_speech_end', finalize_ack_supported=True)
    config['turn_finalization_class'] = ('controlled' if config.get('finalization') in
        ('manual_at_speech_end','force_end_of_utterance_at_speech_end') else 'provider_exception')
    return validate(config)


def validate_profile(config):
    require(config.get('measurement_profile') == MEASUREMENT and config.get('profile_version')=='private-turns-v1', 'Expected private-turns-v1 model profile')
    require(config == controlled_profile(config), 'Turn profile has inconsistent finalization settings')
    return config


def smoke_selection(m):
    ranked=sorted(m['clips'],key=lambda c: hashlib.sha256(('turn-smoke-v1:42:'+c['clip_id']).encode()).hexdigest())
    smoke=[]; sources=set()
    for c in ranked:
        if c['source_id'] not in sources:
            smoke.append(c);sources.add(c['source_id'])
    require(len(smoke)<=10, 'A ten-turn smoke cannot cover all speaker recordings; create a smaller dataset plan')
    smoke += [c for c in ranked if c not in smoke][:max(0,10-len(smoke))]
    return [c['clip_id'] for c in smoke]


def prepare_run_plan(manifest_path, configs, out):
    """An offline run budget. This file never authorizes or dispatches API calls."""
    m=verify_turn_manifest(manifest_path)
    profiles=[]
    for path in configs:
        config=validate_profile(json.loads(Path(path).read_text()))
        profiles.append(dict(path=str(Path(path).resolve()), sha256=sha256(Path(path)),
                             model_id=config.get('model_id',config['model']),
                             finalization_class=config['turn_finalization_class']))
    require(profiles and len({p['model_id'] for p in profiles})==len(profiles), 'Provide distinct model profiles')
    smoke=smoke_selection(m)
    result=dict(schema_version=1, measurement_profile=MEASUREMENT, manifest_path=str(Path(manifest_path).resolve()),
        preparation_mode=m.get('preparation_mode','manual-review-v1'),
        listening_review_verified=m['listening_review_verified'],
        manifest_sha256=sha256(Path(manifest_path)), configs=profiles, attempts_per_turn=1,
        smoke_clip_ids=smoke, full_clip_ids=[c['clip_id'] for c in m['clips'] if c['clip_id'] not in smoke],
        smoke_policy='All smoke turns must be complete, pacing-valid, and have available supported timing before unstarted full turns run.',
        planned_sessions=len(m['clips'])*len(profiles), audio_seconds_per_model=sum(c['submitted_seconds'] for c in m['clips']),
        total_audio_seconds=sum(c['submitted_seconds'] for c in m['clips'])*len(profiles),
        provider_calls_authorized=False, counts=m['counts'])
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    require(not out.exists(),'Use a new run-plan path');write_json(out,result);return result


def smoke_row_passed(row, profile, policy=None):
    if not row.get('valid'):
        return False
    timing = row.get('turn_timing', {})
    if (policy == 'completed-empty-allowed-v1' and timing.get('ttft_status') == 'no_text'
            and timing.get('ttfs_status') == 'no_final_text'
            and timing.get('ttft_ms') is None and timing.get('ttfs_ms') is None):
        return True
    return (timing.get('ttft_ms') is not None and
            (profile['turn_finalization_class'] != 'controlled' or timing.get('ttfs_ms') is not None))


async def run_turns(manifest, config, out, *, dry_run=False, resume=False, pacing_check=None,
                    authorized_private_manifest_sha256=None, selected_clip_ids=None, smoke_receipt=None):
    m=verify_turn_manifest(manifest)
    profile=validate_profile(json.loads(Path(config).read_text()))
    smoke=smoke_selection(m)
    if selected_clip_ids is not None:
        require(selected_clip_ids and len(set(selected_clip_ids))==len(selected_clip_ids) and
                set(selected_clip_ids)<={c['clip_id'] for c in m['clips']}, 'Invalid turn shard')
        if not dry_run and not set(selected_clip_ids)<=set(smoke):
            require(smoke_receipt is not None, 'Full turn shards require completed model smoke evidence')
            require(smoke_receipt.get('manifest_sha256')==sha256(Path(manifest)) and
                    smoke_receipt.get('config_sha256')==sha256(Path(config)), 'Smoke belongs to different inputs')
            rows=smoke_receipt.get('rows',[])
            require(len(rows)==len(smoke) and {r['clip_id'] for r in rows}==set(smoke), 'Incomplete smoke coverage')
            require(all(smoke_row_passed(r, profile, smoke_receipt.get('policy')) for r in rows), 'Full turn shards require passing model smoke')
    return await run(Path(manifest), Path(config), Path(out), dry_run, resume, pacing_check,
                     max_attempts=1, authorized_private_manifest_sha256=authorized_private_manifest_sha256,
                     smoke_clip_ids=smoke if selected_clip_ids is None else None,
                     selected_clip_ids=selected_clip_ids, stop_on_provider_failure=True)
