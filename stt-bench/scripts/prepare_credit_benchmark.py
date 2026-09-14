"""Freeze a small public-only comparison without making provider requests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from stt_bench.data import load_manifest, sha256, write_json

MODELS = {'gradium-default': 'vocera-gradium-default-20260914-ready',
          'reson8-realtime': 'vocera-reson8-realtime-20260914-setup'}
SOURCES = [
    ('pipecat', 'datasets/pipecat-stt-benchmark/3fe50170d520c951957b86996ef082a6ab87b394/full/manifest.json', 'public_anchor', 7500),
    ('fleurs-general', 'datasets/fleurs-en-us-deepgram-v2/manifest.json', 'public_anchor', 3750),
    ('fleurs-entities', 'datasets/fleurs-en-us-deepgram-v2/manifest.json', 'public_entities', 3750),
]


def select(clips, condition, frame_budget):
    candidates = [c for c in clips if c['condition'] == condition and 50 < c['total_frames'] <= 1000]
    ranked = sorted(candidates, key=lambda c: hashlib.sha256(('small-public-v1:42:' + c['clip_id']).encode()).hexdigest())
    selected, used = [], 0
    for clip in ranked:
        if used + clip['total_frames'] <= frame_budget:
            selected.append(clip); used += clip['total_frames']
    return selected


def prepare(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    groups = []
    for label, source, condition, budget in SOURCES:
        path = Path(source); parent = json.loads(path.read_text())
        chosen = select(parent['clips'], condition, budget)
        directory = out / 'datasets' / label; (directory / 'audio').mkdir(parents=True)
        clips = []
        for clip in chosen:
            audio = path.parent / clip['audio']
            if sha256(audio) != clip['audio_sha256']:
                raise ValueError('Frozen source audio changed')
            target = directory / 'audio' / (clip['clip_id'] + '.wav')
            shutil.copyfile(audio, target)
            clips.append(dict(clip, audio='audio/' + target.name))
        manifest = {k: parent[k] for k in ('dataset', 'language', 'split', 'source_revision', 'preprocessing') if k in parent}
        manifest.update(schema_version=2, dataset_id=label, subset='credit-limited-public-v1',
                        source_manifest_sha256=sha256(path), source_manifest_path=str(path),
                        selection=dict(seed=42, algorithm='SHA256 order, greedy whole-clip packing within fixed cohort frame budget',
                                       max_clip_seconds=20, frame_budget=budget, result_independent=True), clips=clips)
        # Annotations are retained per clip, bound by the selected manifest hash.
        # No source manifest, annotation, reference, or audio is modified.
        write_json(directory / 'manifest.json', manifest)
        load_manifest(directory / 'manifest.json')
        groups.append(dict(id=label, manifest=f'datasets/{label}/manifest.json',
                           manifest_sha256=sha256(directory / 'manifest.json'), clips=len(clips),
                           total_frames=sum(c['total_frames'] for c in clips),
                           submitted_seconds=sum(c['total_frames'] for c in clips)/50,
                           reference_words=sum(len(c['reference'].split()) for c in clips),
                           annotated_entities=sum(len(c.get('entities') or []) for c in clips)))
    plan = dict(schema_version=1, max_audio_seconds_per_provider=300, max_attempts_per_clip=1,
                datasets=groups, planned_clips=sum(g['clips'] for g in groups),
                planned_audio_seconds_per_provider=sum(g['total_frames'] for g in groups)/50,
                models=[dict(model=m, sandbox=s, config_sha256=sha256(Path('config/models')/(m+'.json')))
                        for m,s in MODELS.items()],
                authorization='User approved about five minutes per provider; one attempt; public Pipecat and FLEURS cohorts',
                limitations=['Small diagnostic sample; no population-level ranking',
                             'Whole clips up to 20 seconds; selected in seeded hash order without provider results',
                             'Automatic speech boundaries; no independent listening review',
                             'Entity annotations from existing reference-text review; only covered types are measured'])
    if plan['planned_audio_seconds_per_provider'] > 300:
        raise ValueError('Audio budget exceeded')
    write_json(out/'plan.json', plan)
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True)
    prepare(parser.parse_args().out)
