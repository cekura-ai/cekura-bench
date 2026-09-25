"""Package current benchmark code and frozen Pipecat inputs, excluding secrets."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from stt_bench.catalog import dataset_definition
from stt_bench.data import load_manifest, read_fleurs, sha256
from stt_bench.huggingface_data import verify_prepared


def regression_files():
    paths = [Path('test.tsv'), Path('annotations/fleurs-en-us-entities-v1.json')]
    # The existing reproducibility tests inspect every source row, including
    # rejected candidates. Include those fixtures rather than skipping tests.
    paths.extend(Path('test') / row['filename'] for row in read_fleurs(Path('test.tsv'), Path('test')))
    for name in ('fleurs-en-us-smoke-v1', 'fleurs-en-us-deepgram-v2'):
        manifest = Path('datasets') / name / 'manifest.json'
        data = load_manifest(manifest)
        paths.append(manifest)
        paths.extend(manifest.parent / clip['audio'] for clip in data['clips'])
        if data.get('annotations_sha256'):
            paths.append(manifest.parent / 'annotations.json')
    return paths


def bundle_files():
    root = verify_prepared(dataset_definition('pipecat-stt-benchmark'))
    paths = [Path(name) for name in ('pyproject.toml', 'uv.lock', 'README.md')]
    paths.extend(regression_files())
    for directory, pattern in (('src', '*.py'), ('tests', '*.py'), ('scripts', '*.py'), ('config', '*.json'), ('scripts', '*.mjs'), ('tests', '*.mjs')):
        paths.extend(sorted(Path(directory).rglob(pattern)))
    # Select manifests and their referenced audio, rather than arbitrary dataset files.
    paths.append(root / 'prepared.json')
    for subset in ('full', 'smoke'):
        manifest = root / subset / 'manifest.json'
        paths.append(manifest)
        for clip in json.loads(manifest.read_text())['clips']:
            paths.append(manifest.parent / clip['audio'])
        cache = manifest.parent / 'derivatives/pcm24000'
        if cache.exists():
            for record in sorted(cache.glob('*.json')):
                wav = record.with_suffix('.wav')
                if sha256(wav) != json.loads(record.read_text())['output_sha256']:
                    raise ValueError('Derivative hash mismatch')
                paths.extend([record, wav])
    files = sorted(set(paths))
    for p in files:
        if p.is_symlink() or not p.resolve().is_relative_to(Path.cwd().resolve()):
            raise ValueError(f'Bundle cannot include links or files outside the checkout: {p}')
    return files


def build_bundle(out, files):
    if out.exists():
        raise ValueError('Bundle already exists; choose a new path')
    # Conservative: never risk filling the local disk while making a compressed copy.
    required = sum(p.stat().st_size for p in files) + 128 * 1024**2
    if shutil.disk_usage(out.parent).free < required:
        raise ValueError(f'Insufficient free space for bundle: need {required} bytes including reserve')
    try:
        with tarfile.open(out, 'x:gz', compresslevel=1) as tar:
            for p in files:
                tar.add(p, arcname=str(p), recursive=False)
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    out.with_suffix(out.suffix + '.sha256').write_text(sha256(out) + '  ' + out.name + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    files = bundle_files()
    print(json.dumps(dict(files=len(files), uncompressed_bytes=sum(p.stat().st_size for p in files))))
    if args.out:
        build_bundle(args.out, files)
        print(args.out)
