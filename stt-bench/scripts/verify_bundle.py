"""Read-only validation of a launch bundle against current code and frozen inputs."""
import hashlib
from pathlib import Path
import sys
import tarfile
from bundle_remote import bundle_files
from stt_bench.data import sha256


def verify(path):
    files = {str(p): p for p in bundle_files()}
    with tarfile.open(path) as tar:
        members = tar.getmembers()
        if len(members) != len(files) or {m.name for m in members} != set(files):
            raise ValueError('Bundle inventory mismatch; rebuild from current code')
        for m in members:
            if not m.isfile() or hashlib.file_digest(tar.extractfile(m), 'sha256').hexdigest() != sha256(files[m.name]):
                raise ValueError('Bundle file changed')
    expected = Path(str(path) + '.sha256').read_text().split()[0]
    if sha256(Path(path)) != expected:
        raise ValueError('Bundle checksum mismatch')


if __name__ == '__main__':
    verify(sys.argv[1])
