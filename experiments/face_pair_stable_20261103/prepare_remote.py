"""Extract and verify a new isolated experiment; never replace an old code tree."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve(strict=True)
    code = base / 'code'
    if code.exists():
        raise RuntimeError(f'Will not overwrite existing source: {code}')
    manifest = json.loads((base / 'source_manifest.json').read_text())
    with tarfile.open(base / 'source.tar.gz', 'r:gz') as archive:
        for member in archive.getmembers():
            destination = (code / member.name).resolve()
            if not destination.is_relative_to(code) or not member.isfile():
                raise RuntimeError(f'Invalid source member: {member.name}')
        code.mkdir()
        archive.extractall(code, filter='data')
    mismatches = [name for name, digest in manifest.items()
                  if hashlib.sha256((code / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError(f'Source checksum mismatch: {mismatches}')
    (base / 'verification').mkdir(exist_ok=True)
    print(json.dumps({'base': str(base), 'source_files_verified': len(manifest)}))


if __name__ == '__main__':
    main()
