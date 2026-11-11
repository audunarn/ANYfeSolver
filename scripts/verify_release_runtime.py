"""Verify the version-only runtime bridge to the accepted GE delivery wheel.

This does not re-adjudicate or modify the historical qualification artifact.
Git exports may use CRLF; runtime comparisons normalize only line endings and
the exact top-level release version assignment. Everything else must agree.
"""
import argparse
from email.parser import BytesParser
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

ACCEPTED_SHA256 = '699ae0dbaef06be61efa7b9a8f923c2ab5607061c00f7ab9f4066217cc340ef4'


def compare(accepted: Path, release: Path) -> dict:
    old_raw, new_raw = accepted.read_bytes(), release.read_bytes()
    if sha256(old_raw).hexdigest() != ACCEPTED_SHA256:
        raise ValueError('accepted artifact hash mismatch')
    with ZipFile(accepted) as old, ZipFile(release) as new:
        def runtime(z):
            if len(z.namelist()) != len(set(z.namelist())):
                raise ValueError('duplicate wheel entries')
            return {n: z.read(n).replace(b'\r\n', b'\n') for n in z.namelist()
                    if n.startswith('anysolver/') and not n.endswith('/')}
        before, after = runtime(old), runtime(new)
        if before.keys() != after.keys():
            raise ValueError('runtime path set mismatch')
        changes = []
        for name in sorted(before):
            a, b = before[name], after[name]
            if a == b:
                continue
            if name != 'anysolver/__init__.py' or a.count(b'__version__ = "0.4.2"') != 1:
                raise ValueError('runtime changed: ' + name)
            if a.replace(b'__version__ = "0.4.2"', b'__version__ = "0.4.3"') != b:
                raise ValueError('non-version runtime change')
            changes.append(name)
        if changes != ['anysolver/__init__.py']:
            raise ValueError('release version change missing')
        metadata = [n for n in new.namelist() if n.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError('wheel metadata count')
        headers = BytesParser().parsebytes(new.read(metadata[0]))
        if headers['Version'] != '0.4.3' or headers['Name'].lower() != 'anysolver':
            raise ValueError('release metadata mismatch')
    return dict(schema='anysolver.release-runtime-bridge.v1',
                accepted_wheel_sha256=ACCEPTED_SHA256,
                release_wheel_sha256=sha256(new_raw).hexdigest(),
                release_wheel_bytes=len(new_raw), runtime_file_count=len(before),
                normalized_changes=changes, version_only=True,
                qualification_scope_unchanged=True, defaults_unchanged=True)


def compare_manifest(manifest: Path, release: Path) -> dict:
    record = json.loads(manifest.read_text())
    if (set(record) != {'schema', 'accepted_commit', 'accepted_wheel_sha256', 'runtime'}
            or record['schema'] != 'anysolver.release-runtime-manifest.v1'
            or record['accepted_commit'] != '5fc032e48d25c0a0b866363514b73ac7baf8803c'
            or record['accepted_wheel_sha256'] != ACCEPTED_SHA256
            or len(record['runtime']) != 317):
        raise ValueError('accepted runtime manifest authority mismatch')
    with ZipFile(release) as wheel:
        if len(wheel.namelist()) != len(set(wheel.namelist())):
            raise ValueError('duplicate wheel entries')
        paths = {n for n in wheel.namelist() if n.startswith('anysolver/') and not n.endswith('/')}
        if paths != set(record['runtime']):
            raise ValueError('runtime path set mismatch')
        for name in sorted(paths):
            raw = wheel.read(name).replace(b'\r\n', b'\n')
            if name == 'anysolver/__init__.py':
                if raw.count(b'__version__ = "0.4.3"') != 1:
                    raise ValueError('release version assignment mismatch')
                raw = raw.replace(b'__version__ = "0.4.3"', b'__version__ = "0.4.2"')
            if sha256(raw).hexdigest() != record['runtime'][name]:
                raise ValueError('runtime hash mismatch: ' + name)
        names = [n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise ValueError('wheel metadata count')
        headers = BytesParser().parsebytes(wheel.read(names[0]))
        if headers['Version'] != '0.4.3' or headers['Name'].lower() != 'anysolver':
            raise ValueError('release metadata mismatch')
    raw = release.read_bytes()
    return dict(schema='anysolver.release-runtime-bridge.v1',
                accepted_wheel_sha256=ACCEPTED_SHA256,
                release_wheel_sha256=sha256(raw).hexdigest(),
                release_wheel_bytes=len(raw), runtime_file_count=317,
                normalized_changes=['anysolver/__init__.py'], version_only=True,
                qualification_scope_unchanged=True, defaults_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    authority = parser.add_mutually_exclusive_group(required=True)
    authority.add_argument('--accepted-wheel', type=Path)
    authority.add_argument('--manifest', type=Path)
    parser.add_argument('--wheel', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    record = (compare(args.accepted_wheel, args.wheel) if args.accepted_wheel
              else compare_manifest(args.manifest, args.wheel))
    raw = (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    with args.output.open('xb') as stream:
        stream.write(raw)
    print(raw.decode(), end='')


if __name__ == '__main__':
    main()
