"""Authenticated preserved producer -> NEW independent238 reference, never a seed."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PureWindowsPath

W = Path('C:/ANYworkspaceWork/contact_trackB_dependency_repair_20261013/d1-polynomial-reference-work-v15')
REPO = Path('C:/Github/ANYfeSolver/.worktrees/contact-portable-replay')
V15_SOURCE_ROOT = PureWindowsPath('C:/Github/ANYsolver/.worktrees/contact-portable-replay')
V15_BINDING_SHA256 = 'aa1be2f9dab42ccdf08d9ffd4864aad27a311933015924da19516f19e363dd85'
OWNED_EXISTING = ('scripts/general_contact_enriched_reference.py',
    'scripts/general_contact_enriched_reference_geometry.py',
    'scripts/general_contact_enriched_reference_bounds.py',
    'scripts/general_contact_enriched_reference_run.py',
    'tests/test_general_contact_enriched_reference.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mapped_paths(rows, expected, directory):
    """Locate historical files by logical name and authenticate their original bytes."""
    from scripts.general_contact_enriched_reference_geometry import require
    paths = {}
    for row in rows:
        name = row['name']
        require(name in expected and name not in paths, 'missing, duplicate or unknown bound input: '+name)
        path = directory / row['resolved']
        require(row['sha256'] == expected[name] and sha(path) == expected[name],
                'exact historical source/input/producer binding: '+name)
        paths[name] = path
    require(paths.keys() == expected.keys(), 'incomplete historical input map')
    return paths


def verify_binding(binding, *, resolved_inputs=None):
    from scripts.general_contact_enriched_reference_geometry import require
    files = binding['files']
    require(all(binding[key] in files for key in ('source_before', 'producer_raw', 'producer_closure')),
            'producer and baseline files must be bound')
    if resolved_inputs is None:
        direct = {p: Path(p) for p in files}
        for name, path in direct.items():
            require(sha(path) == files[name], 'exact source/input/producer binding: '+name)
        baseline = json.loads(direct[binding['source_before']].read_bytes())
        fresh = {Path(p).resolve(): h for p, h in binding.get('source_exclusions', {}).items()}
        if fresh:
            require(fresh.keys() == {(REPO/name).resolve() for name in OWNED_EXISTING},
                    'exact five prospective owned-source rebindings only')
            require(all(sha(p) == h for p, h in fresh.items()), 'fresh owned reference source identities')
        for name, digest in baseline['files'].items():
            path = (REPO/name).resolve()
            require(sha(path) == fresh.get(path, digest), 'protected/fresh-owned source mutation: '+name)
    else:
        locator = Path(resolved_inputs)
        mapped = json.loads(locator.read_bytes())
        require(set(mapped) == {'direct', 'sources'}, 'invalid historical input map')
        direct = _mapped_paths(mapped['direct'], files, locator.parent)
        baseline = json.loads(direct[binding['source_before']].read_bytes())
        fresh = {PureWindowsPath(p): h for p, h in binding['source_exclusions'].items()}
        require(fresh.keys() == {V15_SOURCE_ROOT/name for name in OWNED_EXISTING},
                'exact five prospective owned-source rebindings only')
        expected = {name: fresh.get(V15_SOURCE_ROOT/name, h) for name, h in baseline['files'].items()}
        require(all(name in expected for name in OWNED_EXISTING), 'missing historical owned source')
        _mapped_paths(mapped['sources'], expected, locator.parent)
    raw = json.loads(direct[binding['producer_raw']].read_bytes())
    closure = json.loads(direct[binding['producer_closure']].read_bytes())
    require(closure['status'] == 'PASS_SCOPED_DEVELOPMENT_CLOSED_NOT_QUALIFICATION'
            and closure['raw_pytest_exit'] == 0 and closure['verification']['active_raw_drift'] == 0,
            'authenticated closed W13 development producer only')
    require(raw['status'] == 'DEVELOPMENT_NCP_CONVERGED'
            and raw['scientific_acceptance'] is False and len(raw['coordinates']) == 238
            and len(raw['nodal_forces']) == 25, 'complete producer response payload')
    return raw


def run(output, binding_path=W/'INPUT_BINDING.json', *, resolved_inputs=None):
    # Binding checks precede model construction; no saved accepted state is installed.
    binding_bytes = Path(binding_path).read_bytes()
    if resolved_inputs is not None and hashlib.sha256(binding_bytes).hexdigest() != V15_BINDING_SHA256:
        raise ValueError('resolved input map requires the exact V15 binding')
    binding = json.loads(binding_bytes)
    raw = verify_binding(binding, resolved_inputs=resolved_inputs)
    from examples.general_contact_enriched_matched import build
    from scripts.general_contact_enriched_reference import assess, publish, serial
    from scripts.general_contact_enriched_reference_geometry import require
    model, system, load, descriptor = build()
    require(system._accepted is None, 'fresh unloaded shared structural owner; no producer checkpoint seed')
    require(json.loads(json.dumps(descriptor, default=serial)) == raw['descriptor'],
            'exact original physical model/control/offset/input producer mapping')
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    def progress(value):
        publish(output/'PROGRESS.json', value)
        if value.get('kind') == 'completed_original_gate':
            publish(output/'ASSESSMENT_PARTIAL.json', value['assessment'])
    try:
        result = assess(model, system, load, descriptor, raw['coordinates'], raw['nodal_forces'], progress=progress)
        require(system._accepted is None, 'reference/shared structural trials never publish accepted state')
        result['producer_raw_sha256'] = binding['files'][binding['producer_raw']]
        result['new_reference_seed'] = 'ZERO_UNLOADED'
        require(Path(binding_path).read_bytes() == binding_bytes, 'reference input binding changed during assessment')
        verify_binding(binding, resolved_inputs=resolved_inputs)
        publish(output/'ASSESSMENT.json', result)
        return result
    except BaseException as primary:
        try:
            publish(output/'EXCEPTION.json', dict(status='NONPASS', type=type(primary).__name__, message=str(primary),
                scientific_acceptance=False, later_gates='UNRUN'))
        except BaseException as secondary:
            primary.add_note('secondary exception publication failure: '+repr(secondary))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--binding', type=Path, default=W/'INPUT_BINDING.json')
    parser.add_argument('--resolved-inputs', type=Path)
    args = parser.parse_args()
    result = run(args.output, args.binding, resolved_inputs=args.resolved_inputs)
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
