"""Standard-library-only source inventory. Never imports or executes mechanics.

--freeze exclusively creates the inventory and rendered matrix from the fixed
release Git objects. --check validates the saved inventory against those objects.
This is planning/source coverage, not a scientific certificate or a call-graph
proof. Test-function references are unexpanded AST definitions, not executed
pytest parameter cases. Unmapped search hits remain visible for follow-up.
"""
from __future__ import annotations
import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MATRIX = 'docs/reference_cases/ge_beam3_legacy_parity_matrix_v1.json'
INVENTORY = 'docs/reference_cases/ge_beam3_legacy_parity_inventory_v1.json'
REPORT = 'docs/GE_BEAM3_LEGACY_PARITY_MATRIX.md'
BASE = '74703a3202251edc0beafb21cd31f52c9304ceb8'
TREE = '64a18a8fe137946ec2f0d4841821a19066a2707f'
LEGACY = {'B2_B3_IMPLEMENTED', 'B2_B3_TESTED', 'B3_TESTED',
          'B2_TESTED_B3_COVERAGE_GAP', 'SHARED_ROUTE_TESTED',
          'LEGACY_COVERAGE_UNESTABLISHED', 'LEGACY_REJECTED', 'LEGACY_PLACEHOLDER'}
GE = {'PARTIAL', 'SCOPED_ACCEPTED', 'GAP', 'UNESTABLISHED'}
ACTION = {'IMPLEMENT', 'RETAIN_AND_INTEGRATE', 'IMPLEMENT_WITH_B3_BASELINE_TEST',
          'VERIFY_LEGACY_FIRST', 'NEW_CAPABILITY_NOT_PROVEN_LEGACY', 'NO_PARITY_OBLIGATION'}
METHOD_ROWS = {
    'num_nodes': ['P01'], 'dofs_per_node': ['P01'], 'total_dofs': ['P01'],
    'get_dof_mapping': ['P01'], 'get_node_coordinates': ['P02', 'U01'],
    'compute_shape_functions': ['P02'], 'compute_stiffness_matrix': ['P03', 'P04'],
    'compute_mass_matrix': ['P06', 'P20'], 'compute_geometric_stiffness_matrix': ['P23'],
    'compute_nonlinear_response': ['P13', 'P14', 'P15'],
    'compute_stresses': ['P19', 'U05'], 'compute_internal_forces': ['U07'],
    'to_dict': ['P18', 'U09'],
}
EXTRA = ['src/anysolver/ge_beam3_element.py', 'src/anysolver/ge_beam3_native.py',
         'src/anysolver/_ge_beam3_native_analysis.py',
         'src/anysolver/_ge_beam3_durable_coupled.py',
         'docs/GE_BEAM3_NATIVE_WORKFLOWS.md', 'docs/GE_BEAM3_DELIVERY_STATUS.md',
         'docs/reference_cases/ge_beam3_final_delivery_independent_review.json']


def canonical(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)+'\n').encode()


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    if canonical(value) != raw.replace(b'\r\n', b'\n'):
        raise ValueError('noncanonical JSON')
    return value


def validate_rows(data):
    if set(data) != {'schema', 'baseline_commit', 'baseline_tree', 'baseline_release',
                     'status', 'scientific_tests_run', 'default_changes', 'scope', 'rows'}:
        raise ValueError('matrix schema keys')
    if (data['schema'] != 'anysolver.ge-beam3.legacy-parity-matrix.v1'
            or data['baseline_commit'] != BASE or data['baseline_tree'] != TREE
            or data['baseline_release'] != '0.4.3'
            or data['status'] != 'FROZEN_PLANNING_INVENTORY_NOT_QUALIFICATION'
            or data['scientific_tests_run'] is not False or data['default_changes'] is not False):
        raise ValueError('planning authority or result claim changed')
    expected = [f'P{i:02d}' for i in range(1, 33)] + [f'U{i:02d}' for i in range(1, 11)]
    if [r['id'] for r in data['rows']] != expected:
        raise ValueError('incomplete, duplicate or reordered rows')
    for row in data['rows']:
        if set(row) != {'id', 'area', 'legacy', 'ge', 'action', 'detail', 'sources', 'legacy_tests', 'ge_tests'}:
            raise ValueError('row schema')
        if row['legacy'] not in LEGACY or row['ge'] not in GE or row['action'] not in ACTION:
            raise ValueError('unknown support disposition')
        if row['id'].startswith('U') and row['action'] not in {'NO_PARITY_OBLIGATION', 'VERIFY_LEGACY_FIRST'}:
            raise ValueError('unsupported baseline became implementation authority')
        if row['legacy'] in {'LEGACY_REJECTED', 'LEGACY_PLACEHOLDER'} and row['action'] != 'NO_PARITY_OBLIGATION':
            raise ValueError('unsupported baseline became implementation authority')
        if not row['sources'] or not row['detail']:
            raise ValueError('missing source or qualification boundary')
        for ref in row['sources'] + row['legacy_tests'] + row['ge_tests']:
            path = ref.split('::')[0]
            if Path(path).is_absolute() or '..' in Path(path).parts or '\\' in path:
                raise ValueError('unsafe source reference')


def git(*args):
    return subprocess.run(['git', '--no-replace-objects', *args], cwd=ROOT,
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=30).stdout


def definitions(raw):
    return {n.name for n in ast.walk(ast.parse(raw))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def build_inventory(data):
    validate_rows(data)
    if git('rev-parse', BASE+'^{tree}').decode().strip() != TREE:
        raise ValueError('baseline tree mismatch; missing local authority is not accepted')
    entries = {}
    for line in git('ls-tree', '-r', BASE).decode().splitlines():
        meta, path = line.split('\t', 1)
        mode, kind, blob = meta.split()
        if kind == 'blob': entries[path] = blob
    refs = sorted({ref for r in data['rows'] for key in ('sources', 'legacy_tests', 'ge_tests')
                   for ref in r[key]})
    paths = sorted({ref.split('::')[0] for ref in refs} | set(EXTRA))
    source = {}
    for path in paths:
        if path not in entries:
            raise ValueError('reference absent from baseline: '+path)
        raw = git('show', BASE+':'+path)
        source[path] = dict(blob=entries[path], bytes=len(raw), sha256=sha256(raw).hexdigest())
        for ref in refs:
            if ref.startswith(path+'::') and ref.split('::')[1] not in definitions(raw):
                raise ValueError('missing test definition: '+ref)
    elements = ast.parse(git('show', BASE+':src/anysolver/elements.py'))
    public = {}
    for node in elements.body:
        if isinstance(node, ast.ClassDef) and node.name in {'Element', 'BeamElement', 'QuadraticBeamElement'}:
            public[node.name] = sorted(n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith('_'))
    if set().union(*map(set, public.values())) != set(METHOD_ROWS):
        raise ValueError('unmapped legacy public member')
    # Enumerate ALL baseline test modules, not just files whose names say beam.
    # Rows attach reviewed references; other modules are not asserted irrelevant.
    corpus = {}
    for path, blob in sorted(entries.items()):
        if path.startswith('tests/test_') and path.endswith('.py'):
            mapped = [r['id'] for r in data['rows'] if any(ref.split('::')[0] == path
                      for ref in r['legacy_tests']+r['ge_tests'])]
            corpus[path] = dict(blob=blob, rows=mapped,
                role='ROW_REFERENCE' if mapped else 'NOT_USED_AS_DIRECT_PARITY_EVIDENCE')
    return dict(schema='anysolver.ge-beam3.legacy-parity-inventory.v1',
        baseline_commit=BASE, baseline_tree=TREE,
        matrix_sha256=sha256(canonical(data)).hexdigest(), source_bindings=source,
        legacy_public_members=public, public_member_rows=METHOD_ROWS,
        test_module_corpus=corpus, mechanics_executed=False)


def render(data, inventory):
    lines = ['# GE-B3 legacy parity matrix — frozen step 1', '',
        'Baseline: ANYsolver 0.4.3, commit `'+BASE+'`, tree `'+TREE+'`.', '',
        'Status: **planning inventory frozen; no new mechanics qualification**. '
        'This step changes no runtime, default, dependency, source equation or historical evidence. '
        'No scientific test was executed. Test links identify source definitions, not fresh passes.', '',
        '## Scope and evidence rules', '',
        '- Legacy B3 is the required element baseline. B2-only and synthetic/shared-framework '
        'tests identify additional obligations but do not certify B3 combinations.',
        '- GE means the accepted native owned workflow. The older straight `ge-beam3` facade '
        'remains separate; neither interface inherits the other\'s acceptance.',
        '- `TESTED` means a relevant test exists and was inspected, not full domain qualification. '
        '`PARTIAL` means scoped GE functionality exists but the broader route remains open.',
        '- `LEGACY_REJECTED` includes explicit rejection or documented warning/skip. '
        '`LEGACY_PLACEHOLDER` is not physical functionality. Neither creates a requirement to copy a defect.',
        '- `LEGACY_COVERAGE_UNESTABLISHED` requires a baseline audit/test before deciding '
        'whether it is parity work or a new capability. Absence of a test is not proof of unsupported code.',
        '- Generic spring, shell-only, manifest and adapter tests are not beam numerical evidence. '
        'Parameter combinations must be expanded and inspected at the implementation gate.',
        '- Full parity is closed only when each in-scope implementation obligation has accepted '
        'source, actual-route tests, reviewed evidence and an installed-wheel check. No percentage '
        'or estimated completion is inferred from this inventory.', '',
        '## Required routes and current gaps', '',
        '| ID | Route | Legacy evidence class | Current GE | Next disposition |',
        '|---|---|---|---|---|']
    for r in data['rows']:
        lines.append('| '+ ' | '.join([r['id'], r['area'], r['legacy'], r['ge'], r['action']])+' |')
    lines += ['', '## Exact route notes and corresponding tests', '']
    for r in data['rows']:
        lines += ['### '+r['id']+' — '+r['area'], '', r['detail'], '',
                  'Source: '+', '.join('['+p+'](../'+p+')' for p in r['sources'])+'.', '',
                  'Legacy test references:']
        lines += ['- `'+t+'`' for t in r['legacy_tests']] or ['- None found for this precise claim; source-only boundary.']
        lines += ['', 'GE test references (scope/implementation pointers, not new qualification):']
        lines += ['- `'+t+'`' for t in r['ge_tests']] or ['- None claimed.']
        lines.append('')
    lines += ['## Completeness and limitations', '',
        f'The inventory binds {len(inventory["source_bindings"])} referenced source/test/status files '
        f'and enumerates all {len(inventory["test_module_corpus"])} baseline `tests/test_*.py` modules by Git blob. '
        'Every public member declared by Element/B2/B3 is mapped to a row, including inherited placeholders. '
        'Constructors, public selectors and shared solver families are covered by the route rows. '
        'This is a closed snapshot inventory, not a proof of every dynamically reachable combination.', '',
        'Other test modules remain visible as `NOT_USED_AS_DIRECT_PARITY_EVIDENCE`; they are not '
        'deleted, excluded from CI, or classified as scientifically irrelevant. The matrix does '
        'not claim to inventory live sibling repositories. ANYfem/ANYstructure consumer audits '
        'are an explicit P31 follow-up, not silently counted as covered by solver adapter tests.', '',
        'Canonical matrix: [JSON](reference_cases/ge_beam3_legacy_parity_matrix_v1.json). '
        'Hash/object authority and complete test-module index: '
        '[inventory](reference_cases/ge_beam3_legacy_parity_inventory_v1.json).', '',
        '## Recommended next step — general static integration contract', '',
        'Freeze the current-core FEModel integration design for P01/P07/P08/P09/P10/P18/P20. '
        'Specify shared SO(3) state ownership, supported constraints, static internal elimination, '
        'retained physical inertia, heterogeneous sections and checkpoint transactions before coding. '
        'Begin with elastic straight/curved two-element and beam-network cases; add a B3 fixture '
        'where existing proof is B2-only. Reuse accepted GE records without rerunning closed campaigns.', '',
        'Later gates: nonlinear/material and joint parity; transient/contact/activity; production '
        'scale and ecosystem consumers. Full finite-velocity dynamics and stable postbuckling '
        'must not be mislabelled as already-established legacy requirements.', '',
        'Implementation remains a subsequent user-requested step. After each completed step, '
        'report its actual outcome and recommend the next gate. Scientific children remain bounded '
        'at 600 seconds/24 GiB/one numerical thread, at most three concurrent workers, '
        '1800 seconds per wave and no automatic retry. This static audit needs no mechanics run.', '']
    return '\n'.join(lines).encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    if args.freeze == args.check: p.error('choose exactly one mode')
    data = strict((ROOT / MATRIX).read_bytes())
    inventory = build_inventory(data)
    products = {INVENTORY: canonical(inventory), REPORT: render(data, inventory)}
    if args.freeze:
        if any((ROOT / name).exists() for name in products):
            raise ValueError('freeze outputs already exist; no overwrite')
        for name, raw in products.items():
            with (ROOT / name).open('xb') as f: f.write(raw)
    else:
        for name, raw in products.items():
            if (ROOT / name).read_bytes().replace(b'\r\n', b'\n') != raw:
                raise ValueError('frozen parity product mismatch: '+name)
    print(json.dumps(dict(rows=len(data['rows']), bound_files=len(inventory['source_bindings']),
        indexed_test_modules=len(inventory['test_module_corpus']), mechanics_executed=False)))


if __name__ == '__main__': main()
