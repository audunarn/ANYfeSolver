"""Actual retained arch traversal and separately computed continuum response."""
from dataclasses import asdict
from hashlib import sha256
import json
import numpy as np
import pytest
from docs.reference_cases import ge_beam3_retained_arch_case as case
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_schur_line_program import save


@pytest.mark.parametrize('macros', (2, 4, 8, 12))
def test_actual_arch_traversal(macros, tmp_path):
    made = case.model(macros); p = case.program(macros)
    print(dict(stage='actual-arc-start', macros=macros), flush=True)
    result = case.arc.solve(made, p, progress=lambda row: print(row, flush=True))
    save(tmp_path/'checkpoint.json', result.checkpoint)
    save(tmp_path/'process-disposition.json', dict(status=result.status, failure=result.failure,
        cursor=result.completed_steps, production_qualified=False))
    assert result.status == 'completed', result.failure
    c = case.arc.Context(made, p)
    state, records = c.restore(result.checkpoint, expected_sha256=sha256(result.checkpoint).hexdigest())
    assert c.checkpoint(records) == result.checkpoint
    previous = None; rows = []; references = []
    # States are freshly issued by replay, never reconstructed or cast by the test.
    accepted, prefix = c.initial, ()
    for raw in records:
        row = json.loads(raw)
        accepted, regenerated = c.stage(c.physical.make(row['mechanical'], decoded=True),
            row['parameter'], accepted, prefix, row['iterations'])
        assert regenerated == raw
        prefix = (*prefix, regenerated)
        compared, previous, recovered = case.compare(c, accepted, previous=previous)
        rows.append(compared); references.append(asdict(previous))
        save(tmp_path/('point-%02d.json'%accepted.completed_steps), dict(row=compared,
            reference=asdict(previous), recovered=recovered))
        print(dict(stage='reference-complete', macros=macros, **{k:compared[k] for k in
            ('step', 'drop', 'load', 'load_error', 'slope', 'reference_slope')}), flush=True)
        assert max(*compared['metrics'], abs(compared['arc_gap']), compared['correction'],
            compared['material_error'], compared['force_balance'], compared['moment_balance'], compared['work_error'],
            *compared['geometry'].values()) <= 1e-11
    assert all(a['drop'] < b['drop'] for a,b in zip(rows, rows[1:]))
    assert rows[0]['slope'] > 0. > rows[-1]['slope']
    assert rows[0]['reference_slope'] > 0. > rows[-1]['reference_slope']
    assert any(b['load'] < a['load'] for a,b in zip(rows, rows[1:]))
    maximums = {key:max(row[key] for row in rows) for key in ('load_error', 'recovery_error', 'energy_error')}
    accurate = all(v < .02 for v in maximums.values())
    save(tmp_path/'comparison.json', dict(macros=macros, rows=rows, maximums=maximums,
        engineering_2_percent_pass=accurate, actual_arc_from_virgin=True, post_limit_traversed=True,
        replay_identical=True, full_spatial_stability=False, independent_review='PENDING', production_qualified=False))
    # Coarse meshes are diagnostics; only the registered finest mesh can close
    # this engineering gate. No coarse failure is relabeled as qualification.
    if macros == 12: assert accurate, maximums


def test_reference_comparison_cannot_supply_load_to_controller():
    import ast, inspect
    tree = ast.parse(inspect.getsource(case.continuum))
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert set(imports) == {'dataclasses', 'time', 'numpy', 'scipy.integrate'}
    for n in (2, 4, 8, 12):
        p = case.program(n)
        assert p.steps == (.01,)*12 and p.nodal_forces.rows == ((n+1, 0., -1., 0.),)


@pytest.mark.parametrize('count', (True, 1, 3, 16))
def test_case_extent(count):
    with pytest.raises(ValueError): case.model(count)
    with pytest.raises(ValueError): case.program(count)
