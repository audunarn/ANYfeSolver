"""Real material paths through model-owned definitions, never state conversion."""
from hashlib import sha256
import json
import numpy as np
import pytest

from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver import _ge_beam3_analysis_fibre_translation as route
from anyfesolver.control import CancellationToken
from test_ge_beam3_retained_fibre import make_model
from test_ge_beam3_retained_fibre_control import program


def digest(raw):
    return sha256(raw).hexdigest()


def analysis(contrast=1.):
    source = make_model(contrast)
    definitions = tuple(NativeBeamDefinition.capture(NativeFibreStaticElement(i, tuple(e.node_ids),
        e.operator.reference, e.section, order=e.operator.order), np.diag([2.,2.,2.,.07,.09,.11]))
        for i,e in sorted(source.mesh.elements.items()))
    return NativeBeamAnalysis(definitions, tuple(source.boundary_conditions))


@pytest.mark.parametrize('contrast', (1., 1.e12))
def test_reconstruction_preserves_operators_but_not_driver_or_state_owner(contrast):
    made = analysis(contrast)
    bridge = route._model(made)
    direct = make_model(contrast)
    assert bridge is not made.model
    for eid, element in bridge.mesh.elements.items():
        original = made.model.mesh.elements[eid]
        assert type(original) is NativeFibreStaticElement
        assert element is not original and element.operator is not original.operator
        assert element.section is not original.section
        assert element.operator.identity == original.operator.identity
        assert canonical(element.to_dict()) == canonical(direct.mesh.elements[eid].to_dict())


@pytest.fixture(scope='module')
def accepted(tmp_path_factory):
    p = program(); made = analysis()
    result = made.solve_translation(p)
    assert result.status == 'completed', result.backend_result.failure
    assert any(row[2] > 0 for cell in result.backend_result.state.histories
               for station in cell.stations for row in station.rows)
    root = tmp_path_factory.mktemp('owned-fibre-path')
    (root/'checkpoint.json').write_bytes(result.checkpoint)
    return p, result


def test_actual_plastic_load_unload_reversal_matches_original_owner(accepted, tmp_path):
    p, result = accepted
    direct = route.owner.solve_translation_program(make_model(), p)
    assert direct.status == 'completed', direct.failure
    assert result.backend_result.checkpoint == direct.checkpoint
    assert canonical(result.backend_result.state) == canonical(direct.state)
    made = analysis()
    imported = made.import_translation_checkpoint(p, direct.checkpoint, expected_sha256=digest(direct.checkpoint))
    assert imported == result.checkpoint
    recovery = made.recover_translation(p, imported, expected_sha256=digest(imported))
    context = route.owner.Context(make_model(), p)
    state, _ = context.restore(direct.checkpoint, expected_sha256=digest(direct.checkpoint))
    assert canonical(recovery) == canonical(context.recover(state))
    (tmp_path/'recovery.json').write_bytes(canonical(recovery))
    assert not result.production_qualified


def test_prefix_restart_retains_exact_material_origin_and_history(accepted):
    p, result = accepted; made = analysis()
    prefix = made.translation_checkpoint_prefix(p, result.checkpoint, 2, expected_sha256=result.checkpoint_sha256)
    resumed = analysis().solve_translation(p, checkpoint=prefix, expected_sha256=digest(prefix))
    assert resumed.status == 'completed' and resumed.checkpoint == result.checkpoint
    assert made.translation_checkpoint_prefix(p, result.checkpoint, 4,
        expected_sha256=result.checkpoint_sha256) == result.checkpoint
    with pytest.raises(ValueError):
        made.translation_checkpoint_prefix(p, result.checkpoint, 5, expected_sha256=result.checkpoint_sha256)
    with pytest.raises(ValueError):
        made.solve_translation(p, checkpoint=result.checkpoint, expected_sha256=result.checkpoint_sha256, stop_after=1)


@pytest.mark.parametrize('stage,cursor', (('before_commit',0),('committed',1)))
def test_cancel_has_only_accepted_history_then_resumes_exactly(accepted, stage, cursor):
    p, whole = accepted; token = CancellationToken()
    def observer(row):
        if row['stage'] == 'fibre-control.'+stage:
            token.cancel()
    run = analysis().solve_translation(p, cancellation_token=token, progress=observer)
    assert run.status == 'cancelled' and run.backend_result.completed_targets == cursor
    resumed = analysis().solve_translation(p, checkpoint=run.checkpoint, expected_sha256=run.checkpoint_sha256)
    assert resumed.status == 'completed' and resumed.checkpoint == whole.checkpoint


@pytest.mark.parametrize('mutation', ('schema','owner','program','seed','graph','hash','qualification',
                                     'extra','duplicate','nonfinite','overflow','whitespace'))
def test_reject_envelope_before_driver_reconstruction(accepted, monkeypatch, mutation):
    p, result = accepted; value = json.loads(result.checkpoint)
    if mutation == 'schema': value['schema'] = route.owner.SCHEMA
    elif mutation == 'owner': value['workflow'] = 'RETAINED_FROM_REFERENCE'
    elif mutation == 'program': value['program']['targets'] = [.1]
    elif mutation == 'seed': value['seed_sha256'] = '0'*64
    elif mutation == 'graph': value['definition_graph_sha256'] = '0'*64
    elif mutation == 'hash': value['backend_sha256'] = '0'*64
    elif mutation == 'qualification': value['production_qualified'] = True
    elif mutation == 'extra': value['extra'] = None
    raw = canonical(value)
    if mutation == 'duplicate': raw = raw.replace(b'{', b'{"schema":"duplicate",', 1)
    elif mutation == 'nonfinite': raw = raw.replace(b'false', b'NaN', 1)
    elif mutation == 'overflow': raw = raw.replace(b'false', b'1e999', 1)
    elif mutation == 'whitespace': raw += b'\n'
    monkeypatch.setattr(route, '_model', lambda *a: pytest.fail('reconstructed before envelope authority'))
    with pytest.raises(ValueError):
        analysis().recover_translation(p, raw, expected_sha256=digest(raw))


@pytest.mark.parametrize('mutation', ('nodes','inertia','material','boundaries','identity'))
def test_changed_analysis_rejected_before_retained_driver(accepted, monkeypatch, mutation):
    p, result = accepted; made = analysis()
    if mutation == 'nodes': made.model.mesh.nodes[1].x += .1
    elif mutation == 'inertia': made._inertias[1] = 2*made._inertias[1]
    elif mutation == 'material': made.model.materials.clear()
    elif mutation == 'boundaries': made.model.boundary_conditions.clear()
    else: made.identity = '0'*64
    monkeypatch.setattr(route, '_model', lambda *a: pytest.fail('reconstructed on changed model'))
    with pytest.raises(ValueError):
        made.solve_translation(p, checkpoint=result.checkpoint, expected_sha256=result.checkpoint_sha256)


def test_no_static_checkpoint_conversion_and_no_generalized_seed(accepted):
    p, result = accepted; made = analysis()
    with pytest.raises(ValueError):
        made.recover(result.checkpoint, expected_sha256=result.checkpoint_sha256)
    raw = made._envelope(result.backend_result.checkpoint)
    with pytest.raises(ValueError):
        made.solve_translation(p, checkpoint=raw, expected_sha256=digest(raw))
    with pytest.raises(TypeError):
        made.solve_translation(p, seed=b'generalized', expected_seed_sha256='0'*64)


def test_mutating_observer_never_returns_checkpoint_and_lock_is_released():
    made = analysis()
    def observer(row):
        made.model.mesh.nodes[1].x += .1
    with pytest.raises(ValueError):
        made.solve_translation(program(), progress=observer)
    assert made._lock.acquire(blocking=False)
    made._lock.release()


@pytest.mark.parametrize('bad', (True, -1, 5))
def test_bad_cursor_rejected_before_driver(bad, monkeypatch):
    monkeypatch.setattr(route, '_model', lambda *a: pytest.fail('constructed on invalid controls'))
    with pytest.raises(ValueError):
        analysis().solve_translation(program(), stop_after=bad)
