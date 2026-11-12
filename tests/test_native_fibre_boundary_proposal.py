"""Focused DEV checks of the private fixed-origin trial proposal."""
from dataclasses import FrozenInstanceError, fields
from decimal import Decimal as D, localcontext
from types import SimpleNamespace
import math
import numpy as np
import pytest

from anyfesolver._ge_beam3_fibre_cell import FibreCellConjugate, FibreTrialState, CellHistory
from anyfesolver._ge_beam3_fibre_section import FlowCurve, canonical
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver


def one_fibre(strain=.49, derivative=.1, *, curve=None, z0=0., p0=0., force=1.):
    curve = curve or FlowCurve.linear(100., 10.)
    fibre = SimpleNamespace(young=200., curve=curve)
    section = SimpleNamespace(fibres=(fibre,), _map=lambda _: [D(1)]+[D(0)]*5)
    origin = CellHistory('one', ())
    cell = SimpleNamespace(identity='one', section=section,
                           _origins=lambda _: (((D.from_float(z0), D.from_float(p0)),),))
    compliance = ((D.from_float(derivative),)+tuple(D(0) for _ in range(17)),)+tuple(tuple(D(0) for _ in range(18)) for _ in range(5))
    payload = FibreTrialState('one', origin, (force,)+tuple(0. for _ in range(17)),
        ((D.from_float(strain),)+tuple(D(0) for _ in range(5)),), (compliance,))
    direction = [1.]+[0.]*17
    return cell, payload, origin, direction, fibre


def proposal(cell, payload, origin, direction):
    return FibreCellConjugate.boundary_amount(cell, payload, direction, origin=origin)


def test_guard_crosses_actual_binary64_yield_without_tuned_slack():
    cell, payload, origin, dp, fibre = one_fibre(force=1e12)
    amount = proposal(cell, payload, origin, dp)
    # Actual source update has multiplication then addition. Large p forces the
    # guard to account for representation, rather than one nextafter(amount).
    actual_change = D.from_float(payload.resultants[0]+amount*dp[0])-D.from_float(payload.resultants[0])
    strain = payload.strains[0][0]+payload.strain_compliances[0][0][0]*actual_change
    assert fibre.curve._return(D(200), strain, D(0), D(0))[2] == 'PLASTIC_POSITIVE'
    assert .1 < amount < .102


def test_single_use_direction_retains_complete_rounding_guard():
    cell, payload, origin, dp, _ = one_fibre(force=1e12)
    reusable=proposal(cell, payload, origin, dp)
    assert proposal(cell, payload, origin, iter(dp)) == reusable


@pytest.mark.parametrize('strain,change,z0,p0,expected', [
    (.49, .1, 0., 0., 'PLASTIC_POSITIVE'),
    (-.49, -.1, 0., 0., 'PLASTIC_NEGATIVE'),
    (.6, -.2, 0., 0., 'ELASTIC'),
    (.7, -3., .2, .3, 'PLASTIC_NEGATIVE'),
])
def test_signed_yield_and_fixed_predecessor_reversal(strain, change, z0, p0, expected):
    cell, payload, origin, dp, fibre = one_fibre(strain, change, z0=z0, p0=p0)
    amount = proposal(cell, payload, origin, dp)
    assert amount is not None and 0 < amount < 1
    with localcontext() as context:
        context.prec = 80
        made = payload.strains[0][0]+payload.strain_compliances[0][0][0]*D.from_float(amount)
        assert fibre.curve._return(D(200), made, D.from_float(z0), D.from_float(p0))[2] == expected


def test_next_table_knot_and_no_event():
    curve = FlowCurve((0., .2, .4), (100., 120., 160.), 100.)
    cell, payload, origin, dp, _ = one_fibre(.65, .3, curve=curve)
    amount = proposal(cell, payload, origin, dp)
    assert .49 < amount < .51
    cell, payload, origin, dp, _ = one_fibre(0., .01)
    assert proposal(cell, payload, origin, dp) is None
    assert proposal(cell, payload, origin, [0.]*18) is None


@pytest.mark.parametrize('bad', [[1.]*17, [float('nan')]+[0.]*17, [True]+[0.]*17])
def test_invalid_correction_refused(bad):
    cell, payload, origin, dp, _ = one_fibre()
    with pytest.raises(ValueError): proposal(cell, payload, origin, bad)


def test_origin_and_cell_binding_and_owned_payload():
    cell, payload, origin, dp, _ = one_fibre()
    with pytest.raises(ValueError, match='origin-bound'):
        proposal(cell, payload, CellHistory('one', ()), dp)
    cell.identity = 'different'
    with pytest.raises(ValueError, match='origin-bound'): proposal(cell, payload, origin, dp)
    with pytest.raises(FrozenInstanceError): payload.resultants = ()
    with pytest.raises(TypeError): payload.strains[0][0] = D(0)


def test_driver_cache_state_origin_generation_and_legacy_dispatch():
    cell, payload, origin, dp, _ = one_fibre()
    cell.boundary_amount = lambda data, direction, **kwargs: FibreCellConjugate.boundary_amount(cell, data, direction, **kwargs)
    operator = SimpleNamespace(cell=cell, guard=lambda: None)
    owner = SimpleNamespace(_bl5_native_material=True, _generation=5, _shell_count=0,
                            _beam_parts=((SimpleNamespace(operator=operator), None, None),))
    driver = object.__new__(FinitePointDriver)
    driver.owner=owner; driver._beam_nodal=0; driver._beam_origins=(origin,)
    state=SimpleNamespace(resultants=np.array([payload.resultants]))
    driver._native_boundary_trial=(state, driver._beam_origins, 5, ((0,payload),))
    driver._shell_origins=()
    driver._q4_boundary_trial=(state, driver._shell_origins, 5, ())
    full=np.r_[np.zeros(6), dp]
    amount=driver._native_boundary_amount(state, full)
    assert amount is not None
    # Proposal scalar remains usable after a candidate invalidates the cache.
    driver._native_boundary_trial=None
    assert math.isfinite(amount)
    with pytest.raises(ValueError, match='stale'): driver._native_boundary_amount(state, full)
    driver._native_boundary_trial=(state, driver._beam_origins, 4, ((0,payload),))
    with pytest.raises(ValueError, match='stale'): driver._native_boundary_amount(state, full)
    driver._native_boundary_trial=(state, tuple([origin]), 5, ((0,payload),))
    with pytest.raises(ValueError, match='stale'): driver._native_boundary_amount(state, full)
    driver._native_boundary_trial=(state, driver._beam_origins, 5, ((0,payload),))
    with pytest.raises(ValueError, match='stale'):
        driver._native_boundary_amount(SimpleNamespace(resultants=state.resultants), full)
    state.resultants[0,0]+=1.
    with pytest.raises(ValueError, match='changed'): driver._native_boundary_amount(state, full)
    owner._bl5_native_material=False
    assert driver._native_boundary_amount(state, full) is None


def test_actual_cell_capture_preserves_legacy_response_bytes_and_origin():
    from anyfesolver import stiffened_shell as ss
    from examples.stiffened_shell_native_fibre_bl5_material import definition
    owner=ss.create_analysis(definition=definition())
    try:
        from anyfesolver._ge_beam3_retained_fibre import RetainedFibreOperator
        element, _, origin=next(part for part in owner._beam_parts if type(part[0].operator) is RetainedFibreOperator)
        cell=element.operator.cell
        before=canonical(origin)
        ordinary=cell.response([0.]*18, origin)
        captured=cell.response([0.]*18, origin, capture_trial=True)
        payload=captured.pop('_trial_state')
        assert canonical(captured)==canonical(ordinary)
        assert payload.origin is origin and canonical(origin)==before
        assert isinstance(payload.strains, tuple) and isinstance(payload.strain_compliances, tuple)
        assert len(payload.strains)==len(cell.stations)
        assert all(len(row)==6 for row in payload.strains)
        assert all(len(row)==6 and all(len(r)==18 for r in row) for row in payload.strain_compliances)
        # Capturing from a one-shot load iterable also owns the validated p.
        generated=cell.response(iter([0.]*18), origin, capture_trial=True)['_trial_state']
        assert generated.resultants == (0.,)*18
        from anyfesolver._ge_beam3_retained_fibre import Evaluation, BoundaryEvaluation
        operator=element.operator
        args=(operator.reference.coordinates, np.zeros((3,3)), operator.reference.nodal_triads,
              np.tile(np.eye(3),(2,1,1)), np.zeros(18))
        legacy=operator.evaluate(*args, origin=origin)
        selected=operator.evaluate(*args, origin=origin, capture_trial=True)
        assert type(legacy) is Evaluation and type(selected) is BoundaryEvaluation
        assert tuple(f.name for f in fields(legacy)) == ('potential','residual','hessian','hessian_low',
            'kinematics','history','material','production_qualified')
        assert legacy.material==selected.material
        np.testing.assert_array_equal(legacy.residual, selected.residual)
        np.testing.assert_array_equal(legacy.hessian, selected.hessian)
        assert selected.trial_material.origin is origin
        with pytest.raises(ValueError,match='capture policy'):
            operator.evaluate(*args, origin=origin, capture_trial=1)
    finally: owner.close()
