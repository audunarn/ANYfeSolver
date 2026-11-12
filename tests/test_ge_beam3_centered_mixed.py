"""Stable-reference local operator tests, not native/full beam qualification."""

import math

import numpy as np
import pytest

from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Centered
from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam
from anyfesolver._ge_beam3_p5.compensated import CompensatedStationaryBeam
from anyfesolver._ge_beam3_p5.mixed import LocalForceAccuracy
from anyfesolver._ge_beam3_p5.section import DirectedHardeningSection
from anyfesolver._ge_beam3_p5.core import canonical
from anyfesolver._ge_beam3_p5_coordinates.positions import from_total
from test_ge_beam3_centered_reference import fixture
from test_ge_beam3_curved_p5_native_chart_probe import sample
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law
from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import rotation


def make(shift, *, plastic=False, loaded=False, total=None):
    points, frames = fixture(translation=shift,spatial=True)
    ref = Centered(points,frames); material = law(.02 if plastic else 1000.)
    section = DirectedHardeningSection(material._elastic,material._direction,material._yield,material._hardening)
    total = sample() if total is None else total
    high, low = from_total(points,total,points+total.reshape(3,6)[:,:3])
    beam = CenteredStationaryBeam(ref,section,order=8,position_low=low,
        line_force=np.array([.001,-.002,.001]) if loaded else None)
    vertex = np.array([rotation(v)@q for v,q in zip(total.reshape(3,6)[:,3:],frames)])
    return beam, high, vertex


@pytest.mark.parametrize('plastic', [False,True])
@pytest.mark.parametrize('loaded', [False,True])
def test_local_stationary_response_is_byte_identical_after_exact_translation(plastic,loaded):
    results = []
    for shift in (0.,2.**20,2.**30,2.**40,-2.**40):
        beam, high, vertex = make(shift,plastic=plastic,loaded=loaded)
        result = beam.solve(high,vertex,force_accuracy=LocalForceAccuracy(1e-12,2.))
        assert result.local_residual_norm <= 1e-11
        assert any(station.response.plastic_active for station in result.stations) == plastic
        results.append(canonical(result))
    assert all(result == results[0] for result in results)


def test_at_origin_stable_operator_agrees_with_preserved_variational_operator():
    beam, high, vertex = make(0.)
    original = CompensatedStationaryBeam(beam.reference,beam.section,order=8,
        position_low=beam.position_low)
    first = original.solve(high,vertex,force_accuracy=LocalForceAccuracy(1e-12,2.))
    second = beam.solve(high,vertex,force_accuracy=LocalForceAccuracy(1e-12,2.))
    for name in ('residual','tangent'):
        a,b = getattr(first,name),getattr(second,name)
        assert np.linalg.norm(a-b) <= 1e-11*max(1.,np.linalg.norm(a))
    assert abs(first.potential-second.potential) <= 1e-11*max(1.,abs(first.potential))


@pytest.mark.parametrize('shift', [0.,2.**30,2.**40])
def test_sub_ulp_translation_dead_load_work_against_analytic_curve_length(shift):
    nodes, frames = fixture(translation=shift)
    ref = Centered(nodes,frames)
    section = DirectedHardeningSection(np.eye(6),np.array([1.,0.,0.,0.,0.,0.]),1.,1.)
    total = np.zeros((3,6)); translation = 2.**-54; total[:,0] = translation
    high, low = from_total(nodes,total.ravel(),nodes+total[:,:3])
    beam = CenteredStationaryBeam(ref,section,order=8,position_low=low,line_force=np.array([1.,0.,0.]))
    result = beam.evaluate(high,frames,np.tile(np.eye(3),(2,1,1)),np.zeros((2,2,3)))
    length = math.sqrt(2.)+math.asinh(1.)
    expected = -length*translation
    assert abs(result.potential-expected) <= 1e-11*abs(expected)
    # The sum of physical nodal translation forces is the integrated applied load.
    resultant = result.residual[:18].reshape(3,6)[:,:3].sum(axis=0)
    assert np.linalg.norm(resultant-[-length,0.,0.]) <= 1e-11*length


def test_stable_dead_load_tangent_and_residual_are_variations_of_same_potential():
    beam, high, vertex = make(2.**40,loaded=True)
    result = beam.solve(high,vertex,force_accuracy=LocalForceAccuracy(1e-12,2.))
    direction = np.cos(np.arange(36)+.1); step = 1e-6
    base = beam.evaluate(high,vertex,result.local_rotations,result.moments)
    plus = beam.evaluate(high,vertex,result.local_rotations,result.moments,increment=step*direction)
    minus = beam.evaluate(high,vertex,result.local_rotations,result.moments,increment=-step*direction)
    assert abs((plus.potential-minus.potential)/(2*step)-base.residual@direction) <= 1e-7*max(1.,abs(base.residual@direction))
    expected = base.hessian@direction
    assert np.linalg.norm((plus.residual-minus.residual)/(2*step)-expected) <= 1e-7*max(1.,np.linalg.norm(expected))
    assert np.linalg.norm(base.hessian-base.hessian.T) <= 1e-11*max(1.,np.linalg.norm(base.hessian))


@pytest.mark.parametrize('mutation', ['order','origin_count','nonfinite_low','nonfinite_force','reference_type'])
def test_stable_operator_rejects_unbound_inputs(mutation):
    beam, _, _ = make(0.)
    reference = beam.reference; options = dict(order=8,position_low=np.zeros((3,3)))
    if mutation == 'order': options['order'] = 12
    if mutation == 'origin_count': options['origins'] = ()
    if mutation == 'nonfinite_low': options['position_low'][0,0] = np.nan
    if mutation == 'nonfinite_force': options['line_force'] = [np.inf,0.,0.]
    if mutation == 'reference_type': reference = object()
    with pytest.raises(ValueError): CenteredStationaryBeam(reference,beam.section,**options)
