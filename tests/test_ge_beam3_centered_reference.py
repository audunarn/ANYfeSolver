"""Independent rational geometry checks; not a beam qualification campaign."""

from fractions import Fraction as F

import numpy as np
import pytest

from anyfesolver import ge_beam3_curved_reference as historical
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Centered


def oracle(nodes, xi, derivative=False):
    x = F(float(xi))
    weights = ((x-F(1,2),-2*x,x+F(1,2)) if derivative else
               (x*(x-1)/2,1-x*x,x*(x+1)/2))
    return tuple(sum((w*F(float(nodes[i,j])) for i,w in enumerate(weights)), F()) for j in range(3))


def fixture(*, translation=0., scale=1., spatial=False):
    nodes = np.array([[-1.,0.,0.],[0.,.5,.25 if spatial else 0.],[1.,0.,0.]])*scale
    frames = []
    for xi in (-1.,0.,1.):
        axis = np.array([float(v) for v in oracle(nodes,xi,True)]); axis /= np.linalg.norm(axis)
        second = np.array([0.,0.,1.]); second -= axis*float(axis@second); second /= np.linalg.norm(second)
        frames.append(np.column_stack((axis,second,np.cross(axis,second))))
    return nodes+translation, np.array(frames)


@pytest.mark.parametrize('spatial', [False,True])
@pytest.mark.parametrize('shift', [0.,2.**20,2.**30,2.**40,-2.**40])
def test_exactly_represented_translation_preserves_all_relative_geometry(spatial, shift):
    nodes, frames = fixture(spatial=spatial); base = Centered(nodes,frames)
    moved = Centered(nodes+shift,frames)
    assert moved.regularity == base.regularity
    points = np.polynomial.legendre.leggauss(8)[0]
    for xi in points:
        for name in ('derivative','tangent','frame','frame_derivative','intrinsic_curvature'):
            assert np.array_equal(getattr(moved,name)(xi),getattr(base,name)(xi)), name
        assert moved.jacobian(xi) == base.jacobian(xi)
        expected = oracle(nodes+shift,xi)
        high, low = moved.position_pair(xi)
        for j in range(3):
            error = abs(F(float(high[j]))+F(float(low[j]))-expected[j])
            assert error <= F(1,10**11)*F(float(base.regularity.characteristic_length))
        derivative = np.array([float(v) for v in oracle(nodes+shift,xi,True)])
        assert np.linalg.norm(moved.derivative(xi)-derivative) <= 1e-11*base.regularity.characteristic_length
    for cell in (0,1):
        for point in points:
            t = float((point+1)/2)
            assert np.array_equal(moved.half_cell_lift(cell,t),base.half_cell_lift(cell,t))


@pytest.mark.parametrize('scale', [2.**-20,1.,2.**20])
def test_analytic_half_lift_agrees_with_independent_exact_interpolation(scale):
    nodes, frames = fixture(translation=2.**20*scale,scale=scale,spatial=True)
    ref = Centered(nodes,frames)
    for cell in (0,1):
        for t in (0.,.125,.375,.5,.75,1.):
            exact = oracle(nodes,cell-1+t)
            expected = np.array([float(exact[j]-(1-F(t))*F(float(nodes[cell,j]))-F(t)*F(float(nodes[cell+1,j]))) for j in range(3)])
            assert np.array_equal(ref.half_cell_lift(cell,t),expected)


def test_frozen_evaluator_translation_defect_remains_reproducible():
    nodes, frames = fixture(); shift = 2.**40
    base = historical.CurvedBeam3ReferenceGeometry(nodes,frames)
    moved = historical.CurvedBeam3ReferenceGeometry(nodes+shift,frames)
    errors = [np.linalg.norm(base.derivative(x)-moved.derivative(x)) for x in np.polynomial.legendre.leggauss(8)[0]]
    assert max(errors) > 1e-8  # Historical witness only, not a gate accepting the defect.


@pytest.mark.parametrize('xi', [-.75,-.25,.25,.75])
def test_analytic_frame_derivative_and_tangent_chain_rule(xi):
    nodes, frames = fixture(translation=2.**40,spatial=True); ref = Centered(nodes,frames)
    step = 1e-6
    actual = ref.frame_derivative(xi)
    difference = (ref.frame(xi+step)-ref.frame(xi-step))/(2*step)
    assert np.linalg.norm(actual-difference) <= 1e-7
    assert np.linalg.norm(ref.frame(xi).T@actual+actual.T@ref.frame(xi)) <= 1e-11
    difference = (ref.tangent(xi+step)-ref.tangent(xi-step))/(2*step)
    assert np.linalg.norm(ref.tangent_derivative(xi)-difference) <= 1e-7


def test_reversal_and_exact_rigid_frame_reexpression_preserve_geometry():
    nodes, frames = fixture(translation=2.**30,spatial=True); ref = Centered(nodes,frames)
    reversed_ref = ref.reversed(); transform = np.diag([-1.,1.,-1.])
    proper = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    moved = ref.rigidly_transformed(proper,np.array([2.**30,-2.**30,0.]))
    for xi in (-.75,-.25,.25,.75):
        assert np.linalg.norm(reversed_ref.frame(xi)-ref.frame(-xi)@transform) <= 1e-11
        assert np.array_equal(reversed_ref.derivative(xi),-ref.derivative(-xi))
        assert np.linalg.norm(moved.frame(xi)-proper@ref.frame(xi)) <= 1e-11
        assert np.array_equal(moved.derivative(xi),proper@ref.derivative(xi))
    assert reversed_ref.reversed().canonical_bytes() == ref.canonical_bytes()


@pytest.mark.parametrize('mutation', ['nonfinite','zero_geometry','bad_frame','bad_tangent','weak_regularity','weak_frame','branch'])
def test_geometry_and_frame_admission_remain_fail_closed(mutation):
    nodes, frames = fixture(); options = {}
    if mutation == 'nonfinite': nodes[0,0] = np.nan
    if mutation == 'zero_geometry': nodes[:] = 0.
    if mutation == 'bad_frame': frames[0,0,0] += .1
    if mutation == 'bad_tangent': frames = np.tile(np.eye(3),(3,1,1))
    if mutation == 'weak_regularity': options['regularity_relative_tolerance'] = 1e-16
    if mutation == 'weak_frame': options['frame_tolerance'] = 1e-6
    if mutation == 'branch': frames[1] = frames[1]@np.diag([1.,-1.,-1.])
    with pytest.raises(ValueError): Centered(nodes,frames,**options)


def test_midpoint_frame_derivative_traces_and_owned_fingerprint():
    nodes, frames = fixture(spatial=True); ref = Centered(nodes,frames)
    with pytest.raises(ValueError): ref.frame_derivative(0.)
    assert np.isfinite(ref.frame_derivative(0.,trace='LEFT')).all()
    assert np.isfinite(ref.frame_derivative(0.,trace='RIGHT')).all()
    before = ref.canonical_bytes(); nodes[:] = 0.; frames[:] = 0.
    assert ref.canonical_bytes() == before
    assert ref.canonical_data()['schema'] == 'GE_BEAM3_CENTERED_Q2_REFERENCE_GEOMETRY_SCHEMA_V2'
    assert ref.evaluation_id in before.decode('ascii')
    for xi in (-1.,0.,1.):
        high, low = ref.position_pair(xi)
        assert np.array_equal(high,ref.coordinates[int(xi)+1]) and not np.any(low)


def test_old_consumers_do_not_silently_claim_the_centered_evaluator():
    from anyfesolver._ge_beam3_p5.compensated import CompensatedStationaryBeam
    from anyfesolver._ge_beam3_p5.section import DirectedHardeningSection
    nodes, frames = fixture(translation=2.**30)
    ref = Centered(nodes,frames)
    section = DirectedHardeningSection(np.eye(6),np.array([1.,0.,0.,0.,0.,0.]),1.,1.)
    made = CompensatedStationaryBeam(ref,section,order=8,position_low=np.zeros((3,3)))
    assert type(made.reference) is historical.CurvedBeam3ReferenceGeometry
    assert not hasattr(made.reference,'evaluation_id')


def test_station_retains_split_position_and_evaluation_identity():
    nodes, frames = fixture(translation=2.**40,spatial=True); ref = Centered(nodes,frames)
    station = ref.station(.3); high, low = ref.position_pair(.3)
    assert np.array_equal(station.position,high) and np.array_equal(station.position_low,low)
    assert np.any(low) and station.evaluation_id == ref.evaluation_id
    assert not station.position_low.flags.writeable
    before = ref.canonical_bytes()
    station.position_low.setflags(write=True); station.position_low[:] = 123.
    assert ref.canonical_bytes() == before


@pytest.mark.parametrize('scale', [1.,1e8,1e-8])
def test_fold_and_scale_relative_near_singularity_remain_rejected(scale):
    frames = np.tile(np.eye(3),(3,1,1))
    for nodes in (np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,0.]]),
                  np.array([[0.,0.,0.],[.25+1e-15,0.,0.],[1.,0.,0.]])):
        with pytest.raises(historical.GeBeam3CurvedGeometryError): Centered(scale*nodes,frames)
