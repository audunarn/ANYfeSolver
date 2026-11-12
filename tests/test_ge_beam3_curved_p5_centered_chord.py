"""Scalar/one- and two-element checks, not another 32-element execution."""

from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver._ge_beam3_mixed_ad import Jet2, constant_matrix
from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from docs.reference_cases.ge_beam3_curved_p5_centered_chord import (
    CenteredChordMixedProbe, CenteredChordAssemblyHistoryProbe, SCHEMA,
)
from docs.reference_cases.ge_beam3_curved_p5_nonlinear_mixed_probe import (
    NonlinearMixedBeamProbe, LocalForceAccuracy, NonlinearLocalError,
)
from docs.reference_cases.ge_beam3_curved_p5_section_probe import DirectedHardeningSectionProbe, SectionHistory
from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import rotation, T3, T6
from docs.reference_cases.ge_beam3_curved_p5_displacement_control_probe import DisplacementControlledAssemblyProbe, DisplacementControlError
from docs.reference_cases.ge_beam3_curved_p5_assembly_history_probe import AssemblyTransactionError
from docs.reference_cases.ge_beam3_curved_p5_history_path_probe import digest
from docs.reference_cases.ge_beam3_curved_p5_arch_refinement_case import make_beam
from test_ge_beam3_curved_p5_algebra_probe import reference, perturbed, section
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law, A
from test_ge_beam3_curved_p5_assembly_history_probe import model, forces
from test_ge_beam3_curved_p5_chord_cancellation_diagnostic import ROOT, INPUTS, direct


def centered(old):
    return CenteredChordAssemblyHistoryProbe(old._references,
        [tuple(int(n) for n in row) for row in old._maps], old._sections,
        fixed_nodes=tuple(int(n) for n in old._fixed), order=old._order, extent=old._extent)


def test_implemented_jet_values_reduce_exact_saved_input_error():
    saved = {}
    for name, (size, sha) in INPUTS.items():
        data = (ROOT/name).read_bytes()
        assert len(data) == size and hashlib.sha256(data).hexdigest() == sha
        saved[name] = json.loads(data)
    refs = np.array(saved['initial.json']['assembly']['positions'])
    raw = saved['failed_last.json'];x = np.array(raw['positions'])
    old_errors, new_errors = [], []
    for index, e in enumerate(raw['response']['elements']):
        # A three-coordinate helper fixture only: no element evaluation,
        # Newton iteration, assembly, spectrum or actual geometry constructor.
        dummy = SimpleNamespace(reference=SimpleNamespace(coordinates=refs[2*index:2*index+3]))
        for cell, (left, right) in enumerate(((0,1),(1,2))):
            chord = x[2*index+right]-x[2*index+left]
            base = refs[2*index+right]-refs[2*index+left]
            u = e['local_rotations'][cell]
            made_u = constant_matrix(u, 3)
            made_d = [Jet2.variable(float(v), i, 3) for i, v in enumerate(chord)]
            old = NonlinearMixedBeamProbe._chord_strain(dummy, made_u, made_d, left, right)
            new = CenteredChordMixedProbe._chord_strain(dummy, made_u, made_d, left, right)
            for i in range(3):
                exact = direct([[Fraction.from_float(v) for v in row] for row in u],
                    list(map(Fraction.from_float, chord)), list(map(Fraction.from_float, base)), i)
                old_errors.append(abs(Fraction.from_float(old[i].value)-exact))
                new_errors.append(abs(Fraction.from_float(new[i].value)-exact))
                np.testing.assert_array_equal(old[i].gradient, new[i].gradient)
                np.testing.assert_array_equal(old[i].hessian, new[i].hessian)
    assert max(old_errors) > 20*max(new_errors)


@pytest.mark.parametrize('height,yield_force,load', [(0.,1000.,None),(.4,1000.,None),(.4,.02,None),(.4,.02,[.1,-.2,.3])])
def test_complete_potential_and_analytic_derivatives_match_direct(height, yield_force, load):
    ref=reference(height);x,q,u=perturbed(ref)
    m=np.cos(np.arange(12)).reshape(2,2,3)/7
    kw=dict(order=8, line_force=load)
    direct_model=NonlinearMixedBeamProbe(ref,law(yield_force),**kw)
    new_model=CenteredChordMixedProbe(ref,law(yield_force),**kw)
    increment=np.sin(np.arange(36)+.4)/300
    a=direct_model.evaluate(x,q,u,m,increment=increment)
    b=new_model.evaluate(x,q,u,m,increment=increment)
    assert abs(a.potential-b.potential) <= 1e-11
    assert np.linalg.norm(a.residual-b.residual) <= 1e-11
    assert np.linalg.norm(a.hessian-b.hessian) <= 1e-11*max(1.,np.linalg.norm(a.hessian))
    assert tuple(s.response.plastic_active for s in a.stations)==tuple(s.response.plastic_active for s in b.stations)


@pytest.mark.parametrize('yield_force',[1000.,.02])
def test_condensed_energy_residual_tangent_directional_agreement(yield_force):
    ref=reference(.4);x,q,_=perturbed(ref)
    m=CenteredChordMixedProbe(ref,law(yield_force),order=8)
    policy=LocalForceAccuracy(1e-12,2.);base=m.solve(x,q,force_accuracy=policy)
    direction=np.sin(np.arange(18)+.2)/8;step=1e-6;responses=[]
    for sign in (1.,-1.):
        d=sign*step*direction.reshape(3,6)
        solved=m.solve(x+d[:,:3],np.array([rotation(d[n,3:])@q[n] for n in range(3)]),force_accuracy=policy)
        responses.append(m.evaluate(x,q,solved.local_rotations,solved.moments,increment=np.r_[d.ravel(),np.zeros(18)]))
    plus,minus=responses
    active=lambda r:tuple(s.response.plastic_active for s in r.stations)
    assert active(plus)==active(base)==active(minus)
    assert abs((plus.potential-minus.potential)/(2*step)-base.residual@direction) <= 1e-7
    assert np.linalg.norm((plus.residual[:18]-minus.residual[:18])/(2*step)-base.tangent@direction) <= 1e-7
    assert np.linalg.norm(base.tangent-base.tangent.T) <= 1e-11


def test_large_common_rigid_rotation_and_reference_reexpression():
    ref=reference(.4);x,q,u=perturbed(ref);mom=np.cos(np.arange(12)).reshape(2,2,3)/7
    g,shift=rotation([1.8,-1.7,2.1]),np.array([2.,-3.,4.])
    base=CenteredChordMixedProbe(ref,law(),order=8).evaluate(x,q,u,mom)
    transform=np.eye(36);transform[:24,:24]=np.kron(np.eye(8),g)
    for new_ref,new_u in ((ref,np.einsum('ij,njk->nik',g,u)),
        (ref.rigidly_transformed(g,shift),np.einsum('ij,njk,kl->nil',g,u,g.T))):
        other=CenteredChordMixedProbe(new_ref,law(),order=8).evaluate(x@g.T+shift,np.einsum('ij,njk->nik',g,q),new_u,mom)
        assert abs(other.potential-base.potential) <= 1e-11
        assert np.linalg.norm(other.residual-transform@base.residual) <= 1e-11
        assert np.linalg.norm(other.hessian-transform@base.hessian@transform.T) <= 1e-11*np.linalg.norm(base.hessian)


def test_connectivity_reversal_preserves_station_origins_and_work():
    ref=reference(.4);x,q,u=perturbed(ref);mom=np.cos(np.arange(12)).reshape(2,2,3)/7
    origins=tuple(SectionHistory(.0001*i,.0002*i) for i in range(16))
    base=CenteredChordMixedProbe(ref,law(),order=8,origins=origins).evaluate(x,q,u,mom)
    reversed_law=DirectedHardeningSectionProbe(T6@section()@T6.T,T6@A,.02,.4)
    flip=np.diag([-1.,1.,-1.])
    other=CenteredChordMixedProbe(ref.reversed(),reversed_law,order=8,origins=origins[::-1]).evaluate(
        x[::-1],q[::-1]@flip,u[::-1],mom[::-1,::-1]@T3.T)
    assert abs(base.potential-other.potential) <= 1e-11
    for a,b in zip(other.stations,base.stations[::-1]):
        assert a.response.origin==b.response.origin
        assert np.linalg.norm(a.response.resultants-T6@b.response.resultants) <= 1e-11


def test_two_element_plastic_transactions_and_schema_are_deterministic():
    first,second=centered(model()),centered(model())
    for amplitude in (.1,0.):
        a,b=first.trial(amplitude*forces()),second.trial(amplitude*forces())
        assert digest(a)==digest(b)
        assert all(e.accuracy_schema==SCHEMA for e in a.response.elements)
        first.commit(a);second.commit(b)
        assert digest(first.replay())==digest(a.response)
        assert digest(first.committed)==digest(second.committed)
    from docs.reference_cases.ge_beam3_curved_p5_force_accurate_assembly import ForceAccurateAssemblyHistoryProbe
    old=ForceAccurateAssemblyHistoryProbe(first._references,[(0,1,2),(2,3,4)],first._sections,order=8)
    old._checkpoint=deepcopy(first._checkpoint)
    with pytest.raises(AssemblyTransactionError):old.replay()


def test_explicit_centered_displacement_control_and_failure_preserve_state():
    old,_=make_beam(2);control=DisplacementControlledAssemblyProbe(centered(old),node=2)
    trial=control.trial(.095);control.commit(trial)
    assert digest(control.committed_model.replay())==digest(trial.assembly.response)
    before=digest(control.committed_model.committed)
    with pytest.raises(DisplacementControlError):control.trial(.09,max_mixed_evaluations=0)
    assert digest(control.committed_model.committed)==before
    ref=reference(.4);x,q,_=perturbed(ref);m=CenteredChordMixedProbe(ref,law(),order=8)
    with pytest.raises(NonlinearLocalError):m.solve(x,q,max_evaluations=0)
    assert digest(control.committed_model.committed)==before


@pytest.mark.parametrize('scale',[1.,1/32])
@pytest.mark.parametrize('shift',[[0.,0.,0.],[2.,-3.,4.]])
def test_stress_free_short_and_long_curved_members_under_rigid_motion(scale,shift):
    base=reference(.1);ref=CurvedBeam3ReferenceGeometry(scale*base.coordinates,base.nodal_triads)
    g=rotation([1.8,-1.7,2.1]);q=np.einsum('ij,njk->nik',g,ref.nodal_triads)
    law=DirectedHardeningSectionProbe(np.diag([1000.,400.,400.,.02,.01,.02]),[1.,0.,0.,0.,0.,0.],1e6,1.)
    probe=CenteredChordMixedProbe(ref,law,order=8)
    response=probe.evaluate(ref.coordinates@g.T+shift,q,np.tile(g,(2,1,1)),np.zeros((2,2,3)))
    # Unit-aware normalized invariant checks from the development plan.
    # Moment stationarity and moment-dual rotation gaps have different units;
    # do not form an unscaled norm mixing all 36 residual coordinates.
    length=2*scale;force_scale=1000.
    physical=response.residual[:18].reshape(3,6).copy();physical[:,3:]/=length
    assert np.linalg.norm(physical)/force_scale <= 1e-11
    assert np.linalg.norm(response.residual[18:24])/(force_scale*length) <= 1e-11
    assert np.linalg.norm(response.residual[24:]) <= 1e-11
    assert abs(response.potential)/(force_scale*length) <= 1e-11


def test_absolute_rigid_motion_residual_floor_is_preserved_as_open_diagnostic():
    # Do not relabel this state as passing the assembly's absolute force gate.
    base=reference(.1);ref=CurvedBeam3ReferenceGeometry(base.coordinates/32,base.nodal_triads)
    g=rotation([1.8,-1.7,2.1]);q=np.einsum('ij,njk->nik',g,ref.nodal_triads)
    law=DirectedHardeningSectionProbe(np.diag([1000.,400.,400.,.02,.01,.02]),[1.,0.,0.,0.,0.,0.],1e6,1.)
    for cls in (NonlinearMixedBeamProbe,CenteredChordMixedProbe):
        e=cls(ref,law,order=8).evaluate(ref.coordinates@g.T+[2.,-3.,4.],q,np.tile(g,(2,1,1)),np.zeros((2,2,3)))
        assert np.linalg.norm(e.residual[:18]) > 1e-11
