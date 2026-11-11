"""Explicit paired-mode consumers; unchanged continuum and preload gates."""
from types import SimpleNamespace
import numpy as np
import pytest
from anyfesolver import _ge_beam3_native_generalized_paired_modal as paired
from anyfesolver._native_paired_factor_chain_modes import apply_mode_map
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases.ge_beam3_reference_modal_groups import principal_correlations
from docs.reference_cases import ge_beam3_continuum_frequency_shooting as reference
import test_ge_beam3_native_generalized_modal as inherited
import test_ge_beam3_native_spectral_prestress as prestress
import test_ge_beam3_native_modal_groups as groups
from test_ge_beam3_native_continuum_frequencies import CASES,stations
from test_ge_beam3_schur_line_program import save


def facade():
    def prepare(*args,**kwargs):
        packet,guard=paired.prepare(*args,**kwargs)
        return packet.base,guard
    def solve(*args,**kwargs):
        packet,result=paired.solve_modes(*args,bounds=(-100.,1e6),**kwargs)
        return packet.base,result
    return SimpleNamespace(prepare=prepare,solve_modes=solve)


@pytest.mark.parametrize('curved',(False,True))
def test_free_body_paired_rigid_modes(curved,tmp_path):
    model,states,inertias=inherited.make(curved)
    before=canonical(states)
    packet,modes=paired.solve_modes(model,states,np.zeros(18),inertias,np.zeros(18),
        bounds=(-100.,1e6),num_modes=15)
    rigid=np.zeros((24,6))
    for i,x in enumerate(model.mesh.elements[1].operator.reference.coordinates):
        rigid[6*i:6*i+3,:3]=np.eye(3)
        rigid[6*i:6*i+3,3:]=-inherited.skew(x)
        rigid[6*i+3:6*i+6,3:]=np.eye(3)
    rigid[18:21,3:]=rigid[21:24,3:]=np.eye(3)
    reference_field=packet.kinetic@rigid
    field=apply_mode_map(modes,packet.kinetic)
    assert np.linalg.norm(field.T@field-np.eye(15))<=1e-11
    covariance=field[:,:6].T@reference_field
    squared=principal_correlations(field[:,:6].T@field[:,:6],covariance,
        reference_field.T@reference_field)
    assert np.min(squared)>=1.-1e-11
    scale=max(1.,np.linalg.norm(packet.base.stiffness)/np.linalg.norm(packet.base.mass))
    assert np.max(abs(modes.eigenvalues[:6]))<=1e-11*scale
    assert np.min(modes.eigenvalues[6:])>1e-11*scale
    assert modes.high_modes.shape==(24,15) and len(packet.base.algebraic_dofs)==9
    assert canonical(states)==before
    save(tmp_path/'rigids.json',dict(packet=packet,modes=modes,rigid_principal_correlations=squared,
        reference_rigid_kinetic=reference_field.T@reference_field,production_qualified=False))


def test_connected_shared_trace_paired(tmp_path):
    model,states,inertias=inherited.make(True,2)
    n=model.mesh.dof_manager.total_dofs;before=canonical(states)
    packet,modes=paired.solve_modes(model,states,np.zeros(n),inertias,np.zeros(n),
        bounds=(-100.,1e6),num_modes=15)
    assert packet.base.mass.shape==(42,42) and len(packet.base.algebraic_dofs)==15
    assert modes.high_modes.shape==(42,15)
    field=apply_mode_map(modes,packet.kinetic)
    assert np.linalg.norm(field.T@field-np.eye(15))<=1e-11
    assert np.max(abs(modes.eigenvalues[:6]))<1e-8 and np.min(modes.eigenvalues[6:])>1e-8
    assert canonical(states)==before
    save(tmp_path/'connected.json',dict(packet=packet,modes=modes,production_qualified=False))


@pytest.mark.parametrize('reversed',(False,True))
def test_paired_frame_covariance_and_reversal(reversed,tmp_path,monkeypatch):
    monkeypatch.setattr(inherited,'modal',facade())
    inherited.test_frame_covariance_and_reversal(reversed,tmp_path)


def test_paired_loaded_conservative_state(tmp_path,monkeypatch):
    monkeypatch.setattr(inherited,'modal',facade())
    inherited.test_loaded_conservative_state_preserves_history(tmp_path)


def weighted_fields(model,packet,modes,continuum,height,inertia,macros):
    """Map both halves to physical station velocity before final rounding."""
    native=[];exact=[];lookup={float(t):i for i,t in enumerate(continuum.sites)}
    layout=dict(packet.internal_layout)
    for eid,cell,weight,t,measure in stations(macros):
        e=model.mesh.elements[eid];coords=e.operator.reference.coordinates
        lift=.5*weight*(weight-1)*(coords[0]-2*coords[1]+coords[2])
        left=model.mesh.dof_manager.get_node_dofs(e.node_ids[cell])
        right=model.mesh.dof_manager.get_node_dofs(e.node_ids[cell+1])
        spin=list(layout[eid][3*cell:3*cell+3])
        mapping=np.zeros((6,len(packet.mass)))
        mapping[:3,list(left[:3])]=(1-weight)*np.eye(3)
        mapping[:3,list(right[:3])]=weight*np.eye(3)
        mapping[:3,spin]=-inherited.skew(lift)
        mapping[3:,spin]=np.eye(3)
        jac,_,rotation=reference.geometry(t,height)
        factor=np.sqrt(measure*jac)*np.linalg.cholesky(inertia).T@rotation.T
        native.append(apply_mode_map(modes,factor@mapping))
        exact.append(factor@continuum.fields[:,lookup[t],:6].T)
    return np.vstack(native),np.vstack(exact)


@pytest.mark.parametrize('case',CASES)
def test_paired_continuum_closed_groups(case,tmp_path,monkeypatch):
    monkeypatch.setattr(groups,'modal',facade())
    monkeypatch.setattr(groups,'weighted_fields',weighted_fields)
    groups.test_complete_native_modal_groups(case,tmp_path)


@pytest.mark.parametrize('macros',(1,2,4,8))
def test_paired_actual_buckling_refinement(macros,tmp_path,monkeypatch):
    monkeypatch.setattr(prestress,'modal',facade())
    prestress.test_buckling_refinement(macros,tmp_path)
