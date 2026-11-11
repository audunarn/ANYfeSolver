"""Bounded actual-state spectral boundaries and separate critical-load search."""
from copy import deepcopy
import numpy as np
import pytest
from anyfesolver import _ge_beam3_native_generalized_modal as modal
from anyfesolver._ge_beam3_native_arc import make_store
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern,assemble_distributed_trial
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_native_generalized_program import solve_distributed_model
from anyfesolver.nonlinear_state import native_trial_full_coordinates
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from anyfesolver.fe_core import FEModel
from test_ge_beam3_native_generalized_modal import make,close
from test_ge_beam3_schur_line_program import save

EMPTY=DistributedPattern(LinePattern(()),())


def axial_state(strain,macros,*,yield_force=1e6):
    source,_,masses=make(macros=macros,clamped=True,coupled=False)
    m=FEModel('native-spectral-preload')
    for i,node in source.mesh.nodes.items():m.add_node(i,*node.coords())
    for i,old in source.mesh.elements.items():
        section=EllipsoidalGeneralizedSection(old.section.elastic,np.eye(6),yield_force,1.)
        e=NativeGeneralizedStaticElement(i,old.node_ids,old.operator.reference,section,order=4)
        m.add_element(i,e);m.materials[e.material_name]=section
    for bc in source.boundary_conditions:m.add_boundary_condition(deepcopy(bc))
    n=m.mesh.dof_manager.total_dofs
    initial={i:e.init_model_bound_nonlinear_state(m.mesh,e.section,1) for i,e in m.mesh.elements.items()}
    store=make_store(m,initial,np.zeros(n))
    total=np.zeros(n)
    for node in sorted(m.mesh.nodes):
        total[m.mesh.dof_manager.get_node_dofs(node)[0]]=strain*m.mesh.nodes[node].coords()[0]
    force,_,_,external=assemble_distributed_trial(m,total,store,EMPTY)
    assert not np.any(external)
    # Independent one-dimensional specialization of the associated hardening law.
    trial=1000.*abs(strain)
    magnitude=trial if trial<=yield_force else (1000.*abs(strain)+1000.*yield_force)/1001.
    stress=float(np.copysign(magnitude,strain))
    expected=np.zeros(n)
    expected[m.mesh.dof_manager.get_node_dofs(1)[0]]=-stress
    expected[m.mesh.dof_manager.get_node_dofs(max(m.mesh.nodes))[0]]=stress
    error=close(force,expected)
    token=store.active_trial_token()
    store.commit(token,accepted_full_displacement=total,
        accepted_full_coordinates=native_trial_full_coordinates(store,m,total))
    states=store.materialize();tip=expected.copy()
    tip[m.mesh.dof_manager.get_node_dofs(1)[0]]=0.
    assert store.generation==1 and all(s['epoch']==1 for s in states.values())
    return m,states,masses,total,tip,store,error


@pytest.mark.parametrize('kind',('yield-boundary','plastic'))
def test_actual_material_boundary_is_not_a_frequency(kind,tmp_path):
    strain=.125/1000 if kind=='yield-boundary' else .001
    m,states,masses,total,force,store,error=axial_state(strain,1,yield_force=.125)
    before=canonical(states);generation=store.generation
    with pytest.raises(ValueError,match='current-rest (perturbation|spectra|material branch)') as caught:
        modal.solve_modes(m,states,total,masses,force)
    assert canonical(states)==before and canonical(store.materialize())==before
    assert store.generation==generation and not store.has_active_trial
    positive=sum(sum(h.accumulated)>0 for s in states.values() for h in s['response'].history.stations)
    if kind=='plastic':assert positive==8
    save(tmp_path/'boundary.json',dict(kind=kind,error=str(caught.value),rejected=True,
        committed_state_sha256=sha(states),positive_plastic_stations=positive,
        axial_equilibrium_error=error,history_unchanged=True))


def test_deformed_kinetic_work_and_live_trial_ownership(tmp_path):
    m,_,masses=make(True,clamped=True)
    pattern=DistributedPattern(LinePattern(((1,.05,-.02,.03),)),())
    result,_=solve_distributed_model(m,pattern,steps=1)
    assert result.status=='completed',result.info
    store=make_store(m,result.element_states,result.displacements)
    committed=store.materialize();before=canonical(committed)
    e=m.mesh.elements[1]
    trial=result.displacements.copy();trial[12]+=.0001
    _,_,pending,_=assemble_distributed_trial(m,trial,store,pattern)
    pending_bytes=canonical(dict(pending));issued=e._issued;validator=e._validator;token=store.active_trial_token()
    generation=store.generation
    try:
        packet,guard=modal.prepare(m,committed,result.displacements,masses,np.zeros(18))
        assert store.active_trial_token()==token and store.generation==generation
        assert e._validator is validator and e._issued is issued
        assert canonical(dict(pending))==pending_bytes and canonical(store.materialize())==before
        speed=np.sin(np.arange(24)+.37);energy=0.
        state=committed[1];points,weights=np.polynomial.legendre.leggauss(64)
        coords=e.operator.reference.coordinates;b=coords[0]-2*coords[1]+coords[2]
        for cell in (0,1):
            u=state['response'].rotations[cell];spin=speed[18+3*cell:21+3*cell]
            for p,w in zip(points,weights):
                t=float((p+1)/2);xi=cell-1+t;offset=u@(.5*t*(t-1)*b)
                velocity=(1-t)*speed[6*cell:6*cell+3]+t*speed[6*(cell+1):6*(cell+1)+3]+np.cross(spin,offset)
                q=u@e.operator.reference.frame(xi);v=np.r_[q.T@velocity,q.T@spin]
                energy+=float(w*e.operator.reference.jacobian(xi)*(v@masses[1]@v)/4)
        error=close(np.array(speed@packet.mass@speed/2),np.array(energy));guard()
    finally:
        store.discard_trial(token)
    assert canonical(store.materialize())==before and not store.has_active_trial
    save(tmp_path/'live-state.json',dict(kinetic_error=error,independent_energy=energy,
        packet=packet,accepted_history_unchanged=True,pending_trial_unchanged=True,
        validator_unchanged=True,discard_succeeded=True))


@pytest.mark.parametrize('macros',(1,2,4,8))
def test_buckling_refinement(macros,tmp_path):
    # Clamped-free length 2, weaker bending rigidity 1. Compression is positive.
    euler=float(np.pi**2/16)
    rows=[]
    def evaluate(compression):
        print(dict(stage='preload',macros=macros,compression=compression),flush=True)
        m,states,masses,total,force,store,error=axial_state(float(-compression/1000.),macros)
        before=canonical(states)
        packet,modes=modal.solve_modes(m,states,total,masses,force,num_modes=1)
        assert canonical(states)==before and not store.has_active_trial
        lam=float(modes.eigenvalues[0])
        row=dict(index=len(rows),compression=compression,lambda_min=lam,
            state_sha256=sha(states),equilibrium_error=error)
        save(tmp_path/('point-%02d.json'%len(rows)),dict(row=row,packet=packet,modes=modes,total=total,
            manufactured_uniform_axial_equilibrium=True,global_newton_programme_run=False))
        rows.append(row)
        print(dict(stage='spectrum',**row),flush=True)
        return lam
    tension=evaluate(-.5*euler);unloaded=evaluate(0.);compressed=evaluate(.5*euler)
    assert tension>unloaded>compressed>0.
    low,high=0.,2*euler
    assert evaluate(high)<0.
    low_value,high_value=unloaded,rows[-1]['lambda_min']
    for _ in range(18):
        midpoint=float((low+high)/2);value=evaluate(midpoint)
        if value>0.:low,low_value=midpoint,value
        else:high,high_value=midpoint,value
    critical=float((low+high)/2)
    assert low_value>0. and high_value<=0. and (high-low)/euler<1e-5
    error=abs(critical/euler-1.)
    if macros==8:assert error<.02
    save(tmp_path/'buckling.json',dict(macros=macros,rows=rows,critical_bracket=[low,high],
        bracket_eigenvalues=[low_value,high_value],critical_midpoint=critical,euler_reference=euler,
        relative_euler_error=error,finest_engineering_gate=(macros==8),
        search='18_BISECTIONS_OF_ACTUAL_PRELOADED_SIGNED_PENCIL',
        current_frequency_is_not_a_load_factor=True,production_qualified=False,
        full_spatial_postbuckling_qualified=False))
