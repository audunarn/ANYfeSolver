"""Actual model-owned native force, restart, recovery and separate modal routes."""
from copy import deepcopy
from hashlib import sha256
import json
import numpy as np
import pytest

from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis, NativeBeamAnalysisError, NativeBeamWorkflowError
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_native_generalized_program import solve_distributed_model
from anyfesolver._ge_beam3_native_generalized_modal import solve_modes
from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver.boundary import LoadCase, BoundaryCondition
from test_ge_beam3_native_generalized import problem as generalized_problem, pattern
from test_ge_beam3_native_fibre_static_element import problem as fibre_problem
from test_ge_beam3_native_generalized_modal import make as modal_model


def definitions(model, masses=None):
    return tuple(NativeBeamDefinition.capture(e, np.diag([2.,2.,2.,.07,.09,.11]) if masses is None else masses[i])
                 for i,e in sorted(model.mesh.elements.items()))


def analysis(model, masses=None):
    return NativeBeamAnalysis(definitions(model, masses), tuple(model.boundary_conditions))


def comparable(result):
    return canonical(dict(status=result.status, displacement=result.displacements, states=result.element_states))


@pytest.mark.parametrize('family', ('generalized','fibre'))
@pytest.mark.parametrize('curved', (False,True))
def test_definition_graph_owns_real_native_model(family,curved):
    source,e=(generalized_problem if family=='generalized' else fibre_problem)(curved,True)
    made=analysis(source);other=analysis(source)
    assert made.identity==other.identity
    assert made.model is not source and made.model.mesh.elements[1] is not e
    assert type(made.model.mesh.elements[1]) is type(e)
    assert made.model.mesh.dof_manager.total_dofs==18
    made._guard();other._guard()


@pytest.mark.parametrize('mutation', ('node','boundary','material','inertia','identity','owner','elements'))
def test_model_mutation_rejected_before_state_initialization(mutation,monkeypatch):
    source,e=generalized_problem();made=analysis(source)
    def forbidden():raise AssertionError('state initialized on changed model')
    monkeypatch.setattr(made,'_initial',forbidden)
    if mutation=='node':made.model.mesh.nodes[1].x+=.1
    elif mutation=='boundary':made.model.boundary_conditions[0].name+='changed'
    elif mutation=='material':made.model.materials[e.material_name]=object()
    elif mutation=='inertia':made._inertias[1]=2*made._inertias[1]
    elif mutation=='identity':made.identity='0'*64
    elif mutation=='owner':made._family='PHYSICAL_FIBRE_NODAL'
    else:made.model.mesh.elements[1]=source.mesh.elements[1]
    with pytest.raises(ValueError):made.reference_modes()


def test_concurrent_use_rejected_before_state_initialization(monkeypatch):
    made=analysis(generalized_problem()[0])
    monkeypatch.setattr(made,'_initial',lambda: (_ for _ in ()).throw(AssertionError('entered state initialization')))
    made._lock.acquire()
    try:
        with pytest.raises(NativeBeamAnalysisError,match='already in use'):made.reference_modes()
    finally:made._lock.release()


@pytest.mark.parametrize('steps,iterations', ((True,24),(0,24),(17,24),(2,True),(2,0),(2,25)))
def test_bad_solver_controls_fail_before_state(steps,iterations,monkeypatch):
    made=analysis(generalized_problem()[0])
    monkeypatch.setattr(made,'_initial',lambda: (_ for _ in ()).throw(AssertionError('mechanics entered')))
    with pytest.raises(NativeBeamAnalysisError):made.solve_distributed(pattern(),steps=steps,max_iterations=iterations)


def test_no_cross_family_workflow_or_legacy_fallback():
    gen=analysis(generalized_problem()[0]);fib=analysis(fibre_problem()[0])
    with pytest.raises(NativeBeamWorkflowError):gen.solve_nodal(((3,1.,0.,0.),))
    with pytest.raises(NativeBeamWorkflowError):fib.solve_distributed(pattern())
    with pytest.raises(NativeBeamWorkflowError):fib.reference_modes()
    with pytest.raises(NativeBeamAnalysisError):NativeBeamAnalysis((),())
    with pytest.raises(NativeBeamAnalysisError):NativeBeamAnalysis((object(),),())


def test_definition_graph_hash_rejects_inertia_rebinding_before_backend_decode(monkeypatch):
    made=analysis(generalized_problem()[0])
    raw=made._envelope(b'not-a-real-backend')
    row=json.loads(raw);row['definition_graph_sha256']='0'*64
    raw=canonical(row)
    with pytest.raises(NativeBeamAnalysisError,match='owner mismatch'):
        made.recover(raw,expected_sha256=sha256(raw).hexdigest())
    with pytest.raises(NativeBeamAnalysisError,match='external'):
        made.recover(raw,expected_sha256='0'*64)


@pytest.mark.parametrize('kind', ('missing_node','duplicate_node','unknown_dof','nonzero','boolean','partial_rotation','cached_mismatch'))
def test_unsupported_boundaries_rejected_during_construction(kind):
    model,_=generalized_problem();node_ids=[1];values={'ux':0.}
    if kind=='missing_node':node_ids=[99]
    elif kind=='duplicate_node':node_ids=[1,1]
    elif kind=='unknown_dof':values={'magic':0.}
    elif kind=='nonzero':values={'ux':.01}
    elif kind=='boolean':values={'ux':False}
    elif kind=='partial_rotation':values={'rx':0.}
    boundary=BoundaryCondition('invalid',node_ids,values)
    if kind=='cached_mismatch':boundary._dof_indices={}
    with pytest.raises(NativeBeamAnalysisError):NativeBeamAnalysis(definitions(model),(boundary,))


def test_free_force_model_rejected_before_initial_state(monkeypatch):
    source,_=generalized_problem();made=NativeBeamAnalysis(definitions(source),())
    monkeypatch.setattr(made,'_initial',lambda: (_ for _ in ()).throw(AssertionError('mechanics entered')))
    with pytest.raises(NativeBeamAnalysisError,match='before state initialization'):
        made.solve_distributed(pattern())


def test_generalized_actual_native_solve_restart_and_recovery(tmp_path):
    source,_=generalized_problem(True,True);made=analysis(source);load=pattern()
    whole=made.solve_distributed(load)
    assert whole.status=='completed',whole.backend_result.info
    direct,_=solve_distributed_model(source,load)
    assert comparable(whole.backend_result)==comparable(direct)
    recovered=made.recover(whole.checkpoint,expected_sha256=whole.checkpoint_sha256)
    assert recovered[1]['fibre_stress_status']=='RESULTANT_LEVEL_ONLY_NO_FIBRE_STRESSES_SUPPLIED'
    changed=analysis(source,{1:2*np.diag([2.,2.,2.,.07,.09,.11])})
    with pytest.raises(NativeBeamAnalysisError,match='owner mismatch'):
        changed.recover(whole.checkpoint,expected_sha256=whole.checkpoint_sha256)
    prefix=made.checkpoint_prefix(whole.checkpoint,1,expected_sha256=whole.checkpoint_sha256)
    half=DistributedPattern(LinePattern(tuple((eid,*map(float,.5*np.array(force))) for eid,*force in load.line.rows)),
        tuple((eid,*map(float,.5*np.array(moment))) for eid,*moment in load.couples))
    resumed=analysis(source).solve_distributed(half,steps=1,checkpoint=prefix,expected_sha256=sha256(prefix).hexdigest())
    assert comparable(resumed.backend_result)==comparable(whole.backend_result)
    # Distributed spatial couples cannot become a conservative modal claim.
    with pytest.raises(ValueError,match='nonconservative'):
        made.current_modes(whole.checkpoint,expected_sha256=whole.checkpoint_sha256)
    assert made.recover(whole.checkpoint,expected_sha256=whole.checkpoint_sha256)[1]['state_sha256']==recovered[1]['state_sha256']
    (tmp_path/'checkpoint.json').write_bytes(whole.checkpoint)
    (tmp_path/'native.json').write_bytes(canonical(dict(response=json.loads(comparable(direct)),recovery=recovered,
        resume_equal=True,conservative_misclassification_rejected=True,production_qualified=False)))


def test_fibre_actual_native_solve_restart_and_recovery(tmp_path):
    source,_=fibre_problem(True,True);made=analysis(source);forces=((3,.35,-.012,.006),)
    whole=made.solve_nodal(forces)
    assert whole.status=='completed',whole.backend_result.info
    load=LoadCase('direct-native');load.add_nodal_load(3,forces=np.array(forces[0][1:]))
    direct=solve_static_nonlinear(source,load,num_steps=2,max_iterations=24,tolerance=1e-12,
        num_layers=1,min_step_fraction=1.,record_increment_snapshots=True,equilibrate_initial_state=False)
    assert comparable(whole.backend_result)==comparable(direct)
    recovery=made.recover(whole.checkpoint,expected_sha256=whole.checkpoint_sha256)
    assert recovery[1]['fibre_stress_status']=='PHYSICAL_SECTION_SUPPLIED_PAIRED_STRESSES'
    prefix=made.checkpoint_prefix(whole.checkpoint,1,expected_sha256=whole.checkpoint_sha256)
    resumed=analysis(source).solve_nodal(forces,steps=1,checkpoint=prefix,expected_sha256=sha256(prefix).hexdigest())
    assert comparable(resumed.backend_result)==comparable(whole.backend_result)
    (tmp_path/'checkpoint.json').write_bytes(whole.checkpoint)
    (tmp_path/'native.json').write_bytes(canonical(dict(response=json.loads(comparable(direct)),recovery=recovery,
        resume_equal=True,production_qualified=False)))


def test_failed_native_force_preserves_only_valid_accepted_prefix(tmp_path):
    source,_=fibre_problem(True,True);made=analysis(source)
    failed=made.solve_nodal(((3,.35,-.012,.006),),steps=1,max_iterations=1)
    assert failed.status!='completed'
    chain=made._decode(failed.checkpoint,failed.checkpoint_sha256)
    assert len(chain)==1 and chain[0]['states'][1]['epoch']==0
    assert np.array_equal(chain[0]['displacements'],np.zeros(18))
    made.recover(failed.checkpoint,expected_sha256=failed.checkpoint_sha256)
    (tmp_path/'failure-prefix.json').write_bytes(failed.checkpoint)


@pytest.mark.parametrize('curved',(False,True))
def test_actual_reference_modal_retains_cell_inertia(curved,tmp_path):
    source,states,masses=modal_model(curved=curved,clamped=False)
    made=analysis(source,masses)
    packet,modes=made.reference_modes(num_modes=15)
    original,original_modes=solve_modes(source,states,np.zeros(18),masses,np.zeros(18),num_modes=15)
    assert packet.mass.shape==(24,24) and packet.internal_layout==((1,tuple(range(18,24))),)
    assert packet.mass.tobytes()==original.mass.tobytes()
    assert packet.stiffness.tobytes()==original.stiffness.tobytes()
    assert modes.eigenvalues.tobytes()==original_modes.eigenvalues.tobytes()
    assert len(packet.algebraic_dofs)==9 and np.linalg.matrix_rank(packet.mass)==15
    scale=max(1.,np.linalg.norm(packet.stiffness)/np.linalg.norm(packet.mass))
    assert np.max(np.abs(modes.eigenvalues[:6]))<=1e-11*scale and min(modes.eigenvalues[6:])>1e-11*scale
    (tmp_path/'modal.json').write_bytes(canonical(dict(packet=packet,modes=modes,actual_descriptor_route=True,
        static_mass_substitution=False,production_qualified=False)))
