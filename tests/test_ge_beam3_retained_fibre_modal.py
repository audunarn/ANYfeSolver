"""Physical fibre current-rest operators, history safety and model dispatch."""
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest
from scipy.linalg import eigh

from anyfesolver import _ge_beam3_retained_fibre_modal as modal
from anyfesolver import _ge_beam3_retained_fibre_control as control
from anyfesolver import _ge_beam3_analysis_fibre_translation as route
from anyfesolver import _ge_beam3_fibre_reference_modal as reference_modal
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_retained_fibre import make_model
from test_ge_beam3_retained_fibre_control import program, transformed_model
from test_ge_beam3_analysis_fibre_translation import analysis


def digest(raw):return sha256(raw).hexdigest()
def masses(model):return {i:np.diag([2.,2.,2.,.07,.09,.11]) for i in model.mesh.elements}
def capture(model,p,raw,**kw):
    return modal.prepare(model,p,raw,masses(model),expected_sha256=digest(raw),**kw)


@pytest.fixture(scope='module')
def unloaded():
    p=replace(program(),targets=(.01,.02,.019))
    run=control.solve_translation_program(make_model(),p)
    assert run.status=='completed',run.failure
    assert any(row[2]>0 for cell in run.state.histories for station in cell.stations for row in station.rows)
    return p,run


def test_unloaded_plastic_state_equals_full_stationary_schur(unloaded,tmp_path):
    p,run=unloaded;m=make_model();packet,guard=capture(m,p,run.checkpoint)
    context=control.Context(m,p);state,records=context.restore(run.checkpoint,expected_sha256=digest(run.checkpoint))
    before=canonical(state)
    residual,h,metrics,responses=context.assemble(state.mechanical,state.parameter,state.histories,p.targets[-1])
    layout=context.layout;coordinates=list(range(layout.nodal_count));resultants=[]
    for i in range(len(layout.elements)):
        coordinates.extend(range(layout.nodal_count+24*i,layout.nodal_count+24*i+6))
        resultants.extend(range(layout.nodal_count+24*i+6,layout.nodal_count+24*(i+1)))
    expected=h[np.ix_(coordinates,coordinates)]-h[np.ix_(coordinates,resultants)]@np.linalg.solve(
        h[np.ix_(resultants,resultants)],h[np.ix_(resultants,coordinates)])
    error=np.linalg.norm(expected-packet.stiffness)/max(1.,np.linalg.norm(expected))
    assert error<=1e-11 and max(metrics)<=1e-11
    assert packet.completed_targets==3 and packet.parameter==state.parameter
    assert packet.material_policy==modal.MATERIAL_POLICY and not packet.control_constraint_in_physical_stiffness
    assert 24 in packet.free_dofs and not np.any(packet.mass[:,packet.algebraic_dofs])
    assert canonical(state)==before and context.checkpoint(records)==run.checkpoint
    for response,origin in zip(responses,state.histories):assert canonical(response.history)==canonical(origin)
    guard();(tmp_path/'full-schur.json').write_bytes(canonical(dict(packet=packet,full_schur_error=float(error))))
    (tmp_path/'unloaded-checkpoint.json').write_bytes(run.checkpoint)


def skew(v):
    x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def test_reference_operator_has_six_rigids_and_physical_mass():
    m=make_model();p=program();context=control.Context(m,p);raw=context.checkpoint(())
    packet,guard=capture(m,p,raw);n=context.layout.nodal_count
    rigid=np.zeros((len(packet.mass),6))
    for i,node in enumerate(context.layout.node_ids):
        rigid[6*i:6*i+3,:3]=np.eye(3);rigid[6*i:6*i+3,3:]=-skew(m.mesh.nodes[node].coords())
        rigid[6*i+3:6*i+6,3:]=np.eye(3)
    for i in range(len(context.layout.elements)):
        rigid[n+6*i:n+6*i+3,3:]=np.eye(3);rigid[n+6*i+3:n+6*i+6,3:]=np.eye(3)
    assert np.linalg.norm(packet.stiffness@rigid)/max(1.,np.linalg.norm(packet.stiffness))<1e-11
    assert np.linalg.matrix_rank(packet.stiffness)==len(packet.mass)-6
    np.linalg.cholesky(rigid.T@packet.mass@rigid)
    assert np.linalg.matrix_rank(packet.mass)==len(packet.mass)-3*len(context.layout.node_ids)
    guard()


def test_model_owned_modes_equal_complete_physical_pencil(unloaded,tmp_path):
    p,run=unloaded;made=analysis()
    envelope=made.import_translation_checkpoint(p,run.checkpoint,expected_sha256=digest(run.checkpoint))
    result=made.translation_modes(p,envelope,expected_sha256=digest(envelope),bounds=(-100.,1e6))
    packet=result.packet
    algebraic=list(packet.algebraic_dofs);physical=[d for d in packet.free_dofs if d not in algebraic]
    k=packet.stiffness;m=packet.mass
    reduced=k[np.ix_(physical,physical)]-k[np.ix_(physical,algebraic)]@np.linalg.solve(
        k[np.ix_(algebraic,algebraic)],k[np.ix_(algebraic,physical)])
    expected=eigh(reduced,m[np.ix_(physical,physical)],eigvals_only=True)[:6]
    np.testing.assert_allclose(result.modes.eigenvalues,expected,rtol=1e-11,atol=1e-11)
    assert result.definition_graph_sha256==made.identity and result.checkpoint_sha256==digest(envelope)
    repeated=analysis().translation_modes(p,envelope,expected_sha256=digest(envelope),bounds=(-100.,1e6))
    assert canonical(result)==canonical(repeated)
    assert not result.production_qualified
    (tmp_path/'owned-modes.json').write_bytes(canonical(result))
    (tmp_path/'owned-checkpoint.json').write_bytes(envelope)


def test_loaded_yield_boundary_rejected_not_relabelled_as_elastic(unloaded):
    p,run=unloaded;context=control.Context(make_model(),p);_,records=context.restore(run.checkpoint)
    with pytest.raises(ValueError,match='history|yield|elastic'):
        capture(make_model(),p,context.checkpoint(records[:2]))


@pytest.mark.parametrize('mutation',('branch','yield_margin','history','increment','fibre_id','derivative'))
def test_material_admission_checks_every_fibre(mutation):
    m=make_model();context=control.Context(m,program());operator=m.mesh.elements[1].operator
    state=context.initial.mechanical;origin=context.initial.histories[0];nodes=context.layout.nodes[0]
    response=operator.evaluate(state.positions[nodes],state.position_low[nodes],state.nodal_frames[nodes],
        state.cell_rotations[0],state.resultants[0],origin=origin)
    data=json.loads(response.material);field=data['stations'][-1]['fibres'][-1]
    if mutation=='branch':field['branch']='PLASTIC_POSITIVE'
    elif mutation=='yield_margin':field['stress']=[operator.section.fibres[-1].curve.flow_stress[0],0.]
    elif mutation=='increment':field['plastic_increment']=[.001,0.]
    elif mutation=='fibre_id':field['fibre_id']='another-fibre'
    elif mutation=='derivative':data['derivative_kind']='SEMISMOOTH_BRANCH_SELECTION'
    else:
        history=replace(origin.stations[-1],rows=tuple((.1,0.,.1,0.) for _ in origin.stations[-1].rows))
        response=replace(response,history=replace(origin,stations=(*origin.stations[:-1],history)))
    response=replace(response,material=canonical(data))
    with pytest.raises(ValueError):modal.elastic_interior(operator,response,origin)


@pytest.mark.parametrize('mutation',('hash','inertia','model','packet','token','force_checkpoint','section_content'))
def test_custody_and_cancellation_fail_closed(mutation):
    m=make_model();p=program();context=control.Context(m,p);raw=context.checkpoint(())
    if mutation=='hash':
        with pytest.raises(ValueError,match='authority'):modal.prepare(m,p,raw,masses(m),expected_sha256='0'*64)
    elif mutation=='token':
        token=CancellationToken();token.cancel()
        with pytest.raises(SolveCancelled):capture(m,p,raw,cancellation_token=token)
    elif mutation=='force_checkpoint':
        other=context.physical.checkpoint(())
        with pytest.raises(ValueError):capture(m,p,other)
    else:
        inertia=masses(m);packet,guard=modal.prepare(m,p,raw,inertia,expected_sha256=digest(raw))
        if mutation=='inertia':inertia[1][0,0]*=2
        elif mutation=='model':m.mesh.nodes[1].x+=.1
        elif mutation=='section_content':
            section=m.mesh.elements[1].section
            object.__setattr__(section,'fibres',(replace(section.fibres[0],young=60.),*section.fibres[1:]))
        else:object.__setattr__(packet,'control_constraint_in_physical_stiffness',True)
        with pytest.raises(ValueError):guard()


def test_fibre_factor_covariance_under_common_rotation():
    transform=rotation([2.6,.8,-.3]);p=program();a=make_model();b=transformed_model(transform)
    moved=replace(p,direction=tuple(map(float,transform@np.array(p.direction))),
        nodal_forces=tuple((n,*map(float,transform@np.array(f))) for n,*f in p.nodal_forces))
    ca=control.Context(a,p);cb=control.Context(b,moved)
    pa,ga=capture(a,p,ca.checkpoint(()));pb,gb=capture(b,moved,cb.checkpoint(()))
    rotation_map=np.kron(np.eye(len(pa.mass)//3),transform)
    for name in ('stiffness','mass'):
        expected=rotation_map@getattr(pa,name)@rotation_map.T
        assert np.linalg.norm(expected-getattr(pb,name))/max(1.,np.linalg.norm(expected))<=1e-11
    ga();gb()


def test_reference_modes_match_virgin_control_without_creating_a_checkpoint(tmp_path):
    made=analysis();p=program();context=control.Context(route._model(made),p)
    controlled,guard=modal.prepare(context.layout.model,p,context.checkpoint(()),made._inertias,
        expected_sha256=digest(context.checkpoint(())))
    packet,modes=made.reference_modes(bounds=(-100.,1e6))
    for key in ('left','right','geometric','kinetic','stiffness','mass'):
        np.testing.assert_array_equal(getattr(packet,key),getattr(controlled,key))
    assert packet.free_dofs==controlled.free_dofs and packet.algebraic_dofs==controlled.algebraic_dofs
    assert packet.definition_graph_sha256==made.identity and not hasattr(packet,'checkpoint_sha256')
    assert np.all(modes.eigenvalues>0.)
    guard();(tmp_path/'reference-modes.json').write_bytes(canonical(dict(packet=packet,modes=modes)))


def test_free_body_fibre_reference_has_six_zero_modes_and_elastic_modes(tmp_path):
    supported=analysis()
    free=NativeBeamAnalysis(tuple(NativeBeamDefinition(raw) for raw in supported._definitions),())
    packet,modes=free.reference_modes(bounds=(-100.,1e6),num_modes=10)
    assert np.max(np.abs(modes.eigenvalues[:6]))<=1e-11
    assert np.all(modes.eigenvalues[6:]>1e-8)
    assert packet.free_dofs==tuple(range(len(packet.mass)))
    assert len(packet.algebraic_dofs)==15
    assert not modes.production_qualified
    (tmp_path/'free-reference.json').write_bytes(canonical(dict(packet=packet,modes=modes)))


def test_reference_capture_guards_model_inertia_and_cancellation():
    made=analysis();packet,guard=reference_modal.prepare(made)
    made._inertias[1]=2*made._inertias[1]
    with pytest.raises(ValueError):guard()
    token=CancellationToken();token.cancel()
    with pytest.raises(SolveCancelled):analysis().reference_modes(bounds=(-100.,1e6),cancellation_token=token)
