"""Physical B2 five-row footprint batches: source development controls."""
from dataclasses import replace,FrozenInstanceError
import json,os
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
from examples.stiffened_shell_b2_physical_rigid_footprint import definition,preload,batch_workflow

SELECT=api.B2PhysicalRigidFootprintSelection(("physical-left",))
DOFS=(12,13,14,36,37,41)

def load():
    value=preload();value[13]=.7;value[14]=.2
    return value

@pytest.fixture
def owner():
    value=api.create_analysis(definition=definition());yield value;value.close()

@pytest.mark.parametrize("count",(1,10,100))
def test_public_batch_counts_selected_outputs_and_envelopes(owner,count):
    result=owner.solve_linear_many(tuple((f"case-{i}",(i+1)*load()) for i in range(count)),selected_dofs=DOFS,
        selected_footprints=SELECT,recover_members={"beam-right":(0.,.5,1.)})
    assert type(result)is api.B2PhysicalRigidFootprintBatchResult and result.status=="completed",result.failure_reason
    assert result.factorization_count==1 and result.solve_count==count and result.maximum_rhs_chunk==1
    assert len(result.cases)==count and len(result.constraint_rows)==20
    assert result.footprint_law_id==api.B2_PHYSICAL_RIGID_FOOTPRINT_DIRECTOR_REFERENCE_V1_EXPERIMENTAL
    assert "solve_linear_many" in owner.capabilities.operations
    reference=api.create_analysis(definition=definition())
    try:single=reference.solve_linear(load(),selected_footprints=SELECT,recover_members={"beam-right":(0.,.5,1.)})
    finally:reference.close()
    assert single.status=="completed"
    for i,case in enumerate(result.cases):
        assert type(case)is api.B2PhysicalRigidFootprintSelectedCase
        assert not hasattr(case,"displacements") and not hasattr(case,"affine_reactions")
        assert len(case.selected_displacements)==len(DOFS) and len(case.selected_affine_reactions)==len(DOFS)
        assert len(case.beam_resultants_local)==1 and case.beam_resultants_local[0][0]=="beam-right"
        assert len(case.constraint_multipliers)==len(case.constraint_gaps)==20
        np.testing.assert_allclose(case.selected_displacements,np.asarray(single.displacements)[list(DOFS)]*(i+1),rtol=1e-8,atol=1e-12)
        np.testing.assert_allclose(case.selected_affine_reactions,np.asarray(single.affine_reactions)[list(DOFS)]*(i+1),rtol=1e-8,atol=1e-12)
        np.testing.assert_allclose(case.beam_resultants_local[0][1],np.array(single.beam_resultants_local[0][1])*(i+1),rtol=1e-8,atol=1e-12)
        w=case.footprint_wrenches[0]
        np.testing.assert_allclose(w.beam_moment_Nm,np.array(single.footprint_wrenches[0].beam_moment_Nm)*(i+1),rtol=1e-8,atol=1e-12)
        assert abs(w.beam_moment_Nm[2])>.1 # shell-arm load path, not grounded beam-tip couple
        assert max(abs(np.array(case.constraint_gaps)))<=1e-9
    for envelope,j in zip(result.displacement_envelopes,range(len(DOFS))):
        values=[case.selected_displacements[j] for case in result.cases]
        assert envelope.minimum==min(values) and envelope.maximum==max(values)
        assert envelope.minimum_case_id==result.cases[values.index(min(values))].case_id
    assert len(result.member_resultant_envelopes)==18
    for envelope in result.member_resultant_envelopes:
        channel=("N","Qy","Qz","Mx","My","Mz").index(envelope.component)
        values=[dict(case.beam_resultants_local)[envelope.member_id]
                [envelope.station_index][channel] for case in result.cases]
        assert envelope.minimum==min(values) and envelope.maximum==max(values)
        assert envelope.minimum_case_id==result.cases[values.index(min(values))].case_id
        assert envelope.maximum_case_id==result.cases[values.index(max(values))].case_id
    with pytest.raises(FrozenInstanceError):result.cases[0].case_id="changed"


def test_no_unselected_recovery_and_first_case_tie(owner,monkeypatch):
    import anyfesolver._stiffened_shell_b2_q4_panel_experimental as module
    def forbidden(*args,**kwargs):raise AssertionError("unselected wrench recovered")
    monkeypatch.setattr(module,"physical_selected_wrenches",forbidden)
    for _,(member,_) in owner._members:monkeypatch.setattr(type(member.operator),"recover_sections",forbidden)
    result=owner.solve_linear_many((("first",load()),("second",load())),selected_dofs=DOFS)
    assert result.status=="completed" and result.constraint_rows==()
    assert result.member_resultant_envelopes==()
    for case in result.cases:assert case.footprint_wrenches==case.beam_resultants_local==case.constraint_multipliers==case.constraint_gaps==()
    for envelope in result.displacement_envelopes+result.affine_reaction_envelopes:
        assert envelope.minimum_case_id==envelope.maximum_case_id=="first"

@pytest.mark.parametrize("stop",("cancel","close","final_close","final_cancel"))
def test_complete_case_prefix_lifetime(owner,stop):
    token=CancellationToken();calls=[]
    def callback(case):
        calls.append(case)
        if stop=="cancel":token.cancel("complete case")
        elif stop=="close":owner.close()
        elif len(calls)==3:
            if stop=="final_cancel":token.cancel("last complete case")
            else:owner.close()
    result=owner.solve_linear_many(tuple((str(i),load()*(i+1)) for i in range(3)),selected_dofs=DOFS,selected_footprints=SELECT,cancellation_token=token,on_case=callback)
    count=3 if stop.startswith("final_") else 1
    assert result.status==("completed" if count==3 else "cancelled") and len(result.cases)==count
    assert result.factorization_count==1 and result.solve_count==count
    assert all(len(case.footprint_wrenches)==1 for case in result.cases)
    if stop in ("close","final_close"):assert owner._handle is None


def test_precancel_and_selection_refusal_do_not_factor(owner):
    factors=owner._handle.factorization_count;solves=owner._handle.solve_count
    token=CancellationToken();token.cancel()
    result=owner.solve_linear_many((("one",load()),),selected_dofs=DOFS,selected_footprints=SELECT,cancellation_token=token)
    assert result.status=="cancelled" and result.cases==() and result.factorization_count==factors and result.solve_count==solves
    with pytest.raises(ValueError):owner.solve_linear_many((("one",load()),),selected_dofs=DOFS,selected_footprints=api.B2NodalRigidFootprintSelection(("physical-left",)))
    assert owner._handle.factorization_count==factors and owner._handle.solve_count==solves


def test_submission_inventory_and_recovery_are_owned(owner):
    first=load();second=load()*2;cases=[["one",first],["two",second]];dofs=list(DOFS);recovery={"beam-right":[0.,1.]}
    def callback(case):
        if case=="one":cases[1][:]=["replacement",np.zeros(54)];dofs[:]=[0];recovery["beam-right"][:]=[.5]
    result=owner.solve_linear_many(cases,selected_dofs=dofs,selected_footprints=SELECT,recover_members=recovery,on_case=callback)
    assert result.status=="completed" and tuple(c.case_id for c in result.cases)==("one","two") and result.selected_dofs==DOFS
    assert all(len(c.beam_resultants_local[0][1])==2 for c in result.cases)


def test_later_load_mutation_and_callback_errors_propagate(owner):
    values=[load(),load()*2]
    def mutate(case):values[1][13]+=1.
    with pytest.raises(ValueError,match="changed after submission"):
        owner.solve_linear_many((("one",values[0]),("two",values[1])),selected_dofs=DOFS,on_case=mutate)
    def explode(case):raise RuntimeError("callback programming error")
    with pytest.raises(RuntimeError,match="callback programming error"):
        owner.solve_linear_many((("one",load()),),selected_dofs=DOFS,on_case=explode)

@pytest.mark.parametrize("after",(0,1))
def test_factor_failure_never_publishes_offset_only_case(owner,monkeypatch,after):
    original=owner._handle.solve;calls=[]
    def failed(rhs):
        if len(calls)==after:raise RuntimeError("deliberate failed factor solve")
        calls.append(1);return original(rhs)
    monkeypatch.setattr(owner._handle,"solve",failed)
    result=owner.solve_linear_many((("one",load()),("two",load()*2)),selected_dofs=DOFS,selected_footprints=SELECT)
    assert result.status=="failed" and len(result.cases)==after and "deliberate failed factor" in result.failure_reason
    assert result.solve_count==after


def transformed_definition(length=1.,force=1.,rotation=None,reverse=False):
    d=definition();r=np.eye(3) if rotation is None else rotation
    nodes={i:tuple(length*(r@x)) for i,x in d.shell_nodes_m.items()}
    shells=tuple(replace(shell,E=shell.E*force/length**2,thickness_m=shell.thickness_m*length,reference_normal=tuple(r@np.array(shell.reference_normal))) for shell in d.shells)
    section=api.LinearSection.isotropic(E=2000.*force/length**2,nu=.25,area=.04*length**2,Iy=.04*.2**2/12.*length**4,Iz=.04*.2**2/12.*length**4,J=2*.04*.2**2/12.*length**4,ky=.8,kz=.8)
    members=[]
    for member in d.members:
        ids=member.node_ids[::-1] if reverse else member.node_ids;points=member.reference_points_m[::-1] if reverse else member.reference_points_m
        members.append(replace(member,node_ids=ids,reference_points_m=tuple(tuple(length*(r@x)) for x in points),local_y_global=tuple(r@np.array(member.local_y_global)),section=section))
    footprints=tuple(replace(g,reference_director=r@g.reference_director,tangent_basis=r@g.tangent_basis) for g in d.physical_footprints)
    return replace(d,shell_nodes_m=nodes,shells=shells,members=tuple(members),physical_footprints=footprints)

@pytest.mark.parametrize("length,force,rotated,reverse",((.25,2.,False,False),(4.,3.,False,False),(1.,1.,True,False),(1.,1.,True,True)))
def test_public_full_model_dimensional_frame_and_member_reversal(length,force,rotated,reverse):
    r=Rotation.from_rotvec((.3,-.2,.4)).as_matrix() if rotated else np.eye(3)
    base=api.create_analysis(definition=definition());new=api.create_analysis(definition=transformed_definition(length,force,r,reverse))
    transformed=load().reshape(-1,6).copy();transformed[:,:3]=force*transformed[:,:3]@r.T;transformed[:,3:]=length*force*transformed[:,3:]@r.T
    try:
        before=base.solve_linear_many((("one",load()),("two",2*load())),selected_dofs=tuple(range(54)),selected_footprints=SELECT,recover_members={})
        after=new.solve_linear_many((("one",transformed.ravel()),("two",2*transformed.ravel())),selected_dofs=tuple(range(54)),selected_footprints=SELECT,recover_members={})
        assert before.status==after.status=="completed",after.failure_reason
        for a,b in zip(before.cases,after.cases):
            expected=np.array(a.selected_displacements).reshape(-1,6);expected[:,:3]=length*expected[:,:3]@r.T;expected[:,3:]=expected[:,3:]@r.T
            np.testing.assert_allclose(b.selected_displacements,expected.ravel(),rtol=1e-8,atol=1e-9*length)
            reactions=np.array(a.selected_affine_reactions).reshape(-1,6);reactions[:,:3]=force*reactions[:,:3]@r.T;reactions[:,3:]=length*force*reactions[:,3:]@r.T
            np.testing.assert_allclose(b.selected_affine_reactions,reactions.ravel(),rtol=1e-8,atol=1e-8*force*max(1.,length))
            wa=a.footprint_wrenches[0];wb=b.footprint_wrenches[0]
            np.testing.assert_allclose(wb.beam_force_N,force*r@wa.beam_force_N,rtol=1e-8,atol=1e-8*force)
            np.testing.assert_allclose(wb.beam_moment_Nm,length*force*r@wa.beam_moment_Nm,rtol=1e-8,atol=1e-8*length*force)
            assert abs(b.internal_strain_energy-length*force*a.internal_strain_energy)<=1e-8*length*force*abs(a.internal_strain_energy)
    finally:base.close();new.close()


def test_public_batch_example():
    result=batch_workflow(10)
    assert result["status"]=="completed" and result["case_count"]==result["solve_count"]==10 and result["factorization_count"]==1
    if os.environ.get("B2_BATCH_EXAMPLE_RECORD"):
        from pathlib import Path
        Path(os.environ["B2_BATCH_EXAMPLE_RECORD"]).write_text(json.dumps(result,indent=2))


def test_invalid_later_load_and_recovery_prevalidate_without_factor(owner):
    factors=owner._handle.factorization_count;solves=owner._handle.solve_count
    with pytest.raises(ValueError):owner.solve_linear_many((("one",load()),("bad",np.zeros(53))),selected_dofs=DOFS)
    assert owner._handle.factorization_count==factors and owner._handle.solve_count==solves
    with pytest.raises(ValueError):owner.solve_linear_many((("one",load()),),selected_dofs=DOFS,recover_members={"unknown":(0.,)})
    assert owner._handle.factorization_count==factors and owner._handle.solve_count==solves



def test_factory_factor_failure_with_offset_propagates(monkeypatch):
    import anyfesolver._stiffened_shell_b2_q4_panel_experimental as module
    d=definition();offset=np.array(d.prescribed_offset);offset[36]=.1
    calls=[]
    def failed(*args,**kwargs):calls.append(1);raise RuntimeError("deliberate factorization failure")
    monkeypatch.setattr(module,"factorize_cached",failed)
    with pytest.raises(RuntimeError,match="deliberate factorization failure"):
        api.create_analysis(definition=replace(d,prescribed_offset=offset))
    assert len(calls)==1 # no owner or offset-only successful case can escape


def test_nonfirst_disjoint_pad_compact_global_provenance(monkeypatch):
    from scipy import sparse
    import anyfesolver._stiffened_shell_b2_q4_panel_experimental as module
    d=definition();nodes={1+i:(float(i),0.,0.) for i in range(4)};nodes.update({5+i:(float(i),1.,0.) for i in range(4)})
    shells=tuple(api.Q4PanelShell(f"s-{i}",(i+1,i+2,i+6,i+5),1000.,.3,.15) for i in range(3))
    members=tuple(api.PhysicalB2Member(f"b-{i}",f"src-{i}",(10+i,11+i),((.5+i,.5,0.),(1.5+i,.5,0.)),(0.,1.,0.),"square",d.members[0].section) for i in range(3))
    pads=(api.B2PhysicalRigidFootprint("first",shells[0].node_ids,10,(0.,0.,1.),((1.,0.),(0.,1.),(0.,0.)),(shells[0].node_ids,)),api.B2PhysicalRigidFootprint("second",shells[2].node_ids,13,(0.,0.,1.),((1.,0.),(0.,1.),(0.,0.)),(shells[2].node_ids,)))
    free=[i for i in range(72) if i not in range(48,54)]
    model=api.B2PhysicalRigidFootprintDefinition(d.profile,nodes,shells,members,pads,sparse.eye(72,format="csr")[:,free],np.zeros(72))
    owner=api.create_analysis(definition=model);selection=api.B2PhysicalRigidFootprintSelection(("second",));applied=np.zeros(72);applied[19]=.7;applied[18]=-1.
    try:
        single=owner.solve_linear(applied,selected_footprints=selection,recover_members={});assert single.status=="completed",single.failure_reason
        original=module.physical_selected_wrenches;calls=[]
        def selected_only(incidences,selected,*args):
            assert selected==("second",);calls.append(selected);return original(incidences,selected,*args)
        monkeypatch.setattr(module,"physical_selected_wrenches",selected_only)
        result=owner.solve_linear_many((("one",applied),),selected_dofs=(18,19),selected_footprints=selection)
        assert result.status=="completed" and len(result.cases)==1
        indices=tuple(row[0] for row in result.constraint_rows)
        assert indices==tuple(range(20,40)) and all(row[1]=="second" for row in result.constraint_rows)
        case=result.cases[0]
        assert len(single.constraint_multipliers)==40 and len(case.constraint_multipliers)==len(case.constraint_gaps)==20
        np.testing.assert_allclose(case.constraint_multipliers,np.array(single.constraint_multipliers)[list(indices)],rtol=1e-8,atol=1e-12)
        np.testing.assert_allclose(case.constraint_gaps,np.array(single.constraint_gaps)[list(indices)],rtol=1e-8,atol=1e-12)
        assert calls==[("second",)] and tuple(w.footprint_id for w in case.footprint_wrenches)==("second",)
        assert {a.shell_node_id for a in case.footprint_wrenches[0].node_actions}==set(shells[2].node_ids)
    finally:owner.close()



def test_typed_body_request_refuses_before_any_rhs(owner,monkeypatch):
    # Metadata-only refusal: no acceleration map/body driver is evaluated.
    typed=api.B2ReferenceBodyLoad((0.,0.,0.),owner.shell_node_ids,owner.beam_node_ids,np.zeros(54),owner._body_model_identity)
    factors=owner._handle.factorization_count;solves=owner._handle.solve_count
    def forbidden(*args):raise AssertionError("typed-body rejection must precede RHS snapshot")
    monkeypatch.setattr(owner,"_body_load_snapshot",forbidden)
    with pytest.raises(ValueError,match="body-load driver not admitted"):
        owner.solve_linear_many((("first",load()),("typed",typed)),selected_dofs=DOFS)
    assert owner._handle.factorization_count==factors and owner._handle.solve_count==solves
