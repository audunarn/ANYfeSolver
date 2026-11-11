"""Loaded-point owner controls against a separate fixed-end boundary solve."""
from dataclasses import replace
import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.transform import Rotation
from anyfesolver import stiffened_shell as api
from anyfesolver._b2_f10_f11_point_attachment import compile_reference_point
from anyfesolver.control import CancellationToken
from test_stiffened_shell_b2_q4_panel_experimental import _panel, _independent_dense
from test_stiffened_shell_b2_body_acceleration import _mass

POLICY = "B2_FIXED_END_LOADED_POINT_TRACE_V1_DEV"


def _definition(*, same_member=False, endpoints=False, affine=False, rotated=False):
    nodes, shells, members, points, free = _panel()
    shells = tuple(replace(s, density_kg_m3=2.) for s in shells)
    members = tuple(replace(m, mass=_mass()) for m in members)
    if same_member:
        points = (replace(points[0], station=.25),
                  replace(points[1], member_id=members[0].member_id, station=.75))
    if endpoints:
        points = tuple(replace(p, station=1.) for p in points)
    t = sparse.eye(54, format="csc")[:, free]
    u0 = np.zeros(54)
    if affine:
        u0[0], u0[6*6+1] = 1e-4, -2e-4
    if rotated:
        r = Rotation.from_rotvec((.25,-.3,.4)).as_matrix()
        nodes = {i: tuple(r@np.asarray(x)) for i,x in nodes.items()}
        shells = tuple(replace(s, reference_normal=tuple(r@np.asarray(s.reference_normal))) for s in shells)
        members = tuple(replace(m,
            reference_points_m=tuple(tuple(r@np.asarray(x)) for x in m.reference_points_m),
            local_y_global=tuple(r@np.asarray(m.local_y_global))) for m in members)
        transform = sparse.kron(sparse.eye(18), sparse.csr_matrix(r), format="csr")
        t = transform@t
        u0 = transform@u0
    return api.B2Q4PanelDefinition(api.B2_E4_PL_FAST_LINEAR_V1,nodes,shells,members,points,t,u0,
                                 point_trace_policy=POLICY)


def _fixed_end_pose(member, stations, uniform, actions, *, nodal=None, strains=False):
    """Independent equilibrium and six initial-resultant boundary unknowns.

    No candidate load-bubble maps, element trace polynomials or point Green
    matrices. Integrate a piecewise quadratic resultant and its compliance,
    then impose all six end poses through a separate 6x6 boundary solve.
    """
    length = member.length_m
    invs = np.linalg.solve(np.asarray(member.section.stiffness), np.eye(6))
    ex = np.array((1.,0.,0.))
    qx, qw = np.polynomial.legendre.leggauss(6)
    actions = tuple((float(a)*length,np.asarray(w)) for a,w in actions)
    uniform = np.asarray(uniform)
    nodal = np.zeros(12) if nodal is None else np.asarray(nodal)

    def section(x, initial):
        n = initial[:3]-x*uniform[:3]
        m = initial[3:]-x*np.cross(ex,initial[:3])-x*uniform[3:]+.5*x*x*np.cross(ex,uniform[:3])
        for at,wrench in actions:
            if x > at:
                n = n-wrench[:3]
                m = m+(x-at)*np.cross(ex,wrench[:3])-wrench[3:]
        return invs@np.r_[n,m]

    def pose(x, initial):
        out = np.r_[nodal[:3]+x*np.cross(nodal[3:6],ex),nodal[3:6]].copy()
        splits = sorted(set((0.,x)+tuple(a for a,_ in actions if 0.<a<x)))
        for a,b in zip(splits[:-1],splits[1:]):
            for xi,wi in zip(qx,qw):
                at = a+(b-a)*(xi+1.)/2.
                e = section(at,initial)
                out += (b-a)*wi/2.*np.r_[e[:3]+(x-at)*np.cross(e[3:],ex),e[3:]]
        return out
    end = pose(length,np.zeros(6))
    columns = np.column_stack([pose(length,np.eye(6)[i])-end for i in range(6)])
    initial = np.linalg.solve(columns,nodal[6:]-end)
    evaluate = section if strains else pose
    return np.array([evaluate(float(s)*length,initial) for s in stations])


def _oracle(definition, applied, acceleration=None):
    nodes,shells,members,points = (definition.shell_nodes_m,definition.shells,
                                  definition.members,definition.points)
    k,c = _independent_dense(nodes,shells,members,points)
    t = definition.transformation.toarray()
    u0 = np.asarray(definition.prescribed_offset)
    h = np.zeros((len(c),len(c))); a = np.zeros(len(c))
    maps = {}
    for i,p in enumerate(points):
        m = next(m for m in members if m.member_id==p.member_id)
        shell = next(s for s in shells if s.shell_id==p.shell_id)
        law = compile_reference_point(m,np.array([nodes[n] for n in shell.node_ids]),
                                      station=p.station,offset_local_m=p.offset_local_m)
        b = law.beam_constraint_differential@np.kron(np.eye(2),m.local_frame_rows().T)
        maps[i] = (m,b)
        if acceleration is not None:
            uniform = m.mass.matrix_at_reference_axis()[:, :3]@m.local_frame_rows()@acceleration
            axis = _fixed_end_pose(m,(p.station,),uniform,())[0]
            a[6*i:6*i+6] = b@np.r_[axis[:3]+np.cross(axis[3:],p.offset_local_m),axis[3:]]
        for j,other in enumerate(points):
            if other.member_id != p.member_id: continue
            other_m,other_b = maps[j] if j in maps else (None,None)
            if other_b is None:
                ss = next(s for s in shells if s.shell_id==other.shell_id)
                ll = compile_reference_point(m,np.array([nodes[n] for n in ss.node_ids]),
                                             station=other.station,offset_local_m=other.offset_local_m)
                other_b = ll.beam_constraint_differential@np.kron(np.eye(2),m.local_frame_rows().T)
            g = []
            for w in np.eye(6):
                axis_wrench = np.r_[w[:3],w[3:]+np.cross(other.offset_local_m,w[:3])]
                if other.station in (0.,1.):
                    axis = np.zeros(6)
                else:
                    axis = _fixed_end_pose(m,(p.station,),np.zeros(6),((other.station,axis_wrench),))[0]
                g.append(np.r_[axis[:3]+np.cross(axis[3:],p.offset_local_m),axis[3:]])
            h[6*i:6*i+6,6*j:6*j+6] = b@np.array(g).T@other_b.T
    operator = np.block([[t.T@k@t,(c@t).T],[c@t,-h]])
    rhs = np.r_[t.T@(applied-k@u0),-c@u0-a]
    answer = np.linalg.solve(operator,rhs)
    return k,c,h,a,t@answer[:t.shape[1]]+u0,answer[t.shape[1]:],maps


@pytest.mark.parametrize("same_member,affine,rotated",((False,False,False),(True,False,False),
                                                      (True,True,True)))
@pytest.mark.parametrize("body",(False,True))
def test_coupled_loaded_pose_matches_independent_boundary_kkt(same_member,affine,rotated,body):
    definition = _definition(same_member=same_member,affine=affine,rotated=rotated)
    analysis = api.create_analysis(definition=definition)
    try:
        acceleration = np.array((.3,-.4,-.7)) if body else None
        load = analysis.body_acceleration_load(acceleration) if body else np.zeros(54)
        if not body:
            load[6*8+1],load[6*5+2] = .08,-.16
        applied = load.load if body else load
        k,c,h,a,expected,lam,maps = _oracle(definition,applied,acceleration)
        result = analysis.solve_linear(load,recover_members={m.member_id:(0.,.25,.5,.75,1.) for m in definition.members})
        assert result.status=="completed", result.failure_reason
        np.testing.assert_allclose(result.displacements,expected,rtol=1e-10,atol=1e-12)
        actual_lam = np.concatenate([v for _,v in result.point_multipliers])
        np.testing.assert_allclose(actual_lam,lam,rtol=1e-10,atol=2e-11)
        np.testing.assert_allclose(c@result.displacements+a-h@actual_lam,0.,atol=2e-12)
        np.testing.assert_allclose(result.affine_reactions,k@expected+c.T@lam-applied,rtol=1e-10,atol=2e-11)
        if same_member:
            assert np.max(abs(h[:6,6:])) > 0.
        poses = {v.member_id:v for v in result.loaded_member_poses}
        shell_ids = sorted(definition.shell_nodes_m)
        beam_ids = sorted({n for m in definition.members for n in m.node_ids})
        energy = .5*np.asarray(result.displacements)@k@np.asarray(result.displacements)
        bubble_energy = body_bubble_work = beam_energy = homogeneous_beam_energy = 0.
        for member in definition.members:
            m = member
            station = poses[m.member_id].stations
            slots = [6*(len(shell_ids)+beam_ids.index(n))+d for n in m.node_ids for d in range(6)]
            local = m.local_dof_map()@np.asarray(result.displacements)[slots]
            uniform = (m.mass.matrix_at_reference_axis()[:, :3]@m.local_frame_rows()@acceleration
                       if body else np.zeros(6))
            actions = []
            for i,p in enumerate(definition.points):
                if p.member_id!=m.member_id or p.station in (0.,1.): continue
                w = -maps[i][1].T@actual_lam[6*i:6*i+6]
                actions.append((p.station,np.r_[w[:3],w[3:]+np.cross(p.offset_local_m,w[:3])]))
            bubble = _fixed_end_pose(m,station,uniform,actions)
            homogeneous = m.local_operator().trace(local,stations=station).pose
            np.testing.assert_allclose(poses[m.member_id].poses_local,homogeneous+bubble,rtol=1e-10,atol=2e-12)
            # Independent fixed-end load work is twice bubble strain energy.
            xi,wi = np.polynomial.legendre.leggauss(6)
            splits = sorted(set((0.,1.)+tuple(s for s,_ in actions)))
            for left,right in zip(splits[:-1],splits[1:]):
                ss = left+(right-left)*(xi+1.)/2.
                bb = _fixed_end_pose(m,ss,uniform,actions)
                measure=m.length_m*(right-left)/2.*wi
                body_work=measure@(bb@uniform)
                body_bubble_work += body_work
                bubble_energy += .5*body_work
                e=_fixed_end_pose(m,ss,uniform,actions,nodal=local,strains=True)
                eh=_fixed_end_pose(m,ss,np.zeros(6),(),nodal=local,strains=True)
                beam_energy += .5*measure@np.einsum("ni,ij,nj->n",e,np.asarray(m.section.stiffness),e)
                homogeneous_beam_energy += .5*measure@np.einsum("ni,ij,nj->n",eh,np.asarray(m.section.stiffness),eh)
            for at,w in actions:
                bubble_energy += .5*w@_fixed_end_pose(m,(at,),uniform,actions)[0]
        assert result.internal_strain_energy==pytest.approx(energy,rel=1e-11)
        assert result.loaded_strain_energy==pytest.approx(energy+bubble_energy,rel=1e-10,abs=2e-12)
        assert result.loaded_strain_energy==pytest.approx(
            energy-homogeneous_beam_energy+beam_energy,rel=1e-10,abs=2e-12)
        # Full external plus prescribed-support work, not just bubble work.
        total_work=np.asarray(applied)@np.asarray(result.displacements)
        total_work+=np.asarray(result.affine_reactions)@np.asarray(result.displacements)+body_bubble_work
        assert 2.*result.loaded_strain_energy==pytest.approx(total_work,rel=1e-10,abs=2e-12)
        assert len(result.equilibrated_sections)==len(definition.members)
    finally:
        analysis.close()


def test_default_and_raw_nodal_semantics_and_binding_remain_distinct():
    d = _definition()
    legacy = api.create_analysis(definition=replace(d,point_trace_policy=None))
    new = api.create_analysis(definition=d)
    try:
        oldbody=legacy.body_acceleration_load((.3,-.4,-.7))
        old1=legacy.solve_linear(oldbody);old2=legacy.solve_linear(oldbody.load)
        assert old1.displacements==old2.displacements
        with pytest.raises(ValueError):
            new.solve_linear(oldbody)
        body=new.body_acceleration_load((.3,-.4,-.7))
        typed=new.solve_linear(body);raw=new.solve_linear(body.load)
        assert typed.status==raw.status=="completed"
        assert not np.allclose(typed.displacements,raw.displacements,rtol=1e-8,atol=1e-12)
        assert body.model_identity!=oldbody.model_identity
    finally:
        new.close();legacy.close()


@pytest.mark.parametrize("count",(1,10,100))
def test_loaded_batch_selection_and_factor_reuse(count):
    a=api.create_analysis(definition=_definition(same_member=True))
    try:
        load=a.body_acceleration_load((.3,-.4,-.7))
        result=a.solve_linear_many(tuple((str(i),load) for i in range(count)),
            selected_dofs=(49,50),selected_points=("joint-left",),
            recover_members={"beam-left":(.25,.75)})
        assert result.status=="completed",result.failure_reason
        assert len(result.cases)==count and result.factorization_count==1
        assert all(len(c.loaded_member_poses)==1 and len(c.equilibrated_sections)==1 for c in result.cases)
        assert all(c.loaded_member_poses[0].stations==(.25,.75) for c in result.cases)
        assert result.cases[-1].loaded_strain_energy==result.cases[0].loaded_strain_energy
        assert len(result.member_resultant_envelopes)==12
        assert all(item.member_id=="beam-left" and
                   item.minimum_case_id==item.maximum_case_id=="0"
                   for item in result.member_resultant_envelopes)
    finally:a.close()


def test_loaded_batch_cancellation_returns_complete_case():
    a=api.create_analysis(definition=_definition());token=CancellationToken()
    try:
        load=a.body_acceleration_load((.3,-.4,-.7))
        result=a.solve_linear_many((("one",load),("two",load)),selected_dofs=(49,),
            recover_members={"beam-left":(.5,)},cancellation_token=token,on_case=lambda _:token.cancel())
        assert result.status=="cancelled" and len(result.cases)==1
        assert len(result.cases[0].loaded_member_poses)==1
        assert len(result.member_resultant_envelopes)==6
        assert all(item.minimum_case_id==item.maximum_case_id=="one"
                   for item in result.member_resultant_envelopes)
    finally:a.close()


def test_endpoint_green_is_zero_and_modal_request_rejected():
    d=_definition(endpoints=True);a=api.create_analysis(definition=d)
    legacy=api.create_analysis(definition=replace(d,point_trace_policy=None))
    try:
        # Raw nodal loading has no fixed-end interior action at endpoint joints.
        load=np.zeros(54);load[49]=.08;load[32]=-.16
        actual=a.solve_linear(load);old=legacy.solve_linear(load)
        assert actual.status==old.status
        if actual.status=="completed":
            np.testing.assert_allclose(actual.displacements,old.displacements,rtol=1e-11,atol=1e-12)
        else:
            assert actual.failure_reason==old.failure_reason
        with pytest.raises(ValueError,match="loaded|policy|spectral|mode"):
            a.modes(api.B2ReferenceModesRequest(count=2))
    finally:a.close();legacy.close()




def test_equivalent_m_n_and_mm_kn_loaded_owner():
    original=_definition(same_member=True,affine=True)
    length_scale, force_scale = 1000., .001
    d=np.diag((1.,1.,1.,length_scale,length_scale,length_scale))
    members=tuple(replace(m,
        reference_points_m=tuple(tuple(length_scale*np.asarray(x)) for x in m.reference_points_m),
        section=api.LinearSection.generalized(force_scale*np.asarray(m.section.stiffness)*np.outer(d.diagonal(),d.diagonal())),
        mass=api.MassMetadata(m.mass.source_id,
            tuple(length_scale*np.asarray(m.mass.origin_local_m)),
            force_scale/length_scale**2*np.asarray(m.mass.matrix_per_m)*np.outer(d.diagonal(),d.diagonal())))
        for m in original.members)
    shell_nodes={i:tuple(length_scale*np.asarray(x)) for i,x in original.shell_nodes_m.items()}
    shells=tuple(replace(s,E=s.E*force_scale/length_scale**2,
        thickness_m=s.thickness_m*length_scale,
        density_kg_m3=s.density_kg_m3*force_scale/length_scale**4) for s in original.shells)
    pose_scale=np.tile((length_scale,)*3+(1.,)*3,9)
    converted=replace(original,shell_nodes_m=shell_nodes,shells=shells,members=members,
        points=tuple(replace(p,offset_local_m=tuple(length_scale*np.asarray(p.offset_local_m)))
                     for p in original.points),
        transformation=sparse.diags(pose_scale)@original.transformation,
        prescribed_offset=pose_scale*original.prescribed_offset)
    a=api.create_analysis(definition=original);b=api.create_analysis(definition=converted)
    try:
        result=a.solve_linear(a.body_acceleration_load((.3,-.4,-.7)))
        other=b.solve_linear(b.body_acceleration_load(length_scale*np.array((.3,-.4,-.7))))
        assert result.status==other.status=="completed",(result.failure_reason,other.failure_reason)
        np.testing.assert_allclose(np.asarray(other.displacements)/pose_scale,result.displacements,
                                   rtol=1e-10,atol=2e-12)
        wrench_scale=np.tile((force_scale,)*3+(force_scale*length_scale,)*3,9)
        np.testing.assert_allclose(np.asarray(other.affine_reactions)/wrench_scale,result.affine_reactions,
                                   rtol=1e-10,atol=2e-10)
        assert other.loaded_strain_energy/(length_scale*force_scale)==pytest.approx(
            result.loaded_strain_energy,rel=1e-10)
    finally:a.close();b.close()


def test_reject_unknown_trace_policy_before_solve():
    with pytest.raises(ValueError,match="policy|trace"):
        api.create_analysis(definition=replace(_definition(),point_trace_policy="unknown"))



def test_physical_green_allows_more_point_rows_than_free_nodal_coordinates():
    d=_definition(same_member=True)
    d=replace(d,transformation=sparse.eye(54,format="csc")[:,(32,)],
              prescribed_offset=np.zeros(54))
    load=np.zeros(54);load[32]=-.16
    a=api.create_analysis(definition=d)
    try:
        _,c,h,_,expected,lam,_=_oracle(d,load)
        assert np.linalg.matrix_rank(c@d.transformation.toarray())==1
        assert np.linalg.matrix_rank(h)==12
        result=a.solve_linear(load)
        assert result.status=="completed",result.failure_reason
        np.testing.assert_allclose(result.displacements,expected,rtol=1e-10,atol=2e-12)
        np.testing.assert_allclose(np.concatenate([v for _,v in result.point_multipliers]),
                                   lam,rtol=1e-10,atol=2e-11)
    finally:a.close()
    with pytest.raises(ValueError,match="affine"):
        api.create_analysis(definition=replace(d,point_trace_policy=None))



def _steel_definition():
    # Public physical SI example, not a rescaled easy-section fixture.
    from pathlib import Path
    import runpy
    build=runpy.run_path(str(Path(__file__).resolve().parents[1]/
        "examples/stiffened_shell_b2_body_acceleration.py"))["definition"]
    d=build("q4")
    return api.B2Q4PanelDefinition(d.profile,d.shell_nodes_m,d.shells,d.members,d.points,
        d.transformation,d.prescribed_offset,point_trace_policy=POLICY)


def test_physical_steel_body_work_requires_bounded_resolution_certificate(monkeypatch):
    d=_steel_definition();a=api.create_analysis(definition=d)
    old=api.create_analysis(definition=replace(d,point_trace_policy=None))
    try:
        calls=[];certificate=a._explicit_load_resolved
        def checked(*args,**kwargs):
            value=certificate(*args,**kwargs);calls.append(value);return value
        monkeypatch.setattr(a,"_explicit_load_resolved",checked)
        body=a.body_acceleration_load((0.,0.,-9.81))
        result=a.solve_linear(body)
        assert result.status=="completed",result.failure_reason
        assert calls==[True] and result.loaded_strain_energy>0.
        # Current legacy numerical guard and identity retain their meaning.
        legacy=old.solve_linear(old.body_acceleration_load((0.,0.,-9.81)))
        assert legacy.status=="completed",legacy.failure_reason
        assert not hasattr(legacy,"loaded_member_poses")
        monkeypatch.setattr(a,"_explicit_load_resolved",lambda *args,**kwargs:False)
        unresolved=a.solve_linear(body)
        assert unresolved.status=="failed" and unresolved.displacements==()
        assert unresolved.failure_reason=="panel mixed-load resolution insufficient"
    finally:a.close();old.close()


def test_resolution_certificate_rejects_lost_applied_work():
    a=api.create_analysis(definition=_steel_definition())
    try:
        body=a.body_acceleration_load((0.,0.,-9.81))
        applied=np.asarray(body.load);bg=np.zeros(a._C.shape[0])
        for member in a._loaded_members.values():
            values,_=member.body_constraint(body.acceleration_m_s2)
            ix=[6*i+c for i in member.point_indices for c in range(6)]
            bg[ix]=values
        rhs=np.r_[a._project(applied-a._ku0),-a._cu0-bg]
        # An answer that discarded the whole nonzero load cannot be certified.
        assert not a._explicit_load_resolved(np.zeros_like(rhs),rhs,abs(a._project(applied)))
    finally:a.close()
