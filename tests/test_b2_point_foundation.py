"""Prospective finite-support F10/F11 checks; no nodal/line qualification."""
from dataclasses import replace
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.spatial.transform import Rotation
from scipy.sparse.linalg import splu

from anyfesolver._b2_f10_f11_point_attachment import compile_reference_point
from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._stiffened_shell_b2_q4_panel_experimental import _scaled_point_kkt
from test_b2_f10_f11_point_attachment import specimen, VERTICES
from test_b2_f10_f11_s3_point_attachment import _specimen, TRIANGLE
from test_b2_physical_linear_trace import _bvp_pose


FORMS = ("base", "shift", "rotate", "units", "winding")
STEPS = (1.e-4, 3.e-5, 1.e-5)
REFERENCE_LENGTH_M = 2.
REFERENCE_FORCE_N = 1.
ABSOLUTE_NORMALIZED_FLOOR = 1.e-12


def skew(v):
    x,y,z=v
    return np.array(((0.,-z,y),(z,0.,-x),(-y,x,0.)))


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),ABSOLUTE_NORMALIZED_FLOOR))


@lru_cache(None)
def base(kind,coupled):
    if kind=="q4":
        member,_,shell_k,_=specimen();vertices=VERTICES.copy()
    else:
        member,_,vertices,shell_k,_=_specimen()
    section=member.section
    if coupled:
        s=np.asarray(section.stiffness).copy()
        for i,j,value in ((0,4,.08),(1,5,-.06),(2,3,.04)):
            s[i,j]=s[j,i]=value*np.sqrt(s[i,i]*s[j,j])
        section=LinearSection.generalized(s)
        member=replace(member,section=section)
    d=member.local_dof_map()
    beam_k=d.T@member.local_operator().evaluate(np.zeros(12)).tangent@d
    return member,vertices,shell_k,beam_k


@lru_cache(None)
def bvp_port(kind,coupled,sign):
    member,_,_,_=base(kind,coupled)
    offset=sign*np.array((0.,.08,-.03))
    return np.column_stack([_bvp_pose(member.section,np.eye(12)[i],.4,offset)
                            for i in range(12)])


def independent_shell_port(vertices):
    radii=vertices-vertices.mean(axis=0)
    g=sum((r@r*np.eye(3)-np.outer(r,r) for r in radii),np.zeros((3,3)))
    p=np.zeros((6,6*len(vertices)))
    for i,r in enumerate(radii):
        p[:3,6*i:6*i+3]=np.eye(3)/len(vertices)
        p[3:,6*i:6*i+3]=np.linalg.solve(g,skew(r))
    return p


@pytest.mark.parametrize("kind", ("q4","s3"))
@pytest.mark.parametrize("sign", (-1,1))
@pytest.mark.parametrize("coupled", (False,True))
@pytest.mark.parametrize("form", FORMS)
def test_point_foundation(kind,sign,coupled,form):
    member,vertices,shell_k,beam_k=base(kind,coupled)
    n_shell=6*len(vertices);n=n_shell+12
    physical_points=np.asarray(member.reference_points_m).copy()
    physical_points[:,2]*=sign
    rotation=(Rotation.from_rotvec((.37,-.23,.19)).as_matrix()
              if form=="rotate" else np.eye(3))
    shift=np.array((3.,-2.,.7)) if form in ("shift","rotate") else np.zeros(3)
    length_units,force_units=(1000.,.001) if form=="units" else (1.,1.)
    work_units=length_units*force_units
    order=(np.array((0,3,2,1)) if kind=="q4" else np.array((0,2,1))) if form=="winding" else np.arange(len(vertices))
    slots=np.r_[np.concatenate([np.arange(6*i,6*i+6) for i in order]),np.arange(n_shell,n)]
    perm=np.eye(n)[slots]
    rotate=linalg.block_diag(*([rotation]*(n//3)))
    units=np.tile((length_units,)*3+(1.,)*3,n//6)
    transform=perm@rotate@np.diag(units)
    shell_vertices=length_units*(vertices[order]@rotation.T+shift)
    points=length_units*(physical_points@rotation.T+shift)
    section=(LinearSection.generalized(np.asarray(member.section.stiffness)*
             np.outer((1.,)*3+(length_units,)*3,(1.,)*3+(length_units,)*3)*force_units)
             if form=="units" else member.section)
    changed=replace(member,reference_points_m=tuple(map(tuple,points)),
                    local_y_global=tuple(rotation[:,1]),section=section)
    offset=length_units*sign*np.array((0.,.08,-.03))
    law=compile_reference_point(changed,shell_vertices,station=.4,offset_local_m=offset)
    ps=independent_shell_port(shell_vertices)
    pose_scale=np.diag((length_units,)*3+(1.,)*3)
    pose_rotation=linalg.block_diag(rotation,rotation)
    pb=pose_rotation@pose_scale@bvp_port(kind,coupled,sign)@np.linalg.inv(
        rotate[n_shell:,n_shell:]@np.diag(units[n_shell:]))
    origin=shell_vertices.mean(axis=0)
    point=points[0]+.4*(points[1]-points[0])+rotation@offset
    tangent=shell_vertices[1]-shell_vertices[0];tangent/=np.linalg.norm(tangent)
    normal=np.cross(shell_vertices[1]-shell_vertices[0],shell_vertices[-1]-shell_vertices[0]);normal/=np.linalg.norm(normal)
    rs=np.column_stack((tangent,np.cross(normal,tangent),normal));rb=rotation
    js=np.block([[-rs.T,rs.T@skew(point-origin)],[np.zeros((3,3)),-rb.T]])
    jb=linalg.block_diag(rs.T,rb.T)
    expected_c=np.column_stack((js@ps,jb@pb))
    row_units=np.r_[(length_units,)*3,(1.,)*3]
    qscale=np.tile((REFERENCE_LENGTH_M,)*3+(1.,)*3,n//6)
    pose_characteristic=np.r_[(REFERENCE_LENGTH_M,)*3,(1.,)*3]
    pose_inverse=np.linalg.inv(pose_rotation@pose_scale)
    normalized=lambda c:(c@transform)*qscale[None,:]/(row_units*pose_characteristic)[:,None]
    shell_normalized=lambda p:(pose_inverse@p@transform[:n_shell,:n_shell])*qscale[None,:n_shell]/pose_characteristic[:,None]
    beam_normalized=lambda p:(pose_inverse@p@transform[n_shell:,n_shell:])*qscale[None,n_shell:]/pose_characteristic[:,None]
    metrics={"constraint_relative":relative(normalized(law.constraint),normalized(expected_c)),
             "shell_port_relative":relative(shell_normalized(law.shell_port),shell_normalized(ps)),
             "beam_port_relative":relative(beam_normalized(law.beam_port),beam_normalized(pb))}
    direction=transform@(np.sin(np.arange(n)+.3)*.01)
    multiplier=work_units*np.array((.17,-.23,.31,.07,-.05,.11))/row_units
    force=np.r_[*law.forces(multiplier)]
    master_wrench=js.T@multiplier
    radii=shell_vertices-origin
    g=sum((r@r*np.eye(3)-np.outer(r,r) for r in radii),np.zeros((3,3)))
    couple=np.linalg.solve(g,master_wrench[3:])
    shell_action=np.zeros((len(vertices),6))
    shell_action[:,:3]=master_wrench[:3]/len(vertices)+np.cross(couple,radii)
    action_normalized=lambda f,section:transform[section,section].T@f*qscale[section]/(work_units*REFERENCE_FORCE_N*REFERENCE_LENGTH_M)
    metrics["shell_action_relative"]=relative(action_normalized(force[:n_shell],slice(0,n_shell)),action_normalized(shell_action.ravel(),slice(0,n_shell)))
    metrics["beam_action_relative"]=relative(action_normalized(force[n_shell:],slice(n_shell,n)),action_normalized(pb.T@jb.T@multiplier,slice(n_shell,n)))
    work_expected=float(multiplier@(expected_c@direction))
    metrics["work_relative"]=abs(force@direction-work_expected)/max(abs(work_expected),ABSOLUTE_NORMALIZED_FLOOR*work_units*REFERENCE_FORCE_N*REFERENCE_LENGTH_M)
    xyz=np.vstack((shell_vertices,points))/length_units
    actions=force.reshape(-1,6).copy();actions[:,:3]/=force_units;actions[:,3:]/=work_units
    total_f=actions[:,:3].sum(axis=0)
    total_m=(actions[:,3:]+np.cross(xyz,actions[:,:3])).sum(axis=0)
    metrics["wrench_scaled"]=float(max(np.linalg.norm(total_f),np.linalg.norm(total_m)/REFERENCE_LENGTH_M)/REFERENCE_FORCE_N)
    rigid_errors=[]
    for axis in np.eye(6):
        translation=axis[:3]*(.01*REFERENCE_LENGTH_M*length_units)
        spin=axis[3:]*.01
        rigid=np.array([np.r_[translation+np.cross(spin,x),spin]
                        for x in np.vstack((shell_vertices,points))]).ravel()
        rigid_errors.append(float(np.max(np.abs(law.constraint@rigid/
            (row_units*pose_characteristic)))))
    metrics["rigid_scaled"]=max(rigid_errors)
    poses=np.array((ps@direction[:n_shell],pb@direction[n_shell:]))
    a0=rs.T@rb;e0=rs.T@(point-origin)

    def objective_gap(positions,frames):
        return np.r_[frames[0].T@(positions[1]-positions[0])-e0,
            Rotation.from_matrix(a0.T@frames[0].T@frames[1]).as_rotvec()]

    metrics["reference_scaled"]=float(np.max(np.abs(objective_gap(
        np.array((origin,point)),(rs,rb))/(row_units*pose_characteristic))))
    predicted=(law.constraint@direction)/(row_units*pose_characteristic)
    derivative_errors=[]
    for step in STEPS:
        gaps=[]
        for sense in (1.,-1.):
            frames=np.array([Rotation.from_rotvec(sense*step*p[3:]).as_matrix()@r
                             for p,r in zip(poses,(rs,rb))])
            gaps.append(objective_gap(np.array((origin,point))+sense*step*poses[:,:3],frames))
        observed=(gaps[0]-gaps[1])/(2*step*row_units*pose_characteristic)
        derivative_errors.append(relative(observed,predicted))
    metrics["actual_update_plateau"]=derivative_errors
    k_base=linalg.block_diag(shell_k,beam_k)
    inverse=np.linalg.inv(transform)
    k=work_units*inverse.T@k_base@inverse
    fixed=set(range(6))|set(range(n_shell,n_shell+6))
    fixed.update(6*i+5 for i in range(1,len(vertices)))
    free=np.array([i for i in range(n) if i not in fixed])
    t0=np.eye(n)[:,free];t0[free[0],1]+=.25
    t=transform@t0
    prescribed=np.cos(np.arange(n)+.2)*2.e-4
    applied=np.sin(np.arange(n)+.7)*.1
    prescribed[5:n_shell:6]=0.;applied[5:n_shell:6]=0.
    u0=transform@prescribed
    load=work_units*inverse.T@applied
    kr=t.T@k@t;cf=law.constraint@t
    rhs=np.r_[t.T@(load-k@u0),-law.constraint@u0]
    scaled,scale=_scaled_point_kkt(sparse.csr_matrix(kr),sparse.csr_matrix(cf))
    answer=scale*splu(scaled).solve(scale*rhs)
    q,lam=answer[:len(free)],answer[len(free):]
    co=expected_c@t
    particular=np.linalg.lstsq(co,-expected_c@u0,rcond=None)[0]
    z=linalg.null_space(co)
    oracle=particular+z@np.linalg.solve(z.T@kr@z,z.T@(rhs[:len(free)]-kr@particular))
    dual=np.linalg.lstsq(co.T,rhs[:len(free)]-kr@oracle,rcond=None)[0]
    u=t@q+u0;uo=t@oracle+u0
    reaction=k@u+law.constraint.T@lam-load
    ro=k@uo+expected_c.T@dual-load
    multiplier_characteristic=REFERENCE_FORCE_N*np.r_[(1.,)*3,(REFERENCE_LENGTH_M,)*3]
    metrics.update(displacement_relative=relative(inverse@u/qscale,inverse@uo/qscale),
        multiplier_relative=relative(lam*row_units/(work_units*multiplier_characteristic),dual*row_units/(work_units*multiplier_characteristic)),
        reaction_relative=relative(transform.T@reaction*qscale/(work_units*REFERENCE_FORCE_N*REFERENCE_LENGTH_M),transform.T@ro*qscale/(work_units*REFERENCE_FORCE_N*REFERENCE_LENGTH_M)),
        equilibrium_scaled=float(np.linalg.norm((t.T@reaction)*qscale[free])/np.linalg.norm((t.T@load)*qscale[free])),
        gap_scaled=float(np.max(np.abs(law.constraint@u/(row_units*pose_characteristic)))),
        support_work_scaled=float(abs(u@k@u-u@load-u0@reaction)/max(abs(u@k@u),abs(u@load)+abs(u0@reaction),ABSOLUTE_NORMALIZED_FLOOR*work_units*REFERENCE_FORCE_N*REFERENCE_LENGTH_M)))
    report={"shell":kind,"offset_sign":sign,"coupled":coupled,"form":form,
        "law_id":law.law_id,"units":{"length":length_units,"force":force_units},
        "scales":{"length_m":REFERENCE_LENGTH_M,"force_n":REFERENCE_FORCE_N,
                  "moment_nm":REFERENCE_LENGTH_M*REFERENCE_FORCE_N,"normalized_floor":ABSOLUTE_NORMALIZED_FLOOR},
        "steps":STEPS,"metrics":metrics,"rank":int(np.linalg.matrix_rank(cf)),
        "bending_rotation_norm":float(np.linalg.norm((inverse@(u-u0)).reshape(-1,6)[1:len(vertices),3:5]))}
    output=os.environ.get("B2_POINT_FOUNDATION_REPORTS")
    if output:
        target=Path(output);target.mkdir(exist_ok=True)
        (target/f'{kind}-{sign}-{int(coupled)}-{form}.json').write_text(json.dumps(report,indent=2)+'\n')
    for key in ("constraint_relative","shell_port_relative","beam_port_relative",
                "shell_action_relative","beam_action_relative",
                "displacement_relative","multiplier_relative","reaction_relative"):
        assert metrics[key]<=1.e-10,(key,report)
    for key in ("work_relative","wrench_scaled","equilibrium_scaled","support_work_scaled"):
        assert metrics[key]<=1.e-8,(key,report)
    assert metrics["reference_scaled"]<=1.e-9 and metrics["gap_scaled"]<=1.e-9 and metrics["rigid_scaled"]<=1.e-9,report
    assert max(derivative_errors)<=1.e-6,report
    assert report["rank"]==6 and report["bending_rotation_norm"]>1.e-8,report


NEGATIVE_CASES = ("nonfinite_vertices","collinear","malformed","zero_edge",
                  "negative_station","large_station","nonfinite_station",
                  "nonfinite_offset","bad_offset","unrepresentable_offset")


def write_negative(name,exception,rejected):
    output=os.environ.get("B2_POINT_FOUNDATION_REPORTS")
    if output:
        target=Path(output);target.mkdir(exist_ok=True)
        (target/f'negative-{name}.json').write_text(json.dumps({
            "case":name,"rejected":rejected,"exception":type(exception).__name__,
            "message":str(exception)},indent=2)+'\n')


def require_rejection(name,factory):
    try:
        answer=factory()
    except ValueError as exception:
        write_negative(name,exception,True)
    except Exception as exception:
        write_negative(name,exception,False)
        raise
    else:
        if hasattr(answer,"close"):answer.close()
        write_negative(name,None,False)
        raise AssertionError('invalid input accepted: '+name)


@pytest.mark.parametrize("kind", ("q4","s3"))
@pytest.mark.parametrize("defect", NEGATIVE_CASES)
def test_invalid_point_foundation(kind,defect):
    member,vertices,_,_=base(kind,False)
    vertices=vertices.copy();station=.4;offset=(0.,.08,-.03)
    if defect=="nonfinite_vertices":vertices[0,0]=np.nan
    elif defect=="collinear":vertices[:,1:]=0.
    elif defect=="malformed":vertices=vertices[:,:2]
    elif defect=="zero_edge":vertices[1]=vertices[0]
    elif defect=="negative_station":station=-.1
    elif defect=="large_station":station=1.1
    elif defect=="nonfinite_station":station=np.inf
    elif defect=="nonfinite_offset":offset=(0.,np.nan,0.)
    elif defect=="bad_offset":offset=(0.,0.)
    else:
        rotation=Rotation.from_rotvec((.37,-.23,.19)).as_matrix()
        member=replace(member,reference_points_m=tuple(map(tuple,
            np.asarray(member.reference_points_m)@rotation.T)),local_y_global=tuple(rotation[:,1]))
        vertices=vertices@rotation.T
        offset=(np.finfo(float).max,)*3
    require_rejection(f'{kind}-{defect}',lambda:compile_reference_point(
        member,vertices,station=station,offset_local_m=offset))


def test_nonplanar_q4_support_rejected():
    member,vertices,_,_=base("q4",False)
    vertices=vertices.copy();vertices[2,2]=.2
    require_rejection('q4-nonplanar',lambda:compile_reference_point(member,vertices,station=.4))


@pytest.mark.parametrize("defect", ("dimension","nonfinite_t","nonfinite_u0","dependent_rows"))
def test_invalid_affine_point_input(defect):
    from anyfesolver._stiffened_shell_b2_q4_point_experimental import create_experimental_b2_q4_point_analysis
    member,vertices,_,_=base("q4",False)
    t=sparse.eye(36,format="csr")[:,:12];u0=np.zeros(36)
    if defect=="dimension":t=t[:35]
    elif defect=="nonfinite_t":t.data[0]=np.inf
    elif defect=="nonfinite_u0":u0[0]=np.nan
    else:t=sparse.csr_matrix((36,12))
    require_rejection('affine-'+defect,lambda:create_experimental_b2_q4_point_analysis(member,vertices,
            shell_E=1000.,shell_nu=.3,shell_thickness_m=.15,station=.4,
            transformation=t,prescribed_offset=u0))
