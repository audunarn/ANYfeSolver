"""Independent weak-P0 C3D20 development oracle; no external execution.

Same specimen/load as the immutable released reference. Twenty arclength rows
replace its twenty-five strong rows; full body displacement spaces are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium, _scaled_constraint_gap
from scripts import r8_bl5_released_director_reference as original
from scripts import r8_bl5_matched_native_reference as matched

IDENTITY='R8_BL5_WEAK_P0_C3D20_REFERENCE_DEV_V1'
SUPPORT_LENGTH=.25
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
STRONG_MANIFEST_SHA='344c358780e8e103230414e244e9d280c1779484ce0cb41cc29439ab9f262c2b'
STRONG_PARSED_SHA='f0956217c49728d948023bf7d7c654750f15f28af7198f1a7751c02728e82816'
NATIVE_CALIBRATION_SHA='ffdbe095aa64a303c9d67e75f84db1fab8ca3b6dbac0f87b609d6d826478f7ee'


def specimen():
    _,model=original.build_reference()
    return model


def layout(model):
    order=tuple(sorted(model['physical_nodes'])); index={n:3*i for i,n in enumerate(order)}
    fixed=np.array([index[n]+j for n in model['root_nodes'] for j in range(3)])
    free=np.setdiff1d(np.arange(3*len(order)),fixed)
    load=np.zeros(3*len(order));load[index[model['load_node']]:index[model['load_node']]+3]=model['load_N']
    return order,index,fixed,free,load


def assemble_body(model):
    """Full sparse isotropic C3D20 K; local B^T D B, no inverse or repair."""
    model=original.require_metadata(model);order,index,*_=layout(model)
    d=original._elastic_matrix();points,weights=np.polynomial.legendre.leggauss(3)
    rows,cols,values=[],[],[];elements={}
    for eid,ids in model['elements'].items():
        original._brick(ids,model['physical_nodes'])
        xyz=np.array([model['physical_nodes'][n] for n in ids]);ke=np.zeros((60,60))
        for ii in itertools.product(range(3),repeat=3):
            _,grad=original._shape(tuple(points[i] for i in ii));jac=xyz.T@grad
            det=float(np.linalg.det(jac))
            if not np.isfinite(det) or det<=0:raise ValueError('positive finite Jacobian required')
            gradient=np.linalg.solve(jac.T,grad.T).T
            b=np.zeros((6,60))
            for j,(gx,gy,gz) in enumerate(gradient):
                b[:,3*j:3*j+3]=((gx,0,0),(0,gy,0),(0,0,gz),(gy,gx,0),(0,gz,gy),(gz,0,gx))
            ke+=(b.T@d@b)*det*math.prod(weights[i] for i in ii)
        if not np.isfinite(ke).all():raise ValueError('nonfinite body operator')
        slots=np.array([index[n]+j for n in ids for j in range(3)])
        rows.extend(np.repeat(slots,60));cols.extend(np.tile(slots,60));values.extend(ke.ravel())
        elements[eid]=(slots,ke)
    k=sparse.coo_matrix((values,(rows,cols)),shape=(3*len(order),3*len(order))).tocsr()
    k.sum_duplicates();k.eliminate_zeros()
    return k,elements


def face_row(model,point,body):
    """One physical C3D20 face interpolation; shared material face is continuous."""
    _,index,*_=layout(model);choices=[]
    for eid in model[body+'_elements']:
        ids,low,high,center,span=original._brick(model['elements'][eid],model['physical_nodes'])
        if all(low[j]-1e-14<=point[j]<=high[j]+1e-14 for j in range(3)):
            reference=2*(np.asarray(point)-center)/span
            # A material face has an exact reference coordinate even when the
            # physical center/span arithmetic rounds just inside that face.
            # Preserve its C3D20 trace zeros before assembling weak rows.
            for j in range(3):
                if abs(point[j]-low[j])<=1e-14:
                    reference[j]=-1.
                elif abs(point[j]-high[j])<=1e-14:
                    reference[j]=1.
            shape,_=original._shape(reference)
            row=np.zeros((3,3*len(index)))
            for n,weight in zip(ids,shape):
                for j in range(3):row[j,index[n]+j]+=weight
            choices.append(row)
    if not choices:raise ValueError('material face point outside body')
    if any(np.max(abs(r-choices[0]))>1e-13 for r in choices):
        raise ValueError('inconsistent shared material face interpolation')
    return choices[0]


def five_rows(model,x):
    sb=face_row(model,(x,0.,-.01),'shell');st=face_row(model,(x,0.,.01),'shell')
    bb=face_row(model,(x,0.,.01),'beam');bt=face_row(model,(x,0.,.07),'beam')
    return np.vstack((bb-st,((bt-bb)/.06-(st-sb)/.02)[:2]))


def constraints(model,kind):
    if kind=='strong':
        c=np.vstack([five_rows(model,x) for x in (0.,.25,.5,.75,1.)]);scale=np.ones(25)
    elif kind=='weak':
        c=np.zeros((20,3*len(model['physical_nodes'])))
        for edge in range(4):
            for alpha in (.5-.5/math.sqrt(3),.5+.5/math.sqrt(3)):
                values=.125*five_rows(model,(edge+alpha)*.25)
                for channel in range(5):c[4*channel+edge]+=values[channel]
        scale=np.full(20,.25) # mean gap, position normalized by specimen length1m
    else:raise ValueError('explicit strong or weak representation required')
    return sparse.csr_matrix(c),scale


def rigid_fields(model,body):
    order,index,*_=layout(model);vectors=np.zeros((3*len(order),6))
    for n in model[body+'_nodes']:
        p=np.array(model['physical_nodes'][n])
        for j in range(3):
            vectors[index[n]:index[n]+3,j]=np.eye(3)[j]
            vectors[index[n]:index[n]+3,3+j]=np.cross(np.eye(3)[j],p)
    return vectors


def rank_checks(model,k,c,scale):
    _,_,_,free,_=layout(model)
    # Physical coordinate scale1m and support-mean position/director gaps.
    singular=np.linalg.svd(c[:,free].toarray()/scale[:,None],compute_uv=False)
    rank=int(np.sum(singular>singular[0]*max(c.shape)*np.finfo(float).eps))
    beam=rigid_fields(model,'beam');motion=np.linalg.svd(c@beam/scale[:,None],compute_uv=False)
    rigid={}
    for body in ('shell','beam'):
        v=rigid_fields(model,body);norm=max(float(abs(k).max()),1.)*max(float(np.max(abs(v))),1.)
        rigid[body]=float(np.max(abs(k@v)))/norm
    return dict(free_row_rank=rank,free_row_singular_min=float(singular[-1]),
        free_row_singular_max=float(singular[0]),bar_rigid_motion_rank=int(np.linalg.matrix_rank(c@beam/scale[:,None])),
        bar_rigid_singular_min=float(motion[-1]),body_rigid_relative_residual=rigid,
        physical_scaling='all continuum coordinate scales1m; weak gap averages by .25m support; model length1m')


def field_dict(model,vector):
    order,index,*_=layout(model)
    return {n:np.array(vector[index[n]:index[n]+3],copy=True) for n in order}


def solve(model,k,c,scale,*,diagnostics=None):
    """Sparse indefinite LU, diagonal congruence equilibration, bounded refinement."""
    _,index,fixed,free,load=layout(model);kf=k[free,:][:,free].tocsc();cf=c[:,free].tocsc()
    diagonal=kf.diagonal()
    if not np.isfinite(diagonal).all() or np.any(diagonal<=0):raise ValueError('positive finite displacement diagonal required; no SPD claim')
    dq=1/np.sqrt(diagonal);norm=np.sqrt(np.asarray((cf@sparse.diags(dq)).power(2).sum(axis=1)).ravel())
    if np.any(norm<=0) or not np.isfinite(norm).all():raise ValueError('nonzero finite constraint rows required')
    dl=1/norm;dp=sparse.diags(dq);dm=sparse.diags(dl)
    a=sparse.bmat([[dp@kf@dp,dp@cf.T@dm],[dm@cf@dp,None]],format='csc')
    rhs=np.r_[dq*load[free],np.zeros(c.shape[0])];lu=splu(a)
    z=lu.solve(rhs)
    for _ in range(3):z+=lu.solve(rhs-a@z)
    q=np.zeros(len(load));q[free]=dq*z[:len(free)];lam=dl*z[len(free):]
    internal=k@q;interface=-c.T@lam;residual=internal-interface-load
    support=np.zeros_like(q);support[fixed]=residual[fixed]
    eq=_linear_equilibrium(residual[free],load[free],np.ones(len(free)))
    gap=_scaled_constraint_gap(c@q,scale)
    work=float(load@q);denominator=max(abs(work),1e-12)
    wr=lambda v,body=None,origin=(0.,0.,0.):matched.common_origin_wrench(model['physical_nodes'],
        {n:f for n,f in field_dict(model,v).items() if body is None or n in model[body+'_nodes']},origin)
    cuts={body:dict(internal_wrench=wr(internal,body).tolist(),interface_wrench=wr(interface,body).tolist(),
        applied_wrench=wr(load,body).tolist(),support_wrench=wr(support,body).tolist()) for body in ('shell','beam')}
    result=dict(identity=IDENTITY,status='CALCULATED_NOT_VALIDATED',comparison_qualified=False,
        equation_ratio=eq,constraint_max=gap,raw_gap_max=float(np.max(abs(c@q))),
        applied_pairing_J=work,internal_pairing_J=float(q@internal),interface_pairing_J=float(q@interface),
        work_denominator_J=denominator,internal_work_relative_error=abs(float(q@internal)-work)/denominator,
        interface_work_relative_error=abs(float(q@interface))/denominator,
        total_wrench=wr(load+interface+support).tolist(),domain_cuts=cuts,
        tip_centroid_uz_m=float(q[index[model['load_node']]+2]),
        algebraic_multipliers=lam.tolist(),interface_exerted_loads=field_dict(model,interface),
        support_exerted_loads=field_dict(model,support),displacement=field_dict(model,q),
        polarity='C is beam-minus-shell; Kq+C.T lambda_alg=f; exerted interface=-C.T lambda_alg',
        numerical_method='sparse indefinite LU; diagonal congruence only; three residual corrections; no inverse/penalty/symmetrization')
    if diagnostics is not None:
        Path(diagnostics).write_text(json.dumps(result,default=jsonable,indent=2)+'\n')
    if not np.isfinite(q).all() or not np.isfinite(lam).all() or eq>1e-8 or gap>1e-9:
        raise ValueError('equation/constraint gates failed: '+str((eq,gap)))
    if result['internal_work_relative_error']>1e-8 or result['interface_work_relative_error']>1e-8:
        raise ValueError('work gates failed')
    if np.max(np.abs(result['total_wrench']))>1e-8:raise ValueError('physical wrench gate failed')
    result['status']='DEVELOPMENT_SOLVED_NOT_ACCEPTED'
    return result,q,lam


def saved_strong(saved,model):
    """Reuse EXACT previously verified execution/provenance; no new execution.

    Trusted immutable manifest and parsed-action SHA are explicit inputs. This
    does not establish a new executable run by a caller-supplied manifest.
    """
    saved=Path(saved);manifest=json.loads((saved/'execution-binding.json').read_text())
    if SHA(saved/'execution-binding.json')!=STRONG_MANIFEST_SHA or SHA(saved/'parsed-v2.json')!=STRONG_PARSED_SHA:
        raise ValueError('prior trusted execution/action identity changed')
    if (manifest['schema']!='R8_BL5_RELEASED_REFERENCE_EXECUTION_V1' or manifest['model_sha256']!=model['model_sha256'] or manifest['status']!='completed' or manifest['exit_code']!=0
        or manifest['executable_sha256']!=original.CCX_SHA256 or manifest['command'][1:]!=['-i','released']
        or Path(manifest['working_directory']).resolve()!=saved.resolve()
        or Path(manifest['command'][0]).name.lower()!='ccx.exe'
        or set(manifest['files'])!={'released.inp','released.frd','released.sta','stdout.log','stderr.log'}):
        raise ValueError('saved strong specimen execution mismatch')
    if any(SHA(saved/name)!=expected for name,expected in manifest['files'].items()):
        raise ValueError('saved strong bytes changed')
    if manifest['files']['released.inp']!=model['deck_sha256']:raise ValueError('saved deck mismatch')
    u,rf=original.read_double_output((saved/'released.frd').read_bytes(),model)
    return np.concatenate([u[n] for n in sorted(u)]),u,rf


def strong_action_error(saved,model,k,q):
    if SHA(Path(saved)/'parsed-v2.json')!=STRONG_PARSED_SHA:raise ValueError('prior action bytes changed')
    prior=json.loads((Path(saved)/'parsed-v2.json').read_text())
    g0={n:np.zeros(3) for n in model['physical_nodes']}
    for element in prior['element_internal_covectors'].values():
        for n,f in element.items():g0[int(n)]+=np.asarray(f)
    vector=np.concatenate([g0[n] for n in sorted(g0)])
    denominator=max(float(np.max(abs(vector))),1.)
    return float(np.max(abs(k@q-vector)))/denominator,denominator


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--saved-strong',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv);args.output.mkdir(exist_ok=False)
    model=specimen();k,_=assemble_body(model);cs,ss=constraints(model,'strong');cw,sw=constraints(model,'weak')
    rank=rank_checks(model,k,cw,sw)
    (args.output/'rank.json').write_text(json.dumps(rank,indent=2)+'\n')
    if rank['free_row_rank']!=20 or rank['bar_rigid_motion_rank']!=6 or max(rank['body_rigid_relative_residual'].values())>1e-12:
        raise ValueError('weak rank or body rigid prerequisite failed')
    saved,_,_=saved_strong(args.saved_strong,model)
    strong,qs,_=solve(model,k,cs,ss)
    error=float(np.max(abs(qs-saved)))/max(float(np.max(abs(saved))),1e-12)
    action_error,_=strong_action_error(args.saved_strong,model,k,qs)
    if error>1e-8 or action_error>1e-8:raise ValueError('same-body strong saved reconstruction mismatch')
    weak,_,_=solve(model,k,cw,sw)
    for name,result in (('strong',strong),('weak',weak)):
        (args.output/(name+'.json')).write_text(json.dumps(result,default=jsonable,indent=2)+'\n')
    (args.output/'comparison.json').write_text(json.dumps(dict(strong_saved_U_relative_error=error,
        strong_saved_internal_relative_error=action_error,
        strong_tip_uz=strong['tip_centroid_uz_m'],weak_tip_uz=weak['tip_centroid_uz_m'],
        physical_accuracy_accepted=False,remaining='different body/primal spaces and arbitrary broken native director sides'),indent=2)+'\n')


if __name__=='__main__':main()
