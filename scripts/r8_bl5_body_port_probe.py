"""Saved-density effective body-port compliance discriminator; ordinary DEV only.

No coupled native calibration, external execution, production dispatch or accuracy
claim. C_ref is beam-minus-shell; C_native is shell-minus-beam. Their transpose
loads agree on the previously proven common trace subset, not on arbitrary fields.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_ge_experimental import _linear_equilibrium
from scripts import r8_bl5_matched_native_reference as matched
from scripts import r8_bl5_weak_continuum_reference as continuum

IDENTITY='R8_BL5_COMMON_DENSITY_BODY_PORT_DEV_V1'
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SAVED={
    'input-v1.json':'849c3e6fb49c6bdc11145f5dd24c3ae507dd2f8a7c640a3e29b97346aee01615',
    'operators-v1.npz':'f4b421a6cc21fbf530d9d027928af0b5376e74c7150d6da9751eb0bc190296aa',
    'results-v3/weak-v1.json':'a1fe3a9391e041deb74707d6278f27cea9a8b1e3488bdae3e59bd878d37e3711',
}
FLOORS=dict(equilibrium_ratio=1e-8,physical_wrench_N_Nm=1e-8,
    work_relative=1e-8,work_denominator_J=1e-12,material_compatibility_m_rad=1e-9,
    physical_reference_length_m=1.,force_floor_N=1.,moment_floor_Nm=1.)


def load_saved(root):
    """Strict immutable saved-byte identity; no newly calculated continuum solve."""
    root=Path(root)
    for name,digest in SAVED.items():
        if SHA(root/name)!=digest:raise ValueError('saved continuum bytes changed: '+name)
    model=continuum.original.require_metadata(json.loads((root/'input-v1.json').read_text()))
    result=json.loads((root/'results-v3/weak-v1.json').read_text())
    if result['identity']!=continuum.IDENTITY or result['status']!='DEVELOPMENT_SOLVED_NOT_ACCEPTED':
        raise ValueError('saved weak result is not the trusted DEV result')
    with np.load(root/'operators-v1.npz',allow_pickle=False) as a:
        k=sparse.csr_matrix((a['K_data'],a['K_indices'],a['K_indptr']),shape=tuple(a['K_shape']))
        c=sparse.csr_matrix(a['C_weak'])
    order,index,fixed,free,f=continuum.layout(model)
    u=np.concatenate([result['displacement'][str(n)] for n in order])
    lam=np.asarray(result['algebraic_multipliers'],dtype=float)
    if c.shape!=(20,len(u)) or lam.shape!=(20,) or not np.isfinite(u).all() or not np.isfinite(lam).all():
        raise ValueError('saved density/displacement layout mismatch')
    interface=-c.T@lam
    saved_interface=np.concatenate([result['interface_exerted_loads'][str(n)] for n in order])
    if not np.array_equal(interface,saved_interface):raise ValueError('saved physical action identity mismatch')
    if _linear_equilibrium((k@u-interface-f)[free],f[free],np.ones(len(free)))>1e-8:
        raise ValueError('saved equilibrium check failed')
    return model,k,c,u,lam,f


def continuum_fields(saved):
    model,k,c,u,lam,f=saved;order,index,*_=continuum.layout(model)
    physical=f-c.T@lam;body={}
    for name in ('shell','beam'):
        slots=np.array([index[n]+j for n in model[name+'_nodes'] for j in range(3)])
        load=physical[slots];v=u[slots];kb=k[slots,:][:,slots]
        pairing=float(load@v);internal=float(v@(kb@v));den=max(abs(pairing),1e-12)
        body[name]=dict(paired_work_J=pairing,stored_energy_J=.5*internal,
            work_relative_error=abs(internal-pairing)/den,
            load_wrench=(continuum.rigid_fields(model,name).T@physical).tolist())
        if body[name]['work_relative_error']>1e-8:raise ValueError('saved body work failed: '+name)
    points=[]
    for edge in range(4):
        for alpha in (.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)):
            x=(edge+alpha)*.25
            points.append(dict(x_m=float(x),point_gap_position_m_director_rad=(continuum.five_rows(model,x)@u).tolist()))
    return dict(body=body,integrated_gap_position_m2_director_m=(c@u).tolist(),point_gaps=points,
        warning='Integrated zero gap does not imply pointwise continuity; energies are saved continuum fields only.')


def native_rigid_fields(owner):
    """Six actual beam rigid pose fields; retained p=0, all cell spins included."""
    out=[]
    for j in range(6):
        translation=np.eye(3)[j] if j<3 else np.zeros(3)
        spin=np.eye(3)[j-3] if j>=3 else np.zeros(3)
        q=matched.common_vector(owner,lambda p:np.zeros(3),np.zeros(3),
            lambda p:translation+np.cross(spin,p),spin)
        out.append(q[owner._shell_count:])
    return np.column_stack(out)


def shell_rigid_fields(owner):
    r=np.zeros((owner._shell_count,6))
    for i,p in enumerate(owner._shell_xyz):
        r[6*i:6*i+3,:3]=np.eye(3)
        for j in range(3):r[6*i:6*i+3,3+j]=np.cross(np.eye(3)[j],p)
        r[6*i+3:6*i+6,3:]=np.eye(3)
    return r


def require_balanced(load,rigid):
    """No projection: physical six-mode pairings at declared 1N/1Nm floors."""
    pairings=rigid.T@load
    if not np.isfinite(pairings).all() or np.max(abs(pairings))>1e-8:
        raise ValueError('unsupported beam load has nonzero physical rigid work: '+str(pairings))
    return pairings


def prepare(saved):
    if SHA(matched.__file__)!='068ae3ae16fe167d3052a02e8ecf0b7c6c0c3ba83f0db1f79ab5dfd07a75d9a8' or SHA(continuum.__file__)!='ecaedbab83249f46a390a4bb2bc0da269d54459f9d7722233b0675208cb8c645':
        raise ValueError('historical preparation helper identity changed')
    owner=ss.create_analysis(definition=matched.definition())
    model,k,c,u,lam,f=saved
    if owner._C.shape!=(20,168) or owner._shell_count!=90 or owner._beam_count!=78:
        raise ValueError('unchanged native specimen layout required')
    native_f=np.zeros(168);native_f[90+6*owner._beam_ids.index(105)+2]=1.
    loads=native_f+owner._C.T@lam
    rigid=native_rigid_fields(owner);beam_k=owner._K[90:,:][:,90:]
    coefficient=float(np.max(abs(beam_k@rigid)))/max(float(abs(beam_k).max()),1.)
    if coefficient>1e-8:raise ValueError('native rigid-field body identity failed')
    pairings=require_balanced(loads[90:],rigid)
    native_wrenches={'shell':shell_rigid_fields(owner).T@loads[:90],'beam':pairings}
    reference_load=f-c.T@lam
    common_wrench={body:continuum.rigid_fields(model,body).T@reference_load for body in ('shell','beam')}
    if any(np.max(abs(native_wrenches[body]-common_wrench[body]))>1e-8 for body in common_wrench):
        raise ValueError('common-origin physical body load mapping failed')
    return owner,loads,rigid,dict(beam_rigid_load_pairings_N_Nm=pairings.tolist(),
        full_native_physical_load_covector=loads.tolist(),
        native_body_load_wrenches_N_Nm={body:v.tolist() for body,v in native_wrenches.items()},
        continuum_body_load_wrenches_N_Nm={body:v.tolist() for body,v in common_wrench.items()},
        native_body_rigid_coefficient_ratio=coefficient,density_channel_major=lam.tolist(),
        polarity='continuum f-C_ref.T lambda_ref; native f+C_native.T lambda_ref',
        row_measure='4 x .25m P0 supports, arclength-integrated; position m2/director m',
        density_units=['N/m']*12+['N']*8,criteria=FLOORS,
        scope='Effective body-port compliance; arbitrary broken native trace space remains unmatched.')


def precise_residual(k,q,load):
    """64-digit reduction of ORIGINAL binary64 K/q/load, no operator change.

    Quiet loaded rows must retain their original relative equilibrium guard.
    A CSR dot rounds its large cancellation before subtracting a quiet load.
    This evaluates that same stored operator with a longer accumulator instead.
    The ordinary CSR residual is separately recorded, never silently erased.
    """
    k=k.tocsr();out=np.zeros(k.shape[0])
    with localcontext() as context:
        context.prec=64
        values=[Decimal.from_float(float(v)) for v in q]
        for row in range(k.shape[0]):
            lo,hi=k.indptr[row:row+2]
            total=sum((Decimal.from_float(float(v))*values[int(j)] for v,j in zip(k.data[lo:hi],k.indices[lo:hi])),Decimal(0))
            out[row]=float(total-Decimal.from_float(float(load[row])))
    return out


def solve_body(k,load,coordinate_scale,*,fixed=(),rigid=None,diagnostics=None):
    """Original mixed body K, no condensation; indefinite LU with congruence only."""
    n=len(load);fixed=np.asarray(fixed,dtype=int);free=np.setdiff1d(np.arange(n),fixed)
    scale=np.asarray(coordinate_scale)[free]
    if not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('positive finite declared coordinate scale required')
    if rigid is not None:
        require_balanced(load,rigid)
        g=rigid.T # explicit physical pose gauge, no added body support/stiffness
        if np.linalg.matrix_rank(g)!=6:raise ValueError('six independent physical beam gauge fields required')
    else:g=np.zeros((0,n))
    dp=sparse.diags(scale);a0=dp@k[free,:][:,free]@dp
    rowmax=np.asarray(abs(a0).max(axis=1).toarray()).ravel()
    if np.any(rowmax<=0) or not np.isfinite(rowmax).all():raise ValueError('nonzero finite mixed body rows required')
    d=scale/np.sqrt(rowmax);dp=sparse.diags(d);gf=sparse.csr_matrix(g[:,free])
    dg=1/np.sqrt(np.asarray((gf@dp).power(2).sum(axis=1)).ravel()) if len(g) else np.zeros(0)
    dm=sparse.diags(dg)
    a=sparse.bmat([[dp@k[free,:][:,free]@dp,dp@gf.T@dm],[dm@gf@dp,None]],format='csc')
    rhs=np.r_[d*load[free],np.zeros(len(g))];lu=splu(a);z=lu.solve(rhs)
    q=np.zeros(n);q[free]=d*z[:len(free)];eta=dg*z[len(free):]
    for _ in range(3):
        # Refine physical original KKT residual, not the rounded congruent A.
        residual=precise_residual(k,q,load)
        if len(g):residual+=precise_residual(sparse.csr_matrix(g.T),eta,np.zeros(n))
        gauge=precise_residual(sparse.csr_matrix(g),q,np.zeros(len(g)))
        dz=lu.solve(-np.r_[d*residual[free],dg*gauge])
        q[free]+=d*dz[:len(free)];eta+=dg*dz[len(free):]
    ordinary_residual=k@q-load;residual=precise_residual(k,q,load)
    internal=residual+load;reaction=g.T@eta
    eq=_linear_equilibrium(residual[free],load[free],np.asarray(coordinate_scale)[free])
    work=float(load@q);den=max(abs(work),1e-12)
    report=dict(status='CALCULATED_NOT_VALIDATED',physical_coordinates=q.tolist(),
        full_unscaled_residual=residual.tolist(),gauge_multipliers=eta.tolist(),
        full_physical_load_covector=load.tolist(),ordinary_CSR_residual=ordinary_residual.tolist(),
        residual_reduction='64-digit original binary64 K/q/load products and sum; no K/load projection or modification',
        gauge_covector=reaction.tolist(),gauge_pose=(g@q).tolist(),
        equation_ratio=eq,paired_work_J=work,internal_pairing_J=float(q@internal),
        work_denominator_J=den,work_relative_error=abs(float(q@internal)-work)/den,
        support_covector={str(int(i)):float(residual[i]) for i in fixed},
        factorization='sparse indefinite LU, congruence and three original-operator physical residual corrections; no inverse/condensation/penalty')
    if diagnostics is not None:write(diagnostics,report)
    if not np.isfinite(q).all() or eq>1e-8 or report['work_relative_error']>1e-8 or np.max(abs(reaction),initial=0)>1e-8:
        raise ValueError('body equilibrium/work/gauge reaction failed')
    if np.max(abs(g@q),initial=0)>1e-9:raise ValueError('physical rigid-pose gauge failed')
    report['status']='DEVELOPMENT_BODY_SOLVED_NOT_ACCEPTED'
    return q,report


def beam_material(owner,q):
    """Reference-linear J q = grad Psi*(p); actual virgin section energy at p.

    Do not evaluate finite rotations from a linear calibration increment. The
    local reference Jacobian and unchanged section conjugate supply this check.
    """
    rows=[];energy=0.
    for element,slots,origin in owner._beam_parts:
        op=element.operator;ref=op.reference;local=q[slots];p=local[24:]
        response=op.evaluate(ref.coordinates,np.zeros((3,3)),ref.nodal_triads,
            np.tile(np.eye(3),(2,1,1)),np.zeros(18),origin=origin)
        material=op.cell.response(p.tolist(),origin,check=op.guard)
        gradient=np.asarray(material['gradient'][0])+np.asarray(material['gradient'][1])
        kinematic=response.hessian[24:,:24]@local[:24]
        residual=kinematic-gradient
        recovered=op.recover_material(np.tile(np.eye(3),(2,1,1)),p,origin=origin)
        stored=float(sum(recovered.stored_energy));energy+=stored
        dissipation=float(sum(recovered.incremental_dissipation))
        if origin!=op.cell.virgin() or recovered.history!=origin:
            raise ValueError('probe must retain exact virgin predecessor and returned history')
        fibres=[f for station in recovered.stations for f in station['fibres']]
        if any(f.branch!='ELASTIC' or any(f.plastic_increment) for f in fibres):
            raise ValueError('every fibre must witness strictly elastic branch and zero plastic increment')
        row=dict(element_id=element.element_id if hasattr(element,'element_id') else element.node_ids,
            compatibility_m_rad=residual.tolist(),resultants=p.tolist(),kinematics=kinematic.tolist(),
            stored_material_energy_J=stored,incremental_dissipation_J=dissipation,
            exact_virgin_history=True,fibre_elastic_witnesses=[dict(fibre_id=f.fibre_id,branch=f.branch,
                plastic_increment=list(f.plastic_increment),stress_Pa=list(f.stress)) for f in fibres],
            stations=[dict(strain=(s['strain']+s['strain_low']).tolist(),
                resultants=(s['resultants']+s['resultants_low']).tolist()) for s in recovered.stations])
        rows.append(row)
        if np.max(abs(residual))>1e-9 or dissipation!=0.:raise ValueError('virgin material stationarity/domain failed')
    return dict(stored_material_energy_J=energy,elements=rows,
        warning='Physical section energy after stationarity, never a negative saddle subblock energy.')


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=continuum.jsonable)+'\n')


def run_probe(saved,prepared,output):
    """Exactly one RHS per native body; diagnostic outputs precede validation."""
    output=Path(output);output.mkdir(exist_ok=False)
    owner,loads,rigid,preflight=prepared
    write(output/'preflight.json',preflight)
    reference=continuum_fields(saved);write(output/'saved-continuum-fields.json',reference)
    ns=owner._shell_count
    shell_q,shell=solve_body(owner._K[:ns,:ns],loads[:ns],owner._coordinate_scale[:ns],
        fixed=owner._fixed,diagnostics=output/'shell-calculated.json')
    write(output/'shell-solved.json',shell)
    r_shell=shell_rigid_fields(owner);shell_support=np.zeros(ns)
    for i,value in shell['support_covector'].items():shell_support[int(i)]=value
    shell_wrench=r_shell.T@(loads[:ns]+shell_support)
    write(output/'shell-wrench.json',dict(load=(r_shell.T@loads[:ns]).tolist(),
        support=(r_shell.T@shell_support).tolist(),residual=shell_wrench.tolist()))
    if np.max(abs(shell_wrench))>1e-8:raise ValueError('shell physical load/support wrench failed')
    beam_q,beam=solve_body(owner._K[ns:,ns:],loads[ns:],owner._coordinate_scale[ns:],
        rigid=rigid,diagnostics=output/'beam-calculated.json')
    write(output/'beam-solved.json',beam)
    material=beam_material(owner,beam_q);write(output/'beam-material.json',material)
    denominator=max(abs(beam['paired_work_J']),1e-12)
    energy_error=abs(2*material['stored_material_energy_J']-beam['paired_work_J'])/denominator
    if energy_error>1e-8:raise ValueError('physical material energy does not close beam work')
    if np.max(abs(rigid.T@np.asarray(beam['full_unscaled_residual'])))>1e-8:
        raise ValueError('beam physical wrench residual failed')
    q=np.r_[shell_q,beam_q];point=[];state=matched.reference_state(owner)
    for op,slots,weight,edge,alpha in owner._line_samples:
        response=op.evaluate(np.zeros((2,6)),state,np.zeros(5))
        point.append(dict(edge=int(edge),alpha=float(alpha),weight_m=float(weight),
            linear_point_gap_position_m_director_rad=(response.gap_jacobian@q[slots]).tolist()))
    report=dict(identity=IDENTITY,status='DEVELOPMENT_DISCRIMINATOR_NOT_ACCEPTED',
        compute_inventory=dict(native_shell_RHS=1,native_beam_RHS=1,coupled_calibration=0,external_runs=0),
        native_body_paired_work_J={name:value['paired_work_J'] for name,value in (('shell',shell),('beam',beam))},
        continuum_body_paired_work_J={name:value['paired_work_J'] for name,value in reference['body'].items()},
        effective_port_work_ratio={name:value['paired_work_J']/reference['body'][name]['paired_work_J'] for name,value in (('shell',shell),('beam',beam))},
        material_energy_relative_error=energy_error,native_point_gaps=point,
        native_integrated_gap=(owner._C@q).tolist(),
        decision_scope='Effective body-port work at fixed saved density, not the coupled native response and not isolated constitutive compliance.',
        limitations='Common action identity only on proven global affine/director subset; broken native cell traces unmatched. No accuracy/convergence/qualification.')
    write(output/'result.json',report)
    return report
