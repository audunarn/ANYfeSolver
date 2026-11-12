"""One saved coarse projected-density positive-norm assessment.

Physical work is derived independently from density/frame/arms. Mechanical K
is never inverted or used as this norm. Every retained halfcell spin remains
in the kinematic norm; physical nodal rotations/resultants remain in actual KKT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import linalg, sparse
from threadpoolctl import threadpool_info

from scripts.r8_bl5_projected_auxiliary_norm_dev import (
    CELLS, ELL, LENGTH, NX, NY, SUPPORTS, WIDTH, integrated_p1, lower_gram,
    projection, sha)
from scripts.b3_e4_bl5_riesz_controls_dev import beam_norm, shell_scalar
from scripts.b3_e4_bl5_reference_accuracy_dev import beam_bracket
from scripts.b3_e4_bl5_modal_reference_dev import p0_shell_bracket
from scripts.r8_bl5_projected_density_patch_dev import compile_saved_projection


def assess(root, registration, output, wheel):
    contract = json.loads(registration.read_text())
    if (contract['status'] != 'ELIGIBLE_PHYSICAL_L2_REPRESENTATION_DEV' or
            contract['fixture'] != dict(nx=NX,ny=NY,halfcells=CELLS,supports=SUPPORTS)):
        raise ValueError('eligible prospective coarse representation required')
    scalar_path=registration.parent/'scalar.json'
    if sha(scalar_path)!=contract['scalar_registration_sha256']:
        raise ValueError('changed scalar registration')
    if sha(registration.parent/'representation.json')!=contract['representation_sha256']:
        raise ValueError('changed canonical representation evidence')
    scalar_contract=json.loads(scalar_path.read_text())
    if (contract['safe_case_floor']!=scalar_contract['safe_case_floor'] or
            contract['shell_modes']!=scalar_contract['reference_preparation'][-1]['modes']):
        raise ValueError('changed independent scalar bound/reference')
    if contract['criteria']!=dict(reference_error=1e-3,inverse_eigen_symmetry=1e-10,
        condition_budget=1e4,beta_absolute_arithmetic_margin=1e-6,
        variational_ceiling=(1+1e-6)**2):
        raise ValueError('changed prospective numerical criteria')
    data, _, _, _, c, _, _, beam_work, fixed, _, records, *_ = compile_saved_projection(root)
    if data['layout']['nx'] != NX or len(data['policy']['support_endpoints_m']) != SUPPORTS+1:
        raise ValueError('saved coarse fixture only; no automatic refinement')
    xyz=np.asarray(data['xyz_m'])
    shell_xyz=np.asarray([(i/NX,j*WIDTH/NY,0.) for j in range(NY+1) for i in range(NX+1)])
    beam_xyz=np.c_[np.linspace(0.,LENGTH,CELLS+1),np.zeros(CELLS+1),np.full(CELLS+1,.0675)]
    if (xyz.shape!=(len(shell_xyz)+len(beam_xyz),3) or
            not np.allclose(xyz,np.r_[shell_xyz,beam_xyz],atol=1e-13,rtol=0.) or
            not np.allclose(data['policy']['support_endpoints_m'],
                            np.linspace(0.,LENGTH,SUPPORTS+1),atol=1e-13,rtol=0.) or
            data['layout']['counts']['physical_dofs']!=6*(len(shell_xyz)+len(beam_xyz))):
        raise ValueError('changed physical geometry/layout/reference supports')
    local = records[0][0]
    frame = local.port.frame; n = frame@local.reference_director
    ab = frame@local.reference_arm; ass = local.thickness*n/2
    np.testing.assert_allclose(ab,(0.,0.,-.05),atol=1e-13,rtol=0.)
    np.testing.assert_allclose(ass,(0.,0.,.0175),atol=1e-13,rtol=0.)
    for op,*_ in records:
        if (np.linalg.norm(op.port.lift)>1e-13 or
                not np.allclose(op.port.frame,frame,atol=1e-13,rtol=0.) or
                not np.allclose(frame@op.reference_arm,ab,atol=1e-13,rtol=0.)):
            raise ValueError('constant frame/arms and zero lift required')
    p=projection(); cols=10*SUPPORTS; scale=np.repeat((1.,1.,1.,ELL,ELL),2*SUPPORTS)
    np.testing.assert_allclose(contract['physical_L2_basis_scale'],
        1/np.sqrt(np.diag(p.T@p/CELLS)),rtol=0.,atol=1e-14)
    densities=np.zeros((CELLS,5,cols))
    for channel in range(5):
        densities[:,channel,2*SUPPORTS*channel:2*SUPPORTS*(channel+1)]=p*scale[2*SUPPORTS*channel]
    # Independent exerted beam/shell wrenches: include force-arm and direct work.
    fb=np.einsum('ij,cjk->cik',frame,densities[:,:3])
    director_vectors=np.einsum('ij,cjk->cik',frame@local.director_basis.T,densities[:,3:])
    couple=np.cross(n,director_vectors.transpose(0,2,1)).transpose(0,2,1)
    mb=np.cross(ab,fb.transpose(0,2,1)).transpose(0,2,1)+couple
    fs=-fb
    ms=np.cross(ass,fs.transpose(0,2,1)).transpose(0,2,1)-couple
    knots,hb,ba,_=beam_norm(LENGTH,ELL,CELLS,1,ab,.1)
    xs,_,hs,sa=shell_scalar(LENGTH,WIDTH,ELL,NX,NY)
    count=(NX+1)*(NY+1)
    shell=np.zeros((5,count,cols))
    for channel in range(5):
        values=fs[:,channel] if channel<3 else ms[:,channel-3]
        shell[channel,:NX+1]=integrated_p1(xs,values,knots)
    beam_u=np.stack([integrated_p1(knots,fb[:,a],knots) for a in range(3)],axis=1)
    beam=np.concatenate((beam_u.reshape(-1,cols),(mb/CELLS).reshape(-1,cols)))
    sa5=np.concatenate([a*count+sa for a in range(5)])
    full_work=np.concatenate((shell.reshape(-1,cols),beam))
    physical=int(data['layout']['counts']['physical_dofs'])
    shell_slots=np.concatenate([6*np.arange(count)+a for a in range(5)])
    beam_slots=np.concatenate([6*count+6*i+np.arange(3) for i in range(CELLS+1)])
    spins=np.concatenate([physical+24*member+3*half+np.arange(3)
                          for member in range(NX) for half in (0,1)])
    all_fields=np.r_[shell_slots,beam_slots,spins]
    active=np.r_[sa5,5*count+ba]
    slots=all_fields[active]
    if not np.array_equal(np.setdiff1d(all_fields,slots),np.intersect1d(fixed,all_fields)):
        raise ValueError('actual and auxiliary physical supports differ')
    absent=np.setdiff1d(np.arange(c.shape[1]),all_fields)
    omitted=c[:,absent].data
    omitted_max=float(np.max(np.abs(omitted))) if len(omitted) else 0.
    if omitted_max>1e-13 or np.intersect1d(slots,fixed).size:
        raise ValueError('physical quotient/support mapping failed')
    complete_native=-c[:,all_fields].T@sparse.diags(scale)
    work_error=float(np.max(np.abs(full_work-complete_native.toarray())))
    if work_error>1e-12:
        raise ArithmeticError('independent ALL-column physical work mismatch')
    h=sparse.block_diag((hs,hs,hs,ELL**2*hs,ELL**2*hs,hb),format='csc')
    g,residual,skew=lower_gram(h,full_work[active])
    scalar_lower,scalar_upper,shell_info=p0_shell_bracket(knots,LENGTH,WIDTH,ELL,contract['shell_modes'])
    shell_lo=np.zeros((cols,cols));shell_hi=np.zeros_like(shell_lo)
    for channel in range(5):
        coef=fs[:,channel] if channel<3 else ms[:,channel-3]/ELL
        shell_lo+=coef.T@scalar_lower@coef
        shell_hi+=coef.T@scalar_upper@coef
    preparation=[]
    for subdivision in (1,2,4,8,16,32):
        bl,bu,info=beam_bracket(fb,mb/CELLS,length=LENGTH,ell=ELL,arm=ab,width=.1,
                              subdivisions=subdivision)
        lower,upper=shell_lo+bl,shell_hi+bu
        error=float(linalg.eigvalsh(upper-lower,lower)[-1])
        preparation.append(dict(beam=info,total_relative_reference_error=error))
        if error<=1e-3: break
    else:
        raise ArithmeticError('preset independent reference budget exhausted')
    d=np.diag(np.tile(contract['physical_L2_basis_scale'],5))
    gn,mn=d@g@d,d@upper@d
    values,vectors=linalg.eigh(gn,mn)
    eig_residual=float(np.linalg.norm(gn@vectors-(mn@vectors)*values)/np.linalg.norm(gn@vectors))
    beta=float(np.sqrt(values[0])) if values[0]>0 else None
    conditions=dict(native=float(np.linalg.cond(gn)),reference=float(np.linalg.cond(mn)))
    safe=contract['safe_case_floor']
    # Independent scalar extremizer exercised in the normal-force channel.
    scalar_vector=np.asarray(contract['scalar_extremizer'])
    extremizer=np.zeros(cols);extremizer[4*SUPPORTS:6*SUPPORTS]=scalar_vector
    directional=float(np.sqrt((extremizer@g@extremizer)/(extremizer@upper@extremizer)))
    # Director couple cancels arm moment without erasing translational work.
    couple_map=np.cross(n,(frame@local.director_basis.T).T).T
    force=np.array((1.,0.,0.))
    direct=np.linalg.lstsq(couple_map,-np.cross(ab,force),rcond=None)[0]
    cancellation=np.zeros(cols)
    for support in range(SUPPORTS):
        for channel,value in enumerate(np.r_[frame.T@force,direct]/np.array((1.,1.,1.,ELL,ELL))):
            cancellation[2*(channel*SUPPORTS+support)]=value
    cancelled=np.einsum('cik,k->ci',mb,cancellation)
    if np.max(np.abs(cancelled))>1e-13 or cancellation@g@cancellation<=0.:
        raise ArithmeticError('arm cancellation lost surviving force work')
    # Bare physical yaw has positive auxiliary size and is NOT quotiented out.
    yaw=beam_work.T[:,5]
    yaw_trace=yaw[slots]
    yaw_norm=float(yaw_trace@(h@yaw_trace))
    if yaw_norm<=0. or np.linalg.norm(c@yaw)==0.:
        raise ArithmeticError('physical yaw norm/work treatment failed')
    predicates=dict(all_column_physical_work=work_error<=1e-12,
        reference_error=error<=1e-3,inverse_residual=residual<=1e-10,
        eigen_residual=eig_residual<=1e-10,gram_skew=skew<=1e-10,
        physical_L2_Gram_condition=max(conditions.values())<=1e4,
        all_eigenvalues_positive=bool(np.all(values>0)),
        scalar_product_lower_bound=beta is not None and beta>=safe,
        variational_upper_bound=bool(values[-1]<=(1+1e-6)**2))
    result=dict(status='COMPLETED_COARSE_DEVELOPMENT_ONLY' if all(predicates.values())
        else 'UNRESOLVED_COARSE_DEVELOPMENT_CHECK',predicates=predicates,
        beta=beta,safe_case_floor=safe,conditions=conditions,
        raw_conditions=dict(native=float(np.linalg.cond(g)),reference=float(np.linalg.cond(upper))),
        all_generalized_eigenvalues=values.tolist(),eigen_residual=eig_residual,
        norm_inverse_residual=residual,norm_rows=h.shape[0],gram_relative_skew=skew,
        all_column_work_max_abs=work_error,omitted_slots_max_abs=omitted_max,
        independent_normal_force_extremizer_directional_beta=directional,
        arm_cancellation_max_cell_moment=float(np.max(np.abs(cancelled))),
        arm_cancellation_positive_native_dual=float(cancellation@g@cancellation),
        bare_yaw_positive_auxiliary_norm=yaw_norm,
        reference_preparation=preparation,shell_reference=shell_info,
        source_sha256=sha(__file__),registration_sha256=sha(registration),
        helper_sha256={Path(f.__code__.co_filename).name:sha(f.__code__.co_filename)
                       for f in (projection,beam_norm,shell_scalar,beam_bracket,
                                 p0_shell_bracket,compile_saved_projection)},
        wheel_sha256=sha(wheel),imported_anyfesolver=sys.modules['anyfesolver'].__file__,
        imported_local_operators={name:dict(path=sys.modules[name].__file__,
            sha256=sha(sys.modules[name].__file__)) for name in (
            'anyfesolver._stiffened_shell_bl5_local_experimental',
            'anyfesolver._stiffened_shell_l1_local_experimental',
            'anyfesolver._ge_beam3_retained_generalized_state')},
        input_sha256={name:sha(root/'native'/name) for name in ('inputs.json','operators.npz')},
        limitations='coarse positive auxiliary norm only; physical yaw is NOT quotiented out. No structural SPD, growing-policy, locking, converged transfer or F03 acceptance.')
    np.savez(output.with_suffix('.npz'),native_lower=g,reference_lower=lower,
             reference_upper=upper,physical_density=densities,all_column_work=full_work,
             norm_action=full_work[active],slots=slots,physical_L2_scale=d)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wheel',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError(args.output)
    pools=threadpool_info()
    if not pools or any(row['num_threads']!=1 for row in pools):
        raise RuntimeError('effective single-thread norm/reference pools required')
    result=assess(args.root,args.registration,args.output,args.wheel)
    print(json.dumps({name:result[name] for name in ('status','beta','safe_case_floor','conditions','predicates')}))
    raise SystemExit(0 if all(result['predicates'].values()) else 1)
