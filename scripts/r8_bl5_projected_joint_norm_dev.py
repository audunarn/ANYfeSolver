"""Independent bound then native norm for ONE growing projected32 fixture.

Whole B3 member:shell interval1:1, halfcell:shell2:1. Four shell intervals
per support; projected coarse/fine spaces are NOT nested. Recompute every Gram.
This is a development discriminator, not formalF03 or multiplier selection.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import linalg,sparse
from threadpoolctl import threadpool_info
from scripts.r8_bl5_projected_auxiliary_norm_dev import integrated_p1,lower_gram
from scripts.b3_e4_bl5_riesz_controls_dev import shell_scalar,beam_norm
from scripts.b3_e4_bl5_modal_reference_dev import p0_shell_bracket
from scripts.b3_e4_bl5_reference_accuracy_dev import beam_bracket
from scripts.r8_bl5_projected_reference_action_dev import integrated_projected_weights
from scripts.r8_bl5_external_affine_reference_dev import shape_weights

IDENTITY='BL5_PROJECTED_FOUR_SHELL_INTERVALS_JOINT32_DEV_V1'
NX,NY,CELLS,SUPPORTS=32,8,64,8
LENGTH,WIDTH,ELL=1.,.2,.4
FIXTURE=dict(nx=NX,ny=NY,halfcells=CELLS,supports=SUPPORTS)
CRITERIA=dict(reference_error=1e-3,inverse_eigen_symmetry=1e-10,condition_budget=1e4,
    arithmetic_beta_margin=1e-6,variational_ceiling=(1+1e-6)**2)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def helpers():
    return {Path(f.__code__.co_filename).name:sha(f.__code__.co_filename) for f in
        (integrated_p1,lower_gram,shell_scalar,beam_norm,p0_shell_bracket,beam_bracket,integrated_projected_weights,shape_weights)}


def projection():
    p=np.zeros((CELLS,2*SUPPORTS))
    for cell in range(CELLS):
        support=cell//8
        midpoint=F(2*cell+1,2*CELLS)
        p[cell,2*support]=1.
        p[cell,2*support+1]=float(2*(midpoint-F(support,SUPPORTS))*SUPPORTS-1)
    return p


def register(output):
    # This stage imports no native operator and inspects no native beta.
    p=projection();xs,_,h,active=shell_scalar(LENGTH,WIDTH,ELL,NX,NY)
    knots=np.linspace(0.,LENGTH,CELLS+1);boundary=integrated_p1(xs,p,knots)
    full=np.zeros(((NX+1)*(NY+1),2*SUPPORTS));full[:NX+1]=boundary
    action=full[active];g,residual,skew=lower_gram(h,action)
    mass=p.T@p/CELLS;expected=np.tile((1/SUPPORTS,21/(64*SUPPORTS)),SUPPORTS)
    np.testing.assert_array_equal(mass,np.diag(expected))
    d=np.diag(1/np.sqrt(expected));preparation=[]
    for modes in (4096,8192,16384):
        ml,mu,info=p0_shell_bracket(knots,LENGTH,WIDTH,ELL,modes)
        lower,upper=p.T@ml@p,p.T@mu@p
        error=float(linalg.eigvalsh(upper-lower,lower)[-1])
        preparation.append(dict(modes=modes,projected_relative_tail=error,
            complete_halfcell_relative_tail=info['relative_loewner_tail_bound']))
        if error<=CRITERIA['reference_error']:break
    else:raise ArithmeticError('preset independent shell reference budget exhausted')
    gd,md=d@g@d,d@upper@d;values,vectors=linalg.eigh(md,gd)
    eigen_residual=float(np.linalg.norm(md@vectors-(gd@vectors)*values)/np.linalg.norm(md@vectors))
    cs2=float(values[-1]);rho=np.hypot(.05,.1/2)/ELL
    product2=max(2*cs2,max(2.,1+2*rho*rho)*max(2*cs2,1+rho*rho))
    safe=float(1/np.sqrt(product2)/np.sqrt(1.001)-1e-6)
    conditions=dict(discrete=float(np.linalg.cond(gd)),reference=float(np.linalg.cond(md)))
    exact=np.zeros_like(boundary)
    for j in range(SUPPORTS):
        weights,_=integrated_projected_weights(F(j,SUPPORTS),F(j+1,SUPPORTS),NX,CELLS)
        for degree,rows in enumerate(weights):
            for node,w in rows.items():exact[node,2*j+degree]=float(w)
    endpoint_error=float(np.max(np.abs(boundary-exact)))
    quarter=np.arange(SUPPORTS)*4
    if np.any(exact[quarter+1,2*np.arange(SUPPORTS)+1]==0) or np.any(exact[quarter+3,2*np.arange(SUPPORTS)+1]==0):
        raise ArithmeticError('independent interior affine action absent')
    # A genuine supported-Q1 checkerboard outside this density subspace.
    checker=np.tile((1.,-3.,3.,-1.),CELLS//4)
    checker_action=integrated_p1(xs,checker[:,None],knots)[:,0]
    checker_defect=float(np.max(np.abs(checker_action[1:])))
    checker_reference=float(checker@ml@checker)
    duplicate_rank=int(np.linalg.matrix_rank(np.c_[action,action[:,0]]))
    change=np.diag(np.geomspace(.2,5.,2*SUPPORTS))
    transformed=linalg.eigvalsh(change@upper@change,change@g@change)
    basis_error=float(np.max(np.abs(transformed-values))/max(1.,cs2))
    unit=1000.;xu,_,hu,au=shell_scalar(unit*LENGTH,unit*WIDTH,unit*ELL,NX,NY)
    fullu=np.zeros_like(full);fullu[:NX+1]=integrated_p1(xu,p,unit*knots)
    gu,_,_=lower_gram(hu,fullu[au])
    mlu,muu,_=p0_shell_bracket(unit*knots,unit*LENGTH,unit*WIDTH,unit*ELL,modes)
    unitvalues=linalg.eigvalsh(p.T@muu@p,gu)
    unit_error=float(np.max(np.abs(unitvalues-values))/max(1.,cs2))
    predicates=dict(full_scalar_positive=bool(np.all(values>0)),reference_error=error<=1e-3,
        inverse_eigen_symmetry=max(residual,eigen_residual,skew)<=1e-10,
        physical_L2_condition=max(conditions.values())<=1e4,independent_work=endpoint_error<=1e-13,
        checkerboard=checker_defect<=1e-13 and checker_reference>0.,
        duplicate_deficiency=duplicate_rank==2*SUPPORTS,
        basis_and_units=max(basis_error,unit_error)<=1e-10)
    np.savez(output.with_suffix('.npz'),projection=p,scalar_lower=g,reference_lower=lower,
        reference_upper=upper,density_L2_scale=d,scalar_action=action,independent_boundary=exact)
    result=dict(status='REGISTERED_JOINT_REFINEMENT_DEVELOPMENT_BOUND' if all(predicates.values()) else 'UNRESOLVED_JOINT_BOUND',
        identity=IDENTITY,fixture=FIXTURE,coordinate_norm=dict(length_m=LENGTH,width_m=WIDTH,ell_m=ELL),
        physical_ratio=dict(member_shell='1:1',halfcell_shell='2:1',shell_intervals_per_support=4,halfcells_per_support=8),
        predicates=predicates,fixed_criteria=CRITERIA,safe_case_floor=safe,cs_squared_upper=cs2,
        conditions=conditions,reference_preparation=preparation,shell_modes=modes,
        all_scalar_eigenvalues=values.tolist(),physical_L2_basis_scale=np.diag(d).tolist(),
        numerical=dict(reference_error=error,inverse_residual=residual,eigen_residual=eigen_residual,gram_skew=skew,
            endpoint_error=endpoint_error,checker_defect=checker_defect,checker_reference=checker_reference,
            basis_error=basis_error,unit_error=unit_error),
        arrays_sha256=sha(output.with_suffix('.npz')),source_sha256=sha(__file__),helper_sha256=helpers(),
        limitations='new fine scalar bound only; no native beta inspected; not a uniform floor, trend threshold or F03 acceptance')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def assess(root,registration,output,wheel):
    contract=json.loads(registration.read_text())
    if (contract['status']!='REGISTERED_JOINT_REFINEMENT_DEVELOPMENT_BOUND' or contract['identity']!=IDENTITY
        or contract['fixture']!=FIXTURE or contract['fixed_criteria']!=CRITERIA
        or contract['source_sha256']!=sha(__file__) or contract['arrays_sha256']!=sha(registration.with_suffix('.npz'))
        or contract['helper_sha256']!=helpers() or not all(contract['predicates'].values())):
        raise ValueError('unchanged independently eligible fine registration required')
    owner=json.loads((root/'summary.json').read_text())
    if (owner['status']!='COMPLETED_INSTALLED_DEVELOPMENT_ONLY' or owner['identity']!=IDENTITY
        or owner['fixture']!=FIXTURE or not all(owner['predicates'].values())
        or owner['registration']['json_sha256']!=sha(registration) or owner['wheel_sha256']!=sha(wheel)
        or owner['source_sha256']!=sha(Path(owner['source_root'])/'scripts/r8_bl5_projected_joint_owner_dev.py')):
        raise ValueError('completed bound fine installed owner required')
    for name,value in owner['native_input_sha256'].items():
        if sha(root/'native'/name)!=value:raise ValueError('changed actual fine operator input')
    p=projection()
    with np.load(registration.with_suffix('.npz'),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['projection'],p)
        cs2=float(linalg.eigvalsh(z['reference_upper'],z['scalar_lower'])[-1]);rho=np.hypot(.05,.05)/ELL
        product2=max(2*cs2,max(2.,1+2*rho*rho)*max(2*cs2,1+rho*rho))
        if abs(contract['safe_case_floor']-(1/np.sqrt(product2)/np.sqrt(1.001)-1e-6))>1e-12:
            raise ValueError('changed independent scalar floor')
    # This existing generic helper already admits nx32/eight supports. Do not
    # modify any historical/coarse admission or invoke its main/loaded screen.
    from scripts.r8_bl5_projected_density_patch_dev import compile_saved_projection
    data,_,actual,_,c,_,_,beam_work,fixed,_,records,*_=compile_saved_projection(root)
    if c.shape!=(80,2940):raise ValueError('declared fine operator dimensions required')
    for name,module in sys.modules.copy().items():
        if name.split('.')[0]=='anyfesolver' and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve();site=Path(owner['site']).resolve()
            if not path.is_relative_to(site) or sha(path)!=owner['module_sha256'].get(path.relative_to(site).as_posix()):
                raise ValueError('changed installed product import: '+name)
    if (data['layout']['nx'],data['layout']['ny'],len(data['layout']['beam_node_ids'])-1,len(data['policy']['support_endpoints_m'])-1)!=(NX,NY,CELLS,SUPPORTS):
        raise ValueError('fixed fine geometry required')
    if sparse.linalg.norm(actual-c)>1e-12:raise ArithmeticError('actual installed projected action differs')
    xyz=np.asarray(data['xyz_m']);count=(NX+1)*(NY+1)
    expected_shell=np.array([(i/NX,j*WIDTH/NY,0.) for j in range(NY+1) for i in range(NX+1)])
    expected_beam=np.c_[np.linspace(0.,LENGTH,CELLS+1),np.zeros(CELLS+1),np.full(CELLS+1,.0675)]
    if not np.allclose(xyz,np.r_[expected_shell,expected_beam],rtol=0.,atol=1e-13):raise ValueError('changed reference geometry')
    np.testing.assert_allclose(data['policy']['support_endpoints_m'],np.linspace(0.,1.,SUPPORTS+1),rtol=0.,atol=1e-13)
    if data['line']['basis']!='discontinuous_p1_cell_average' or data['line']['multiplier_span']!=8:
        raise ValueError('changed complete projected resolution rule')
    local=records[0][0];frame=local.port.frame;n=frame@local.reference_director
    ab=frame@local.reference_arm;ass=local.thickness*n/2
    np.testing.assert_allclose(ab,(0,0,-.05),rtol=0.,atol=1e-13);np.testing.assert_allclose(ass,(0,0,.0175),rtol=0.,atol=1e-13)
    for op,*_ in records:
        if np.linalg.norm(op.port.lift)>1e-13 or not np.allclose(op.port.frame,frame,rtol=0.,atol=1e-13) or not np.allclose(frame@op.reference_arm,ab,rtol=0.,atol=1e-13):
            raise ValueError('constant frame/arms and zero lift required')
    cols=10*SUPPORTS;scale=np.repeat((1.,1.,1.,ELL,ELL),2*SUPPORTS)
    densities=np.zeros((CELLS,5,cols))
    for channel in range(5):densities[:,channel,2*SUPPORTS*channel:2*SUPPORTS*(channel+1)]=p*scale[2*SUPPORTS*channel]
    fb=np.einsum('ij,cjk->cik',frame,densities[:,:3])
    director=np.einsum('ij,cjk->cik',frame@local.director_basis.T,densities[:,3:])
    couple=np.cross(n,director.transpose(0,2,1)).transpose(0,2,1)
    mb=np.cross(ab,fb.transpose(0,2,1)).transpose(0,2,1)+couple;fs=-fb
    ms=np.cross(ass,fs.transpose(0,2,1)).transpose(0,2,1)-couple
    knots,hb,ba,_=beam_norm(LENGTH,ELL,CELLS,1,ab,.1);xs,_,hs,sa=shell_scalar(LENGTH,WIDTH,ELL,NX,NY)
    shell=np.zeros((5,count,cols))
    for channel in range(5):shell[channel,:NX+1]=integrated_p1(xs,fs[:,channel] if channel<3 else ms[:,channel-3],knots)
    beam_u=np.stack([integrated_p1(knots,fb[:,a],knots) for a in range(3)],axis=1)
    beam=np.concatenate((beam_u.reshape(-1,cols),(mb/CELLS).reshape(-1,cols)))
    full_work=np.concatenate((shell.reshape(-1,cols),beam));physical=data['layout']['counts']['physical_dofs']
    all_fields=np.r_[np.concatenate([6*np.arange(count)+a for a in range(5)]),
        np.concatenate([6*count+6*i+np.arange(3) for i in range(CELLS+1)]),
        np.concatenate([physical+24*member+3*half+np.arange(3) for member in range(NX) for half in (0,1)])]
    active=np.r_[np.concatenate([a*count+sa for a in range(5)]),5*count+ba];slots=all_fields[active]
    if not np.array_equal(np.setdiff1d(all_fields,slots),np.intersect1d(fixed,all_fields)):raise ValueError('physical supports differ')
    omitted=c[:,np.setdiff1d(np.arange(c.shape[1]),all_fields)].data
    omitted_max=float(np.max(np.abs(omitted))) if len(omitted) else 0.
    if omitted_max>1e-13 or np.intersect1d(slots,fixed).size:raise ValueError('quotient/support mapping failed')
    work_error=float(np.max(np.abs(full_work-(-c[:,all_fields].T@sparse.diags(scale)).toarray())))
    if work_error>1e-12:raise ArithmeticError('independent all-column physical work mismatch')
    h=sparse.block_diag((hs,hs,hs,ELL**2*hs,ELL**2*hs,hb),format='csc');g,residual,skew=lower_gram(h,full_work[active])
    sl,su,shell_info=p0_shell_bracket(knots,LENGTH,WIDTH,ELL,contract['shell_modes'])
    shell_lo=np.zeros((cols,cols));shell_hi=np.zeros_like(shell_lo)
    for channel in range(5):
        coef=fs[:,channel] if channel<3 else ms[:,channel-3]/ELL
        shell_lo+=coef.T@sl@coef;shell_hi+=coef.T@su@coef
    preparation=[]
    for subdivision in (1,2,4,8,16,32):
        bl,bu,info=beam_bracket(fb,mb/CELLS,length=LENGTH,ell=ELL,arm=ab,width=.1,subdivisions=subdivision)
        lower,upper=shell_lo+bl,shell_hi+bu;error=float(linalg.eigvalsh(upper-lower,lower)[-1])
        preparation.append(dict(beam=info,total_relative_reference_error=error))
        if error<=1e-3:break
    else:raise ArithmeticError('preset complete physical reference budget exhausted')
    d=np.diag(np.tile(contract['physical_L2_basis_scale'],5));gn,mn=d@g@d,d@upper@d
    values,vectors=linalg.eigh(gn,mn);beta=float(np.sqrt(values[0])) if values[0]>0 else None
    er=float(np.linalg.norm(gn@vectors-(mn@vectors)*values)/np.linalg.norm(gn@vectors))
    conditions=dict(native=float(np.linalg.cond(gn)),reference=float(np.linalg.cond(mn)))
    yaw=beam_work.T[:,5];yaw_norm=float(yaw[slots]@(h@yaw[slots]))
    if yaw_norm<=0. or np.linalg.norm(c@yaw)==0.:raise ArithmeticError('bare yaw omitted from work/norm')
    predicates=dict(all_column_work=work_error<=1e-12,reference_error=error<=1e-3,
        inverse_eigen_symmetry=max(residual,er,skew)<=1e-10,physical_L2_condition=max(conditions.values())<=1e4,
        full_spectrum_positive=bool(np.all(values>0)),independent_scalar_floor=beta is not None and beta>=contract['safe_case_floor'],
        variational_ceiling=bool(values[-1]<=CRITERIA['variational_ceiling']))
    np.savez(output.with_suffix('.npz'),native_lower=g,reference_lower=lower,reference_upper=upper,
        physical_density=densities,all_column_work=full_work,norm_action=full_work[active],slots=slots,physical_L2_scale=d)
    result=dict(status='COMPLETED_JOINT_DEVELOPMENT_NORM' if all(predicates.values()) else 'UNRESOLVED_JOINT_NORM',
        identity=IDENTITY,fixture=FIXTURE,predicates=predicates,beta=beta,safe_case_floor=contract['safe_case_floor'],
        conditions=conditions,numerical=dict(all_column_work=work_error,inverse_residual=residual,eigen_residual=er,
            skew=skew,reference_error=error,omitted_work=omitted_max,bare_yaw_positive_norm=yaw_norm),
        all_generalized_eigenvalues=values.tolist(),reference_preparation=preparation,shell_reference=shell_info,
        registration_sha256=sha(registration),source_sha256=sha(__file__),helper_sha256=helpers()|
            {Path(compile_saved_projection.__code__.co_filename).name:sha(compile_saved_projection.__code__.co_filename)},
        input_sha256={name:sha(root/'native'/name) for name in ('inputs.json','operators.npz')},wheel_sha256=sha(wheel),
        imported_anyfesolver=sys.modules['anyfesolver'].__file__,
        limitations='ONE joint32 development level; no interlacing/coarseGram reuse, mechanical kernel claim, uniform stability or formalF03')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('register','assess'))
    for arg in ('root','registration','wheel'):p.add_argument('--'+arg,type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.npz').exists():raise FileExistsError(a.output)
    if not threadpool_info() or any(v['num_threads']!=1 for v in threadpool_info()):raise RuntimeError('effective one-thread numerical pools required')
    r=register(a.output) if a.stage=='register' else assess(a.root,a.registration,a.output,a.wheel)
    print(json.dumps({k:r[k] for k in ('status','safe_case_floor','conditions')}|({'beta':r['beta']} if 'beta' in r else {'cs_squared_upper':r['cs_squared_upper']})))
    raise SystemExit(0 if all(r['predicates'].values()) else 1)
