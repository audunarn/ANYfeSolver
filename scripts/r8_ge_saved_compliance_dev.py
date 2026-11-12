"""One saved general two-RHS compliance diagnostic; no model or mechanics evaluation."""
from pathlib import Path
import argparse,ast,json,math,os,struct,sys,warnings,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import r8_ge_paired_fixed_diagnostic_dev as previous
DERIVATIVE=previous.DOC/'derivative-v1'
DOC=previous.DOC/'saved-compliance-v1'
INPUT=DOC/'COMPLIANCE_INPUT.json'
PLAN=dict(evaluation=14,generation=11,bordered_solve=5,control_dof=6,
         general_service='scipy.sparse.linalg.spsolve',rhs_columns=['original_p_f','original_R_f'],
         source_cap_s=20,installed_cap_s=60,budget_s=180,reserve_s=60,
         assembly=False,material_update=False,equilibrium=False,new_direction=False,
         conditioning='two-RHS directional lower bounds only',near_zero_c='diagnostic only')
SOURCE_NAMES=('test_source_saved_identity','test_source_general_sign_and_gap',
 'test_source_unresolved_zero_and_nonfinite','test_source_component_backward_error',
 'test_source_actual_service_once_and_failure','test_source_dimensionless_scaling',
 'test_source_inventory_no_inverse_or_mechanics','test_source_adoption_and_layout_refusal')
def sha(path):return previous.sha(path)
def write(path,value):previous.write(path,value)
def arrays(path):
    result={}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            raw=archive.read(name)
            if raw[:6]!=b'\x93NUMPY':raise ValueError('exact saved array framing required')
            offset=10 if raw[6]==1 else 12
            size=struct.unpack_from('<H' if raw[6]==1 else '<I',raw,8)[0]
            header=ast.literal_eval(raw[offset:offset+size].decode('latin1'))
            if header['fortran_order']:raise ValueError('actual saved C-order arrays required')
            fmt={'<f8':'d','<i8':'q','<i4':'i','<u8':'Q'}[header['descr']]
            result[name[:-4]]=list(struct.unpack('<'+str(math.prod(header['shape']))+fmt,raw[offset+size:]))
    return result
def require_adoption(value):
    if value.get('status')!='ADOPTED' or value.get('plan')!=PLAN:raise ValueError('exact root mathematical/resource adoption required')
def validate(value):
    free=value['free'];n=len(value['R'])
    if n!=132 or len(free)!=120 or len(set(free))!=len(free) or any(type(i) is not int or not 0<=i<n for i in free):raise ValueError('original complete free-row map required')
    if value['control_dof']!=6 or 6 not in free or value['K']['shape']!=[n,n]:raise ValueError('original controlled row/operator shape required')
    if len(value['p'])!=n or len(value['W'])!=n or len(value['S'])!=n:raise ValueError('complete original work and column scales required')
    if value['gradient']!=[float(i==6) for i in range(n)]:raise ValueError('exact stored control gradient required')
    if not all(math.isfinite(x) and x>0 for x in value['W']+value['S']):raise ValueError('positive finite frozen scales required')
    if not all(math.isfinite(x) for x in value['R']+value['p']+[value['g']]):raise ValueError('finite original RHS/gap required')
    matrix=value['K']
    if not len(matrix['data'])==len(matrix['row'])==len(matrix['col']) or not all(math.isfinite(x) for x in matrix['data']):raise ValueError('finite complete saved sparse entries required')
    if any(type(i) is not int or not 0<=i<n for i in matrix['row']+matrix['col']):raise ValueError('saved sparse index outside original layout')
    roles=[i for group in value['roles'].values() for i in group]
    if sorted(roles)!=sorted(free):raise ValueError('complete disjoint physical/retained/POINT free-role layout required')
def input_data():
    identity=json.loads((DOC/'COMPLIANCE_INPUT_IDENTITY.json').read_text())
    if sha(INPUT)!=identity['sha256']:raise ValueError('saved matrix input drift')
    value=json.loads(INPUT.read_text());validate(value)
    for name,h in value['source_input_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('saved source evidence drift: '+name)
    return value
def derive(z,y,g,index):
    if len(z)!=len(y) or not 0<=index<len(z):raise ValueError('complete matching two-RHS solution required')
    c=float(z[index]);b=float(y[index])
    if not math.isfinite(c) or not math.isfinite(b):return dict(status='UNRESOLVED_NONFINITE_PROJECTION',c=c,b=b)
    if not math.isfinite(g):return dict(status='UNRESOLVED_NONFINITE_GAP',c=c,b=b)
    try:numerator=math.fsum((b,-g))
    except OverflowError:return dict(status='UNRESOLVED_NONREPRESENTABLE_NUMERATOR',c=c,b=b)
    if c==0.:return dict(status='UNRESOLVED_ZERO_COMPLIANCE',c=c,b=b,numerator=numerator)
    amplitude=numerator/c
    if not math.isfinite(amplitude):return dict(status='UNRESOLVED_NONREPRESENTABLE_AMPLITUDE',c=c,b=b,numerator=numerator)
    try:step=[math.fsum((float(a)*amplitude,-float(v))) for a,v in zip(z,y)]
    except (OverflowError,ValueError):return dict(status='UNRESOLVED_NONREPRESENTABLE_DIRECTION',c=c,b=b,numerator=numerator)
    if not all(math.isfinite(v) for v in step):return dict(status='UNRESOLVED_NONREPRESENTABLE_DIRECTION',c=c,b=b,numerator=numerator)
    return dict(status='FINITE_COMPLIANCE_CAPTURED',c=c,b=b,numerator=numerator,amplitude=amplitude,direction=step)
def componentwise_backward(residual,magnitude):
    if len(residual)!=len(magnitude):raise ValueError('complete componentwise denominator required')
    values=[]
    for r,m in zip(residual,magnitude):
        if m==0:
            if r!=0:return dict(status='UNRESOLVED_ZERO_ROW_SCALE',maximum=None,ratios=values)
            values.append(0.)
        else:values.append(abs(r)/m)
    return dict(status='OBSERVED',maximum=max(values,default=0.),ratios=values,
                meaning='standard componentwise |Ax-b|/(|A||x|+|b|), no denominator floor or acceptance cutoff')
def scaled_norms(x,b,W,S,matrix_norm1):
    if not len(x)==len(b)==len(W)==len(S):raise ValueError('complete frozen row/column scales required')
    xn=math.fsum(abs(a/s) for a,s in zip(x,S));bn=math.fsum(abs(w*a) for w,a in zip(W,b))
    return dict(scaled_solution_norm1=xn,scaled_rhs_norm1=bn,
        directional_condition_lower_bound=None if bn==0 else matrix_norm1*xn/bn,
        interpretation='observed two-RHS directional condition indicator; lower bound conditional on accurate solutions (exact-arithmetic exact-solution inequality), never certified from approximate x alone or an inverse upper bound')
def service_once(solver,matrix,rhs):
    # Exactly one call with two original RHS, never a global inverse or probe.
    return solver(matrix,rhs)
def prepare():
    if INPUT.exists():raise ValueError('preparation already frozen')
    DOC.mkdir(parents=True,exist_ok=True)
    original=json.loads((DERIVATIVE/'DERIVATIVE_INPUT.json').read_text())
    work=arrays(DERIVATIVE/'attempt-installed-derivative1/actual-direction-work-map.npz')
    energy=work['energy'][0];W=work['weights'];S=[energy*x for x in W]
    # Exact saved border RHS final entry is -g. Never reset a small gap to zero.
    solve=original['solve'];evaluation=original['evaluation']
    evidence=[DERIVATIVE/'DERIVATIVE_INPUT.json',DERIVATIVE/'attempt-installed-derivative1/actual-direction-work-map.npz',DERIVATIVE/'DERIVATIVE_TERMINAL_SELECTED.json']
    value=dict(K=evaluation['tangent'],R=evaluation['residual'],p=solve['direction'],free=solve['free'],
        gradient=solve['gradient'],control_dof=6,g=-solve['rhs'][-1],W=W,S=S,energy=energy,
        coordinate_scale_reconstruction='S=E*W, frozen binary64 column scales from exact saved corrector work map; primal coordinates plus E/constraint_scale multiplier scales, not asserted bit-identical to pre-division owner coordinate_scale',
        saved_direction=solve['full_correction'],saved_amplitude_delta=solve['amplitude_delta'],
        original_frames_and_material_trial=evaluation['state'],
        roles={k:v for k,v in work.items() if k not in ('direction','amplitude_delta','external','action','magnitude','weights','energy','free')},
        source_input_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in evidence})
    validate(value);write(INPUT,value);write(DOC/'COMPLIANCE_INPUT_IDENTITY.json',dict(sha256=sha(INPUT)))
    write(DOC/'COMPLIANCE_PLAN.json',PLAN)
def connected(out):
    import numpy as np
    from scipy import sparse
    # The frozen installed corrector's actual general sparse service.
    from anyfesolver._stiffened_shell_displacement_corrector import spla
    if out!=DOC/'attempt-installed-compliance1':raise ValueError('sole exact saved compliance attempt required')
    previous.artifact();v=input_data();report=dict(status='STARTED',scientific_acceptance=False,plan=PLAN)
    try:
        matrix=v['K'];K=sparse.coo_matrix((matrix['data'],(matrix['row'],matrix['col'])),shape=matrix['shape']).tocsc()
        free=np.asarray(v['free']);Kff=K[free,:][:,free].tocsc()
        rhs=np.column_stack((np.asarray(v['p'])[free],np.asarray(v['R'])[free]))
        W=np.asarray(v['W'])[free];S=np.asarray(v['S'])[free];index=int(np.flatnonzero(free==6)[0])
        report.update(free=free,control_free_index=index,stored_gap=v['g'],stored_amplitude=v['saved_amplitude_delta'],
                      service=dict(module=spla.spsolve.__module__,name=spla.spsolve.__name__,calls=1,rhs_columns=2,
                                   interpretation='general original uncondensed free mixed system; no SPD assumption'))
        write(out/'compliance.json',report)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',spla.MatrixRankWarning)
                solution=service_once(spla.spsolve,Kff,rhs)
        except spla.MatrixRankWarning as error:
            report.update(status='UNRESOLVED_SINGULAR_FREE_SUBSYSTEM',error=str(error),
                limitation='Kff singularity does not reject a valid original bordered controlled system')
            write(out/'compliance.json',report);return report
        if solution.shape!=rhs.shape or not np.isfinite(solution).all():
            report.update(status='UNRESOLVED_NONFINITE_GENERAL_SOLUTION');write(out/'compliance.json',report);return report
        z,y=solution[:,0],solution[:,1]
        residual=Kff@solution-rhs;magnitude=abs(Kff)@abs(solution)+abs(rhs)
        np.savez(out/'solutions-equation-observations.npz',free=free,z=z,y=y,rhs=rhs,residual=residual,magnitude=magnitude,W=W,S=S)
        if not np.isfinite(residual).all() or not np.isfinite(magnitude).all():
            report.update(status='UNRESOLVED_NONREPRESENTABLE_EQUATION_OBSERVATION');write(out/'compliance.json',report);return report
        scaled=sparse.diags(W)@Kff@sparse.diags(S)
        scaled_matrix_norm1=float(np.max(np.asarray(abs(scaled).sum(axis=0)).ravel()))
        observations=[]
        for j,name in enumerate(('original_load_direction','current_residual')):
            x=solution[:,j];b=rhs[:,j];r=residual[:,j]
            rhs_work=float(np.linalg.norm(W*b));error_work=float(np.linalg.norm(W*r))
            observations.append(dict(rhs=name,work_rhs_norm=rhs_work,work_equation_error_absolute=error_work,
                work_equation_error_relative=None if rhs_work==0 else error_work/rhs_work,
                componentwise_backward=componentwise_backward(r.tolist(),magnitude[:,j].tolist()),
                scaled_matrix_norm1=scaled_matrix_norm1,**scaled_norms(x.tolist(),b.tolist(),W.tolist(),S.tolist(),scaled_matrix_norm1)))
        result=derive(z,y,v['g'],index);report.update(result)
        zscale=float(np.linalg.norm(z/S,1));control_scale=float(S[index]);cscaled=abs(float(z[index])/control_scale)
        report.update(rhs_observations=observations,c_units='m per dimensionless coefficient of original500N-x/20N-z pattern',
            b_units='m',amplitude_units='dimensionless',scaled_control_compliance=cscaled,
            scaled_load_response_norm1=zscale,tiny_c_machine_resolution_indicator=bool(cscaled<=np.finfo(float).eps*zscale),
            tiny_c_interpretation='epsilon times solution vector1norm is an observational indicator only; no clipping/floor, no assertion c accurate from backward error',
            sign_formula='K*d-p*da=-R; e6*d=-g => da=(e6*Kff^-1*R-g)/(e6*Kff^-1*p)',
            original_bordered_equation_was_observed_solved=True,
            scope='this free-system diagnostic does not certify original bordered forward accuracy or tangent correctness')
        if result['status']=='FINITE_COMPLIANCE_CAPTURED':
            d=np.asarray(result['direction']);a=result['amplitude'];saved_d=np.asarray(v['saved_direction'])[free]
            recon=Kff@d-rhs[:,0]*a+rhs[:,1];control=float(d[index]+v['g'])
            compare=d-saved_d
            report.update(amplitude_difference=a-v['saved_amplitude_delta'],control_equation_error=control,
                reconstructed_equation_work_absolute=float(np.linalg.norm(W*recon)),
                direction_difference_scaled_norm1=float(np.linalg.norm(compare/S,1)),
                direction_difference_scaled_max=float(np.max(abs(compare/S))),
                reconstruction_componentwise_backward=componentwise_backward(recon.tolist(),(abs(Kff)@abs(d)+abs(rhs[:,0]*a)+abs(rhs[:,1])).tolist()),
                control_cancellation_magnitude=float(abs(z[index]*a)+abs(y[index])+abs(v['g'])),
                inferred_mechanism='interpret c,b,numerator and observed conditioning together; no physical defect or whole tangent correctness inferred')
        write(out/'compliance.json',report);return report
    except BaseException as error:
        report.update(status='FAILED_UNEXPECTED',error=dict(type=type(error).__name__,message=str(error)))
        write(out/'compliance.json',report);raise
def bindings():
    paths={ROOT/name for name in previous.bindings()}
    paths.update((Path(__file__).resolve(),ROOT/'tests/test_r8_ge_saved_compliance_dev.py',INPUT,
        DOC/'COMPLIANCE_INPUT_IDENTITY.json',DOC/'COMPLIANCE_PLAN.json',DOC/'COMPLIANCE_READINESS.json',DOC/'COMPLIANCE_PREPARATION.md',DOC/'prepare_compliance_readiness.py'))
    for name in json.loads(INPUT.read_text())['source_input_sha256']:paths.add(ROOT/name)
    if len(paths)!=len({path.name for path in paths}):raise ValueError('snapshot basename collision')
    return {path.relative_to(ROOT).as_posix():sha(path) for path in sorted(paths)}
def child(out,stage):
    import psutil
    psutil.Process().cpu_affinity([0]);sys.path[:0]=[str(previous.PACKAGE/'site'),str(ROOT/'tests')]
    if json.loads((out/'bindings.json').read_text())!=bindings():raise ValueError('child input/source drift')
    os.environ.update(GE_SAVED_COMPLIANCE_STAGE=stage,GE_SAVED_COMPLIANCE_OUTPUT=str(out))
    from scripts.r8_f06_log_domain_check_dev import runtime
    runtime(out,'runtime-before',previous.PACKAGE/'site')
    import pytest,xml.etree.ElementTree as ET
    names=SOURCE_NAMES if stage=='source-controls' else ('test_installed_one_saved_general_pair',)
    try:
        code=int(pytest.main([str(ROOT/'tests/test_r8_ge_saved_compliance_dev.py')+'::'+name for name in names]+['-q','-o','pythonpath=','-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'tmp')]))
        cases=list(ET.parse(out/'junit.xml').iter('testcase'))
        counts=dict(tests=len(cases),errors=sum(len(list(x.iter('error'))) for x in cases),failures=sum(len(list(x.iter('failure'))) for x in cases),skipped=sum(len(list(x.iter('skipped'))) for x in cases))
        write(out/'counts.json',counts)
        if len(cases)!=len(names) or counts['skipped']:raise ValueError('exact no-skip inventory required')
        return code
    finally:
        rt=runtime(out,'runtime-after',previous.PACKAGE/'site');tree=previous.artifact()
        for path in rt['imports'].values():
            if sha(path)!=tree['src/'+Path(path).relative_to(previous.PACKAGE/'site').as_posix()]:raise ValueError('installed import drift')
def launch(stage):
    from scripts.r8_f06_fibre_log_domain_dev import run
    path=DOC/('COMPLIANCE_'+('SOURCE' if stage=='source-controls' else 'INSTALLED')+'_AUTHORITY.json')
    captured=path.read_bytes();grant=json.loads(captured);slot=DOC/'COMPLIANCE_SLOT_TRANSFER.json'
    require_adoption(json.loads((DOC/'COMPLIANCE_ADOPTION.json').read_text()))
    if grant.get('status')!='AUTHORIZED' or json.loads(slot.read_text()).get('status')!='TRANSFERRED':raise ValueError('fresh source/installed authority and slot required')
    if grant.get('bindings_sha256')!=sha(DOC/'COMPLIANCE_BINDINGS.json') or grant.get('slot_sha256')!=sha(slot) or grant.get('adoption_sha256')!=sha(DOC/'COMPLIANCE_ADOPTION.json'):raise ValueError('exact adopted input/source/slot identities required')
    if (grant.get('budget_s'),grant.get('reserve_s'),grant.get('process_cap_s'))!=(180,60,20 if stage=='source-controls' else 60):raise ValueError('fresh resource envelope required')
    if list(DOC.glob('attempt-'+stage+'*')):raise ValueError('sole attempt consumed')
    if stage=='installed-compliance' and (json.loads((DOC/'attempt-source-controls1/counts.json').read_text())!=dict(tests=8,errors=0,failures=0,skipped=0) or json.loads((DOC/'attempt-source-controls1/receipt.json').read_text())['status']!='PASS'):raise ValueError('exact8 sourcePASS required')
    try:return run(stage,doc=DOC,entry=__file__,bind=bindings,authority_file=path.name)
    finally:
        out=DOC/('attempt-'+stage+'1')
        if out.exists():
            (out/'used-authority.json').write_bytes(captured)
            if stage=='installed-compliance' or json.loads((out/'receipt.json').read_text())['status']!='PASS':
                lp=DOC/'ledger.json';v=json.loads(lp.read_text());v.update(status='CLOSED',unused_discarded_s=v['budget_s']-v['used_s']);write(lp,v)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--child',action='store_true');p.add_argument('--out');p.add_argument('--stage',choices=('source-controls','installed-compliance'));a=p.parse_args()
    if a.prepare:prepare()
    else:raise SystemExit(child(Path(a.out),a.stage) if a.child else launch(a.stage))
