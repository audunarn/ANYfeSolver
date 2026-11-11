"""Current installed40x16 Fz10/My1 producer; no automatic solver execution."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from decimal import Decimal,localcontext
from pathlib import Path
import sys
import zipfile
import numpy as np
from scripts.r8_e02_native_rigid_tip import geometry
from scripts.r8_e02_rigid_acceptance import identity,measure,registration,matched_port_work
from scripts.r8_e02_rigid_precision import (
    build_precision_deck,_validate_candidate,_ENERGY,_HEADER,
    root_elimination_proof,root_wrench,ELIMINATED,
)
from scripts.r8_e02_double_output import read_double_output
from scripts.r8_e02_rigid_tip_reference import CCX_SHA256,ReferenceParseError,_printed_number,parse_result
from scripts.r8_e02_rigid_torque_fields import WHEEL_SHA

ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'docs/agent_plans/development/r8-b2-work-guard-installed-20261009/site'
CCX=Path('C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/calculix-223/CalculiX-2.23.0-win-x64/bin/ccx.exe')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
def admitted(load):
    if load not in ('Fz','My'):raise ValueError('only current40/+offset/+Fz10N or+My1Nm authorized')

def binding():
    b=json.loads((SITE.parent/'installed-v2/binding.json').read_text())
    wheel=SITE.parent/'dist/anysolver-0.4.7-py3-none-any.whl'
    if sha(wheel)!=WHEEL_SHA or sha(CCX)!=CCX_SHA256:raise ValueError('pinned executable/wheel changed')
    with zipfile.ZipFile(wheel) as z:
        for rel,value in b['module_sha256'].items():
            if sha(SITE/rel)!=value or hashlib.sha256(z.read(rel)).hexdigest()!=value:
                raise ValueError('installed source/wheel module changed:'+rel)
    return b,dict(native_source=identity(dict(head=b['source_head'],modules=b['module_sha256'])),
        native_wheel=WHEEL_SHA,native_site_manifest=identity(b['module_sha256']),external_executable=CCX_SHA256)

def prepare(load,path):
    admitted(load);path=Path(path);path.mkdir(parents=True,exist_ok=False)
    b,a=binding();c=registration(load,40,16,2,a)
    deck,m=build_precision_deck(40,16,1,load,nz=2)
    save(path/'contract.json',c);save(path/'native-input.json',c['model']);save(path/'candidate.json',m)
    (path/'reference.inp').write_bytes(deck.encode('ascii'))
    save(path/'artifact-binding.json',dict(artifacts=a,installed_source_head=b['source_head'],module_sha256=b['module_sha256']))
    return c,deck,m,a

def staged(load,path):
    admitted(load);path=Path(path);b,a=binding()
    c=json.loads((path/'contract.json').read_text());deck=(path/'reference.inp').read_text();m=json.loads((path/'candidate.json').read_text())
    expected=registration(load,40,16,2,a);text,manifest=build_precision_deck(40,16,1,load,nz=2)
    if c!=json.loads(json.dumps(expected)) or deck!=text or m!=json.loads(json.dumps(manifest)):
        raise ValueError('staged current principal candidate differs from admitted bytes')
    native_input=json.loads((path/'native-input.json').read_text())
    if identity(native_input)!=c['native_input_sha256']:raise ValueError('staged native construction changed')
    return c,deck,m,a

def observation(contract,domain):
    return dict(registration_sha256=contract['registration_sha256'],status='completed',domain=domain,
        reaction_provenance='direct-zero-SPC-physical-root',precision='binary64_physical_fields',
        frame_rotation=np.eye(3).tolist(),frame_origin_SI=[0.,0.,0.],units=dict(length_to_m=1.,force_to_N=1.),
        applied_wrench=[contract['wrench_SI']])

def ordinary_rigid_observation(points,displacements,port,force,couple):
    """Unadjudicated binary64 diagnostics of the historical centered equations."""
    p=np.asarray(points,float);u=np.asarray(displacements,float);port=np.asarray(port,float)
    force=np.asarray(force,float);couple=np.asarray(couple,float)
    center=p.mean(axis=0);mean=u.mean(axis=0);arms=p-center
    inertia=np.eye(3)*np.sum(arms*arms)-arms.T@arms
    theta=np.linalg.solve(inertia,np.sum(np.cross(arms,u-mean),axis=0))
    translation=mean+np.cross(theta,port-center)
    dual=np.linalg.solve(inertia,couple+np.cross(port-center,force))
    nodal=force/len(p)+np.cross(dual,arms)
    resultant=nodal.sum(axis=0);moment=np.cross(p-port,nodal).sum(axis=0)
    pairing=math.fsum(float(a*b) for a,b in zip(nodal.flat,u.flat));work=float(force@translation+couple@theta)
    limits=[1e-12*max(1.,np.linalg.norm(force)),1e-12*max(1.,np.linalg.norm(couple)),1e-12*max(abs(pairing),abs(work),1e-100)]
    errors=[float(np.max(np.abs(resultant-force))),float(np.max(np.abs(moment-couple))),abs(pairing-work)]
    return dict(translation_m=translation.tolist(),theta_rad=theta.tolist(),
        rigid_mapping_max_gap_m=float(np.max(np.abs(mean+np.cross(theta,arms)-u))),
        external_final_pairing_J=pairing,port_pairing_J=work,load_resultant_N=resultant.tolist(),
        load_moment_about_port_Nm=moment.tolist(),dual_errors=errors,dual_limits=limits,
        dual_checks_passed=[bool(a<=b) for a,b in zip(errors,limits)],arithmetic='ordinary binary64 diagnostic only')

def accurate_rigid_fit(points,displacements,port,force=None,couple=None):
    """Same centered rigid equations, Decimal80 arithmetic on exact input floats."""
    p=np.asarray(points,dtype=float);u=np.asarray(displacements,dtype=float);port=np.asarray(port,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or p.shape!=u.shape or len(p)<3 or port.shape!=(3,) or not all(np.isfinite(x).all() for x in (p,u,port)):
        raise ReferenceParseError('complete finite physical rigid-fit inputs required')
    with localcontext() as ctx:
        ctx.prec=80;D=lambda x:Decimal.from_float(float(x))
        P=[[D(x) for x in row] for row in p];U=[[D(x) for x in row] for row in u];n=Decimal(len(P))
        center=[sum(x[j] for x in P)/n for j in range(3)];mean=[sum(x[j] for x in U)/n for j in range(3)]
        X=[[x[j]-center[j] for j in range(3)] for x in P];V=[[x[j]-mean[j] for j in range(3)] for x in U]
        cross=lambda x,y:[x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]]
        trace=sum(sum(t*t for t in x) for x in X)
        A=[[((trace if i==j else Decimal(0))-sum(x[i]*x[j] for x in X)) for j in range(3)] for i in range(3)]
        b=[sum(cross(x,y)[j] for x,y in zip(X,V)) for j in range(3)]
        def solve(rhs):
            matrix=[row[:] for row in A];rhs=rhs[:]
            for i in range(3):
                pivot=matrix[i][i]
                if pivot<=0:raise ReferenceParseError('degenerate physical rigid-fit geometry')
                for j in range(i,3):matrix[i][j]/=pivot
                rhs[i]/=pivot
                for k in range(3):
                    if k==i:continue
                    multiplier=matrix[k][i]
                    for j in range(i,3):matrix[k][j]-=multiplier*matrix[i][j]
                    rhs[k]-=multiplier*rhs[i]
            return rhs
        b=solve(b)
        correction=cross(b,[D(port[j])-center[j] for j in range(3)])
        T=[mean[j]+correction[j] for j in range(3)]
        translation=np.array([float(x) for x in T]);theta=np.array([float(x) for x in b]);dual_record={}
        if (force is None)!=(couple is None):raise ReferenceParseError('complete declared force/couple required')
        if force is not None:
            if any(np.asarray(x).shape!=(3,) or not np.isfinite(x).all() for x in (force,couple)):
                raise ReferenceParseError('finite declared force/couple required')
            F=[D(x) for x in force];M=[D(x) for x in couple];offset=[D(port[j])-center[j] for j in range(3)]
            lever=cross(offset,F);dual=solve([M[j]+lever[j] for j in range(3)])
            nodal=[[F[j]/n+cross(dual,x)[j] for j in range(3)] for x in X]
            resultant=[sum(f[j] for f in nodal) for j in range(3)]
            moments=[cross([p[j]-D(port[j]) for j in range(3)],f) for p,f in zip(P,nodal)]
            moment=[sum(m[j] for m in moments) for j in range(3)]
            pairing=sum(sum(f[j]*u[j] for j in range(3)) for f,u in zip(nodal,U))
            work=sum(F[j]*T[j]+M[j]*b[j] for j in range(3))
            errors=[max(abs(resultant[j]-F[j]) for j in range(3)),max(abs(moment[j]-M[j]) for j in range(3)),abs(pairing-work)]
            limits=[D(1e-12*max(1.,np.linalg.norm(force))),D(1e-12*max(1.,np.linalg.norm(couple))),D(1e-12)*max(abs(pairing),abs(work),D(1e-100))]
            if any(e>limit for e,limit in zip(errors,limits)):raise ReferenceParseError('force/moment/work dual map failed')
            dual_record=dict(external_final_pairing_J=float(pairing),port_pairing_J=float(work),
                load_resultant_N=[float(x) for x in resultant],load_moment_about_port_Nm=[float(x) for x in moment],
                dual_errors=[str(x) for x in errors],dual_limits=[str(x) for x in limits],dual_checks_passed=[True]*3,
                dual_action_provenance='same declared-load transpose map; not recovered interface traction')
    gap=float(np.max(np.abs(u-translation-np.cross(theta,p-port))))
    if gap>max(128*math.ulp(float(np.max(np.abs(u)))),1e-8*float(np.max(np.abs(u)))):
        raise ReferenceParseError('physical patch is not a rigid displacement map')
    return dict(translation_m=translation.tolist(),theta_rad=theta.tolist(),rigid_mapping_max_gap_m=gap,**dual_record,
        arithmetic='Decimal80 centered fit of exact binary64 U and reference coordinates')

def parse_current_solid(data,dat,stdout,deck,manifest,*,principal_observation=False):
    """D6-private literal D2 structural checks with adopted accurate pose fit.

    The original six print intervals/64ulp allowance are unchanged. Ordinary
    fit diagnostics remain recorded; old helper/failed execution are untouched.
    """
    base=_validate_candidate(deck,manifest);proof=root_elimination_proof(base)
    if manifest['variant']!=ELIMINATED or (base['nx'],base['ny'],base['nz'],base['sign'])!=(40,16,2,1) or base['load_name'] not in ('Fz','My'):
        raise ReferenceParseError('D6 fixed-zero-root40/nz2 +Fz/My scope required')
    headers=list(_ENERGY.finditer(dat))
    if len(headers)!=1:raise ReferenceParseError('missing/duplicate/unsupported integrated ELSE dataset')
    header=headers[0]
    if _printed_number(header[1])[0]!=1.:raise ReferenceParseError('ELSE is not at completed load time')
    end=min((h.start() for h in _HEADER.finditer(dat) if h.start()>=header.end()),default=len(dat));rows={}
    for row in dat[header.end():end].splitlines():
        if not row.strip():continue
        fields=row.split()
        if len(fields)!=2 or not re.fullmatch(r'[1-9]\d*',fields[0]):raise ReferenceParseError('unsupported or malformed ELSE row')
        e=int(fields[0])
        if e in rows:raise ReferenceParseError('duplicate ELSE element')
        value,bound,_=_printed_number(fields[1])
        if value<0:raise ReferenceParseError('negative elastic internal energy')
        rows[e]=(value,bound)
    if set(rows)!={int(e) for e in base['elements']}:raise ReferenceParseError('incomplete/unexpected integrated energy elements')
    printed=parse_result(dat[:header.start()]+dat[end:],stdout,base);u,rf=read_double_output(data,base)
    physical={int(n):p for n,p in base['physical_nodes'].items()};ids=base['patch_nodes']
    if set(u)!=set(physical):raise ReferenceParseError('missing/extra physical displacement data')
    controls={int(n):p for n,p in base['control_nodes'].items()};port=controls[base['controls']['patch_ref']]
    args=([physical[n] for n in ids],[u[n] for n in ids],port,base['applied_force_N'],base['applied_moment_Nm'])
    ordinary=ordinary_rigid_observation(*args);accurate=accurate_rigid_fit(*args)
    comparisons=[]
    for key,pk,bk,labels in (('translation_m','patch_ref_displacement_m','patch_ref_displacement_rounding_bound_m',('Ux','Uy','Uz')),
        ('theta_rad','patch_rot_theta_rad','patch_rot_theta_rounding_bound_rad',('thetaX','thetaY','thetaZ'))):
        for label,x,old,y,bound in zip(labels,accurate[key],ordinary[key],printed[pk],printed[bk]):
            allowance=bound+64*max(math.ulp(x),math.ulp(y));oldallowance=bound+64*max(math.ulp(old),math.ulp(y))
            row=dict(component=label,accurate=x,ordinary=old,printed_center=y,half_print_unit=bound,
                accurate_difference=x-y,ordinary_difference=old-y,allowance=allowance,
                accurate_pass=abs(x-y)<=allowance,ordinary_pass=abs(old-y)<=oldallowance)
            comparisons.append(row)
    print_passed=all(x['accurate_pass'] for x in comparisons)
    if not print_passed:
        failed=[x['component'] for x in comparisons if not x['accurate_pass']]
        if not (principal_observation is True and base['load_name']=='My' and failed==['thetaX']):
            raise ReferenceParseError('accurate physical port differs from original printed control rounding')
        # Prospective separation was adopted only for this completed saved pair.
        # These bind observations; they do not fit a tolerance to the response.
        if (hashlib.sha256(data).hexdigest()!='656e66491128432f2ec9d9d3fa08663f3e7a686b29674a3ac0a9f8a08cd1617a'
            or hashlib.sha256(dat.encode('utf-8')).hexdigest()!='bffc710d59679794e63b805a55821474e5ad1a2463671ab3ed1a81805a16c5ed'):
            raise ReferenceParseError('principal print separation requires exact adopted saved My observations')
    pose=dict(ordinary,**accurate)
    wrench=root_wrench(base,rf);expected=tuple(base['expected_root_force_N'])+tuple(base['expected_root_moment_Nm'])
    return dict(status='D6_PARSED_PHYSICAL_OBSERVATIONS',acceptance=False,port=pose,ordinary_port=ordinary,
        six_component_print_comparison=comparisons,ordinary_strict_comparison_passed=all(x['ordinary_pass'] for x in comparisons),
        supporting_print_interval_status='PASSED' if print_passed else 'FAILED',
        principal_observation_route=bool(principal_observation),
        physical_admissibility_does_not_certify_supporting_print_intervals=bool(principal_observation and not print_passed),
        root_elimination_proof=proof,root_binary_field_is_reaction=True,root_binary_field_wrench=wrench,
        root_wrench_residual=[x-y for x,y in zip(wrench,expected)],root_displacement_max_m=max(abs(v) for n in base['root_nodes'] for v in u[n]),
        independent_integrated_internal_energy_J=math.fsum(v for v,b in rows.values()),energy_print_rounding_bound_J=math.fsum(b for v,b in rows.values()),
        can_certify_energy_work_1e_minus_8=False,paired_printed_control_result=printed,
        provenance='D6-only accurate reconstruction; historical strict helper/failure and print intervals unchanged')

def native(load,path):
    c,deck,m,a=staged(load,path);path=Path(path)
    sys.path.insert(0,str(SITE))
    import anyfesolver,os,psutil
    from anyfesolver import b3_ge,stiffened_shell as api
    from threadpoolctl import threadpool_info
    if not Path(anyfesolver.__file__).resolve().is_relative_to(SITE):raise RuntimeError('mixed source/site import')
    pools=threadpool_info()
    if psutil.Process().cpu_affinity()!=[0] or os.environ.get('NUMBA_DISABLE_JIT')!='1' or any(p['num_threads']!=1 for p in pools):
        raise RuntimeError('effective resource configuration differs')
    save(path/'native-runtime-preflight.json',dict(python=sys.version,pools=pools,affinity=[0],jit_disabled=True,artifacts=a))
    nodes,cells,patch,selected,root=geometry(40,16);e,nu,t=210e9,.3,.02;g=e/(2*(1+nu))
    section=b3_ge.EllipsoidalGeneralizedSection(np.diag((e*.0004,g*.0004*5/6,g*.0004*5/6,
        g*2.249e-8,e*1.3333333333333335e-8,e*1.3333333333333335e-8)),np.eye(6),1e20,1e9)
    member=b3_ge.define_beam('b3-ge',1,(100001,100002,100003),
        np.array(c['model']['beam_positions']),np.array(c['model']['beam_frames']),section,np.eye(6))
    panel=api.ShellPanel(tuple(nodes),tuple(nodes.values()),cells,t,e,nu,7850.)
    footprint=api.NodalRigidFootprint('rigid-tip',selected,100001,patch)
    definition=api.LinearDefinition(api.B3_GE_E4_PL_STIFFENED_SHELL_V1,panel,(member,),(),
        tuple(api.FixedCoordinate('shell',n,d) for n in root for d in range(6)),
        q4_pullback_policy=api.Q4_ALGORITHMIC_PULLBACK_V1,footprints=(footprint,))
    owner=api.create_analysis(definition=definition)
    try:
        result=owner.solve_linear((api.NodalAction('beam',100001,tuple(c['wrench_SI'])),))
        if result.status!='completed':raise RuntimeError('native physical solve incomplete')
        shell=np.array(result.shell_displacement,copy=True).reshape(697,6)
        beam=np.array(result.beam_coordinates,copy=True);lam=np.array(result.multipliers,copy=True)
        q=np.r_[shell.ravel(),beam];Kq=np.array(owner._K@q,copy=True);Ctlam=np.array(owner._C.T@lam,copy=True)
        f=np.zeros(len(q));f[shell.size:shell.size+6]=c['wrench_SI'];eq=Kq+Ctlam-f
        rows=np.array([6*(n-1)+j for n in root for j in range(6)])
        support=np.array(result.reactions,copy=True)
        audit=dict(free_equilibrium_max=float(np.max(np.abs(eq[owner._free]))),
            root_Kq_minus_published_support_max=float(np.max(np.abs(Kq[rows]-support[rows]))),
            root_constraint_transpose_max=float(np.max(np.abs(Ctlam[rows]))),
            patch_Kq_plus_constraint_transpose_max=float(np.max(np.abs((Kq+Ctlam)[:shell.size].reshape(697,6)[np.array(selected)-1]))),
            public_free_equilibrium_ratio=float(result.free_equilibrium_ratio),public_constraint=float(result.constraint_max))
        record=observation(c,'native');record.update(port_pose=[beam[:6].tolist()],beam_coordinates=beam.tolist(),
            internal_energy_SI_J=.5*float(q@Kq),energy_provenance='independent-internal-operator',
            patch_action_meaning='actual reference-linear shell operator Kq physical rows; not fabricated dual representative')
        for name,ids in (('root',root),('patch',selected)):
            indices=np.array(ids)-1
            record[name]=dict(node_ids=list(ids),coordinates=[nodes[n] for n in ids],
                poses=shell[indices].tolist(),actions=Kq[:shell.size].reshape(697,6)[indices].tolist())
        stub=owner._K[shell.size:,shell.size:]@beam;pair=float(np.array(c['wrench_SI'])@beam[:6])
        guards=dict(stub_force_max=float(np.max(np.abs(stub))),stub_energy_J=.5*float(beam@stub),
            spin_max=float(np.max(np.abs(shell[:,5]))),port_offset_error=[float(beam[0]-.05*beam[4]),float(beam[1]+.05*beam[3])])
        save(path/'native-fields.json',dict(node_ids=list(nodes),coordinates=list(nodes.values()),shell=shell.tolist(),
            beam=beam.tolist(),multipliers=lam.tolist(),Kq=Kq.tolist(),Ct_lambda=Ctlam.tolist(),load=f.tolist(),audit=audit,guards=guards))
        save(path/'native-observation.json',record)
        metrics=measure(c,record,c['model'],deck,a);save(path/'native-metrics.json',metrics)
        imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
            if name.startswith('anyfesolver') and getattr(module,'__file__',None)}
        finalpools=threadpool_info();save(path/'native-runtime-final.json',dict(imports=imports,pools=finalpools,affinity=psutil.Process().cpu_affinity()))
        if any(not Path(p).is_relative_to(SITE) for p in imports.values()) or any(p['num_threads']!=1 for p in finalpools):raise RuntimeError('mixed imports/effective pools')
        if (not metrics['diagnostic_limits_satisfied'] or pair<=0 or guards['stub_force_max']>1e-7 or
            abs(guards['stub_energy_J'])>1e-10*pair or guards['spin_max']>1e-12 or max(abs(v) for v in guards['port_offset_error'])>1e-9 or
            audit['root_Kq_minus_published_support_max']>1e-12 or audit['root_constraint_transpose_max']>1e-12 or
            audit['patch_Kq_plus_constraint_transpose_max']>1e-8 or audit['public_free_equilibrium_ratio']>1e-8 or audit['public_constraint']>1e-9):
            raise RuntimeError('native completed physical/audit guard failure; no automatic rerun')
        print(json.dumps(dict(load=load,work_J=pair,metrics=metrics['metrics'],audit=audit)))
    finally:owner.close()

def solid(load,path,*,principal_observation=False,output_path=None):
    """Parse saved completed binary physical fields; never invokes CCX."""
    c,deck,m,a=staged(load,path);path=Path(path)
    data=(path/'reference.frd').read_bytes();stdout=(path/'stdout.txt').read_text();dat=(path/'reference.dat').read_text()
    parsed=parse_current_solid(data,dat,stdout,deck,m,principal_observation=principal_observation);u,rf=read_double_output(data,m['base'])
    pose=parsed['port'];record=observation(c,'solid')
    record['port_pose']=[pose['translation_m']+pose['theta_rad']]
    physical={int(n):p for n,p in m['base']['physical_nodes'].items()};patch=m['base']['patch_nodes'];root=m['base']['root_nodes']
    xyz=np.array([physical[n] for n in patch]);arms=xyz-xyz.mean(axis=0)
    inertia=np.eye(3)*np.sum(arms*arms)-arms.T@arms;force=np.array(c['wrench_SI'][:3]);moment=np.array(c['wrench_SI'][3:])
    dual=np.linalg.solve(inertia,moment+np.cross(np.array(c['model']['case']['port_m'])-xyz.mean(axis=0),force))
    applied=force/len(patch)+np.cross(dual,arms)
    record['patch']=dict(node_ids=patch,coordinates=xyz.tolist(),poses=[u[n] for n in patch],actions=applied.tolist())
    record['root']=dict(node_ids=root,coordinates=[physical[n] for n in root],poses=[u[n] for n in root],actions=[rf[n] for n in root])
    record['patch_action_meaning']='declared port load dual representative on complete rigid patch; not recovered CCX FORC'
    output=path if output_path is None else Path(output_path);output.mkdir(parents=True,exist_ok=True)
    record['supporting_print_interval_status']=parsed['supporting_print_interval_status']
    record['six_component_print_comparison']=parsed['six_component_print_comparison']
    save(output/'solid-observation.json',record);metrics=measure(c,record,c['model'],deck,a)
    save(output/'solid-metrics.json',metrics);save(output/'solid-parse.json',parsed)
    if not metrics['diagnostic_limits_satisfied'] or not parsed['root_binary_field_is_reaction'] or parsed['root_displacement_max_m']!=0:
        raise RuntimeError('solid completed physical guard failure; no automatic rerun')
    print(json.dumps(dict(load=load,work_J=metrics['port_work_J'],metrics=metrics['metrics'])))

def compare_saved(path,*,observation_path=None,output_path=None):
    path=Path(path);folder=ROOT/'docs/agent_plans/development/r8-e02-rigid-tip-reference-20261001'
    inventory=json.loads((folder/'raw-v1-identity.json').read_text());archive=folder/'raw-v1.zip'
    if sha(archive)!=inventory['sha256']:raise ValueError('original archive identity changed')
    member='r8-e02-rigid-tip-dev-v3/native-ge-40x16-s1/outputs/result.json'
    with zipfile.ZipFile(archive) as z:raw=z.read(member)
    digest=hashlib.sha256(raw).hexdigest();entry,=[v for v in inventory['files'] if v['path']==member]
    if digest!=entry['sha256']:raise ValueError('actual original40 member changed')
    old=json.loads(raw);rows={}
    for load in ('Fz','My'):
        c,_,_,_=staged(load,path/load)
        obs=path/load if observation_path is None else Path(observation_path)/load
        native_metrics=json.loads((path/load/'native-metrics.json').read_text());solid_metrics=json.loads((obs/'solid-metrics.json').read_text())
        parsed=json.loads((obs/'solid-parse.json').read_text())
        if not native_metrics['diagnostic_limits_satisfied'] or not solid_metrics['diagnostic_limits_satisfied']:
            raise RuntimeError('complete physical measurement guards required for comparison')
        actual=native_metrics['port_work_J'];prior=old['results'][load]['external_final_pairing_J']
        parity=abs(actual-prior)/abs(prior)
        rows[load]=dict(current_native_J=actual,original_native40_J=prior,relative_native_parity=parity,
            native_parity_within_1e_minus_10=parity<=1e-10,
            matched_principal=matched_port_work(actual,solid_metrics['port_work_J']),
            native_physical_metrics=native_metrics,solid_physical_metrics=solid_metrics,
            supporting_print_interval_status=parsed['supporting_print_interval_status'],
            six_component_print_comparison=parsed['six_component_print_comparison'])
        if parity>1e-10:raise RuntimeError('current/source-era principal parity failure:'+load)
    report=dict(status='CURRENT_PRINCIPAL_OBSERVATIONS_ONLY',qualification=False,archive_sha256=inventory['sha256'],
        original_native_member=member,original_native_member_sha256=digest,original_source=inventory['solver_product'],
        original_wheel=inventory['solver_wheel_sha256'],rows=rows,
        scope='current40/+offset/+pureFz10N andMy1Nm; root owns scientific adjudication; no additional convergence claim')
    output=path if output_path is None else Path(output_path);output.mkdir(parents=True,exist_ok=True)
    save(output/'comparison.json',report);print(json.dumps({load:dict(parity=row['relative_native_parity'],comparison=row['matched_principal']) for load,row in rows.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('prepare','native','solid'));p.add_argument('load',choices=('Fz','My'));p.add_argument('path',type=Path)
    args=p.parse_args();globals()[args.operation](args.load,args.path)
