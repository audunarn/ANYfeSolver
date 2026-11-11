"""One fixed 40x16/+Mx E02 physical-field discriminator, development only.

Native reference-linear spatial rotations act on e_z as theta cross e_z.
The solid top/bottom displacement secant is a declared lower-order thickness
tilt, not a unique solid material director. Geometric normal slopes are kept
separate. Finite differences across ground/patch neighborhoods are observation
stencils, never constitutive tangent or continuum-convergence evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

from scripts.r8_e02_native_rigid_tip import geometry
from scripts.r8_e02_rigid_precision import ELIMINATED, _validate_candidate
from scripts.r8_e02_double_output import read_double_output

WHEEL_SHA = '89cfba60de1d967507636eb46e711af52d825a51ce0c1286f22133bac357ded4'
CASE = dict(nx=40,ny=16,nz=2,load_name='Mx',sign=1,E_Pa=210e9,nu=.3,
            thickness_m=.02,port_m=[.9,0.,.05],wrench=[0.,0.,0.,1.,0.,0.])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_native(native):
    nodes,_,_,patch,root = geometry(40,16)
    if native.get('case')!=CASE or native.get('node_ids')!=list(nodes):
        raise ValueError('exact fixed torque case and native node identity required')
    xyz=np.asarray(native.get('coordinates_m'),dtype=float)
    q=np.asarray(native.get('shell_coordinates'),dtype=float)
    if xyz.shape!=(697,3) or not np.array_equal(xyz,np.array(list(nodes.values()))):
        raise ValueError('changed native physical geometry')
    if q.shape!=(697,6) or not np.isfinite(q).all():
        raise ValueError('complete finite native six-coordinate field required')
    if native.get('patch_nodes')!=list(patch) or native.get('root_nodes')!=list(root):
        raise ValueError('changed complete patch/root coverage')
    if native.get('director_contract')!='reference-linear spatial theta; delta d=theta cross e_z':
        raise ValueError('explicit native director contract required')
    return q.reshape(17,41,6).transpose(1,0,2)


def external_common_fields(data,deck,manifest):
    b=_validate_candidate(deck,manifest)
    if (manifest['variant']!=ELIMINATED or
            any(b[k]!=CASE[k] for k in ('nx','ny','nz','load_name','sign'))):
        raise ValueError('only saved fixed-root 40x16/+Mx/nz2 positive-offset reference admitted')
    u,_=read_double_output(data,b)
    positions={int(n):p for n,p in b['physical_nodes'].items()}
    at={}
    for n,p in positions.items():
        key=(round(p[0]*80),round((p[1]+.1)*160),round(p[2]*200))
        if key in at:raise ValueError('ambiguous physical lattice position')
        at[key]=n
    middle=np.empty((41,17,3)); tilt=np.empty((41,17,2))
    thickness=np.empty((41,17,3))
    for i in range(41):
        for j in range(17):
            values=[]
            for z in (0,2,-2):
                key=(2*i,2*j,z)
                if key not in at:raise ValueError('missing common through-thickness physical station')
                n=at[key]
                expected=(i/40,-.1+.2*j/16,z/200)
                if max(abs(a-b) for a,b in zip(positions[n],expected))>4e-15:
                    raise ValueError('common station geometry changed')
                values.append(np.array(u[n]))
            middle[i,j]=values[0]
            secant=(values[1]-values[2])/.02
            thickness[i,j]=secant
            tilt[i,j]=(-secant[1],secant[0])
    return middle,tilt,thickness


def _observations(u,tilt):
    w=u[:,:,2]
    slope_x=np.gradient(w,.025,axis=0,edge_order=2)
    slope_y=np.gradient(w,.0125,axis=1,edge_order=2)
    twist=np.gradient(slope_y,.025,axis=0,edge_order=2)
    return dict(uz_m=w,theta_x_rad=tilt[:,:,0],theta_y_rad=tilt[:,:,1],
        normal_theta_x_rad=slope_y,normal_theta_y_rad=-slope_x,
        transverse_mixed_twist_per_m=twist,
        shear_y_indicator=slope_y-tilt[:,:,0],
        shear_x_indicator=slope_x+tilt[:,:,1])


def compare_fields(native,data,deck,manifest):
    q=validate_native(native)
    external,tilt,thickness=external_common_fields(data,deck,manifest)
    n=_observations(q[:,:,:3],q[:,:,3:5]); e=_observations(external,tilt)
    weights=np.ones(17);weights[[0,-1]]=.5;weights/=16
    profile={}
    for key in n:
        nm=n[key]@weights;em=e[key]@weights
        profile[key]={'native_width_average':nm.tolist(),'external_width_average':em.tolist(),
            'native_minus_external':(nm-em).tolist()}
    delta=np.array(profile['theta_x_rad']['native_minus_external'])
    increments=np.diff(delta[:33])
    total=float(delta[32]-delta[0])
    # Allocate the observed accumulated director-tilt difference by disjoint
    # intervals. These bands were chosen prospectively, not optimized to data.
    bands=(('root_neighborhood',0,2),('free_span_interior',2,30),('patch_neighborhood',30,32))
    allocation=[]
    for name,a,b in bands:
        contribution=float(np.sum(increments[a:b]))
        allocation.append(dict(band=name,x_interval_m=[a*.025,b*.025],
            accumulated_tilt_difference_rad=contribution,
            signed_fraction_of_free_span_difference=(contribution/total if total else None)))
    masks={'root_neighborhood':(0,3),'smooth_free_span':(3,30),
           'patch_derivative_neighborhood':(30,35),'rigid_patch_interior':(35,41)}
    region_metrics={}
    for name,(a,b) in masks.items():
        region_metrics[name]={key:{'rms_native_minus_external':float(np.sqrt(np.mean((n[key][a:b]-e[key][a:b])**2))),
            'max_abs_native_minus_external':float(np.max(np.abs(n[key][a:b]-e[key][a:b])))} for key in n}
    return dict(status='PHYSICAL_FIELD_DEVELOPMENT_ONLY',acceptance=False,case=CASE,
        native_guards=native['guards'],external_candidate_sha256=manifest['deck_sha256'],
        common_node_count=697,x_m=[i*.025 for i in range(41)],y_m=[-.1+j*.0125 for j in range(17)],
        profile=profile,region_metrics=region_metrics,
        mid_surface_translation_delta_max_m=np.max(np.abs(q[:,:,:3]-external),axis=(0,1)).tolist(),
        external_thickness_secant_max_abs=np.max(np.abs(thickness),axis=(0,1)).tolist(),
        free_span_width_average_tilt_difference_rad=total,
        tilt_difference_allocation=allocation,
        native_port_Mx_rotation_rad=native['port_coordinates'][3],
        derivative_stencils='shared second-order finite differences at coincident physical nodes; separate one-sided edge and interface neighborhoods',
        director_observation='native spatial theta versus external top/bottom thickness secant tilt; not a unique solid director or assumed geometric normal',
        causality='spatial mismatch allocation only; does not prove formulation error or a unique mechanical cause',
        remaining_blocker='E02 torque in-plane convergence/connection interpretation and scientific acceptance remain open')


def installed_native_field(site,output):
    """ONE public installed reference-linear solve; complete physical fields saved."""
    site=Path(site).resolve();output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    sys.path.insert(0,str(site))
    from anyfesolver import b3_ge,stiffened_shell as api
    import anyfesolver
    from threadpoolctl import threadpool_info
    import os
    import psutil
    if not Path(anyfesolver.__file__).resolve().is_relative_to(site):
        raise RuntimeError('effective ANYfeSolver import escaped installed artifact')
    evidence=site.parent
    binding=json.loads((evidence/'installed-v2/binding.json').read_text())
    wheel=evidence/'dist/anysolver-0.4.7-py3-none-any.whl'
    if digest(wheel.read_bytes())!=WHEEL_SHA or binding['wheel_sha256']!=WHEEL_SHA:
        raise RuntimeError('guarded wheel identity differs')
    with zipfile.ZipFile(wheel) as archive:
        for relative,expected in binding['module_sha256'].items():
            if digest((site/relative).read_bytes())!=expected or digest(archive.read(relative))!=expected:
                raise RuntimeError('installed module differs from verified guarded wheel: '+relative)
    pools=threadpool_info()
    if any(p['num_threads']!=1 for p in pools) or os.environ.get('NUMBA_DISABLE_JIT')!='1' or psutil.Process().cpu_affinity()!=[0]:
        raise RuntimeError('effective CPU/thread/JIT settings differ')
    runtime=dict(python=sys.version,executable=sys.executable,anyfesolver=str(Path(anyfesolver.__file__).resolve()),
        wheel_sha256=WHEEL_SHA,source_head=binding['source_head'],numerical_pools=pools,
        affinity=psutil.Process().cpu_affinity(),jit_disabled=os.environ.get('NUMBA_DISABLE_JIT'),
        source_contract='spatial exponential(jets[i][3:]) in _ge_beam3_variational_shell.deformation; reference-linear zero-anchor theta')
    (output/'runtime-preflight.json').write_text(json.dumps(runtime,indent=2))
    nodes,cells,patch,patch_nodes,root=geometry(40,16)
    e,nu,t=210e9,.3,.02;g=e/(2*(1+nu))
    section=b3_ge.EllipsoidalGeneralizedSection(np.diag((e*.0004,g*.0004*5/6,g*.0004*5/6,
        g*2.249e-8,e*1.3333333333333335e-8,e*1.3333333333333335e-8)),np.eye(6),1e20,1e9)
    member=b3_ge.define_beam('b3-ge',1,(100001,100002,100003),
        np.array(((.9,0.,.05),(.95,0.,.05),(1.,0.,.05))),np.tile(np.eye(3),(3,1,1)),section,np.eye(6))
    panel=api.ShellPanel(tuple(nodes),tuple(nodes.values()),cells,t,e,nu,7850.)
    group=api.NodalRigidFootprint('rigid-tip',patch_nodes,100001,patch)
    definition=api.LinearDefinition(api.B3_GE_E4_PL_STIFFENED_SHELL_V1,panel,(member,),(),
        tuple(api.FixedCoordinate('shell',n,d) for n in root for d in range(6)),
        q4_pullback_policy=api.Q4_ALGORITHMIC_PULLBACK_V1,footprints=(group,))
    owner=api.create_analysis(definition=definition)
    try:
        result=owner.solve_linear((api.NodalAction('beam',100001,tuple(CASE['wrench'])),))
        if result.status!='completed':raise RuntimeError('native solve did not complete')
        shell=np.array(result.shell_displacement).reshape(697,6)
        beam=np.array(result.beam_coordinates);port=beam[:6]
        q=np.r_[shell.ravel(),beam];count=shell.size
        force=owner._K[count:,count:]@beam
        stub_energy=.5*float(beam@force)
        energy=.5*float(q@(owner._K@q));pair=float(port[3])
        support_force=np.array([math.fsum(float(result.reactions[6*(n-1)+j]) for n in root) for j in range(3)])
        moments=[result.reactions[6*(n-1)+3:6*n]+np.cross(nodes[n],result.reactions[6*(n-1):6*(n-1)+3]) for n in root]
        support_moment=np.array([math.fsum(float(m[j]) for m in moments) for j in range(3)])
        transfer,=owner.recover(selection=api.NodalRigidFootprintSelection(('rigid-tip',)))
        patch_gap=max(float(np.max(np.abs(shell[n-1,:3]-port[:3]-np.cross(port[3:],np.array(nodes[n])-CASE['port_m'])))) for n in patch_nodes)
        root_residual=np.r_[support_force,support_moment+np.array((1.,0.,0.))]
        guards=dict(root_wrench_residual=root_residual.tolist(),root_coordinate_max=float(np.max(np.abs(shell[np.array(root)-1]))),
            normalized_scientific_acceptance='NOT_EVALUATED',
            free_equilibrium_ratio=float(result.free_equilibrium_ratio),constraint_max=float(result.constraint_max),
            physical_patch_translation_gap_m=patch_gap,stub_force_max=float(np.max(np.abs(force))),
            stub_energy_J=stub_energy,internal_quadratic_energy_J=energy,external_final_pairing_J=pair,
            energy_work_relative_gap=abs(2*energy-pair)/abs(pair),
            footprint_wrench_error=(np.array(transfer.shell_action)-CASE['wrench']).tolist(),
            shell_normal_spin_max=float(np.max(np.abs(shell[:,5]))),
            port_offset_identity_m=[float(port[0]-.05*port[4]),float(port[1]+.05*port[3])])
        fields=dict(status='completed',case=CASE,node_ids=list(nodes),coordinates_m=list(nodes.values()),
            shell_coordinates=shell.tolist(),beam_coordinates=beam.tolist(),port_coordinates=port.tolist(),
            patch_nodes=list(patch_nodes),root_nodes=list(root),guards=guards,
            director_contract='reference-linear spatial theta; delta d=theta cross e_z',runtime=runtime)
        (output/'native-fields.json').write_text(json.dumps(fields,indent=2))
        if (pair<=0 or guards['energy_work_relative_gap']>1e-8 or patch_gap>1e-9 or
                max(abs(x) for x in root_residual)>1e-8 or guards['root_coordinate_max']!=0 or
                guards['stub_force_max']>1e-7 or abs(stub_energy)>1e-10*pair or
                max(abs(x) for x in guards['footprint_wrench_error'])>1e-8 or
                guards['shell_normal_spin_max']>1e-12 or max(abs(x) for x in guards['port_offset_identity_m'])>1e-9):
            raise RuntimeError('native physical field guards failed; completed fields preserved, no retry')
        runtime['imported_modules']={name:str(Path(m.__file__).resolve()) for name,m in sys.modules.items()
            if name.startswith('anyfesolver') and getattr(m,'__file__',None)}
        if any(not Path(p).is_relative_to(site) for p in runtime['imported_modules'].values()):
            raise RuntimeError('mixed source/installed runtime imports')
        runtime['final_numerical_pools']=threadpool_info()
        if any(p['num_threads']!=1 for p in runtime['final_numerical_pools']):
            raise RuntimeError('effective final numerical pool count changed')
        (output/'runtime-final.json').write_text(json.dumps(runtime,indent=2))
        print(json.dumps(dict(status='COLLECTED_DEVELOPMENT_ONLY',pairing_J=pair,guards=guards)))
    finally:owner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    installed_native_field(args.site,args.output)
