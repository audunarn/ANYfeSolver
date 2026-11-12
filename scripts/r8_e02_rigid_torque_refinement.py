"""One admitted native 80x32/+Mx refinement, with labelled common samples.

No CCX, other case, extrapolation or numerical acceptance is supplied. Actual
2673-node/2560-cell native fields remain distinct from the 697 sampled stations.
The unchanged common-station derivative stencils and external thickness-secant
tilt are development observations, not unique solid directors or causal proof.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np

from scripts.r8_e02_rigid_torque_fields import (
    CASE as CASE40,WHEEL_SHA,_observations,digest,external_common_fields,validate_native,
)

CASE80=dict(CASE40,nx=80,ny=32)
DIRECTOR='reference-linear spatial theta; delta d=theta cross e_z'


def geometry80():
    nodes={1+j*81+i:(i/80,-.1+.2*j/32,0.) for j in range(33) for i in range(81)}
    cells=[];patch=[]
    for j in range(32):
        for i in range(80):
            a=1+j*81+i;c=(a,a+1,a+82,a+81);cells.append(c)
            if i>=64:patch.append(c)
    selected=tuple(sorted({n for c in patch for n in c}))
    return nodes,tuple(cells),tuple(patch),selected,tuple(1+j*81 for j in range(33))


def geometry_identity():
    nodes,cells,patch,selected,root=geometry80()
    payload=dict(case=CASE80,node_ids=list(nodes),coordinates=list(nodes.values()),
        cells=cells,patch_cells=patch,patch_nodes=selected,root_nodes=root,
        beam_nodes=[100001,100002,100003],beam_positions=[[.9,0.,.05],[.95,0.,.05],[1.,0.,.05]],
        beam_section=dict(area=.0004,Iy=1.3333333333333335e-8,Iz=1.3333333333333335e-8,J=2.249e-8,ky=5/6,kz=5/6),
        director_contract=DIRECTOR,profile='B3_GE_E4_PL_STIFFENED_SHELL_V1',
        pullback='STIFFENED_SHELL_Q4_ALGORITHMIC_PULLBACK_V1',fixed='all six root coordinates zero')
    return digest(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()),payload


def common_sample(fine):
    nodes,cells,patch,selected,root=geometry80()
    if fine.get('case')!=CASE80 or fine.get('node_ids')!=list(nodes):
        raise ValueError('only actual admitted80x32/+Mx native mesh permitted')
    if fine.get('element_count')!=2560 or fine.get('shell_node_count')!=2673:
        raise ValueError('actual fine physical inventory required')
    if fine.get('geometry_sha256')!=geometry_identity()[0]:
        raise ValueError('fine geometry/support/section/load identity changed')
    if fine.get('patch_nodes')!=list(selected) or fine.get('root_nodes')!=list(root) or fine.get('director_contract')!=DIRECTOR:
        raise ValueError('fine director or complete root/patch contract changed')
    xyz=np.asarray(fine.get('coordinates_m'),dtype=float)
    q=np.asarray(fine.get('shell_coordinates'),dtype=float)
    if xyz.shape!=(2673,3) or not np.array_equal(xyz,np.array(list(nodes.values()))):
        raise ValueError('fine physical coordinates changed')
    if q.shape!=(2673,6) or not np.isfinite(q).all():
        raise ValueError('complete finite2673×6 actual native fields required')
    beam=np.asarray(fine.get('beam_coordinates'),dtype=float)
    if beam.shape!=(42,) or not np.isfinite(beam).all():
        raise ValueError('complete native nodal plus retained beam variables required')
    grid=q.reshape(33,81,6).transpose(1,0,2)
    ids=[1+2*j*81+2*i for j in range(17) for i in range(41)]
    return grid[::2,::2].copy(),dict(actual_case=CASE80,actual_shell_nodes=2673,actual_cells=2560,
        observation_grid=dict(nx=40,ny=16,sampled_points=697),fine_node_ids=ids,
        reference_positions_m=[nodes[n] for n in ids],interpolation='none: coincident physical native vertices')


def mismatch_variation(delta):
    delta=np.asarray(delta,dtype=float)
    if delta.shape!=(41,) or not np.isfinite(delta).all():
        raise ValueError('finite41-station profile required')
    increments=np.diff(delta[:33]);net=float(np.sum(increments));variation=float(np.sum(np.abs(increments)))
    bands=[]
    for name,a,b in (('root_neighborhood',0,2),('free_span_interior',2,30),('patch_neighborhood',30,32)):
        signed=float(np.sum(increments[a:b]));absolute=float(np.sum(np.abs(increments[a:b])))
        bands.append(dict(name=name,x_interval_m=[a*.025,b*.025],signed_increment_rad=signed,
            signed_fraction=(signed/net if net else None),absolute_variation_rad=absolute,
            absolute_variation_fraction=(absolute/variation if variation else None)))
    return dict(net_increment_rad=net,total_absolute_variation_rad=variation,bands=bands,
        minimum=dict(x_m=float(np.argmin(delta))*.025,value_rad=float(np.min(delta))),
        maximum=dict(x_m=float(np.argmax(delta))*.025,value_rad=float(np.max(delta))),
        max_abs_rad=float(np.max(np.abs(delta))))


def compare_refinement(fine,coarse,data,deck,manifest):
    q80,sampling=common_sample(fine);q40=validate_native(coarse)
    external,tilt,secant=external_common_fields(data,deck,manifest)
    obs={'native40':_observations(q40[:,:,:3],q40[:,:,3:5]),
         'native80_common':_observations(q80[:,:,:3],q80[:,:,3:5]),
         'external40_solid':_observations(external,tilt)}
    weights=np.ones(17);weights[[0,-1]]=.5;weights/=16
    profiles={}
    for key in obs['native40']:
        values={name:(fields[key]@weights) for name,fields in obs.items()}
        profiles[key]={name:value.tolist() for name,value in values.items()}
        profiles[key]['native80_minus_native40']=(values['native80_common']-values['native40']).tolist()
        profiles[key]['native80_minus_external40']=(values['native80_common']-values['external40_solid']).tolist()
        profiles[key]['native40_minus_external40']=(values['native40']-values['external40_solid']).tolist()
    work40=float(coarse['guards']['external_final_pairing_J']);work80=float(fine['guards']['external_final_pairing_J'])
    if not math.isfinite(work40+work80) or min(work40,work80)<=0:raise ValueError('finite positive physical work required')
    region={}
    for name,(a,b) in {'root_neighborhood':(0,3),'smooth_free_span':(3,30),
            'patch_derivative_neighborhood':(30,35),'rigid_patch_interior':(35,41)}.items():
        region[name]={}
        for key in obs['native40']:
            changes={}
            for label,left,right in (('refinement','native80_common','native40'),
                                     ('fine_vs_external','native80_common','external40_solid')):
                d=obs[left][key][a:b]-obs[right][key][a:b]
                changes[label]=dict(rms=float(np.sqrt(np.mean(d*d))),max_abs=float(np.max(np.abs(d))))
            region[name][key]=changes
    changes=q80-q40
    return dict(status='NATIVE_TORQUE_REFINEMENT_DEVELOPMENT_ONLY',acceptance=False,
        actual_native_case=CASE80,common_observation_view=sampling,
        coarse_native_case=CASE40,external_case=CASE40,external_input_sha256=manifest['deck_sha256'],
        native80_guards=fine['guards'],native40_guards=coarse['guards'],
        native40_work_J=work40,native80_work_J=work80,work_refinement_relative=(work80-work40)/work40,
        common_field_delta_component_max=np.max(np.abs(changes),axis=(0,1)).tolist(),
        common_field_delta_component_rms=np.sqrt(np.mean(changes*changes,axis=(0,1))).tolist(),
        x_m=[i*.025 for i in range(41)],profiles=profiles,regional_changes=region,
        native40_external_tilt_variation=mismatch_variation(profiles['theta_x_rad']['native40_minus_external40']),
        native80_external_tilt_variation=mismatch_variation(profiles['theta_x_rad']['native80_minus_external40']),
        refinement_tilt_variation=mismatch_variation(profiles['theta_x_rad']['native80_minus_native40']),
        normalized_scientific_acceptance='NOT_EVALUATED',
        scope='single native40-to80 refinement; same697 coincident stations/stencils; external solid remains40x16/nz2',
        interpretation='signed net increments and absolute spatial variation differ; no causal, asymptotic or continuum-equivalence claim')


def collect80(site,output):
    """One installed public solve; no changed laws or model-wide dense arrays."""
    site=Path(site).resolve();output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);sys.path.insert(0,str(site))
    from anyfesolver import b3_ge,stiffened_shell as api
    import anyfesolver,os,psutil
    from scipy import sparse
    from threadpoolctl import threadpool_info
    evidence=site.parent;binding=json.loads((evidence/'installed-v2/binding.json').read_text())
    wheel=evidence/'dist/anysolver-0.4.7-py3-none-any.whl'
    if not Path(anyfesolver.__file__).resolve().is_relative_to(site) or digest(wheel.read_bytes())!=WHEEL_SHA:
        raise RuntimeError('effective import/guarded wheel identity changed')
    with zipfile.ZipFile(wheel) as archive:
        for relative,expected in binding['module_sha256'].items():
            if digest((site/relative).read_bytes())!=expected or digest(archive.read(relative))!=expected:
                raise RuntimeError('installed dependency identity changed: '+relative)
    pools=threadpool_info()
    if any(p['num_threads']!=1 for p in pools) or os.environ.get('NUMBA_DISABLE_JIT')!='1' or psutil.Process().cpu_affinity()!=[0]:
        raise RuntimeError('effective numerical resources changed')
    gid,gpayload=geometry_identity();nodes,cells,patch,selected,root=geometry80()
    runtime=dict(python=sys.version,executable=sys.executable,site=str(site),wheel_sha256=WHEEL_SHA,
        installed_source_head=binding['source_head'],geometry_sha256=gid,numerical_pools=pools,
        affinity=psutil.Process().cpu_affinity(),jit_disabled=os.environ.get('NUMBA_DISABLE_JIT'),
        dependency_module_sha256=binding['module_sha256'])
    (output/'geometry.json').write_text(json.dumps(gpayload,indent=2))
    (output/'runtime-preflight.json').write_text(json.dumps(runtime,indent=2))
    e,nu,t=210e9,.3,.02;g=e/(2*(1+nu))
    section=b3_ge.EllipsoidalGeneralizedSection(np.diag((e*.0004,g*.0004*5/6,g*.0004*5/6,
        g*2.249e-8,e*1.3333333333333335e-8,e*1.3333333333333335e-8)),np.eye(6),1e20,1e9)
    member=b3_ge.define_beam('b3-ge',1,(100001,100002,100003),
        np.array(((.9,0.,.05),(.95,0.,.05),(1.,0.,.05))),np.tile(np.eye(3),(3,1,1)),section,np.eye(6))
    panel=api.ShellPanel(tuple(nodes),tuple(nodes.values()),cells,t,e,nu,7850.)
    group=api.NodalRigidFootprint('rigid-tip',selected,100001,patch)
    definition=api.LinearDefinition(api.B3_GE_E4_PL_STIFFENED_SHELL_V1,panel,(member,),(),
        tuple(api.FixedCoordinate('shell',n,d) for n in root for d in range(6)),
        q4_pullback_policy=api.Q4_ALGORITHMIC_PULLBACK_V1,footprints=(group,))
    owner=api.create_analysis(definition=definition)
    try:
        sparse_inventory={}
        for name in ('_K','_C','_A'):
            matrix=getattr(owner,name)
            if not sparse.issparse(matrix):raise RuntimeError('production model-wide operator is not sparse: '+name)
            sparse_inventory[name]=dict(shape=matrix.shape,nnz=matrix.nnz,format=matrix.format)
        (output/'sparse-pre-solve.json').write_text(json.dumps(sparse_inventory,indent=2))
        result=owner.solve_linear((api.NodalAction('beam',100001,tuple(CASE80['wrench'])),))
        if result.status!='completed':raise RuntimeError('native80 solve did not complete')
        shell=np.array(result.shell_displacement).reshape(2673,6);beam=np.array(result.beam_coordinates);port=beam[:6]
        q=np.r_[shell.ravel(),beam];count=shell.size;force=owner._K[count:,count:]@beam
        stub_energy=.5*float(beam@force);energy=.5*float(q@(owner._K@q));pair=float(port[3])
        support_force=np.array([math.fsum(float(result.reactions[6*(n-1)+j]) for n in root) for j in range(3)])
        moments=[result.reactions[6*(n-1)+3:6*n]+np.cross(nodes[n],result.reactions[6*(n-1):6*(n-1)+3]) for n in root]
        support_moment=np.array([math.fsum(float(m[j]) for m in moments) for j in range(3)])
        transfer,=owner.recover(selection=api.NodalRigidFootprintSelection(('rigid-tip',)))
        gap=max(float(np.max(np.abs(shell[n-1,:3]-port[:3]-np.cross(port[3:],np.array(nodes[n])-CASE80['port_m'])))) for n in selected)
        residual=np.r_[support_force,support_moment+np.array((1.,0.,0.))]
        guards=dict(root_wrench_residual=residual.tolist(),root_coordinate_max=float(np.max(np.abs(shell[np.array(root)-1]))),
            normalized_scientific_acceptance='NOT_EVALUATED',free_equilibrium_ratio=float(result.free_equilibrium_ratio),
            constraint_max=float(result.constraint_max),physical_patch_translation_gap_m=gap,
            stub_force_max=float(np.max(np.abs(force))),stub_energy_J=stub_energy,
            internal_quadratic_energy_J=energy,external_final_pairing_J=pair,
            energy_work_relative_gap=(abs(2*energy-pair)/abs(pair) if pair else math.inf),
            footprint_wrench_error=(np.array(transfer.shell_action)-CASE80['wrench']).tolist(),
            shell_normal_spin_max=float(np.max(np.abs(shell[:,5]))),
            port_offset_identity_m=[float(port[0]-.05*port[4]),float(port[1]+.05*port[3])])
        fields=dict(status='completed',case=CASE80,shell_node_count=2673,element_count=2560,geometry_sha256=gid,
            node_ids=list(nodes),coordinates_m=list(nodes.values()),shell_coordinates=shell.tolist(),
            beam_coordinates=beam.tolist(),port_coordinates=port.tolist(),patch_nodes=list(selected),root_nodes=list(root),
            director_contract=DIRECTOR,guards=guards,runtime=runtime,sparse_operators=sparse_inventory)
        (output/'native80-fields.json').write_text(json.dumps(fields,indent=2))
        runtime['imported_modules']={name:str(Path(m.__file__).resolve()) for name,m in sys.modules.items()
            if name.startswith('anyfesolver') and getattr(m,'__file__',None)}
        if any(not Path(p).is_relative_to(site) for p in runtime['imported_modules'].values()):raise RuntimeError('mixed effective imports')
        runtime['final_numerical_pools']=threadpool_info()
        (output/'runtime-final.json').write_text(json.dumps(runtime,indent=2))
        if any(p['num_threads']!=1 for p in runtime['final_numerical_pools']):raise RuntimeError('final effective pools changed')
        if (not np.isfinite(shell).all() or not np.isfinite(beam).all() or pair<=0 or
                guards['energy_work_relative_gap']>1e-8 or gap>1e-9 or max(abs(x) for x in residual)>1e-8 or
                guards['root_coordinate_max']!=0 or guards['stub_force_max']>1e-7 or abs(stub_energy)>1e-10*pair or
                max(abs(x) for x in guards['footprint_wrench_error'])>1e-8 or guards['shell_normal_spin_max']>1e-12 or
                max(abs(x) for x in guards['port_offset_identity_m'])>1e-9):
            raise RuntimeError('native80 completed physical guard failure; fields preserved, no new solve')
        print(json.dumps(dict(status='COLLECTED_NATIVE80_DEVELOPMENT_ONLY',guards=guards,sparse_operators=sparse_inventory)))
    finally:owner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();collect80(args.site,args.output)
