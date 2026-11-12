"""Loaded coupled-state shell chart rebase feasibility, not an owner operation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if not os.environ.get('NUMBA_CACHE_DIR'):
    raise RuntimeError('set an explicit writable NUMBA_CACHE_DIR before import')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import anyfesolver
import numba
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from anyfesolver._ge_beam3_g3c_local_shell import deformation
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_coupled_beam_subdomain import CoupledBeamSubdomain
from anyfesolver._ge_beam3_shell_joint_trial import ShellBeamTrialAssembly
from anyfesolver._ge_beam3_shell_joint_state import CoupledShellBeamAnalysis
from anyfesolver._ge_beam3_retained_generalized_state import Program
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_shell_joint_state import decode
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.fe_core import FEModel


CASES=(('Q4',False),('Q4',True),('S3-V2D',False),('S3-V2D',True))


def fixture(topology,curved):
    """Explicit equivalent of the existing loaded coupled-mode fixture."""
    macros=2 if curved else 1
    section=np.diag((1000.,400.,350.,.8,1.,1.2))
    section[0,3]=section[3,0]=.04
    section[1,5]=section[5,1]=.03
    model=FEModel('g0-f06-native-generalized')
    coordinates=[];frames=[]
    for i,x in enumerate(np.linspace(0.,2.,2*macros+1),start=1):
        position=np.array((x,.2*(1-(x-1)**2) if curved else 0.,0.))
        tangent=np.array((1.,.4*(1-x) if curved else 0.,0.))
        tangent/=np.linalg.norm(tangent)
        z=np.array((0.,0.,1.))
        frame=np.column_stack((tangent,z,np.cross(tangent,z)))
        coordinates.append(position);frames.append(frame)
        model.add_node(i,*position)
    law=EllipsoidalGeneralizedSection(section,np.eye(6),1e6,1.)
    for i in range(macros):
        ids=tuple(range(2*i+1,2*i+4))
        ref=CenteredCurvedBeam3ReferenceGeometry(
            np.asarray(coordinates[2*i:2*i+3]),np.asarray(frames[2*i:2*i+3]))
        element=NativeGeneralizedStaticElement(i+1,ids,ref,law,order=4)
        model.add_element(i+1,element)
        model.materials[element.material_name]=law
    program=Program((1.,),DistributedPattern(LinePattern(()),()))
    beam=CoupledBeamSubdomain(model,program)
    xyz=np.array(((0.,0.,.2),(1.,0.,.2),(1.,1.,.2),(0.,1.,.2)))
    if topology=='S3-V2D':xyz=xyz[:3]
    assembly=ShellBeamTrialAssembly(beam,topology=topology,coordinates=xyz,
        reference_normal=np.array((0.,0.,1.)),thickness=.2,
        elastic_modulus=1000.,poisson_ratio=.3,shell_node=2,beam_node=1,
        variational_shell=True)
    force=np.zeros(assembly.shell_count+beam.nodal_count)
    force[-6:-3]=np.array((0.,.0001,.0002))
    fixed=tuple(range(6)) if topology=='S3-V2D' else (*range(6),*range(18,24))
    return CoupledShellBeamAnalysis(assembly,targets=(.5,1.,0.),
        shell_fixed=fixed,nodal_forces=force)


def error(left,right,floor=1.):
    a=np.asarray(left);b=np.asarray(right)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite or mismatched F06 comparison arrays')
    value=float(np.linalg.norm(a-b)/max(floor,np.linalg.norm(a),np.linalg.norm(b)))
    if not np.isfinite(value):raise ValueError('nonfinite F06 normalized comparison')
    return value


def numeric_state(value):
    records=[]
    def visit(v,path):
        if isinstance(v,Mapping):
            for key in sorted(v):visit(v[key],path+(str(key),))
        elif isinstance(v,np.ndarray):
            for index in np.ndindex(v.shape):visit(v[index],path+tuple(map(str,index)))
        elif isinstance(v,(tuple,list)):
            for i,item in enumerate(v):visit(item,path+(str(i),))
        elif isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_)):
            records.append((path,float(v)))
    visit(value,())
    paths=tuple(p for p,_ in records)
    values=np.asarray([v for _,v in records],dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('missing/nonfinite numeric shell candidate state')
    return paths,values


def shell_spatial(assembly,origin,kin,u):
    material=assembly.model.get_material('joint-shell')
    f,k,candidate=assembly.element.compute_nonlinear_response(
        assembly.model.mesh,material,kin.deformation,origin,assembly.layers,True)
    D,H=kin.differential,kin.second
    g=D.T@f
    coordinate_hessian=D.T@k@D+np.einsum('i,ijk->jk',f,H)
    P=np.eye(len(u));correction=np.zeros_like(coordinate_hessian)
    blocks=[]
    for i,row in enumerate(u.reshape(-1,6)):
        _,A,dA=_exp_terms(row[3:]);sl=slice(6*i+3,6*i+6)
        P[sl,sl]=A;blocks.append((sl,dA))
    spatial=np.linalg.solve(P.T,g)
    for sl,dA in blocks:
        correction[sl,sl]=np.einsum('ijk,i->jk',dA,spatial[sl])
    chart_jac=np.linalg.solve(P.T,coordinate_hessian-correction)
    spatial_jac=chart_jac@np.linalg.inv(P)
    return spatial,spatial_jac,kin.deformation@f,candidate


def case(topology,curved):
    owner=fixture(topology,curved)
    run=owner.solve(stop_after=2)
    if run.status!='paused' or run.state.cursor!=2:
        raise ValueError(f'loaded coupled fixture failed: {run.status}: {run.failure}')
    state=run.state;before=canonical(state.descriptor());checkpoint=run.checkpoint
    a=owner.assembly;u=np.asarray(state.shell_u)
    n=len(a.coordinates);parents=np.tile(np.eye(3),(n,1,1))
    rebased=u.copy();rotations=[]
    for i,row in enumerate(u.reshape(n,6)):
        Q=_exp_terms(row[3:])[0]
        rotations.append(Q)
        rebased[6*i+3:6*i+6]=0.
    rotations=np.asarray(rotations)
    original=deformation(np.asarray(a.coordinates),u,parents)
    alternate=deformation(np.asarray(a.coordinates),rebased,rotations)
    origin=decode(state.shell_history)
    sr0,sj0,work0,c0=shell_spatial(a,origin,original,u)
    sr1,sj1,work1,c1=shell_spatial(a,origin,alternate,rebased)
    parameter=owner.targets[state.cursor-1]
    br,bj,_,responses,_=a.beam.assemble(state.mechanical,parameter,state.beam_origins)
    bi=a.beam.index[a.beam_node]
    shell_position=a.coordinates[a.shell_node]+u.reshape(n,6)[a.shell_node,:3]
    beam_position=state.mechanical.positions[bi]
    physical_shell_frame=rotations[a.shell_node]@a.frame
    joint0=a.joint.evaluate(np.array([shell_position,beam_position]),
                            np.array([physical_shell_frame,state.mechanical.nodal_frames[bi]]),
                            state.multipliers,
                            position_low=np.array([np.zeros(3),state.mechanical.position_low[bi]]))
    joint1=a.joint.evaluate(np.array([a.coordinates[a.shell_node]
                                      +rebased.reshape(n,6)[a.shell_node,:3],beam_position]),
                            np.array([alternate.rotations[a.shell_node]@a.frame,
                                      state.mechanical.nodal_frames[bi]]),state.multipliers,
                            position_low=np.array([np.zeros(3),state.mechanical.position_low[bi]]))
    direct,residual,_,_=owner._evaluate(state,state.mechanical,state.shell_u,
                                        state.multipliers,state.cursor)
    reconstructed=np.r_[sr1,br,np.zeros(6)]
    reconstructed[list(a.slots)]+=joint1.residual
    reconstructed[:len(owner.forces)]-=parameter*owner.forces
    fixed=np.setdiff1d(np.arange(a.count),np.asarray(owner.free))
    metrics={'local_deformation':error(original.deformation,alternate.deformation),
             'positions':error(original.current_positions,alternate.current_positions),
             'physical_rotations':error(original.rotations,alternate.rotations),
             'shell_spatial_residual':error(sr0,sr1),
             'shell_spatial_jacobian':error(sj0,sj1),
             'shell_work':error(work0,work1),
             'joint_gap':error(joint0.constraints,joint1.constraints),
             'joint_spatial_wrench':error(joint0.residual,joint1.residual),
             'joint_spatial_jacobian':error(joint0.spatial_jacobian,joint1.spatial_jacobian),
             'fixed_support_reaction':error(residual[fixed],reconstructed[fixed]),
             'full_loaded_residual':error(residual,reconstructed),
             'beam_native_residual_repeated':error(
                 br,a.beam.assemble(state.mechanical,parameter,state.beam_origins)[0])}
    active=bool(np.linalg.norm(u.reshape(n,6)[:,3:])>1e-8 and
                np.linalg.norm(state.multipliers)>1e-8 and
                abs(work0)>1e-8 and np.linalg.norm(br)>1e-8)
    unchanged=bool(canonical(state.descriptor())==before and run.checkpoint==checkpoint)
    paths0,values0=numeric_state(c0);paths1,values1=numeric_state(c1)
    numeric_norm=float(np.linalg.norm(values0))
    if paths0!=paths1:
        raise ValueError('shell candidate numeric state paths differ')
    if not np.isfinite(numeric_norm):
        raise ValueError('shell candidate numeric state norm nonfinite')
    metrics['shell_candidate_numeric_state']=error(values0,values1)
    if topology=='Q4':
        if numeric_norm>1e-12:
            raise ValueError('Q4 elastic shortcut unexpectedly changed virgin history')
        material_policy='Q4_VIRGIN_ELASTIC_SHORTCUT'
    else:
        if numeric_norm<=1e-8:
            raise ValueError('S3 candidate numeric state lacks nontrivial fields')
        for name in ('layer_strain','layer_stress'):
            left=np.asarray(c0[name]);right=np.asarray(c1[name])
            layer_norm=float(np.linalg.norm(left))
            if not np.isfinite(layer_norm) or layer_norm<=1e-8:
                raise ValueError(f'missing nontrivial shell {name} response')
            metrics['shell_'+name]=error(left,right)
        material_policy='S3_NONZERO_LAYER_HISTORY'
    observed=(np.linalg.norm(u.reshape(n,6)[:,3:]),np.linalg.norm(state.multipliers),
              work0,np.linalg.norm(br),np.linalg.norm(residual[list(owner.free)]))
    if not np.isfinite(observed).all():
        raise ValueError('nonfinite F06 loaded-state observation')
    shell_response_nonzero=bool(np.linalg.norm(sr0)>1e-8 and np.linalg.norm(sj0)>1e-8)
    passed=active and unchanged and shell_response_nonzero and all(v<=1e-9 for v in metrics.values())
    return {'topology':topology,'curved':curved,'cursor':state.cursor,
            'status':'DIAGNOSTIC_F06_PASS_NOT_G0B' if passed else 'DIAGNOSTIC_F06_FAIL_NOT_G0B',
            'nontrivial_loaded_state':active,'accepted_state_checkpoint_unchanged':unchanged,
            'nonzero_shell_force_tangent':shell_response_nonzero,
            'shell_rotation_norm':float(np.linalg.norm(u.reshape(n,6)[:,3:])),
            'joint_multiplier_norm':float(np.linalg.norm(state.multipliers)),
            'shell_work':float(work0),'beam_residual_norm':float(np.linalg.norm(br)),
            'metrics':metrics,'support_count':len(fixed),
            'candidate_numeric_count':len(values0),
            'candidate_numeric_norm':numeric_norm,'candidate_paths_equal':True,
            'material_state_policy':material_policy,
            'coupled_free_residual_norm':float(np.linalg.norm(residual[list(owner.free)])),
            'checkpoint_sha256':hashlib.sha256(checkpoint).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=Path(__file__).read_bytes()
    tracked=('scripts/'+Path(__file__).name,
             'src/anyfesolver/_ge_beam3_g3c_local_shell.py',
             'src/anyfesolver/_ge_beam3_pose_joint.py',
             'src/anyfesolver/_ge_beam3_shell_joint_state.py',
             'src/anyfesolver/_ge_beam3_shell_joint_trial.py',
             'src/anyfesolver/_ge_beam3_coupled_beam_subdomain.py',
             'src/anyfesolver/_ge_beam3_native_generalized_element.py',
             'src/anyfesolver/_ge_beam3_generalized_ellipsoid_section.py')
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],
                                  cwd=ROOT,text=True).strip()
    if dirty:raise RuntimeError('frozen F06 tracked source graph changed')
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in tracked}
    manifest={'source_commit':commit,'source_sha256':hashlib.sha256(source).hexdigest(),
              'tracked_sha256':hashes,'cases':CASES,'loaded_cursor':2,
              'relative_tolerance':1e-9,'normalization_floor':1.,
              'numba_cache':os.environ['NUMBA_CACHE_DIR'],'blas_threads':1,
              'anyfesolver_import':str(Path(anyfesolver.__file__).resolve()),
              'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
              'numba':numba.__version__,'scope':'BOUNDED_REPRESENTATION_V3_NOT_CHECKPOINT_REBASE'}
    args.root.mkdir(parents=True,exist_ok=False)
    (args.root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    failed=False
    with threadpool_limits(limits=1),(args.root/'results.jsonl').open('x',encoding='utf-8') as output:
        for topology,curved in CASES:
            try:
                row=case(topology,curved)
            except Exception as exc:
                row={'topology':topology,'curved':curved,'status':'DIAGNOSTIC_F06_EXCEPTION_NOT_G0B',
                     'failure_type':type(exc).__name__,'failure':str(exc)}
            output.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');output.flush()
            failed |= row['status']!='DIAGNOSTIC_F06_PASS_NOT_G0B'
            print(json.dumps({'topology':topology,'curved':curved,'status':row['status'],
                              'failure':row.get('failure'),'metrics':row.get('metrics')},sort_keys=True),flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
