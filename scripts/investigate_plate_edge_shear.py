"""Fixed-left plate, opposite edge translated parallel to itself.

Development investigation, not finite-strain or buckling qualification. No
out-of-plane restraint or yield law is introduced. The optional sampled nodal
imperfection is 1 mm at the center; the coarse bilinear mesh does not preserve
the analytic zero edge slope. Corotational
kinematics admit rigid rotation with small element deformation, not finite strain.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,platform,time,traceback
import numpy as np
from threadpoolctl import threadpool_info
from anyfesolver import FEModel,BoundaryCondition,solve_linear,solve_static_nonlinear
from anyfesolver.elements import create_shell_element
from anyfesolver.nonlinear_static import _assemble_nonlinear_system

def plate(target,imperfect=False,divisions=2):
    nside=divisions+1
    m=FEModel('fixed-left-edge-shear')
    m.add_material('ideal-elastic',210e9,.3)
    for j in range(nside):
        for i in range(nside):
            x,y=i/divisions,j/divisions
            z=.001*np.sin(np.pi*x)**2*np.sin(np.pi*y) if imperfect else 0.
            m.add_node(1+nside*j+i,x,y,z)
    for j in range(divisions):
        for i in range(divisions):
            n=1+nside*j+i;eid=1+divisions*j+i
            m.add_element(eid,create_shell_element(eid,[n,n+1,n+nside+1,n+nside],'ideal-elastic',thickness=.01))
    m.add_boundary_condition(BoundaryCondition('left-clamp',[1+nside*j for j in range(nside)],dict.fromkeys(['ux','uy','uz','rx','ry','rz'],0.)))
    m.add_boundary_condition(BoundaryCondition('right-shear',[nside*(j+1) for j in range(nside)],{'uy':target}))
    m.apply_boundary_conditions();return m

def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if hasattr(value,'to_dict'):return value.to_dict()
    return str(value)

def investigate(mode,output,divisions=2,tangent_request="consistent",stability_control=None):
    nside=divisions+1
    target=1e-6 if mode.startswith('tiny') else .01
    imperfect=mode=='imperfect-cr'
    kinematics='von_karman' if mode=='tiny-vk' else 'corotational'
    tangent=tangent_request if kinematics=='corotational' else 'auto'
    result={'mode':mode,'target':target,'kinematics':kinematics,'corotational_tangent':tangent,'fixture':{'mesh':[divisions,divisions],'length':1.,'thickness':.01,'E':210e9,'nu':.3,'left':'all six fixed','right':'uy only prescribed; other DOFs free','other_edges':'free','material':'ideal elastic, no yield limit','imperfection':'.001*sin(pi*x)^2*sin(pi*y)' if imperfect else None},'controls':{'num_steps':4,'max_iterations':25,'tolerance':1e-6,'control':'force','load_case':None,'stability_control':stability_control},'python':platform.python_version(),'numpy':np.__version__,'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']}}
    repo=pathlib.Path(__file__).resolve().parents[1]
    paths=[pathlib.Path(__file__)]+[repo/'src/anyfesolver'/n for n in ['nonlinear_static.py','_plate_stability_continuation.py','corotational.py','elements.py','e4_pl_element.py','assembly.py']]
    result['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def save():output.write_text(json.dumps(result,default=jsonable,indent=2))
    events=[]
    def progress(event):
        events.append(event);result['progress']=events;save()
    m=plate(target,imperfect,divisions)
    linear_model=plate(target,imperfect,divisions);start=time.perf_counter();linear,linear_info=solve_linear(linear_model,None)
    result['linear_wall']=time.perf_counter()-start;result['linear_displacements']=linear;save()
    start=time.perf_counter()
    try:
        r=solve_static_nonlinear(m,load_case=None,num_steps=4,max_iterations=25,kinematics=kinematics,corotational_tangent=tangent,record_increment_snapshots=True,progress_callback=progress,stability_control=stability_control)
        result.update(status=r.status,converged=r.converged,solve_wall=time.perf_counter()-start,load_factor=r.load_factor,steps=[s.to_dict() for s in r.steps],displacements=r.displacements,info=r.info)
        driven=np.array([m.mesh.nodes[n].dofs[1] for n in [nside*(j+1) for j in range(nside)]])
        fixed=np.array([d for n in [1+nside*j for j in range(nside)] for d in m.mesh.nodes[n].dofs])
        constrained=np.r_[fixed,driven];free=np.setdiff1d(np.arange(len(r.displacements)),constrained)
        result['snapshots']=[{'factor':s.load_factor,'u':s.displacements,'driven':s.displacements[driven],'clamp_max':float(np.max(abs(s.displacements[fixed])))} for s in r.snapshots]
        result['driven_error']=float(np.max(abs(r.displacements[driven]-target*r.load_factor)))
        result['clamp_error']=float(np.max(abs(r.displacements[fixed])))
        initial=np.array([[node.x,node.y,node.z] for node in m.mesh.nodes.values()])
        current=initial+r.displacements.reshape(-1,6)[:,:3]
        result['initial_coordinates']=initial
        result['current_coordinates']=current
        result['max_out_of_plane']=float(np.max(abs(r.displacements[2::6])))
        result['initial_z']=initial[:,2]
        result['incremental_z']=r.displacements[2::6]
        result['total_z']=current[:,2]
        # Independent geometric diagnostic, NOT a different constitutive law:
        # Green metric strains in the initial surface tangent at 2x2 Gauss points.
        metric_strains=[]
        for j in range(divisions):
            for i in range(divisions):
                n=nside*j+i;indices=[n,n+1,n+nside+1,n+nside]
                for xi in [-1/np.sqrt(3),1/np.sqrt(3)]:
                    for eta in [-1/np.sqrt(3),1/np.sqrt(3)]:
                        deriv=.25*np.array([[-(1-eta),-(1-xi)],[(1-eta),-(1+xi)],[(1+eta),(1+xi)],[-(1+eta),(1-xi)]])
                        a0=initial[indices].T@deriv;a=current[indices].T@deriv
                        _,r0=np.linalg.qr(a0,mode='reduced');f=a@np.linalg.inv(r0)
                        metric_strains.extend(np.linalg.eigvalsh(.5*(f.T@f-np.eye(2))).tolist())
        result['max_abs_surface_metric_strain']=float(np.max(abs(np.array(metric_strains))))
        result['linear_relative_displacement_difference']=float(np.linalg.norm(r.displacements-linear)/np.linalg.norm(linear))
        save()
        fint,_,_= _assemble_nonlinear_system(m,r.displacements,{},5,tangent=False,require_full_coordinates=True,kinematics=kinematics,corotational_tangent=tangent)
        result['internal_force']=fint
        result['free_residual_norm']=float(np.linalg.norm(fint[free]))
        result['relative_free_residual']=float(np.linalg.norm(fint[free])/max(np.linalg.norm(fint[constrained]),1.))
        forces=fint.reshape(-1,6);xyz=np.array([[node.x,node.y,node.z] for node in m.mesh.nodes.values()])+r.displacements.reshape(-1,6)[:,:3]
        result['full_force_resultant']=forces[:,:3].sum(axis=0)
        result['full_current_position_moment_resultant']=(forces[:,3:]+np.cross(xyz,forces[:,:3])).sum(axis=0)
        result['full_moment_definition']='raw rotational force components plus current-position cross translational force'
        if tangent=='work_conjugate':
            from anyfesolver.corotational import spatial_moment_from_rotation_covector
            moments=np.array([spatial_moment_from_rotation_covector(theta,p) for theta,p in zip(r.displacements.reshape(-1,6)[:,3:],forces[:,3:])])
            result['full_spatial_moment_resultant']=(moments+np.cross(xyz,forces[:,:3])).sum(axis=0)
        result['support_moment_definition']='current-position cross support force + constrained rotational reactions; all constrained rotations zero'
        reaction=np.zeros_like(fint);reaction[constrained]=fint[constrained];reaction=reaction.reshape(-1,6)
        result['support_force_resultant']=reaction[:,:3].sum(axis=0)
        result['support_current_position_moment_resultant']=(reaction[:,3:]+np.cross(xyz,reaction[:,:3])).sum(axis=0)
        result['driven_shear_reaction']=float(fint[driven].sum())
        result['threadpools']=threadpool_info();save()
        print(json.dumps({k:result[k] for k in ['mode','status','solve_wall','driven_error','clamp_error','relative_free_residual','linear_relative_displacement_difference','max_out_of_plane','driven_shear_reaction']},default=jsonable),flush=True)
    except Exception:
        result.update(exception=traceback.format_exc(),solve_wall=time.perf_counter()-start);save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['tiny-vk','tiny-cr','moderate-cr','imperfect-cr'],required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--divisions',type=int,choices=[2,4,8,16],default=2);p.add_argument('--tangent',choices=['consistent','work_conjugate'],default='consistent');p.add_argument('--stability-control',choices=['plate_energy'],default=None);a=p.parse_args();investigate(a.mode,a.output,a.divisions,a.tangent,a.stability_control)
