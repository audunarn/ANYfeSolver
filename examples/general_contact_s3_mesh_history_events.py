"""Declared equilibrated S3 friction subincrements; not event-free continuum history."""
from dataclasses import asdict,replace
import argparse,json
from pathlib import Path
import numpy as np
from examples.general_contact_s3_mesh_moving_friction import build_case as original
from anyfesolver import BoundaryCondition,solve_static_nonlinear
from anyfesolver.general_contact import S3MeshHistoryPathFrictionSettings,ContactPathControl,ContactPathNodalForce,ContactPathProgram

A=2.**-14;D=2.**-3

def build_case(ratio=2,workflow="pressure_event"):
    if workflow == "sliding":
        from examples.general_contact_s3_mesh_weak_pressure_friction import build_case as sliding_case
        case=list(sliding_case());case[3]=S3MeshHistoryPathFrictionSettings(**asdict(case[3]),physical_subincrements_per_user_increment=ratio)
        case[5]=dict(case[5],workflow="sliding",physical_subincrements=ratio,Fref_n=.02,Mref_nm=.02)
        return tuple(case)
    if workflow != "pressure_event":raise ValueError("unknown material history workflow")
    m,definition,s,_f,_p,_r=original()
    m.add_boundary_condition(BoundaryCondition('declared-target-slope-anchor',(7,10),{'uz':0.}))
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    K=assemble_stiffness_matrix(m)[0].toarray()
    # Literal BC/path selector plus existing MPC reduction, reused qualified K.
    # Independently build the exact fixture selector/MPC nullspace.
    columns=[]
    for node in (2,3,4,5):
        v=np.zeros(60);v[m.mesh.dof_manager.get_node_dofs(node)[2]]=1.;columns.append(v)
    for node in (7,8,10):
        v=np.zeros(60);v[m.mesh.dof_manager.get_node_dofs(node)[0]]=1.
        v[m.mesh.dof_manager.get_node_dofs(9)[0]]=1. if node in (8,10) else -1.;columns.append(v)
    P=np.array(columns).T
    support=np.array([.01/6,.01/3,.01/6,.01/3]);unit=np.zeros(60)
    for node,w in zip((2,3,4,5),support/.01):unit[m.mesh.dof_manager.get_node_dofs(node)[2]]=-w
    response=P@np.linalg.solve(P.T@K@P,P.T@unit)
    compliance=max(abs(response[m.mesh.dof_manager.get_node_dofs(n)[2]]) for n in (2,3,4,5))
    zseed=min(.001,(A*D/2)/(8*compliance))
    forces=[];poses=[]
    for slope,shift in ((-A,0.),(A,D)):
        u=np.zeros(60)
        for node in range(1,6):u[m.mesh.dof_manager.get_node_dofs(node)[0]]=shift
        for node in (2,3,4,5):
            dd=m.mesh.dof_manager.get_node_dofs(node);x=m.mesh.get_node(node).x
            u[dd[2]]=.015-(m.mesh.get_node(node).z-.025)+slope*(x+shift)
        for node in (8,9):u[m.mesh.dof_manager.get_node_dofs(node)[2]]=slope
        normal=np.array([-slope,0.,1.]);normal/=np.linalg.norm(normal)
        f=K@u
        for node,w in zip((2,3,4,5),support):f[m.mesh.dof_manager.get_node_dofs(node)[:3]]-=zseed*w/.01*normal
        for node,w in zip((7,8,9,10),(.36-shift,.30+shift,0.,.34)):
            f[m.mesh.dof_manager.get_node_dofs(node)[:3]]+=zseed*w*normal
        forces.append(f);poses.append(u)
    nodal=[]
    for node in range(1,11):
        for axis,name in enumerate(('ux','uy','uz')):
            j=m.mesh.dof_manager.get_node_dofs(node)[axis]
            if any(f[j]!=0 for f in forces):nodal.append(ContactPathNodalForce(node,name,(0.,float(forces[0][j]),float(forces[1][j]))))
    p=ContactPathProgram((ContactPathControl.node(2,'ux'),ContactPathControl.node(2,'uy'),ContactPathControl.node(8,'uz')),
                        ((0.,0.,0.),(0.,0.,-A),(D,0.,A)),1,tuple(nodal))
    s=replace(s,path_predictor='control_projection_dev',complementarity_force_scale_n=zseed)
    f=S3MeshHistoryPathFrictionSettings(.3,1000.,(2e-8,)*60,201,pressure_step_n=1e-8,
        finite_material_gap_error_m=5e-5,physical_subincrements_per_user_increment=ratio)
    return m,definition,s,f,p,dict(slope=A,slide=D,normal_seed_n=zseed,compliance_m_per_n=compliance,
        force_endpoints=[x.tolist() for x in forces],pose_endpoints=[x.tolist() for x in poses],
        Fref_n=.02,Mref_nm=.02,physical_subincrements=ratio,structural_K_reused=True,
        contact_endpoint_loads_independent_planar_P1=True,reference_scope='code_preparation_not_nonlinear_reference')

def solve_case(case,through_step=None,restart=None):
    m,d,s,f,p,_=case;total=p.total_steps*f.physical_subincrements_per_user_increment
    return solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,contact_path_program=p,
        max_iterations=18,tolerance=1e-12,max_load_factor=(total if through_step is None else through_step)/total,
        emit_restart_checkpoint=True,record_increment_snapshots=True,restart_checkpoint=restart)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ratio',type=int)
    p.add_argument('--workflow',choices=('pressure_event','sliding'));p.add_argument('--through-step',type=int);p.add_argument('--restart',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnose-material-work',type=Path,help='paired declared saved-data JSON; external evidence required for authenticity; no solve')
    a=p.parse_args()
    if a.diagnose_material_work:
        if a.restart or a.through_step is not None or a.ratio is not None or a.workflow is not None:raise ValueError('saved work diagnosis cannot request solver controls or continuation')
        from anyfesolver.general_contact.material_work_diagnostics import diagnose_work_bundle
        a.output.write_text(json.dumps(diagnose_work_bundle(json.loads(a.diagnose_material_work.read_text())),indent=2))
        return
    incoming=json.loads(a.restart.read_text()) if a.restart else None
    result=solve_case(build_case(2 if a.ratio is None else a.ratio,'pressure_event' if a.workflow is None else a.workflow),a.through_step,incoming)
    a.output.write_text(json.dumps(dict(status=result.status,info=result.info,checkpoint=result.to_restart_checkpoint() if result.status=='completed' else None),indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item()))
    if result.status!='completed':raise RuntimeError(result.info)
if __name__=='__main__':main()
