"""Bounded fixed-plane S3 physical-rim replay; no C5 engineering acceptance."""
import argparse,json
from pathlib import Path
from anyfesolver import BoundaryCondition,create_shell_element,solve_static_nonlinear
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact import ContactDefinition,ContactInteraction,ContactSurface,ContactSolverSettings,ContactPathControl,ContactPathProgram
from anyfesolver.general_contact.s3_sliding_friction import S3SlidingFrictionSettings
from anyfesolver.general_contact.s3_friction_replay import S3RimReplayFrictionSettings

def build_case():
    m=FEModel('S3 fixed-plane physical-rim replay development')
    m.add_material('elastic',1e5,.3,density=1)
    xyz=((.2,.2,.0500002),(.6,.2,.0500002),(.2,.6,.0500002),(0,0,0),(1,0,0),(0,1,0),(1,1,0))
    for n,q in enumerate(xyz,1):m.add_node(n,*q)
    for e,ns in ((1,(1,2,3)),(2,(4,5,6)),(3,(5,7,6))):
        m.add_element(e,create_shell_element(e,ns,'elastic',thickness=.05,formulation='e4-pl-s3-v2d',reference_normal=(0,0,1),reference_surface_offset=.01))
    m.add_boundary_condition(BoundaryCondition('target_plane',(4,5,6,7),{d:0 for d in ('uz','rx','ry','rz')}))
    m.add_boundary_condition(BoundaryCondition('target_origin',(4,),{'ux':0,'uy':0}))
    m.add_boundary_condition(BoundaryCondition('target_orientation',(5,),{'uy':0}))
    m.add_boundary_condition(BoundaryCondition('target_tangential_guide',(6,7),{'uy':0}))
    m.add_boundary_condition(BoundaryCondition('source_guide',(1,2,3),{d:0 for d in ('uy','rx','ry','rz')}))
    for axis in (0,2):
        master=m.mesh.dof_manager.get_node_dofs(1)[axis]
        for n in (2,3):
            row=m.mesh.dof_manager.get_node_dofs(n)[axis]
            m.add_constraint_equation(terms=((row,1.),(master,-1.)),dependent_dof=row,source_id=f'declared_source_translation_{axis}_{n}')
    d=ContactDefinition((ContactSurface('source',(1,),'bottom'),ContactSurface('target',(2,3),'top')),(),
        (ContactInteraction('normal','source','target',1000.,5e-5,normal_law='projected_penalty'),))
    s=ContactSolverSettings(pair_motion='general',normal_discretization='primal_mortar',general_s3_difference_step=2e-7,general_s3_differentiation='jet2_s3_dev',path_predictor='control_projection_dev')
    material=S3SlidingFrictionSettings(.3,1000.,(2e-7,)*42,maximum_candidate_evaluations=145,activation_history='linear_constant_normal_touch_dev')
    f=S3RimReplayFrictionSettings(material,1,2e-9,2e-9,1e-12,event_subdivisions=4,prepared_pressure_paths=True,prepared_partial_pressure_paths=True)
    p=ContactPathProgram((ContactPathControl.node(1,'ux'),ContactPathControl.node(1,'uz')),
        ((0.,0.),(0.,-5.2e-6),(1e-6,-5.2e-6),(.55,-5.2e-6),(1.05,-5.2e-6),(1.04,-5.2e-6),(1.04,5e-6),(.55,-5.2e-6)),1)
    return m,d,s,f,p

def solve_case(case,through_step=None,restart=None,cancellation_token=None):
    m,d,s,f,p=case
    return solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,contact_path_program=p,
        max_iterations=18,tolerance=1e-12,max_load_factor=(p.total_steps if through_step is None else through_step)/p.total_steps,
        emit_restart_checkpoint=True,record_increment_snapshots=True,restart_checkpoint=restart,cancellation_token=cancellation_token)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--through-step',type=int);parser.add_argument('--restart',type=Path);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();incoming=json.loads(a.restart.read_text()) if a.restart else None
    result=solve_case(build_case(),a.through_step,incoming)
    a.output.write_text(json.dumps(dict(status=result.status,info=result.info,checkpoint=result.to_restart_checkpoint() if result.status=='completed' else None),indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item()))
    if result.status!='completed':raise RuntimeError(result.info)
if __name__=='__main__':main()
