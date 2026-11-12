"""Finite exact V2D elastic shell; fixed finite planar tool; ordinary penalty path."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from anyfesolver import (FEModel, BoundaryCondition, ContactDefinition, ContactSurface,
    ContactInteraction, ContactSolverSettings, LoadCase, create_shell_element,
    NonlinearLoadProgram, NonlinearLoadStage, solve_static_nonlinear)
from anyfesolver.general_contact.triangulated_rigid import TriangulatedRigidContactBody

SCHEMA='ANYSOLVER_FINITE_V2D_FIXED_RIGID_EXAMPLE_V1'


def build():
    model=FEModel('finite-v2d-fixed-planar-rigid-penalty')
    model.add_material('elastic',1e5,.3,density=1.)
    coordinates=((.1,.13,.0350002),(.95,.13,.0350002),(.1,.98,.0350002),(.95,.98,.0350002))
    for node,p in enumerate(coordinates,1):model.add_node(node,*p)
    for eid,ids in ((1,(1,2,4)),(2,(1,4,3))):
        model.add_element(eid,create_shell_element(eid,ids,'elastic',thickness=.05,
            formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.),reference_surface_offset=.01))
    # A declared finite rigid seed with zero corotational constitutive strain.
    # All later increments use the ordinary structural/contact driver.
    angle=.5
    c,s=np.cos(angle),np.sin(angle)
    rotation=np.array(((c,0.,s),(0.,1.,0.),(-s,0.,c)))
    origin=np.array((.6,.6,0.))
    q=np.zeros(model.mesh.dof_manager.total_dofs)
    for nid,p in enumerate(coordinates,1):
        ids=model.mesh.dof_manager.get_node_dofs(nid)
        x=np.asarray(p);q[ids[:3]]=origin+rotation@(x-origin)-x
        q[ids[3:6]]=(0.,angle,0.)
    ids=model.mesh.dof_manager.get_node_dofs(1)
    model.add_boundary_condition(BoundaryCondition('physical-clamp',(1,),
        {key:float(value) for key,value in zip(('ux','uy','uz','rx','ry','rz'),q[ids[:6]])}))
    vertices=tuple(tuple(origin+rotation@(np.asarray(p)-origin))
        for p in ((.18,-.02,0.),(1.1,-.02,0.),(.18,1.1,0.),(1.1,1.1,0.)))
    tool=TriangulatedRigidContactBody('fixed-tool',vertices,((1,(0,1,3)),(2,(0,3,2))))
    definition=ContactDefinition((ContactSurface('skin',(1,2),'bottom'),),(tool,),
        (ContactInteraction('finite-penalty','skin','fixed-tool',1e6,5e-5,'penalty'),))
    settings=ContactSolverSettings(pair_motion='general',normal_discretization='primal_mortar',
        general_s3_difference_step=2e-6,general_s3_differentiation='jet2_s3_dev',
        shell_skin_kinematics='corotational_nodal_director_dev')
    normal=rotation@np.array((0.,0.,1.));slide=rotation@np.array((0.,1.,0.))
    stages=[]
    for name,vector in (('press',-5e-5*normal),('slide',1e-5*slide),
                        ('open',1e-4*normal-1e-5*slide),('recontact',-1e-4*normal+1e-5*slide)):
        load=LoadCase(name)
        for node in (2,4):load.add_nodal_load(node,forces=tuple(vector/2))
        stages.append(NonlinearLoadStage(name,load))
    return model,definition,settings,NonlinearLoadProgram(tuple(stages)),q


def solve(model,definition,settings,program,seed,*,through_stage=4,restart_checkpoint=None,**kwargs):
    return solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        load_program=program,max_load_factor=float(through_stage),num_steps=through_stage,
        kinematics='corotational',corotational_tangent='consistent',max_iterations=15,tolerance=1e-11,
        initial_displacements=seed if restart_checkpoint is None else None,
        restart_checkpoint=restart_checkpoint,emit_restart_checkpoint=True,
        record_increment_snapshots=True,**kwargs)


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    return value


def fixture(model,definition,settings,program,seed):
    from anyfesolver.nonlinear_restart import load_case_descriptor
    return dict(nodes=4,elements=2,dofs=model.mesh.dof_manager.total_dofs,
        contact=asdict(definition),settings=asdict(settings),initial_displacements=seed.tolist(),
        initial_rotation_rad=.5,force_scale_N=1e-4,moment_scale_Nm=1e-4,
        numerical_controls=dict(tolerance=1e-11,max_iterations=15,num_layers=5,kinematics='corotational',corotational_tangent='consistent'),
        program=[dict(name=p.name,target_factor=p.target_factor,
            loads=load_case_descriptor(p.load_case)) for p in program.stages],
        scope='fixed finite planar rigid target; shared P1 penalty; physically clamped node1; all other coordinates free')


def report_result(result,model,definition,settings,program,seed):
    poses=[dict(p.to_dict(include_displacements=True),element_states=json_value(p.element_states)) for p in result.snapshots]
    return dict(schema=SCHEMA,status=result.status,failure_reason=result.failure_reason,
        fixture=fixture(model,definition,settings,program,seed),accepted_suffix=poses,
        displacements=json_value(result.displacements),history=json_value(result.info.get('force_displacement_history',[])),
        diagnostics=json_value(result.info),checkpoint=json_value(result.restart_checkpoint),
        snapshot_state_scope='actual producer-returned pre-terminal-seal records',scientific_acceptance=False,released=False)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--through-stage',type=int,choices=range(2,5),default=4)
    parser.add_argument('--restart');parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);output=Path(args.output)
    if output.exists():raise ValueError('output already exists')
    output.parent.mkdir(parents=True,exist_ok=True);result=None
    try:
        model,definition,settings,program,seed=build();checkpoint=None
        if args.restart:
            payload=json.loads(Path(args.restart).read_text())
            if payload.get('schema')==SCHEMA:
                if payload.get('fixture')!=json_value(fixture(model,definition,settings,program,seed)):
                    raise ValueError('saved finite rigid fixture/program mismatch')
                checkpoint=payload.get('checkpoint')
            else:checkpoint=payload
            if not isinstance(checkpoint,dict):raise ValueError('requested checkpoint missing')
        result=solve(model,definition,settings,program,seed,through_stage=args.through_stage,restart_checkpoint=checkpoint)
        report=report_result(result,model,definition,settings,program,seed)
    except Exception as exc:
        report=dict(schema=SCHEMA,status='NONPASS',failure_reason=f'{type(exc).__name__}: {exc}',scientific_acceptance=False,released=False)
        if result is not None:
            report['returned_accepted_suffix']=[p.to_dict(include_displacements=True) for p in result.snapshots]
            report['checkpoint']=json_value(result.restart_checkpoint)
    temporary=output.with_suffix(output.suffix+'.tmp')
    temporary.write_text(json.dumps(report,default=json_value,allow_nan=False));temporary.replace(output)
    print(json.dumps(dict(status=report['status'],accepted_suffix_count=len(report.get('accepted_suffix',[])),output=str(output))))
    return 0 if report['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
