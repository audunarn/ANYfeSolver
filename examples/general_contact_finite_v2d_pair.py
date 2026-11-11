"""Opt-in elastic finite V2D-source/V2D-target press/slide/open/recontact.

All free bending/director coordinates use the existing consistent structural
owner. This development example is not an engineering admission record.
"""
from __future__ import annotations
from dataclasses import asdict
import argparse
import json
import math
from pathlib import Path
import numpy as np
from anyfesolver import FEModel, BoundaryCondition, create_shell_element, solve_static_nonlinear
from anyfesolver.general_contact import ContactSurface, ContactDefinition, ContactInteraction, ContactSolverSettings
from anyfesolver.general_contact.static_path import ContactPathControl, ContactPathNodalForce, ContactPathProgram
from examples.general_contact_finite_rotation_shell_work import rigid_pose, rotation

SCHEMA = 'ANYSOLVER_FINITE_V2D_PAIR_EXAMPLE_V1'


def build():
    model = FEModel('finite-v2d-pair-force-cycle')
    model.add_material('elastic', 1e5, .3, density=1.)
    source=((.1,.13,.0400002),(.95,.13,.0400002),(.1,.98,.0400002),(.95,.98,.0400002))
    target=((0.,0.,-.01),(1.2,0.,-.01),(0.,1.2,-.01),(1.2,1.2,-.01))
    for nid,xyz in enumerate((*source,*target),1):model.add_node(nid,*xyz)
    for eid,nodes in ((1,(1,2,4)),(2,(1,4,3)),(3,(5,6,8)),(4,(5,8,7))):
        model.add_element(eid,create_shell_element(eid,nodes,'elastic',thickness=.05,
            reference_surface_offset=.01,reference_normal=(0.,0.,1.),formulation='e4-pl-s3-v2d'))
    # A physical in-plane yaw support; all target rotations remain genuine
    # owner unknowns except the explicit six-coordinate reference anchor.
    model.add_boundary_condition(BoundaryCondition('target-yaw',(6,),{'uy':0.}))
    definition = ContactDefinition((ContactSurface('v2d_source',(1,2),'bottom'),
                                    ContactSurface('v2d_target',(3,4),'top')),(),
        (ContactInteraction('v2d_pair','v2d_source','v2d_target',1000.,5e-5,'augmented'),))
    settings = ContactSolverSettings(pair_motion='general',normal_discretization='primal_mortar',
        general_s3_differentiation='jet2_s3_dev',general_s3_difference_step=2e-5,
        q4_source_gap_certificate='whole_face',augmentation_support_resolution='exact',
        q4_source_area_tolerance=1e-9,q4_source_weighted_gap_tolerance=1e-10,
        q4_source_support_gradient_tolerance=1e-5,q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5,q4_source_weighted_gap_hessian_tolerance=1e-5,
        maximum_candidate_pairs=8,maximum_q4_source_integration_triangles=128,
        enforcement_solver='monolithic_active_set',complementarity_force_scale_n=1e-4,
        pressure_tolerance=1e-8,shell_pair_orientation='physical_sides_dev',
        shell_skin_kinematics='corotational_nodal_director_dev',
        finite_frame_seed='per_surface_reference_frames_dev')
    names=('ux','uy','uz','rx','ry','rz')
    terms=[(nid,name) for nid in (1,3,5) for name in names]+[(nid,'uz') for nid in (6,7,8)]
    controls=tuple(ContactPathControl.node(nid,name,coordinate_profile='finite_shell_spin_chart_dev')
                   for nid,name in terms)
    angles=[math.pi*i/36. for i in range(13)]+[math.pi/3.]*4
    waypoints=[]
    for step,angle in enumerate(angles):
        q=rigid_pose(model,angle)
        if step>=14:
            for nid in range(1,5): q[model.mesh.dof_manager.get_node_dofs(nid)[1]]+=.02
        waypoints.append(tuple(float(q[model.mesh.dof_manager.get_node_dofs(nid)[names.index(name)]])
                               for nid,name in terms))
    normal=rotation(math.pi/3.)@np.array((0.,0.,1.))
    scalar=(0.,)*13+(-5e-5,-5e-5,5e-5,-5e-5)
    forces=tuple(ContactPathNodalForce(nid,name,tuple(v*normal[axis] for v in scalar))
                  for nid in (2,4) for axis,name in enumerate(names[:3]) if normal[axis]!=0.)
    program=ContactPathProgram(controls,tuple(waypoints),1,forces,control_profile='finite_shell_triplets_dev')
    return model,definition,settings,program


def solve(model,definition,settings,program,**kwargs):
    return solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_path_program=program,kinematics='corotational',corotational_tangent='consistent',
        tolerance=5e-12,max_iterations=18,emit_restart_checkpoint=True,
        record_increment_snapshots=True,**kwargs)


def json_value(value):
    if hasattr(value,'tolist'): return value.tolist()
    if isinstance(value,dict): return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [json_value(v) for v in value]
    return value


def report_result(result,model,definition,settings,program):
    # Persist exactly the returned accepted suffix, even on a non-success.
    # Snapshots precede terminal owner sealing; keep their records distinct.
    poses=[]
    for snapshot in result.snapshots:
        poses.append(dict(snapshot.to_dict(include_displacements=True),
            element_states=json_value(snapshot.element_states)))
    return dict(schema=SCHEMA,status=result.status,failure_reason=result.failure_reason,
        fixture=dict(nodes=8,elements=4,dofs=48,source_elements=[1,2],target_elements=[3,4],pressure_basis='shared_nodal_P1',
                     contact=asdict(definition),settings=asdict(settings),program=program.contract()),
        accepted_suffix=poses,displacements=json_value(result.displacements),
        history=json_value(result.info.get('force_displacement_history',[])),
        diagnostics=json_value(result.info),checkpoint=json_value(result.restart_checkpoint),
        snapshot_state_scope='producer-returned pre-terminal-seal records',
        scientific_acceptance=False,released=False)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--through-step',type=int,choices=range(1,17),default=4)
    parser.add_argument('--restart');parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);output=Path(args.output)
    if output.exists(): raise ValueError('output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    result=None
    try:
        checkpoint=None
        if args.restart:
            payload=json.loads(Path(args.restart).read_text(encoding='utf-8'))
            checkpoint=payload.get('checkpoint') if payload.get('schema')==SCHEMA else payload
            if not isinstance(checkpoint,dict): raise ValueError('requested restart checkpoint missing')
        model,definition,settings,program=build()
        result=solve(model,definition,settings,program,max_load_factor=args.through_step/16.,
                     restart_checkpoint=checkpoint)
        report=report_result(result,model,definition,settings,program)
    except Exception as exc:
        report=dict(schema=SCHEMA,status='NONPASS',failure_reason=f'{type(exc).__name__}: {exc}',
                    scientific_acceptance=False,released=False)
        if result is not None:
            report['returned_accepted_suffix']=[s.to_dict(include_displacements=True) for s in result.snapshots]
            report['checkpoint']=json_value(result.restart_checkpoint)
            report['serialization_failure']=True
    temporary=output.with_suffix(output.suffix+'.tmp')
    temporary.write_text(json.dumps(report,default=json_value,allow_nan=False),encoding='utf-8')
    temporary.replace(output)
    print(json.dumps(dict(status=report['status'],failure_reason=report.get('failure_reason'),
                         accepted_suffix_count=len(report.get('accepted_suffix',[])),output=str(output))))
    return 0 if report['status']=='completed' else 1


if __name__=='__main__': raise SystemExit(main())
