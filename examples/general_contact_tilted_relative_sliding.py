"""Finite Q4 source-only tilted sliding with independent supported target frame.

This new fixture prescribes all six target-anchor coordinates; prior common-frame
support definitions remain unchanged. Every free coordinate is still solved.
"""
from dataclasses import replace
import math,os,json
from pathlib import Path
import numpy as np
from anyfesolver import BoundaryCondition,solve_static_nonlinear
from anyfesolver.general_contact import resolve_automatic_contact_domain
from anyfesolver.general_contact.static_path import ContactPathControl,ContactPathNodalForce,ContactPathProgram
from examples.general_contact_q4_pair_monolithic_cycle import build as base
from examples.general_contact_finite_rotation_shell_work import rotation,rigid_pose


def build():
    model,domain,settings,_,_=base()
    model.name='q4-independent-frames-60deg-source-slide-20mm-free-bending'
    model.clear_boundary_conditions()
    model.add_boundary_condition(BoundaryCondition('target-yaw-anchor',(9,),{'uy':0.}))
    settings=replace(settings,shell_skin_kinematics='corotational_nodal_director_dev',finite_frame_seed='per_surface_reference_frames_dev')
    domain=replace(domain,name='q4-per-surface-frames-tilted-relative-slide')
    resolution=resolve_automatic_contact_domain(model,domain,settings,analysis_control='contact_path')
    terms=[(node,name) for node in (1,4,7) for name in ('ux','uy','uz','rx','ry','rz')]
    terms.extend((node,'uz') for node in (9,13,15))
    controls=tuple(ContactPathControl.node(node,name,coordinate_profile='finite_shell_spin_chart_dev') for node,name in terms)
    angles=[math.pi*i/36. for i in range(13)]+[math.pi/3.]*4
    names=dict(ux=0,uy=1,uz=2,rx=3,ry=4,rz=5);waypoints=[]
    for step,angle in enumerate(angles):
        q=rigid_pose(model,angle)
        if step>=14:
            for node in range(1,7):q[model.mesh.dof_manager.get_node_dofs(node)[1]]+=.02
        waypoints.append(tuple(float(q[model.mesh.dof_manager.get_node_dofs(node)[names[name]]]) for node,name in terms))
    normal=rotation(math.pi/3.)@np.array((0.,0.,1.));scalar=(0.,)*13+(-5e-5,-5e-5,5e-5,-5e-5)
    forces=tuple(ContactPathNodalForce(node,name,tuple(value*normal[axis] for value in scalar)) for node in (3,6) for axis,name in enumerate(('ux','uy','uz')) if normal[axis]!=0.)
    program=ContactPathProgram(controls,tuple(waypoints),1,forces,control_profile='finite_shell_triplets_dev')
    return model,domain,settings,resolution,program


def solve(model,settings,resolution,program,**kwargs):
    return solve_static_nonlinear(model,contact=resolution.definition,contact_settings=settings,contact_path_program=program,
        kinematics='corotational',corotational_tangent='consistent',tolerance=5e-12,max_iterations=18,
        emit_restart_checkpoint=True,record_increment_snapshots=True,**kwargs)


def main(argv=None):
    import argparse
    from examples.general_contact_tilted_relative_output import (load_restart,report_result,write,assess,fixture,controls,finite_vector,json_value)
    parser=argparse.ArgumentParser();parser.add_argument('--through-step',type=int,choices=(4,8,12,13,14,15,16),default=4);parser.add_argument('--restart')
    parser.add_argument('--output');parser.add_argument('--assess-saved')
    args=parser.parse_args(argv);output=args.output or os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if args.output and Path(args.output).exists():raise ValueError('explicit output exists')
    checkpoint=None;prefix=[];restart_source=None
    try:
        if args.assess_saved and args.restart:raise ValueError('assessment and restart are mutually exclusive')
        # Parse/verify requested restart BEFORE constructing any manager/solve.
        if args.restart:checkpoint,prefix,restart_source=load_restart(args.restart,args.through_step)
        model,domain,settings,resolution,program=build()
        if args.assess_saved:
            saved=json.loads(Path(args.assess_saved).read_text(encoding='utf-8'))
            report=assess(saved,model,settings,resolution,program)
            from examples.general_contact_tilted_relative_output import sha
            report['producer_file_sha256']=sha(args.assess_saved)
        else:
            descriptor,_=fixture(model,settings,resolution,program)
            if prefix:
                # Full report fixture identity/control consistency is checked
                # before advancement; earlier fields are still caller evidence.
                saved=json.loads(Path(args.restart).read_text(encoding='utf-8'))
                if saved['identity']!=resolution.identity or saved['program']!=program.contract():raise ValueError('restart fixture/program mismatch')
                assess(saved,model,settings,resolution,program)
            result=solve(model,settings,resolution,program,max_load_factor=args.through_step/16.,restart_checkpoint=checkpoint)
            # Failure-return snapshots are persisted too; no assertion discards
            # an accepted prefix before it can be saved.
            report=report_result(result,model,settings,resolution,program,prefix,restart_source,descriptor=descriptor)
    except Exception as exc:
        report=dict(status='NONPASS',failure_reason=f'{type(exc).__name__}: {exc}',scientific_acceptance=False)
    if output:write(output,report)
    print(json.dumps({k:report.get(k) for k in ('status','failure_reason','accepted_steps','accepted_step_total','assessed_poses','terminal_restart')},default=json_value))
    return 0 if report['status'] in ('completed','PASS') else 1

if __name__=='__main__':raise SystemExit(main())
