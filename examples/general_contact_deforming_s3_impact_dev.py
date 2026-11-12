"""Coupled normal-nodal source/target S3 impact with nonlinear surface work.

All seven physical normal nodal coordinates remain; shell rotations and in-plane
coordinates are prescribed. Explicit development profile, not impact acceptance.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import runpy
import numpy as np
from anyfesolver import ContactTransientSolverControls, TransientConfig, solve_transient_contact
from anyfesolver.matrix_assembly import assemble_stiffness_matrix

_fixture = runpy.run_path(str(Path(__file__).with_name('general_contact_nonuniform_s3_impact_dev.py')))
SUPPORT, KN, GAP = (_fixture[key] for key in ('SUPPORT','KN','GAP'))
MASS = np.diag(np.r_[20.*SUPPORT,np.full(3,49./3)])
CONTROLS = ContactTransientSolverControls(allow_normal_s3_skin_hermite_dev=True,
    save_constraint_reactions_dev=True,equilibrium_tolerance=1.e-10)


def build_model(*,target_xy=None,source_columns=1,normal_stiffness=KN):
    model,definition,_ = _fixture['build_model'](target_mode='xy',target_xy=target_xy,
        source_columns=source_columns,normal_stiffness=normal_stiffness)
    model.constraint_equations.clear()
    count=2*(source_columns+1)+3
    modes = np.zeros((6*count,count))
    for i,node in enumerate(range(1,count+1)):
        modes[model.mesh.dof_manager.get_node_dofs(node)[2],i] = 1.
    return model,definition,modes


def solve(*,dt=2.5e-5,end=.003,checkpoint=None,emit=False,
          controls=CONTROLS,cancellation_token=None,settings=None,target_xy=None,
          initial_normal_coordinates=None,source_columns=1,normal_stiffness=KN):
    model,definition,modes = build_model(target_xy=target_xy,source_columns=source_columns,
        normal_stiffness=normal_stiffness)
    count=modes.shape[1]
    initial=None
    if initial_normal_coordinates is not None:
        values=np.asarray(initial_normal_coordinates)
        if values.shape!=(count,) or values.dtype.kind not in 'fiu' or not np.all(np.isfinite(values)):
            raise ValueError('initial_normal_coordinates must name every finite real normal coordinate')
        initial=modes@values
    result = solve_transient_contact(model,definition,
        TransientConfig(dt,end,initial_displacement=initial,
                       initial_velocity=modes@np.r_[np.full(count-3,-.1),np.zeros(3)]),
        contact_settings=_fixture['SETTINGS'] if settings is None else settings,
        solver_controls=controls,restart_checkpoint=checkpoint,
        emit_restart_checkpoint=emit,cancellation_token=cancellation_token)
    return model,result,modes


def run(*,dt=2.5e-5,end=.003,controls=CONTROLS):
    model,result,modes = solve(dt=dt,end=end,controls=controls)
    q,v,a = (np.asarray(value)@modes for value in
             (result.displacements,result.velocities,result.accelerations))
    structural = modes.T@assemble_stiffness_matrix(model)[0]@modes
    energy = (.5*np.einsum('ij,jk,ik->i',v,MASS,v)
              +.5*np.einsum('ij,jk,ik->i',q,structural,q)
              +np.array([row.regularization_storage_j for row in result.contact]))
    return dict(status='development; not engineering acceptance',profile=result.profile,
        times_s=result.times.tolist(),q_m=q.tolist(),v_m_s=v.tolist(),a_m_s2=a.tolist(),
        maximum_source_normal_variation_m=float(np.max(np.ptp(q[:,:4],axis=1))),
        maximum_target_normal_variation_m=float(np.max(np.ptp(q[:,4:],axis=1))),
        momentum_error_kg_m_s=float(np.max(np.abs(v@np.diag(MASS)-v[0]@np.diag(MASS)))),
        reachable_coordinate_energy_relative_defect=float(np.max(np.abs(energy/energy[0]-1.))),
        contact=[row.to_dict() for row in result.contact],
        swept_minimum_gaps_m=result.swept_minimum_gaps.tolist(),cutbacks=result.cutbacks,
        cutback_diagnostics=result.cutback_diagnostics,
        localized_contact_event_times_s=result.localized_contact_event_times,
        event_endpoint_evaluations=result.event_endpoint_evaluations,
        newmark_stage_evaluations=result.plane_event_stage_evaluations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt',type=float,default=2.5e-5)
    parser.add_argument('--end',type=float,default=.003)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--polygon-pressure-bounds',action='store_true')
    args = parser.parse_args()
    controls=replace(CONTROLS,normal_s3_polygon_pressure_bounds_dev=True) if args.polygon_pressure_bounds else CONTROLS
    text = json.dumps(run(dt=args.dt,end=args.end,controls=controls),indent=2,allow_nan=False)
    if args.output:
        args.output.write_text(text+'\n',encoding='utf-8')
    else:
        print(text)
