"""Headless development: affine Q4 plasticity and fixed-plane normal impact.

The membrane and normal modes are uncoupled. This example observes concurrent
plastic history and contact; it does not establish impact-driven plasticity,
plastic work balance, engineering acceptance or qualification.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction, ContactSurface,
    DNVC208MaterialCurve, FEModel, RigidContactBody, create_shell_element,
)
from anyfesolver.dynamics import TransientConfig
from anyfesolver.general_contact.transient import (
    ContactTransientSolverControls, solve_transient_contact,
)

E, NU = 1.e5, .3


def build_model():
    """Unit square with affine deviatoric strain and rigid normal translation."""
    model = FEModel('affine-plastic-plane-impact-dev')
    curve = DNVC208MaterialCurve(
        sigma_prop=20., sigma_yield=20., sigma_yield_2=20.,
        eps_p_y1=.005, eps_p_y2=.01, K=20., n=.2,
    )
    model.add_material('plastic', E, NU, density=1., yield_stress=20.,
                       hardening_curve=curve)
    for node, (x, y) in enumerate(
            ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)), 1):
        model.add_node(node, x, y, 0.)
    model.add_element(1, create_shell_element(
        1, [1, 2, 3, 4], 'plastic', thickness=.05,
    ))
    model.add_boundary_condition(BoundaryCondition(
        'zero rotations', [1, 2, 3, 4], dict(rx=0., ry=0., rz=0.),
    ))
    modes = np.zeros((model.mesh.dof_manager.total_dofs, 2))
    master_e = model.mesh.dof_manager.get_node_dofs(2)[0]
    master_w = model.mesh.dof_manager.get_node_dofs(1)[2]
    for node in (1, 2, 3, 4):
        x, y = model.mesh.nodes[node].x, model.mesh.nodes[node].y
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        modes[dofs[0], 0], modes[dofs[1], 0], modes[dofs[2], 1] = x, -y, 1.
        for dof, master, factor in ((dofs[0], master_e, 2*x),
                                    (dofs[1], master_e, -2*y),
                                    (dofs[2], master_w, 1.)):
            if dof != master:
                model.add_constraint_equation(
                    terms=((dof, 1.), (master, -factor)),
                )
    definition = ContactDefinition(
        surfaces=(ContactSurface('skin', (1,)),),
        rigid_bodies=(RigidContactBody(
            'floor', 'plane', (0., 0., -.0008), (0., 0., 1.),
        ),),
        interactions=(ContactInteraction('support', 'skin', 'floor', 4.e7, 5.e-5),),
    )
    return model, modes, definition


def source_provenance():
    """Record actual imported source paths and content hashes."""
    names = ('anyfesolver', 'anyfesolver.fe_core', 'anyfesolver.dynamics',
             'anyfesolver.general_contact.transient',
             'anyfesolver.general_contact.core',
             'anyfesolver.general_contact.q4_material_checkpoint',
             'anyfesolver.e4_pl_element', 'anyfesolver.plasticity',
             DNVC208MaterialCurve.__module__)
    sources = {}
    for name in dict.fromkeys(names):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        sources[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    example_path = Path(__file__).resolve()
    return dict(imported_modules=sources, example=dict(
        path=str(example_path), sha256=hashlib.sha256(example_path.read_bytes()).hexdigest(),
    ))


def run(*, dt=2.5e-5, end=.003, events=False):
    model, modes, definition = build_model()
    result = solve_transient_contact(
        model, definition,
        TransientConfig(dt, end, initial_velocity=modes @ np.array([.2, -1.])),
        solver_controls=ContactTransientSolverControls(
            allow_q4_plastic_dev=True, save_q4_material_history_dev=True,
            equilibrium_tolerance=1.e-9,
            q4_plane_event_subdivision_dev=events,
        ),
    )
    projection = modes @ np.linalg.inv(modes.T @ modes)
    coordinates = result.displacements @ projection
    velocities = result.velocities @ projection
    alpha = np.array([np.max(state[1]['alpha'])
                      for state in result.material_state_history])
    plastic_strains = np.array([state[1]['plastic_strain']
                               for state in result.material_state_history])
    # This fixture has uniform [e, -e, 0] membrane strain and no bending.
    # Accepted compact states expose plastic strain, not computed layer stress.
    # Infer stress for diagnostics only; the solve still uses the production law.
    total_strains = np.zeros_like(plastic_strains)
    total_strains[..., 0] = coordinates[:, 0, None]
    total_strains[..., 1] = -coordinates[:, 0, None]
    elastic_matrix = E/(1.-NU*NU)*np.array(
        [[1., NU, 0.], [NU, 1., 0.], [0., 0., (1.-NU)/2.]],
    )
    stresses = (total_strains-plastic_strains) @ elastic_matrix.T
    sx, sy, txy = stresses[..., 0], stresses[..., 1], stresses[..., 2]
    equivalent_stress = np.max(np.sqrt(sx*sx-sx*sy+sy*sy+3*txy*txy), axis=1)
    yielded = alpha > 1.e-12
    active = np.array([item.active_pressure_dofs > 0 for item in result.contact])
    first_yield = np.flatnonzero(yielded)
    final_plastic_strain = np.asarray(result.element_states[1]['plastic_strain'])
    final_stress = stresses[-1]
    tau=stresses[:,0,0]
    energy=(.5/120.*velocities[:,0]**2+.5*.05*velocities[:,1]**2
        +.05*tau*tau/(2.*E/2.6)+.05*20.*alpha
        +np.array([item.penalty_energy for item in result.contact]))
    return dict(
        status='development; not accepted or qualified',
        scope='uncoupled affine membrane plasticity and rigid-normal fixed-plane impact',
        profile=result.profile, dt_s=float(dt),
        initial_time_s=float(result.times[0]), final_time_s=float(result.times[-1]),
        saved_states=len(result.times), accepted_steps=len(result.newton_iterations),
        cutbacks=result.cutbacks,
        localized_contact_event_times_s=list(result.localized_contact_event_times),
        event_endpoint_requests=result.event_endpoint_evaluations,
        plane_event_stage_evaluations=result.plane_event_stage_evaluations,
        energy_diagnostic=dict(scope='fixture plateau branch only; zero external work',
            initial_j=float(energy[0]),final_j=float(energy[-1]),
            maximum_relative_defect=float(np.max(np.abs(energy/energy[0]-1.))),
            final_relative_defect=float(energy[-1]/energy[0]-1.)),
        modal_coordinate_order=['deviatoric_strain', 'normal_displacement_m'],
        initial_modal_coordinates=coordinates[0].tolist(),
        final_modal_coordinates=coordinates[-1].tolist(),
        modal_velocity_order=['deviatoric_strain_rate_per_s', 'normal_velocity_m_s'],
        initial_modal_velocities=velocities[0].tolist(),
        final_modal_velocities=velocities[-1].tolist(),
        maximum_alpha=float(np.max(alpha)),
        final_alpha=float(np.max(result.element_states[1]['alpha'])),
        peak_contact_reaction_n=float(max(np.linalg.norm(item.rigid_force)
                                         for item in result.contact)),
        minimum_endpoint_gap_m=float(min(item.minimum_gap for item in result.contact)),
        minimum_represented_path_gap_m=float(np.min(result.swept_minimum_gaps)),
        final_plastic_strain=dict(component_order=['xx', 'yy', 'engineering_xy'],
            minimum=np.min(final_plastic_strain, axis=0).tolist(),
            maximum=np.max(final_plastic_strain, axis=0).tolist()),
        final_elasticity_inferred_layer_stress_pa=dict(component_order=['xx', 'yy', 'xy'],
            minimum=np.min(final_stress, axis=0).tolist(),
            maximum=np.max(final_stress, axis=0).tolist()),
        observations=dict(
            yield_during_active_contact=bool(np.any(yielded & active)),
            first_yield_time_s=(float(result.times[first_yield[0]]) if len(first_yield) else None),
            elasticity_inferred_stress_decrease_after_yield=bool(np.any(
                yielded[:-1] & (np.diff(equivalent_stress) < -1.e-8))),
            final_elasticity_inferred_equivalent_stress_pa=float(equivalent_stress[-1]),
            peak_elasticity_inferred_equivalent_stress_pa=float(np.max(equivalent_stress)),
            contact_open_at_end=bool(not active[-1]),
            all_endpoint_gap_tolerances_met=bool(all(
                item.gap_tolerance_met for item in result.contact)),
        ),
        provenance=source_provenance(),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt', type=float, default=2.5e-5, help='Newmark step in seconds')
    parser.add_argument('--end', type=float, default=.003, help='End time in seconds')
    parser.add_argument('--events', action='store_true', help='Opt in to Q4 pressure-mode stage events')
    parser.add_argument('--output', type=Path, help='Also save the JSON report at this path')
    args = parser.parse_args()
    report = json.dumps(run(dt=args.dt, end=args.end, events=args.events), indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(report+'\n', encoding='utf-8')
    print(report)
