"""Development: free isotropic sphere and round B2/B3 beam coupled impact."""
import argparse

import numpy as np

from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, RigidContactBody)
from anyfesolver.dynamics import TransientConfig
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.transient import (
    ContactFreeRigidBody, ContactTransientSolverControls, solve_transient_contact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quadratic', action='store_true')
    args = parser.parse_args()
    model = FEModel('free_sphere_beam_impact_dev')
    model.add_material('elastic', 10. if args.quadratic else 1e3, .3, density=1.)
    for node, x in enumerate((-.01, 0., .01) if args.quadratic else (-.01, .01), 1):
        model.add_node(node, x, 0., .025)
    cls = QuadraticBeamElement if args.quadratic else BeamElement
    model.add_element(1, cls(1, list(model.mesh.nodes), 'elastic',
        dict(area=.05, Iy=1e-4, Iz=1e-4, J=1e-4)))
    model.add_boundary_condition(BoundaryCondition('guide', list(model.mesh.nodes),
        {name: 0. for name in ('ux', 'uy', 'rx', 'ry', 'rz')}))
    definition = ContactDefinition(
        surfaces=(ContactSurface('beam', (1,), beam_radius=.01, beam_end_caps='none'),),
        rigid_bodies=(RigidContactBody('sphere', 'sphere', (0., 0., 0.), radius=.01),),
        interactions=(ContactInteraction('pair', 'beam', 'sphere', None, .002, line_stiffness=1e4),))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    dofs = [model.mesh.dof_manager.get_node_dofs(node)[2] for node in model.mesh.nodes]
    initial[dofs] = -.2
    body = ContactFreeRigidBody('sphere', .002, 8e-8, (0., 0., .2), (2., -3., 4.))
    result = solve_transient_contact(model, definition,
        TransientConfig(.00025, .025, initial_velocity=initial),
        contact_settings=ContactSolverSettings(pair_motion='general', line_integration_relative_tolerance=1e-8),
        solver_controls=ContactTransientSolverControls(allow_classical_beam_dev=True, allow_free_sphere_dev=True),
        free_rigid_bodies=(body,))
    weights = .001*np.array([1/6, 2/3, 1/6] if args.quadratic else [.5, .5])
    momentum = [weights @ velocity[dofs]+sample['sphere'].linear_momentum_kg_m_s[2]
                for velocity, sample in zip(result.velocities, result.free_rigid_bodies)]
    print(dict(profile=result.profile, accepted_steps=len(result.newton_iterations),
        maximum_residual_n=float(np.max(result.equilibrium_residuals)),
        minimum_swept_gap_m=float(np.min(result.swept_minimum_gaps)),
        maximum_contact_force_n=float(max(np.linalg.norm(contact.beam_force) for contact in result.contact)),
        normal_momentum_error_kg_m_s=float(np.max(np.abs(np.asarray(momentum)-momentum[0]))),
        final_sphere_velocity_m_s=result.free_rigid_bodies[-1]['sphere'].velocity_m_s,
        final_active_line_cells=result.contact[-1].active_line_cells))


if __name__ == '__main__':
    main()
