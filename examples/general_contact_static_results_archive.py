"""Actual ordinary S3 finite-tool contact result export and detached read."""
from examples.general_contact_triangulated_rigid import case
from anyfesolver import LoadCase, solve_static_nonlinear
from anyfesolver.general_contact import read_static_contact_results


def analysis():
    model, definition, settings = case()
    load = LoadCase('archive_press')
    for node in (2, 3): load.add_nodal_load(node, forces=(0., 0., -.05))
    result = solve_static_nonlinear(model, load, contact=definition, contact_settings=settings,
        kinematics='von_karman', num_steps=2, max_iterations=15,
        emit_restart_checkpoint=True, record_increment_snapshots=True)
    return result, read_static_contact_results(result.to_contact_archive_bytes())


if __name__ == '__main__':
    result, archive = analysis()
    print(result.status, archive.sha256, len(archive.to_bytes()))
