"""Supported virgin elastic Q4/native-GE six-pose-point descriptor modes.

Explicit physical spatial section inertia and shell density are required.
This is an experimental requested-shift result, not a qualified lowest spectrum,
free-free, prestress, finite-history or footprint modal workflow.
"""
import json
import numpy as np
from anyfesolver import b3_ge, stiffened_shell as api


def main():
    e, nu, rho = 210e9, .3, 7850.
    g, b, h = e/(2*(1+nu)), .1, .1
    area = b*h; iy, iz = b*h**3/12., h*b**3/12.
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*area, .8*g*area, .8*g*area, g*(iy+iz), e*iy, e*iz)),
        np.eye(6), 1e20, 1e9)
    # Centroidal rectangular mass per metre, independent of section stiffness.
    inertia = np.diag((rho*area,)*3+(rho*(iy+iz), rho*iy, rho*iz))
    beam = b3_ge.define_beam("b3-ge", 17, (101, 102, 103),
        np.array(((0., 0., .1), (.5, 0., .1), (1., 0., .1))),
        np.tile(np.eye(3), (3, 1, 1)), section, inertia)
    panel = api.ShellPanel((1, 2, 3, 4),
        ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)),
        ((1, 2, 3, 4),), .02, e, nu, rho)
    definition = api.LinearDefinition(api.B3_GE_E4_PL_STIFFENED_SHELL_V1,
        panel, (beam,), (api.PointConnection(1, 101), api.PointConnection(2, 103)),
        tuple(api.FixedCoordinate("shell", node, axis)
              for node in (1, 4) for axis in range(6)))
    owner = api.create_analysis(definition=definition)
    try:
        result = owner.modes(api.GEReferenceModesRequest(count=3, shift_hz=1.))
        if result.status != "completed":
            raise RuntimeError(result.failure_reason)
        print(json.dumps(dict(status=result.status, frequencies_hz=result.frequencies_hz,
            selection=result.selection, mass_identity=result.mass_identity,
            factorization_count=result.factorization_count,
            full_native_coordinate_count=len(result.full_coordinates[0]),
            multiplier_count=len(result.multiplier_modes[0]),
            equation_residuals=result.equation_residuals,
            constraint_residuals=result.constraint_residuals,
            policy=result.policy, production_qualified=False), indent=2))
    finally:
        owner.close()


if __name__ == "__main__": main()
