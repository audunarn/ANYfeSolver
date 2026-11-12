"""Explicit modal-only free-free Q4/native-GE point panel.

Six physical rigid fields and requested flexible pairs; experimental, not
qualified spectrum completeness, prestress, finite dynamics or consumer scope.
The empty-support policy deliberately rejects static and accepted-state writes.
"""
import json
import numpy as np
from anyfesolver import b3_ge, stiffened_shell as api


def main():
    e, nu, rho = 210e9, .3, 7850.
    g, b, h = e/(2*(1+nu)), .1, .1
    area = b*h; iy, iz = b*h**3/12., h*b**3/12.
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*area,.8*g*area,.8*g*area,g*(iy+iz),e*iy,e*iz)),
        np.eye(6),1e20,1e9)
    inertia = np.diag((rho*area,)*3+(rho*(iy+iz),rho*iy,rho*iz))
    beam = b3_ge.define_beam('b3-ge',17,(101,102,103),
        np.array(((0.,0.,.1),(.5,0.,.1),(1.,0.,.1))),
        np.tile(np.eye(3),(3,1,1)),section,inertia)
    shell = api.ShellPanel((1,2,3,4),
        ((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)),
        ((1,2,3,4),),.02,e,nu,rho)
    definition = api.LinearDefinition(api.B3_GE_E4_PL_STIFFENED_SHELL_V1,
        shell,(beam,),(api.PointConnection(1,101),api.PointConnection(2,103)),(),
        modal_boundary_policy=api.GE_FREE_FREE_MODAL_ONLY_V1)
    owner = api.create_analysis(definition=definition)
    try:
        result = owner.modes(api.GEReferenceModesRequest(3,1.))
        if result.status != 'completed': raise RuntimeError(result.failure_reason)
        flexible = result.flexible
        print(json.dumps(dict(status=result.status,policy=result.policy,
            rigid_mode_count=len(result.rigid_coordinates),
            rigid_origin_m=result.rigid_origin_m,
            rigid_kinetic_gram=result.rigid_kinetic_gram.tolist(),
            rigid_equation_residuals=result.rigid_equation_residuals,
            rigid_constraint_residuals=result.rigid_constraint_residuals,
            rigid_flexible_mass_overlap=result.rigid_flexible_mass_overlap,
            frequencies_hz=flexible.frequencies_hz,
            spectral_shift_rad2_s2=result.spectral_shift_rad2_s2,
            equation_residuals=flexible.equation_residuals,
            constraint_residuals=flexible.constraint_residuals,
            factorization_count=flexible.factorization_count,
            selection=result.selection,production_qualified=False),indent=2))
    finally: owner.close()


if __name__ == '__main__': main()
