"""Unregistered B2 weak-shear graph feasibility screen, not F09 evidence."""

from __future__ import annotations

from fractions import Fraction
import json
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from anyfesolver._b2_physical_flexibility_linear import LinearOperator, LinearSection


def screen(members: int, area: float) -> dict:
    if members not in (1, 2):
        raise ValueError('bounded member inventory')
    section = LinearSection.isotropic(E=1., nu=0., area=area,
                                      Iy=1., Iz=1., J=1., ky=1., kz=1.)
    element = LinearOperator(1., section)
    local = element.evaluate(np.zeros(12)).tangent
    rows, columns, data = [], [], []
    for index in range(members):
        dofs=np.r_[np.arange(6*index,6*index+6),
                   np.arange(6*(index+1),6*(index+1)+6)]
        for i, row in enumerate(dofs):
            for j, column in enumerate(dofs):
                rows.append(int(row));columns.append(int(column))
                data.append(float(local[i,j]))
    k=sparse.coo_matrix((data,(rows,columns)),
                        shape=(6*(members+1),)*2).tocsr()
    force_density=Fraction(float(area))/2  # G A ky, with E=1, nu=0.
    force=float(force_density)
    f=np.zeros(k.shape[0]);f[6*members+1]=force
    free=np.arange(6,k.shape[0])
    result={'members':members,'area':area,'force_n':force,
            'weak_modal_stiffness':float(element.modal_stiffness[4]),
            'status':'UNREGISTERED_GRAPH_NOT_F09_ACCEPTANCE'}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',sparse.SparseEfficiencyWarning)
            factor=splu(k[free][:,free].tocsc())
        u=np.zeros(k.shape[0]);u[free]=factor.solve(f[free])
        reaction=k@u-f
        exact_tip=force_density*(Fraction(members**3,3)
                                 +Fraction(members)/force_density)
        exact_rotation=force_density*Fraction(members**2,2)
        exact_energy=force_density*exact_tip/2
        tip=u[6*members+1]
        rotation=u[6*members+5]
        free_force=free[free%6<3]
        free_moment=free[free%6>=3]
        result.update(tip_y_m=float(tip),analytic_tip_y_m=float(exact_tip),
                      tip_relative_error=float(abs(tip-float(exact_tip))
                                               /float(exact_tip)),
                      tip_rz=float(rotation),
                      analytic_tip_rz=float(exact_rotation),
                      rotation_relative_error=float(abs(rotation-float(exact_rotation))
                                                    /float(exact_rotation)),
                      graph_energy_j=float(.5*u@(k@u)),
                      analytic_energy_j=float(exact_energy),
                      normalized_free_force_residual=float(
                          np.linalg.norm(reaction[free_force])/force),
                      normalized_free_moment_residual=float(
                          np.linalg.norm(reaction[free_moment])/(force*members)),
                      normalized_support_force_error=float(abs(reaction[1]+force)/force),
                      normalized_support_moment_error=float(abs(reaction[5]+force*members)
                                                            /(force*members)))
    except RuntimeError as exc:
        result.update(status='UNREGISTERED_GRAPH_FACTOR_FAILURE',
                      failure_type=type(exc).__name__,failure=str(exc))
    return result


if __name__ == '__main__':
    for count in (1,2):
        for area in (1e-4,2.**-80):
            print(json.dumps(screen(count,area),sort_keys=True,allow_nan=False))
