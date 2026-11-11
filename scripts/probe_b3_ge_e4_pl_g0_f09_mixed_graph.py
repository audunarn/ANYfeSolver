"""Exploratory mixed-load B2 graph against a separate cantilever formula.

The exact Fraction oracle integrates diagonal-section force, shear and
bending work; it does not use the candidate element's flexibility matrix.
This is a bounded development diagnostic, not F09 qualification.
"""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

# The private module is self-contained. Avoid importing the whole ANYfeSolver
# package merely to run a local conditioning diagnostic.
_module_path = (Path(__file__).resolve().parents[1] / "src" / "anyfesolver"
                / "_b2_physical_flexibility_linear.py")
_spec = importlib.util.spec_from_file_location("_f09_mixed_graph_local_operator",
                                                _module_path)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
LinearOperator = _module.LinearOperator
LinearSection = _module.LinearSection


def _analytical_tip(length: int, area: float, load: np.ndarray) -> np.ndarray:
    """Castigliano derivative for a straight rooted isotropic cantilever."""
    l = Fraction(length)
    ea = Fraction(float(area))
    ga = ea / 2  # E=1, nu=0, ky=kz=1.
    ei = Fraction(1)
    gj = Fraction(1, 2)
    n, vy, vz, torque, my, mz = (Fraction(float(v)) for v in load)
    return np.array([float(v) for v in (
        n*l/ea,
        vy*l/ga + vy*l**3/(3*ei) + mz*l**2/(2*ei),
        vz*l/ga + vz*l**3/(3*ei) - my*l**2/(2*ei),
        torque*l/gj,
        my*l/ei - vz*l**2/(2*ei),
        mz*l/ei + vy*l**2/(2*ei),
    )])


def screen(members: int, area: float) -> dict:
    if members not in (1, 2) or area not in (2.**-12, 2.**-80):
        raise ValueError("bounded mixed-graph inventory")
    section = LinearSection.isotropic(E=1., nu=0., area=area,
                                      Iy=1., Iz=1., J=1., ky=1., kz=1.)
    operator = LinearOperator(1., section)
    local = operator.evaluate(np.zeros(12)).tangent
    size = 6*(members+1)
    rows, columns, data = [], [], []
    for index in range(members):
        dofs = np.r_[np.arange(6*index, 6*index+6),
                     np.arange(6*(index+1), 6*(index+1)+6)]
        for i, row in enumerate(dofs):
            for j, column in enumerate(dofs):
                rows.append(int(row))
                columns.append(int(column))
                data.append(float(local[i, j]))
    stiffness = sparse.coo_matrix((data, (rows, columns)),
                                  shape=(size, size)).tocsr()
    stiffness.sum_duplicates()
    ea = float(area)
    ga = ea/2
    load = np.array((.75*ea, 1.25*ga, -.75*ga, .125, .125, -.25))
    expected = _analytical_tip(members, area, load)
    force = np.zeros(size)
    force[-6:] = load
    free = np.arange(6, size)
    result = dict(
        status="UNREGISTERED_MIXED_GRAPH_DIAGNOSTIC",
        members=members, area=area, tip_load=load.tolist(),
        analytic_tip=expected.tolist(),
        physical_reference="exact Castigliano diagonal-section cantilever",
        supported_root_dofs=list(range(6)),
        limitations=("straight collinear members", "isotropic scalar section",
                     "not shell attached", "not F09 qualification"),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", sparse.SparseEfficiencyWarning)
            factor = splu(stiffness[free][:, free].tocsc())
        displacement = np.zeros(size)
        displacement[free] = factor.solve(force[free])
        if not np.all(np.isfinite(displacement)):
            return dict(result, status="UNREGISTERED_GRAPH_NONFINITE_RESPONSE")
        reaction = stiffness@displacement-force
        tip = displacement[-6:]
        support = reaction[:6]
        support_expected = np.array((
            -load[0], -load[1], -load[2],
            -load[3], -load[4]+members*load[2],
            -load[5]-members*load[1],
        ))
        component_error = np.abs(tip-expected)/np.maximum(np.abs(expected), 1e-30)
        force_scale = max(np.max(np.abs(load[:3])), 1e-30)
        moment_scale = max(np.max(np.abs(load[3:])), 1e-30)
        result.update(
            computed_tip=tip.tolist(),
            tip_component_relative_error=component_error.tolist(),
            support_reaction=support.tolist(),
            analytic_support_reaction=support_expected.tolist(),
            support_force_error=float(np.linalg.norm(support[:3]-support_expected[:3])
                                      /force_scale),
            support_moment_error=float(np.linalg.norm(support[3:]-support_expected[3:])
                                       /moment_scale),
            free_force_residual=float(np.linalg.norm(reaction[free[free%6<3]])
                                      /force_scale),
            free_moment_residual=float(np.linalg.norm(reaction[free[free%6>=3]])
                                       /moment_scale),
            graph_energy=float(.5*displacement@(stiffness@displacement)),
            analytic_energy=float(.5*load@expected),
        )
    except RuntimeError as exc:
        result.update(status="UNREGISTERED_GRAPH_FACTOR_FAILURE",
                      failure_type=type(exc).__name__, failure=str(exc))
    return result


if __name__ == "__main__":
    for count in (1, 2):
        for area_value in (2.**-12, 2.**-80):
            print(json.dumps(screen(count, area_value), sort_keys=True,
                             allow_nan=False), flush=True)
