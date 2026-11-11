"""Bounded F09 mixed-elbow coordinate-scaling development screen.

No product admission or public B2 behavior is changed by this probe.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import inspect
import types

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import splu

# Load the two private owner modules without executing the unrelated package
# initializer. This probe never substitutes an implementation for either.
_src = Path(__file__).resolve().parents[1] / "src" / "anyfesolver"
if __name__ == "__main__":
    _package = types.ModuleType("anyfesolver")
    _package.__path__ = [str(_src)]
    sys.modules["anyfesolver"] = _package
    for _name in ("_b2_physical_flexibility_linear", "_b2_physical_linear_definition"):
        _spec = importlib.util.spec_from_file_location(f"anyfesolver.{_name}",
                                                       _src / f"{_name}.py")
        assert _spec is not None and _spec.loader is not None
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _module
        _spec.loader.exec_module(_module)
else:
    import anyfesolver

from anyfesolver._b2_physical_flexibility_linear import LinearSection
_elbow_spec = importlib.util.spec_from_file_location(
    "_f09_elbow", Path(__file__).with_name(
        "probe_b3_ge_e4_pl_g0_f09_mixed_elbow_conditioning.py"))
assert _elbow_spec is not None and _elbow_spec.loader is not None
elbow = importlib.util.module_from_spec(_elbow_spec)
_elbow_spec.loader.exec_module(elbow)


# Area is in m², E in Pa. Other dimensions are the predecessor's fixed
# diagnostic elbow: lengths 2 and 1.5 m; Iy = Iz = J = 1 m^4. This is a
# deliberately anisotropic numerical stress family, not a catalog of
# geometrically realizable sections. Loads scale with E so responses remain
# comparable between the numerical and SI stiffness units.
CASES = (
    ("numerical_1e-4", 1.0, 1.0e-4),
    ("numerical_1e-8", 1.0, 1.0e-8),
    ("numerical_2^-40", 1.0, 2.0 ** -40),
    ("numerical_2^-80", 1.0, 2.0 ** -80),
    ("SI_1e-8", 210.0e9, 1.0e-8),
    ("SI_2^-80", 210.0e9, 2.0 ** -80),
)
TIP_TOL = 1.0e-9  # componentwise floor 1 m or 1 rad, diagnostic only
RESIDUAL_TOL = 1.0e-9  # relative to separate max applied N / Nm


class UnrepresentableScale(ValueError):
    """A positive diagonal coordinate transform is not binary64-safe."""


def _scaled_solve(matrix: sparse.csc_matrix, rhs: np.ndarray) -> tuple[np.ndarray, dict]:
    diagonal = matrix.diagonal()
    if not (np.isfinite(matrix.data).all() and np.isfinite(rhs).all()
            and np.isfinite(diagonal).all() and np.all(diagonal > 0)):
        raise UnrepresentableScale("matrix, load, or positive diagonal is not finite")
    # D has physical units sqrt(m/N) for translations and sqrt(rad/Nm)
    # for rotations. Thus D K D is dimensionless in each coordinate's
    # work-conjugate pair; no common N/Nm residual norm is used.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scale = 1.0 / np.sqrt(diagonal)
        transformed = (sparse.diags(scale, format="csc") @ matrix
                       @ sparse.diags(scale, format="csc")).tocsc()
        transformed_rhs = scale * rhs
    if not (np.isfinite(scale).all() and np.all(scale > 0)
            and np.isfinite(transformed.data).all()
            and np.isfinite(transformed_rhs).all()):
        raise UnrepresentableScale("diagonal transform overflow or underflow")
    if np.any((matrix.toarray() != 0) & (transformed.toarray() == 0)):
        raise UnrepresentableScale("diagonal transform erased a nonzero coupling")
    if np.max(np.abs(transformed.diagonal() - 1.0)) > 1.0e-14:
        raise UnrepresentableScale("unit diagonal cannot be represented")
    scaled_solution = splu(transformed, permc_spec="COLAMD",
                           diag_pivot_thresh=1.0).solve(transformed_rhs)
    with np.errstate(over="ignore", invalid="ignore"):
        solution = scale * scaled_solution
    if not np.isfinite(solution).all():
        raise UnrepresentableScale("unscaled response cannot be represented")
    return solution, {"scale_min": float(np.min(scale)),
                      "scale_max": float(np.max(scale))}


def _physical_tip(section: LinearSection, terminal: np.ndarray) -> np.ndarray:
    """Exact cut-wrench Castigliano derivative, independently of B2 matrix."""
    applied = tuple(Fraction(float(value)) for value in terminal)
    compliance = tuple(1 / Fraction(float(section.stiffness[i][i]))
                       for i in range(6))
    tip = [Fraction(0) for _ in range(6)]
    end = tuple(Fraction(x) for x in elbow.POINTS[-1])
    for member, frame in enumerate(elbow.FRAMES):
        start = tuple(Fraction(x) for x in elbow.POINTS[member])
        finish = tuple(Fraction(x) for x in elbow.POINTS[member + 1])
        length = Fraction(2) if member == 0 else Fraction(3, 2)
        for t, weight in ((Fraction(0), 1), (Fraction(1, 2), 4),
                          (Fraction(1), 1)):
            point = tuple(a + t * (b - a) for a, b in zip(start, finish))
            lever = tuple(a - b for a, b in zip(end, point))

            def local_wrench(action):
                force, couple = action[:3], action[3:]
                arm = elbow._cross(lever, force)
                moment = tuple(couple[i] + arm[i] for i in range(3))
                return tuple(sum(Fraction(axis[j]) * vector[j]
                                 for j in range(3))
                             for vector in (force, moment) for axis in frame)

            wrench = local_wrench(applied)
            for axis in range(6):
                unit = tuple(Fraction(int(i == axis)) for i in range(6))
                derivative = local_wrench(unit)
                tip[axis] += length * weight * sum(
                    derivative[j] * compliance[j] * wrench[j]
                    for j in range(6)) / 6
    return np.asarray([float(x) for x in tip])


def _case(name: str, E: float, area: float) -> dict:
    section = LinearSection.isotropic(E=E, nu=0., area=area,
                                      Iy=1., Iz=1., J=1., ky=1., kz=1.)
    terminal = E * np.array((area / 2, area / 3, area / 5,
                             .25, -.125, .375))
    load = np.zeros(18)
    load[12:] = terminal
    _, matrix, reduced = elbow._float_graph(elbow._members(section), load)
    free, scale_info = _scaled_solve(reduced, load[6:])
    displacement = np.zeros(18)
    displacement[6:] = free
    exact_rounded = np.asarray([float(x) for x in elbow._exact_solve(
        reduced.toarray(), load[6:])])[6:]
    physical = _physical_tip(section, terminal)
    tip = displacement[12:]
    residual = (matrix @ displacement - load)[6:]
    force_indices = [0, 1, 2, 6, 7, 8]
    moment_indices = [3, 4, 5, 9, 10, 11]
    force_ratio = float(np.max(np.abs(residual[force_indices])) /
                        np.max(np.abs(terminal[:3])))
    moment_ratio = float(np.max(np.abs(residual[moment_indices])) /
                         np.max(np.abs(terminal[3:])))
    translation_scale = np.maximum(1., np.abs(physical[:3]))
    rotation_scale = np.maximum(1., np.abs(physical[3:]))
    translation_error = float(np.max(np.abs(tip[:3] - physical[:3]) /
                                     translation_scale))
    rotation_error = float(np.max(np.abs(tip[3:] - physical[3:]) /
                                  rotation_scale))
    rounded_error = float(np.max(np.abs(exact_rounded - physical) /
                                 np.maximum(1., np.abs(physical))))
    sparse_rounded_error = float(np.max(np.abs(tip - exact_rounded) /
                                        np.maximum(1., np.abs(exact_rounded))))
    passed = (max(translation_error, rotation_error, rounded_error,
                  sparse_rounded_error) <= TIP_TOL
              and max(force_ratio, moment_ratio) <= RESIDUAL_TOL)
    return {"name": name, "E_Pa": E, "area_m2": area,
            "terminal_wrench_N_Nm": terminal.tolist(),
            "scaled_tip_m_rad": tip.tolist(),
            "exact_rounded_tip_m_rad": exact_rounded.tolist(),
            "independent_force_method_tip_m_rad": physical.tolist(),
            "scaled_vs_physical_translation": translation_error,
            "scaled_vs_physical_rotation": rotation_error,
            "exact_rounded_vs_physical": rounded_error,
            "scaled_vs_exact_rounded": sparse_rounded_error,
            "free_force_residual_to_applied_force": force_ratio,
            "free_moment_residual_to_applied_moment": moment_ratio,
            "passed_declared_diagnostic": bool(passed), **scale_info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    cases = []
    for case in CASES:
        print(f"running {case[0]}", flush=True)
        cases.append(_case(*case))
    negative = None
    try:
        _scaled_solve(sparse.diags([0.0, 1.0], format="csc"),
                      np.array([1.0, 1.0]))
    except UnrepresentableScale as exc:
        negative = str(exc)
    if negative is None:
        raise AssertionError("unrepresentable diagonal was silently admitted")
    subnormal_coupling = None
    try:
        _scaled_solve(sparse.csc_matrix(np.array(
            [[1.0e308, np.nextafter(0., 1.)],
             [np.nextafter(0., 1.), 1.0e308]])), np.ones(2))
    except UnrepresentableScale as exc:
        subnormal_coupling = str(exc)
    if subnormal_coupling is None:
        raise AssertionError("erased subnormal coupling was silently admitted")
    record = {
        "classification": "F09_LEVEL_A_PHYSICAL_RANGE_DEV_NOT_ADMISSION",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "operator_sha256": hashlib.sha256(
            Path(inspect.getfile(LinearSection)).read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "elapsed_s_not_benchmark": time.perf_counter() - started,
        "tip_tolerance": TIP_TOL, "separate_residual_tolerance": RESIDUAL_TOL,
        "cases": cases, "negative_zero_diagonal_rejection": negative,
        "negative_subnormal_coupling_rejection": subnormal_coupling,
        "all_declared_cases_passed": all(x["passed_declared_diagnostic"] for x in cases),
        "limits": ["diagnostic anisotropic sections are not geometric realizability bounds",
                   "one rooted elbow and finite area samples do not establish product admission",
                   "no generalized section, shell attachment, spectra, or installed path"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases),
                      "passed": sum(x["passed_declared_diagnostic"] for x in cases),
                      "negative_rejected": negative is not None,
                      "subnormal_coupling_rejected": subnormal_coupling is not None,
                      "elapsed_s": record["elapsed_s_not_benchmark"],
                      "output": str(args.output)}, sort_keys=True))
    if not record["all_declared_cases_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
