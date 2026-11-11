"""G0 F09-L: bounded local linear B2 physical-flexibility feasibility proof."""

from __future__ import annotations

from fractions import Fraction
import json
import math

import numpy as np

from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import Material


def basic(length: float) -> np.ndarray:
    b = np.zeros((6, 12))
    b[0, (0, 6)] = (-1, 1)
    b[1, (3, 9)] = (-1, 1)
    for row, rot in ((2, 4), (3, 10)):
        b[row, (2, 8, rot)] = (-1 / length, 1 / length, 1)
    for row, rot in ((4, 5), (5, 11)):
        b[row, (1, 7, rot)] = (1 / length, -1 / length, 1)
    return b


def interpolation(length: float, x: float) -> np.ndarray:
    h = np.zeros((6, 6))
    t = x / length
    h[0, 0] = h[3, 1] = 1
    h[2, 2:4] = 1 / length
    h[4, 2:4] = (-1 + t, t)
    h[1, 4:6] = -1 / length
    h[5, 4:6] = (-1 + t, t)
    return h


def scalar_factors(length: float, e: float, nu: float, area: float,
                   iy: float, iz: float, j: float, ky: float,
                   kz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inputs = (length, e, area, iy, iz, j, ky, kz)
    if not all(math.isfinite(v) and v > 0 for v in inputs):
        raise ValueError("finite positive physical scalar inputs required")
    if not math.isfinite(nu) or not -1 < nu < .5:
        raise ValueError("isotropic Poisson ratio outside the admitted range")
    g = e / (2 * (1 + nu))
    s = np.array((e * area, g * area * ky, g * area * kz,
                  g * j, e * iy, e * iz))
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("unrepresentable positive section rigidity")
    b = basic(length)
    ka, kt = s[0] / length, s[3] / length
    kp_y = 1 / (length / (6 * s[4]) + 2 / (s[2] * length))
    km_y = 2 * s[4] / length
    kp_z = 1 / (length / (6 * s[5]) + 2 / (s[1] * length))
    km_z = 2 * s[5] / length
    k = np.array((ka, kt, kp_y, km_y, kp_z, km_z))
    if not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("unrepresentable positive modal stiffness")
    modes = np.stack((b[0], b[1], (b[2] + b[3]) / math.sqrt(2),
                      (b[2] - b[3]) / math.sqrt(2),
                      (b[4] + b[5]) / math.sqrt(2),
                      (b[4] - b[5]) / math.sqrt(2)))
    return s, b, modes * np.sqrt(k)[:, None], k


def rigid_modes(length: float) -> np.ndarray:
    rigid = np.zeros((12, 6))
    for axis in range(3):
        rigid[axis, axis] = rigid[6 + axis, axis] = 1
        rigid[3 + axis, 3 + axis] = rigid[9 + axis, 3 + axis] = 1
    rigid[8, 4] = -length
    rigid[7, 5] = length
    return rigid


def rational_transverse_energy(case: dict, direction: str) -> float:
    r = lambda key: Fraction(case[key])
    e, l, a = r("e"), r("length"), r("area")
    g = e / (2 * (1 + r("nu")))
    ei = e * r("iy" if direction == "z" else "iz")
    shear = g * a * r("kz" if direction == "z" else "ky")
    tangent = 12 * ei * shear / (l * (shear * l * l + 12 * ei))
    return float(tangent * Fraction(.01) ** 2 / 2)


def scalar_case(case: dict) -> dict:
    s, b, rows, modal = scalar_factors(
        case["length"], case["e"], case["nu"], case["area"],
        case["iy"], case["iz"], case["j"], case["ky"], case["kz"])
    length = case["length"]
    results = {}
    section = {"area": case["area"], "Iy": case["iy"],
               "Iz": case["iz"], "J": case["j"],
               "shear_factor_y": case["ky"], "shear_factor_z": case["kz"]}
    public = BeamElement(1, [1, 2], "steel", section)._local_linear_stiffness(
        length, Material("steel", case["e"], case["nu"]))
    for direction, dof in (("y", 7), ("z", 8)):
        u = np.zeros(12)
        u[dof] = .01
        energy = .5 * float(np.linalg.norm(rows @ u) ** 2)
        exact = rational_transverse_energy(case, direction)
        public_energy = .5 * float(u @ public @ u)
        results[direction] = {
            "displacement": u.tolist(), "energy": energy,
            "rational_energy": exact, "public_energy": public_energy,
            "public_over_physical": public_energy / exact,
            "relative_error": abs(energy - exact) / exact,
            "force_work_error": abs(float(u @ (rows.T @ (rows @ u))) - 2 * energy) / (2 * exact),
        }
    null = float(np.linalg.norm(b @ rigid_modes(length)))
    result = {"id": case["id"], "parameters": case,
              "physical_rigidities": s.tolist(),
              "modal_stiffnesses": modal.tolist(),
              "directions": results, "rigid_basic_residual": null,
              "positive_modal_factors": bool(np.all(np.linalg.norm(rows, axis=1) > 0)),
              "basic_rank": int(np.linalg.matrix_rank(b))}
    if case["id"] == "S0":
        local = rows.T @ rows
        result["ordinary_public_matrix_relative"] = float(
            np.linalg.norm(local - public) / np.linalg.norm(public))
        u = np.array((.02, -.04, .03, .01, -.02, .005,
                      -.01, .03, -.05, .015, .02, -.008))
        direction = np.array((.5, -.7, .2, .4, -.6, .3,
                              -.1, .8, -.4, .2, .9, -.5))
        force = rows.T @ (rows @ u)
        flexibility = sum(
            w * interpolation(length, .5 * length * (1 + xi)).T
            @ np.diag(1 / s)
            @ interpolation(length, .5 * length * (1 + xi)) * length / 2
            for xi, w in zip((-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)),
                             (5 / 9, 8 / 9, 5 / 9)))
        resultant = np.linalg.solve(flexibility, b @ u)
        station_work = 0.
        for xi, w in zip((-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)),
                         (5 / 9, 8 / 9, 5 / 9)):
            h = interpolation(length, .5 * length * (1 + xi))
            station_work += w * length / 2 * (
                (h @ resultant) @ ((h @ resultant) / s))
        energy = float(np.linalg.norm(rows @ u) ** 2)
        result["station_complementary_twice_energy"] = station_work
        result["modal_twice_energy"] = energy
        result["basic_force_work"] = float((b @ u) @ resultant)
        result["coordinate_force_work"] = float(u @ force)
        result["station_twice_energy_relative"] = abs(station_work - energy) / energy
        result["basic_work_relative"] = abs(float((b @ u) @ resultant) - energy) / energy
        result["force_work_relative"] = abs(float(u @ force) - energy) / energy
        h = 1e-6
        energies = tuple(.5 * float(np.linalg.norm(rows @ (u + sign * h * direction)) ** 2)
                         for sign in (1, -1))
        direct_work = (energies[0] - energies[1]) / (2 * h)
        physical_force = b.T @ resultant
        result["directional_energy_relative"] = abs(
            direct_work - float(physical_force @ direction)) / abs(float(physical_force @ direction))
        result["directional_energy_slope"] = direct_work
        result["directional_force_work"] = float(physical_force @ direction)
        result["directional_energy_scale"] = abs(float(physical_force @ direction))
        modal_force_derivative = (
            rows.T @ (rows @ (u + h * direction))
            - rows.T @ (rows @ (u - h * direction))) / (2 * h)
        physical_tangent_action = b.T @ np.linalg.solve(flexibility, b @ direction)
        result["directional_tangent_relative"] = float(
            np.linalg.norm(modal_force_derivative - physical_tangent_action)
            / np.linalg.norm(physical_tangent_action))
        result["directional_tangent_scale"] = float(
            np.linalg.norm(physical_tangent_action))
        result["physical_force_relative"] = float(
            np.linalg.norm(force - physical_force) / np.linalg.norm(physical_force))
        result["input_displacement"] = u.tolist()
        result["input_direction"] = direction.tolist()
        result["basic_resultants"] = resultant.tolist()
        result["station_resultants"] = [
            (interpolation(length, x) @ resultant).tolist()
            for x in (0., length / 2, length)]
    return result


def generalized_compliance(section: np.ndarray) -> np.ndarray:
    if (section.shape != (6, 6) or not np.all(np.isfinite(section))
            or not np.allclose(section, section.T, rtol=0, atol=0)):
        raise ValueError("finite symmetric generalized section required")
    try:
        np.linalg.cholesky(section)
    except np.linalg.LinAlgError as exc:
        raise ValueError("positive-definite generalized section required") from exc
    return np.linalg.solve(section, np.eye(6))


def generalized_case() -> dict:
    length = 2.
    section = np.diag((10., 7., 9., 3., 2., 4.))
    section[0, 4] = section[4, 0] = .4
    section[1, 5] = section[5, 1] = .3
    section[2, 3] = section[3, 2] = -.2
    compliance = generalized_compliance(section)
    h0, h1 = interpolation(length, 0), (
        interpolation(length, length) - interpolation(length, 0)) / length
    analytic = (
        length * h0.T @ compliance @ h0
        + length ** 2 / 2 * (h0.T @ compliance @ h1 + h1.T @ compliance @ h0)
        + length ** 3 / 3 * h1.T @ compliance @ h1
    )
    gauss = sum(w * length / 2
                * interpolation(length, .5 * length * (1 + xi)).T
                @ compliance
                @ interpolation(length, .5 * length * (1 + xi))
                for xi, w in zip((-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)),
                                 (5 / 9, 8 / 9, 5 / 9)))
    b = basic(length)
    k = b.T @ np.linalg.solve(gauss, b)
    u = np.array((.02, -.04, .03, .01, -.02, .005,
                  -.01, .03, -.05, .015, .02, -.008))
    direction = np.array((.5, -.7, .2, .4, -.6, .3,
                          -.1, .8, -.4, .2, .9, -.5))
    q = np.linalg.solve(gauss, b @ u)
    dq = np.linalg.solve(gauss, b @ direction)
    station = float(q @ analytic @ dq)
    virtual = float((b.T @ q) @ direction)
    h = 1.e-6
    fplus = b.T @ np.linalg.solve(gauss, b @ (u + h * direction))
    fminus = b.T @ np.linalg.solve(gauss, b @ (u - h * direction))
    residual = (fplus - fminus) / (2 * h) - k @ direction
    return {
        "section": section.tolist(), "length": length,
        "flexibility_relative": float(np.linalg.norm(gauss - analytic) / np.linalg.norm(analytic)),
        "minimum_section_eigenvalue": float(np.min(np.linalg.eigvalsh(section))),
        "minimum_basic_eigenvalue": float(np.min(np.linalg.eigvalsh(gauss))),
        "basic_rank": int(np.linalg.matrix_rank(b)),
        "station_work_relative": abs(station - virtual) / abs(virtual),
        "integrated_station_work": station, "coordinate_virtual_work": virtual,
        "virtual_work_scale": abs(virtual),
        "flexibility_scale": float(np.linalg.norm(analytic)),
        "tangent_skew_relative": float(np.linalg.norm(k - k.T) / np.linalg.norm(k)),
        "tangent_scale": float(np.linalg.norm(k)),
        "directional_relative": float(np.linalg.norm(residual) / np.linalg.norm(k @ direction)),
        "directional_action_scale": float(np.linalg.norm(k @ direction)),
        "rigid_basic_residual": float(np.linalg.norm(b @ rigid_modes(length))),
        "displacement": u.tolist(), "direction": direction.tolist(),
        "basic_resultants": q.tolist(),
        "station_resultants": [
            (interpolation(length, x) @ q).tolist()
            for x in (0., length / 2, length)],
    }


def negative_cases() -> dict:
    failures = {}
    for name, override in (("zero", {"area": 0.}), ("negative", {"iy": -1.}),
                           ("nonfinite", {"j": float("nan")})):
        fields = dict(length=1., e=1., nu=0., area=1., iy=1.,
                      iz=1., j=1., ky=1., kz=1.)
        fields.update(override)
        try:
            scalar_factors(**fields)
        except ValueError as exc:
            failures[name] = {
                "inputs": {key: repr(value) for key, value in fields.items()},
                "error_type": type(exc).__name__, "error_detail": str(exc),
            }
    non_spd = np.diag((10., 7., 9., 3., 2., -1.))
    try:
        generalized_compliance(non_spd)
    except ValueError as exc:
        failures["non_spd"] = {
            "section": non_spd.tolist(),
            "error_type": type(exc).__name__, "error_detail": str(exc),
        }
    return failures


def main() -> int:
    cases = [
        dict(id="S0", e=210e9, nu=.3, length=2.4, area=.012,
             iy=2.3e-6, iz=3.7e-6, j=8e-7, ky=.81, kz=.76),
        dict(id="S1", e=1., nu=0., length=1., area=1e-14,
             iy=1e-12, iz=1e-12, j=1e-12, ky=1., kz=1.),
        dict(id="S2", e=1., nu=0., length=1., area=2.**-80,
             iy=1., iz=1., j=1., ky=1., kz=1.),
    ]
    scalar = [scalar_case(c) for c in cases]
    coupled = generalized_case()
    negatives = negative_cases()
    passed = all(
        row["basic_rank"] == 6 and row["positive_modal_factors"]
        and row["rigid_basic_residual"] <= 1e-12
        and all(v["energy"] > 0 and v["relative_error"] <= 1e-12
                and v["force_work_error"] <= 1e-12
                for v in row["directions"].values())
        for row in scalar
    )
    passed &= all(scalar[0][key] <= 1e-12 for key in (
        "ordinary_public_matrix_relative", "station_twice_energy_relative",
        "basic_work_relative", "force_work_relative", "physical_force_relative"))
    passed &= all(scalar[0][key] <= 1e-6 for key in (
        "directional_energy_relative", "directional_tangent_relative"))
    passed &= all(coupled[key] <= 1e-12 for key in (
        "flexibility_relative", "station_work_relative",
        "tangent_skew_relative", "rigid_basic_residual"))
    passed &= coupled["directional_relative"] <= 1e-6
    passed &= coupled["minimum_section_eigenvalue"] > 0
    passed &= coupled["minimum_basic_eigenvalue"] > 0
    passed &= coupled["basic_rank"] == 6
    passed &= set(negatives) == {"zero", "negative", "nonfinite", "non_spd"}
    print(json.dumps({
        "case": "G0_F09_LINEAR_LOCAL_V1",
        "status": "PREPARATORY_LINEAR_OPERATOR_FEASIBLE" if passed else "FAILED",
        "scalar": scalar, "generalized": coupled,
        "negative_rejections": negatives,
        "scope": "local elastic reference only; no attachment, mass, graph or F09 acceptance",
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
