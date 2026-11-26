"""Tests of the private batched Q4 planar numerics (``_q4_planar_batch``).

Bitwise identity of ``form``/``recover`` with the sealed per-element
``compute_stiffness_components``/``compute_stresses`` on plate, Scordelis-Lo,
pinched-cylinder and irregular-plate meshes, and fail-closed fallback for
hostile inputs and uncertified stationary systems.
"""
import warnings

import numpy as np
import pytest

from anyfesolver import FEModel, LoadCase, solve_linear
from anyfesolver import e4_pl_element as sealed
from anyfesolver import _q4_planar_batch as pb
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.elements import create_shell_element

FORMATION_KEYS = ("core", "pl", "hourglass", "numerical", "total", "jacobian_centre")
DIAGNOSTIC_KEYS = ("input_asymmetry_relative", "row_norm_ratio", "scale_max", "scale_min",
                   "relative_backward_error")


def _grid(P, t, E, nu):
    m = FEModel("q4planarbatch")
    m.add_material("mat", E, nu)
    ni, nj = P.shape[:2]
    N = np.zeros((ni, nj), int)
    k = 0
    for i in range(ni):
        for j in range(nj):
            k += 1
            m.add_node(k, *map(float, P[i, j]))
            N[i, j] = k
    e = 0
    for i in range(ni - 1):
        for j in range(nj - 1):
            e += 1
            m.add_element(e, create_shell_element(
                e, [int(N[i, j]), int(N[i + 1, j]), int(N[i + 1, j + 1]), int(N[i, j + 1])],
                "mat", thickness=t))
    return m, N


def _fix(m, specs):
    bc = {}
    for nodes, dofs in specs:
        for n in np.atleast_1d(nodes):
            for q in dofs:
                bc.setdefault(int(n), {})[q] = 0.0
    for n, d in bc.items():
        m.add_boundary_condition(BoundaryCondition(f"bc{n}", [n], d))


def _cylinder(n, R, L, angle):
    return np.array([[(L / 2 * i / n, R * np.sin(angle * j / n), R * np.cos(angle * j / n))
                      for j in range(n + 1)] for i in range(n + 1)])


def build(case, n):
    lc = LoadCase("lc")
    if case == "plate":
        xs = np.linspace(0, 1.0, n + 1)
        m, N = _grid(np.array([[(x, y, 0.0) for y in xs] for x in xs]), 1e-4, 1.7472e7, 0.3)
        _fix(m, [(N.ravel(), ["ux", "uy", "rz"]), (list(N[-1, :]) + list(N[:, -1]), ["uz"]),
                 (N[0, :], ["ry"]), (N[:, 0], ["rx"])])
        for e in m.mesh.elements:
            lc.add_pressure_load(int(e), 1e-4)
    elif case == "irregular":
        xs = np.linspace(0, 1.0, n + 1)
        P = np.array([[(x, y, 0.0) for y in xs] for x in xs])
        h = 1.0 / n
        P[1:-1, 1:-1, :2] += np.random.default_rng(12345).uniform(-0.2 * h, 0.2 * h,
                                                                  size=(n - 1, n - 1, 2))
        m, N = _grid(P, 0.01, 2.1e11, 0.3)
        edge = list(N[0, :]) + list(N[-1, :]) + list(N[:, 0]) + list(N[:, -1])
        _fix(m, [(edge, ["ux", "uy", "uz", "rx", "ry", "rz"])])
        for e in m.mesh.elements:
            lc.add_pressure_load(int(e), 1e4)
    elif case == "scordelis_lo":
        m, N = _grid(_cylinder(n, 25.0, 50.0, np.radians(40.0)), 0.25, 4.32e8, 0.0)
        _fix(m, [(N[0, :], ["uy", "uz"]), (N[-1, :], ["ux", "ry", "rz"]),
                 (N[:, 0], ["uy", "rx", "rz"])])
        for e in m.mesh.elements:
            lc.add_pressure_load(int(e), -90.0)
    elif case == "pinched":
        m, N = _grid(_cylinder(n, 300.0, 600.0, np.pi / 2), 3.0, 3.0e6, 0.3)
        _fix(m, [(N[0, :], ["uy", "uz"]), (N[-1, :], ["ux", "ry", "rz"]),
                 (N[:, 0], ["uy", "rx", "rz"]), (N[:, -1], ["uz", "rx", "ry"])])
        lc.add_nodal_load(int(N[-1, 0]), [0, 0, -0.25, 0, 0, 0])
    else:
        raise KeyError(case)
    return m, lc


def sealed_inputs(m):
    """Per-element inputs built with the sealed functions (as the sealed entry will)."""
    elements = list(m.mesh.elements.values())
    materials = [m.get_material(e.material_name) for e in elements]
    frames, locals_, constitutive, drill, pl, hourglass, thickness = ([] for _ in range(7))
    for element, material in zip(elements, materials):
        assert type(element) is sealed.QualifiedE4PLShellElement
        assert element._elastic_increment_admitted(m.mesh, material, None)
        assert element.reference_normal is None
        frame, local, warpage = sealed.equation7_frame(element.get_node_coordinates(m.mesh))
        assert warpage <= element.planar_tolerance
        matrix, drill_stiffness = element._constitutive_and_drill_stiffness(material, frame)
        frames.append(frame)
        locals_.append(local)
        constitutive.append(matrix)
        drill.append(drill_stiffness)
        pl.append(element.pl_stabilization)
        hourglass.append(float(element.hourglass_stabilization))
        thickness.append(float(element.thickness))
    arrays = {name: np.ascontiguousarray(np.asarray(values, dtype=np.float64)) for name, values in (
        ("frames", frames), ("locals", locals_), ("constitutive", constitutive),
        ("drill", drill), ("pl_stabilization", pl), ("hourglass_stabilization", hourglass),
        ("thickness", thickness))}
    return elements, materials, arrays


def form_args(a):
    return (a["frames"], a["locals"], a["constitutive"], a["drill"], a["pl_stabilization"],
            a["hourglass_stabilization"])


def recover_args(a, U):
    return a["frames"], a["locals"], a["constitutive"], a["thickness"], U


def _same(reference, batched):
    reference = np.asarray(reference)
    return (reference.dtype == batched.dtype and reference.shape == batched.shape
            and reference.tobytes() == batched.tobytes())


CASES = [(case, n) for case in ("plate", "scordelis_lo", "pinched", "irregular") for n in (4, 8)]


def test_available_and_forced_fallback():
    assert pb.available() is True
    with pb._forced_fallback():
        assert pb.available() is False
        m, _lc = build("plate", 2)
        _e, _m, a = sealed_inputs(m)
        with pytest.raises(pb.PlanarBatchFallback):
            pb.form(*form_args(a))
        with pytest.raises(pb.PlanarBatchFallback):
            pb.recover(*recover_args(a, np.zeros((len(_e), 24))))
    assert pb.available() is True


@pytest.mark.parametrize("case,n", CASES)
def test_form_bitwise_identical(case, n):
    m, _lc = build(case, n)
    elements, materials, a = sealed_inputs(m)
    formed = pb.form(*form_args(a))
    for index, (element, material) in enumerate(zip(elements, materials)):
        reference = element.compute_stiffness_components(m.mesh, material)
        assert reference["warped_formulation"] == "planar_e4_pl"
        for key in FORMATION_KEYS:
            assert _same(reference[key] if key != "jacobian_centre"
                         else np.float64(reference[key]), formed[key][index]), (key, index)
        diagnostics = reference["stationary_solve_diagnostics"]
        for key in DIAGNOSTIC_KEYS:
            assert np.float64(diagnostics[key]).tobytes() == \
                formed["stationary_solve_diagnostics"][key][index].tobytes(), (key, index)


@pytest.mark.parametrize("case,n", CASES)
def test_recover_bitwise_identical(case, n):
    m, lc = build(case, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, _info = solve_linear(m, lc)
    u = np.asarray(u, dtype=float)
    elements, materials, a = sealed_inputs(m)
    U = np.ascontiguousarray(
        np.stack([u[np.asarray(e.get_dof_mapping(m.mesh), dtype=np.intp)] for e in elements]))
    assert np.any(U != 0.0)
    fields = pb.recover(*recover_args(a, U))
    for index, (element, material) in enumerate(zip(elements, materials)):
        reference = element.compute_stresses(m.mesh, U[index], material)
        assert set(fields) <= set(reference)
        # Every array field of the sealed record is produced here, except the
        # constant natural-point table (the sealed caller's).
        array_keys = {k for k, v in reference.items() if isinstance(v, np.ndarray)}
        assert array_keys - set(fields) == {"natural_point_coordinates"}
        for key, values in fields.items():
            assert _same(reference[key], values[index]), (key, index)


def test_inputs_are_validated():
    m, _lc = build("irregular", 2)
    _elements, _materials, a = sealed_inputs(m)
    pb.form(*form_args(a))

    def refused(**changes):
        made = dict(a)
        made.update(changes)
        with pytest.raises(pb.PlanarBatchFallback):
            pb.form(*form_args(made))

    bad = a["frames"].copy()
    bad[1, 0, 0] = np.nan
    refused(frames=bad)
    bad = a["constitutive"].copy()
    bad[0, 2, 2] = np.inf
    refused(constitutive=bad)
    refused(locals=np.zeros((len(a["locals"]), 4, 3)))
    refused(locals=a["locals"][:-1].copy())
    refused(drill=a["drill"].astype(np.float32))
    refused(frames=np.asfortranarray(a["frames"]))
    refused(frames=a["frames"].tolist())
    refused(drill=a["drill"].view(type("Sub", (np.ndarray,), {})))
    refused(drill=-a["drill"])
    refused(frames=np.zeros((0, 3, 3)), locals=np.zeros((0, 4, 2)),
            constitutive=np.zeros((0, 8, 8)), drill=np.zeros(0),
            pl_stabilization=np.zeros(0), hourglass_stabilization=np.zeros(0))
    U = np.zeros((len(a["locals"]), 24))
    U[0, 3] = np.nan
    with pytest.raises(pb.PlanarBatchFallback):
        pb.recover(*recover_args(a, U))
    with pytest.raises(pb.PlanarBatchFallback):
        pb.recover(*recover_args(a, np.zeros((len(a["locals"]), 23))))
    with pytest.raises(pb.PlanarBatchFallback):
        pb.recover(a["frames"], a["locals"], a["constitutive"], -a["thickness"],
                   np.zeros((len(a["locals"]), 24)))


def test_nonpositive_jacobian_falls_back():
    m, _lc = build("irregular", 2)
    _elements, _materials, a = sealed_inputs(m)
    reversed_local = a["locals"][1][::-1].copy()
    c = sealed._coefficients(reversed_local)
    assert min(c["jc"], *(sealed._jacobian(c, r, s)[4] for r, s in sealed._GAUSS)) <= 0.0
    locals_ = a["locals"].copy()
    locals_[1] = reversed_local
    with pytest.raises(pb.PlanarBatchFallback):
        pb.form(*form_args(dict(a, locals=locals_)))
    with pytest.raises(pb.PlanarBatchFallback):
        pb.recover(*recover_args(dict(a, locals=locals_), np.zeros((len(locals_), 24))))


def test_overflowing_stationary_system_falls_back_like_sealed():
    k = 4
    stationary = 1.0e300 * np.eye(k)
    coupling = np.full((24, k), 1.0e307)  # rhs row sums overflow: denominator is inf
    with pytest.raises(ValueError):
        sealed._solve_stationary_system(stationary, coupling)
    with pytest.raises(pb.PlanarBatchFallback):
        pb._solve_stationary(stationary[None], coupling[None])


def test_singular_stationary_system_falls_back_like_sealed():
    m, _lc = build("plate", 2)
    elements, _materials, a = sealed_inputs(m)
    zero = np.zeros_like(a["constitutive"])
    local = a["locals"][0]
    stationary, coupling, _gram = sealed._stationary_blocks(
        local, sealed._coefficients(local), zero[0], compensated=True)
    with pytest.raises(ValueError):
        sealed._solve_stationary_system(stationary, coupling)
    with pytest.raises(pb.PlanarBatchFallback):
        pb.form(*form_args(dict(a, constitutive=zero)))
    with pytest.raises(pb.PlanarBatchFallback):
        pb.recover(*recover_args(dict(a, constitutive=zero), np.zeros((len(elements), 24))))
    # An exactly singular (rank-deficient) stationary matrix.
    singular = np.eye(4)
    singular[3] = singular[2]
    singular[:, 3] = singular[:, 2]
    with pytest.raises(ValueError):
        sealed._solve_stationary_system(singular, np.ones((24, 4)))
    with pytest.raises(pb.PlanarBatchFallback):
        pb._solve_stationary(singular[None], np.ones((1, 24, 4)))


def test_outputs_do_not_alias_inputs_or_each_other():
    m, _lc = build("irregular", 2)
    elements, _materials, a = sealed_inputs(m)
    before = {name: value.copy() for name, value in a.items()}
    formed = pb.form(*form_args(a))
    U = np.ones((len(elements), 24)) * 1.0e-6
    fields = pb.recover(*recover_args(a, U))
    outputs = [v for k, v in formed.items() if k != "stationary_solve_diagnostics"]
    outputs += list(formed["stationary_solve_diagnostics"].values()) + list(fields.values())
    inputs = list(a.values()) + [U]
    for i, out in enumerate(outputs):
        assert type(out) is np.ndarray and out.flags.c_contiguous and out.flags.writeable
        for value in inputs:
            assert not np.shares_memory(out, value)
        for other in outputs[i + 1:]:
            assert not np.shares_memory(out, other)
    fields["physical_director"][:] = 7.0
    formed["total"][:] = 7.0
    for name, value in a.items():
        assert value.tobytes() == before[name].tobytes(), name
    # Mutating inputs after the call does not reach earlier outputs.
    again = pb.form(*form_args(a))
    a["frames"][:] = 0.0
    assert np.all(np.isfinite(again["total"]))
    assert again["core"].tobytes() == pb.form(*form_args(before))["core"].tobytes()


def test_stationary_diagnostics_bitwise_identical_for_dominant_coupling_rows():
    """Codex L4 round 1 P2: the per-element backward error reduces
    ``abs(coupling.T)`` (an F-ordered operand) along its strided axis; the
    batch must reproduce that order, not the contiguous pairwise row sum, so
    every published stationary diagnostic is bit-identical."""

    rng = np.random.default_rng(7)
    compared = 0
    for _trial in range(400):
        q = rng.standard_normal((35, 35))
        stationary = q @ q.T / 35.0 + np.eye(35)
        stationary = 0.5 * (stationary + stationary.T)
        coupling = rng.standard_normal((24, 35))
        coupling[rng.integers(0, 24), :] *= 10.0 ** rng.integers(8, 17)
        try:
            solution, diagnostics = sealed._solve_stationary_system(stationary, coupling)
        except ValueError:
            continue
        batch_solution, batch_diagnostics = pb._solve_stationary(
            np.ascontiguousarray(stationary[None]), np.ascontiguousarray(coupling[None])
        )
        assert batch_solution[0].tobytes() == solution.tobytes()
        for name, values in batch_diagnostics.items():
            assert np.float64(values[0]).tobytes() == np.float64(diagnostics[name]).tobytes(), name
        compared += 1
    assert compared > 300
