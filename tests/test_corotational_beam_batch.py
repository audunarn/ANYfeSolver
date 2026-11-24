"""Bit identity of the stacked corotational beam route against the scalar route."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import FEModel, LoadCase, solve_static_nonlinear
from anyfesolver import _corotational_beam_batch as batch
from anyfesolver import corotational as cr
from anyfesolver import corotational_performance as cp
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.elements import BeamElement
from anyfesolver.nonlinear_static import _assemble_nonlinear_system_unchecked

SECTION = {"area": 0.02, "Iy": 1.6667e-5, "Iz": 6.6667e-5, "J": 2.29e-5}


def _bits(array):
    return np.ascontiguousarray(np.asarray(array, dtype=float)).view(np.uint64)


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.shape(a) == np.shape(b) and np.array_equal(_bits(a), _bits(b))


@pytest.fixture(autouse=True)
def _restore_batch_status():
    saved = dict(batch._STATUS)
    yield
    with batch._LOCK:
        batch._STATUS.clear()
        batch._STATUS.update(saved)


def _frame_model(count=8, seed=0, **section_extra):
    """Beams in assorted directions, some with orientation vectors."""
    rng = np.random.default_rng(seed)
    model = FEModel("batch")
    model.add_material("mat", 2.1e11, 0.3)
    elements = []
    for index in range(count):
        start = rng.uniform(-2.0, 2.0, 3)
        end = start + rng.uniform(0.2, 1.5) * rng.standard_normal(3)
        model.add_node(2 * index + 1, *start)
        model.add_node(2 * index + 2, *end)
        section = dict(SECTION, **section_extra)
        if index % 2:
            section["orientation"] = tuple(np.cross(end - start, rng.standard_normal(3)))
        element = BeamElement(index + 1, [2 * index + 1, 2 * index + 2], "mat", section)
        model.add_element(index + 1, element)
        elements.append(element)
    return model, elements


def _warm(model, elements):
    """Build the scalar reference and local caches exactly as the scalar route does."""
    references = cr._corotational_cache(model)
    material = model.get_material("mat")
    for element in elements:
        element.compute_nonlinear_response(model.mesh, material, np.zeros(12), None, 5, True)
    return references, material


def _kernel_inputs(references, rows):
    X = np.stack([references[e.element_id].coordinates for e, _ in rows])
    U = np.stack([u for _, u in rows])
    T = np.stack([e._nl_cache["T"] for e, _ in rows])
    K = np.stack([e._nl_cache["K_noax"] for e, _ in rows])
    L = np.array([e._nl_cache["L"] for e, _ in rows])
    L2 = np.array([e._nl_cache["L"] ** 2 for e, _ in rows])
    EA = np.array([e._nl_cache["EA"] for e, _ in rows], dtype=float)
    return X, U, T, K, L, L2, EA


def _rigid_state(angle, axis, rng, noise):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    u = rng.standard_normal(12) * noise
    u[3:6] += angle * axis
    u[9:12] += angle * axis
    return u


def _check_rows(model, references, material, rows, tangent):
    f, k, route = batch._kernel(*_kernel_inputs(references, rows), tangent)
    compared = 0
    for position, (element, u) in enumerate(rows):
        if route[position]:
            continue
        f_scalar, k_scalar = batch._scalar_oracle(
            references[element.element_id], element, model.mesh, material, u, tangent
        )
        assert _same(f[position], f_scalar), position
        assert _same(None if k is None else k[position], k_scalar), position
        compared += 1
    return compared, route


@pytest.mark.parametrize("tangent", [True, False])
def test_kernel_matches_scalar_bitwise_on_random_states(tangent):
    model, elements = _frame_model(10, seed=3)
    references, material = _warm(model, elements)
    rng = np.random.default_rng(11)
    rows = []
    for element in elements:
        for scale in (0.0, 1.0e-9, 1.0e-5, 1.0e-3, 4.0e-3, 2.0e-2, 0.2, 1.0, 2.5):
            for _ in range(4):
                rows.append((element, rng.standard_normal(12) * scale))
        for angle in (1.0e-3, 0.4, 1.1, 2.0, 2.9, 3.13):
            rows.append((element, _rigid_state(angle, rng.standard_normal(3), rng, 1.0e-4)))
    compared, route = _check_rows(model, references, material, rows, tangent)
    assert compared == len(rows) - int(route.sum())
    assert compared > 0.95 * len(rows)


def test_every_branch_is_exact_or_routed():
    model, elements = _frame_model(4, seed=5)
    references, material = _warm(model, elements)
    rng = np.random.default_rng(7)
    element = elements[0]
    X = references[element.element_id].coordinates
    t_ref = (X[1] - X[0]) / np.linalg.norm(X[1] - X[0])
    normal = np.cross(t_ref, [0.3, 0.4, 0.5])
    normal /= np.linalg.norm(normal)
    exact = {
        "zero": np.zeros(12),
        "exp_series": rng.standard_normal(12) * 1.0e-4,  # |theta| < 1e-2
        "exp_closed": rng.standard_normal(12) * 5.0e-2,  # |theta| >= 1e-2
        "log_series": _rigid_state(0.2, rng.standard_normal(3), rng, 1.0e-7),
        "log_general": rng.standard_normal(12) * 3.0e-2,
        # Rigid rotation beyond 60 degrees selects R.T - I in
        # _stable_shell_rotation_difference; below it the stable form.
        "difference_fallback": _rigid_state(1.5, rng.standard_normal(3), rng, 1.0e-4),
        "difference_stable": _rigid_state(0.5, rng.standard_normal(3), rng, 1.0e-4),
    }
    routed = {
        "antiparallel": np.r_[np.zeros(3), np.pi * normal, np.zeros(6)],
        "non_finite": np.r_[np.nan, np.zeros(11)],
        "overflowing": np.r_[1.0e200, np.zeros(11)],
    }
    rows = [(element, u) for u in exact.values()] + [(element, u) for u in routed.values()]
    for tangent in (True, False):
        compared, route = _check_rows(model, references, material, rows, tangent)
        assert not route[: len(exact)].any()
        assert route[len(exact):].all()
        assert compared == len(exact)
    # Confirm the two rotation-difference branches really were exercised.
    for name, expected_fallback in (("difference_fallback", True), ("difference_stable", False)):
        _u_d, R, _deformed = cr._corotational_deformation_state(
            references[element.element_id], element, exact[name]
        )
        assert bool(0.5 * (np.trace(R) - 1.0) <= 0.5) is expected_fallback


def test_log_map_branches_exact_and_near_pi_routed():
    rng = np.random.default_rng(13)
    vectors = []
    for angle in (0.0, 1.0e-7, 5.0e-5, 2.0e-4, 0.3, 2.0, 3.0, np.pi - 2.0e-6,
                  np.pi - 5.0e-7, np.pi - 1.0e-9):
        for _ in range(5):
            axis = rng.standard_normal(3)
            vectors.append(angle * axis / np.linalg.norm(axis))
    matrices = np.stack([cr.rotation_matrix_from_vector(v) for v in vectors])
    with np.errstate(all="ignore"):
        theta, route = batch._rotation_vector(matrices)
    for position, matrix in enumerate(matrices):
        scalar = cr.rotation_vector_from_matrix(matrix)
        angle = np.linalg.norm(vectors[position])
        if route[position]:
            assert angle > np.pi - 1.0e-6
        else:
            assert _same(theta[position], scalar)
    assert route.sum() >= 10


def _cantilever(n=12):
    model = FEModel("cantilever")
    model.add_material("mat", 2.1e11, 0.3)
    for i in range(n + 1):
        model.add_node(i + 1, 10.0 * i / n, 0.0, 0.0)
    for i in range(n):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "mat", dict(SECTION)))
    model.add_boundary_condition(
        BoundaryCondition("clamp", [1], {d: 0.0 for d in ("ux", "uy", "uz", "rx", "ry", "rz")})
    )
    return model


def _assemble(model, u, tangent=True, mode="rotated", **kwargs):
    return _assemble_nonlinear_system_unchecked(
        model, u, {}, 5, tangent=tangent, kinematics="corotational",
        corotational_tangent=mode, **kwargs,
    )


def _assembly_pair(model, u, **kwargs):
    batch.set_corotational_beam_batch_enabled(False)
    cp.reset_corotational_performance_status()
    scalar = _assemble(model, u, **kwargs)
    scalar_counts = cp.corotational_performance_status()
    batch.set_corotational_beam_batch_enabled(True)
    cp.reset_corotational_performance_status()
    before = batch.corotational_beam_batch_status()["batched_elements"]
    stacked = _assemble(model, u, **kwargs)
    stacked_counts = cp.corotational_performance_status()
    used = batch.corotational_beam_batch_status()["batched_elements"] - before
    return scalar, stacked, scalar_counts, stacked_counts, used


@pytest.mark.parametrize("tangent", [True, False])
def test_assembly_is_bit_identical_with_identical_counters(tangent):
    model = _cantilever()
    rng = np.random.default_rng(1)
    total = model.mesh.dof_manager.total_dofs
    _assemble(model, np.zeros(total))  # scalar route builds the caches
    for scale in (1.0e-6, 1.0e-2, 0.5):
        u = rng.standard_normal(total) * scale
        scalar, stacked, c_scalar, c_stacked, used = _assembly_pair(model, u, tangent=tangent)
        assert used == len(model.mesh.elements)
        assert _same(scalar[0], stacked[0])
        if tangent:
            assert _same(scalar[1].toarray(), stacked[1].toarray())
            assert _same(scalar[1].data, stacked[1].data)
        else:
            assert scalar[1] is None and stacked[1] is None
        assert c_scalar == c_stacked


def test_solve_is_bit_identical():
    def run(enabled):
        batch.set_corotational_beam_batch_enabled(enabled)
        model = _cantilever(16)
        load = LoadCase("tip")
        load.add_nodal_load(17, [0.0, 3.0e3, -2.0e4, 1.0e4, 0.0, 0.0])
        return solve_static_nonlinear(model, load, num_steps=3, kinematics="corotational")

    scalar = run(False)
    before = batch.corotational_beam_batch_status()["batched_elements"]
    stacked = run(True)
    assert batch.corotational_beam_batch_status()["batched_elements"] > before
    assert _same(scalar.displacements, stacked.displacements)
    assert [(s.load_factor, s.iterations, s.residual_norm) for s in scalar.steps] == [
        (s.load_factor, s.iterations, s.residual_norm) for s in stacked.steps
    ]


def test_forced_mismatch_falls_back_without_side_effects(monkeypatch):
    model = _cantilever()
    total = model.mesh.dof_manager.total_dofs
    u = np.random.default_rng(4).standard_normal(total) * 1.0e-2
    _assemble(model, np.zeros(total))
    batch.set_corotational_beam_batch_enabled(False)
    scalar = _assemble(model, u)
    batch.set_corotational_beam_batch_enabled(True)
    original = batch._kernel

    def perturbed(*args):
        f, k, route = original(*args)
        f[-1, 0] = np.nextafter(f[-1, 0], np.inf)
        return f, k, route

    monkeypatch.setattr(batch, "_kernel", perturbed)
    cp.reset_corotational_performance_status()
    counters = cp.corotational_performance_status()
    assert batch.prepare_corotational_beam_batch(
        model, u, tangent=True, tangent_mode="rotated", num_layers=5,
        stable_shell_rotation=False,
    ) is None
    assert cp.corotational_performance_status() == counters  # nothing recorded
    status = batch.corotational_beam_batch_status()
    assert status["enabled"] is False
    assert status["disabled_reason"] == "spot_check_mismatch"
    stacked = _assemble(model, u)  # whole evaluation on the scalar route
    assert _same(scalar[0], stacked[0])
    assert _same(scalar[1].toarray(), stacked[1].toarray())


def _prepared_ids(model, u, **overrides):
    arguments = dict(tangent=True, tangent_mode="rotated", num_layers=5,
                     stable_shell_rotation=False)
    arguments.update(overrides)
    prepared = batch.prepare_corotational_beam_batch(model, u, **arguments)
    return set() if prepared is None else set(prepared._responses)


def _admission_model():
    model = _cantilever(6)
    total = model.mesh.dof_manager.total_dofs
    _assemble(model, np.zeros(total))
    u = np.random.default_rng(9).standard_normal(total) * 1.0e-3
    batch.set_corotational_beam_batch_enabled(True)
    return model, u


def test_admission_refuses_subclass_and_instance_overrides():
    model, u = _admission_model()
    assert _prepared_ids(model, u) == set(range(1, 7))

    class Spied(BeamElement):
        pass

    original = model.mesh.elements[2]
    spied = Spied(2, list(original.node_ids), "mat", dict(SECTION))
    spied._nl_cache = original._nl_cache
    model.mesh.elements[2] = spied
    assert 2 not in _prepared_ids(model, u)
    model.mesh.elements[2] = original

    element = model.mesh.elements[3]
    element.compute_nonlinear_response = lambda *a, **k: None  # instance override
    assert 3 not in _prepared_ids(model, u)
    del element.compute_nonlinear_response
    assert 3 in _prepared_ids(model, u)


def test_admission_refuses_unsupported_formulations_and_modes():
    model, u = _admission_model()
    element = model.mesh.elements[4]
    element._fiber_plasticity = True
    assert 4 not in _prepared_ids(model, u)
    element._fiber_plasticity = None
    element._geometric_nonlinearity = "corotational"
    assert 4 not in _prepared_ids(model, u)
    element._geometric_nonlinearity = "von_karman"
    element._linear_nodal_offsets = np.zeros((2, 3))
    assert 4 not in _prepared_ids(model, u)
    del element._linear_nodal_offsets
    assert 4 in _prepared_ids(model, u)
    for mode in ("consistent", "work_conjugate"):
        assert _prepared_ids(model, u, tangent_mode=mode) == set()
    assert 5 not in _prepared_ids(model, u, excluded={5})


def test_consistent_mode_assembly_never_uses_batch():
    model, u = _admission_model()
    before = batch.corotational_beam_batch_status()["batched_elements"]
    _assemble(model, u, mode="consistent")
    assert batch.corotational_beam_batch_status()["batched_elements"] == before


def test_patched_scalar_route_disables_batch(monkeypatch):
    model, u = _admission_model()
    monkeypatch.setattr(cr, "corotational_element_response", lambda *a, **k: (None, None, None))
    assert _prepared_ids(model, u) == set()


def test_failed_probe_disables_batch_for_process(monkeypatch):
    model, u = _admission_model()
    batch.reset_corotational_beam_batch_probe()
    monkeypatch.setattr(batch, "_run_probe", lambda: (False, "forced"))
    assert _prepared_ids(model, u) == set()
    status = batch.corotational_beam_batch_status()
    assert status["probe"] is False and status["enabled"] is False
    assert status["disabled_reason"] == "forced"


def test_probe_passes_on_this_platform():
    passed, reason = batch._run_probe()
    assert passed, reason
