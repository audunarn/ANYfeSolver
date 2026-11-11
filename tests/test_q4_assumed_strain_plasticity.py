"""Qualified Q4 plastic shells: assumed-strain layered plasticity.

The historical qualified route added the constant elastic hybrid delta
``Cr = K_hybrid - K_compatible`` to a compatible-strain plastic kernel.  ``Cr``
is indefinite, so once the layers yield the element tangent became negative
along in-plane bending/hourglass modes and the clamped plastic plate
converged to spurious mechanisms (+203 mm at n=8, wrong sign at n=16, against
about 35-39 mm from the legacy element and CalculiX).  The route under test
evaluates the return map at the hybrid's assumed strain; see
``anyfesolver._q4_assumed_strain_plasticity``.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import anyfesolver as fs
from anyfesolver import _q4_assumed_strain_plasticity as AS
from anyfesolver.boundary import FixedSupport, LoadCase
from anyfesolver.elements import create_shell_element
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.nonlinear_performance import NonlinearAssemblyPlan
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from anyfesolver.plasticity import plane_stress_elastic_matrix


def _curve():
    return DNVC208MaterialCurve(
        sigma_prop=3e7, sigma_yield=3.2e7, sigma_yield_2=3.4e7,
        eps_p_y1=0.005, eps_p_y2=0.01, K=4e8, n=0.2,
    )


def _single(nodes=((0, 0), (0.125, 0), (0.125, 0.125), (0, 0.125)), *, plastic=True,
            form=None, thickness=0.003):
    model = fs.FEModel("q4")
    model.add_material(
        "mat", 210e9, 0.3, density=7850.0,
        yield_stress=3e7 if plastic else 0.0,
        hardening_curve=_curve() if plastic else None,
    )
    for index, (x, y) in enumerate(nodes):
        model.add_node(index + 1, float(x), float(y), 0.0)
    kwargs = {} if form is None else {"formulation": form}
    model.add_element(1, create_shell_element(1, [1, 2, 3, 4], "mat", thickness=thickness, **kwargs))
    return model, model.mesh.elements[1], model.get_material("mat")


def _field(nodes, kind, amplitude):
    u = np.zeros(24)
    a = max(float(np.ptp([p[0] for p in nodes])), 1e-12)
    for i, (x, y) in enumerate(nodes):
        if kind == "hourglass":
            u[6 * i] = amplitude * 8.0 * (x - a / 2) * (y - a / 2) / a
        elif kind == "stretch":
            u[6 * i] = amplitude * x
        elif kind == "bend":
            u[6 * i + 2] = amplitude * x * x / a
            u[6 * i + 4] = -2.0 * amplitude * x / a
        elif kind == "twist":
            u[6 * i + 3] = amplitude * x / a
            u[6 * i + 4] = -amplitude * y / a
    return u


SQUARE = ((0, 0), (0.125, 0), (0.125, 0.125), (0, 0.125))
SKEW = ((0, 0), (0.13, 0.02), (0.15, 0.11), (-0.01, 0.12))


@pytest.mark.parametrize("nodes", [SQUARE, SKEW], ids=["square", "skew"])
def test_offset_operator_patch_test_and_hybrid_orthogonality(nodes):
    model, element, material = _single(nodes)
    assert AS.supported(element, model.mesh, material)
    offsets = AS.offset_operator(element, model.mesh, material)
    cache = element._nonlinear_geometry(model.mesh)
    B = np.concatenate((cache["B_m_all"], cache["B_b_all"]), axis=1)
    detw = np.asarray(cache["detw_all"])
    h = element.thickness
    C = plane_stress_elastic_matrix(material.elastic_modulus, material.poisson_ratio)
    D = np.zeros((6, 6))
    D[:3, :3] = h * C
    D[3:, 3:] = h ** 3 / 12.0 * C
    E_hat = B + offsets
    orth = np.einsum("g,gki,kl,glj->ij", detw, E_hat, D, offsets)
    scale = np.abs(np.einsum("g,gki,kl,glj->ij", detw, E_hat, D, E_hat)).max()
    # Exact in exact arithmetic; the bound reflects the certified 35x35
    # stationary solve on distorted facets (square: ~1e-16, skew: ~6e-9).
    assert np.abs(orth).max() <= 1e-7 * scale
    # The qualified stiffness minus its assumed-strain membrane/bending energy
    # (hybrid transverse shear plus PL/hourglass stabilisation) stays positive
    # semidefinite, so a vanishing layered tangent cannot make the element
    # tangent negative; the constant delta Cr itself is indefinite.
    T0 = np.asarray(cache["T0"])
    qualified = np.asarray(element.compute_stiffness_matrix(model.mesh, material))
    membrane_bending = T0.T @ np.einsum("g,gki,kl,glj->ij", detw, E_hat, D, E_hat) @ T0
    rest = qualified - membrane_bending
    assert np.linalg.eigvalsh(0.5 * (rest + rest.T))[0] >= -1e-8 * np.abs(qualified).max()
    correction = element._qualified_linear_correction(model.mesh, material, 5)
    assert np.linalg.eigvalsh(0.5 * (correction + correction.T))[0] < -1e6


def _response(element, model, material, u, state, tangent=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return element.compute_nonlinear_response(model.mesh, material, u, state, 5, tangent)


def _fd_error(element, model, material, u, state, K, h=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(3):
        d = rng.standard_normal(24)
        d *= 1e-3 / np.abs(d).max()
        fp = np.asarray(_response(element, model, material, u + h * d, state, False)[0])
        fm = np.asarray(_response(element, model, material, u - h * d, state, False)[0])
        exact = K @ d
        worst = max(worst, np.linalg.norm((fp - fm) / (2 * h) - exact) / np.linalg.norm(exact))
    return worst


def test_tangent_is_symmetric_and_fd_exact_on_load_unload_reverse_path():
    model, element, material = _single(SKEW)
    path = [_field(SKEW, "hourglass", a) + _field(SKEW, "bend", 5 * a) + _field(SKEW, "stretch", 0.5 * a)
            + _field(SKEW, "twist", 2 * a)
            for a in (1e-3, 2e-3, 3e-3, 1.5e-3, 0.0, -2e-3, -3e-3, 0.0)]
    state = None
    alphas = []
    for u in path:
        force, K, trial = _response(element, model, material, u, state)
        K = np.asarray(K)
        assert np.abs(K - K.T).max() <= 1e-12 * np.abs(K).max()
        assert _fd_error(element, model, material, u, state, K) <= 1e-7
        alphas.append(float(np.max(trial["alpha"])))
        state = trial
    assert alphas[0] > 0.0 and alphas[-1] > alphas[2]  # yielded, reversed plastic flow


def test_plastic_hourglass_mode_keeps_positive_stiffness():
    """Main's route: secant f.v/amp = -4.6e6 N/m (negative stiffness)."""

    model, element, material = _single(SQUARE)
    legacy_model, legacy, _ = _single(SQUARE, form="legacy")
    v = _field(SQUARE, "hourglass", 1.0)
    for amplitude in (0.01, 0.1):
        force, K, trial = _response(element, model, material, amplitude * v, None)
        assert float(np.max(trial["alpha"])) > 0.0
        secant = float(np.asarray(force) @ v) / amplitude
        assert secant > 0.0
        lam = np.linalg.eigvalsh(0.5 * (np.asarray(K) + np.asarray(K).T))[0]
        _f, K_legacy, _t = _response(legacy, legacy_model, material, amplitude * v, None)
        lam_legacy = np.linalg.eigvalsh(0.5 * (np.asarray(K_legacy) + np.asarray(K_legacy).T))[0]
        assert lam >= min(lam_legacy, 0.0) - 1e-6 * np.abs(np.asarray(K)).max()


def test_force_is_continuous_across_first_yield():
    model, element, material = _single(SQUARE)
    amplitudes = np.linspace(0.4e-4, 1.4e-4, 201)
    forces, first = [], None
    for a in amplitudes:
        u = _field(SQUARE, "stretch", a) + _field(SQUARE, "hourglass", 0.3 * a)
        f, _k, trial = _response(element, model, material, u, None, False)
        forces.append(np.asarray(f))
        if first is None and np.any(trial["plastic_strain"]):
            first = a
    assert first is not None and amplitudes[0] < first < amplitudes[-1]
    steps = np.linalg.norm(np.diff(forces, axis=0), axis=1)
    # Linear elastic steps are equal; yielding only lowers the slope (a kink),
    # a force jump at first yield would exceed the elastic step.
    assert steps.max() <= (1.0 + 1e-9) * steps[0]


def test_elastic_material_does_not_enter_the_route():
    model, element, material = _single(SQUARE, plastic=False)
    assert not AS.supported(element, model.mesh, material)
    legacy_model, legacy, legacy_material = _single(SQUARE, form="legacy")
    assert not AS.supported(legacy, legacy_model.mesh, legacy_material)


def _plate(n, thickness, pressure, *, form=None):
    model = fs.FEModel("plate")
    model.add_material("mat", 210e9, 0.3, density=7850.0, yield_stress=3e7, hardening_curve=_curve())
    for j in range(n + 1):
        for i in range(n + 1):
            model.add_node(1 + j * (n + 1) + i, i / n, j / n, 0.0)
    kwargs = {} if form is None else {"formulation": form}
    for j in range(n):
        for i in range(n):
            v = 1 + j * (n + 1) + i
            e = 1 + j * n + i
            model.add_element(e, create_shell_element(e, [v, v + 1, v + n + 2, v + n + 1], "mat",
                                                      thickness=thickness, **kwargs))
    edge = [1 + j * (n + 1) + i for j in range(n + 1) for i in range(n + 1) if i in (0, n) or j in (0, n)]
    model.add_boundary_condition(FixedSupport("clamp", edge))
    load = LoadCase("p")
    for e in model.mesh.elements:
        load.add_pressure_load(e, pressure)
    model.apply_boundary_conditions()
    return model, load


def test_reference_plan_and_scalar_paths_agree_on_plastic_patch():
    model, load = _plate(2, 0.003, 5e4)
    rng = np.random.default_rng(4)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    centre = 4
    u[6 * centre + 2] = 0.03
    u += 2e-4 * rng.standard_normal(u.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_ref, K_ref, states = _assemble_nonlinear_system(
            model, u, {}, 5, tangent=True, require_full_coordinates=True)
        plan = NonlinearAssemblyPlan.build(model, 5)
        assert any(batch.key[0] == "isotropic_q4_assumed_strain_plastic" for batch in plan.shell_batches)
        f_plan, K_plan, _ = plan.assemble(u, {}, tangent=True)
    assert any(np.any(s["plastic_strain"]) for s in states.values())
    K_ref = K_ref.toarray()
    K_plan = K_plan.toarray()
    material = model.get_material("mat")
    F_sc = np.zeros_like(f_ref)
    for eid, element in model.mesh.elements.items():
        dm = np.asarray(element.get_dof_mapping(model.mesh))
        f, _k, _s = _response(element, model, material, u[dm], None)
        F_sc[dm] += np.asarray(f)
    scale = np.abs(f_ref).max()
    assert np.abs(f_plan - f_ref).max() <= 1e-12 * scale
    assert np.abs(F_sc - f_ref).max() <= 1e-12 * scale
    assert np.abs(K_plan - K_ref).max() <= 1e-12 * np.abs(K_ref).max()


# Owner case C1: clamped 1 x 1 m plate, t = 3 mm, 50 kPa, von Karman.
# Legacy element (oracle), 1 step: n=8 38.974 mm, n=16 36.380 mm; CalculiX S4
# n=64 35.37 mm.  Main returned +203.1 mm (n=8) and -42.8 mm (n=16).
LEGACY_C1 = {8: 0.038974, 16: 0.036380}


def _centre_deflection(n, steps):
    model, load = _plate(n, 0.003, 5e4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fs.solve_static_nonlinear(
            model, load, num_steps=steps, max_iterations=25, kinematics="von_karman")
    u = np.asarray(result.displacements, dtype=float)
    centre = (n // 2) * (n + 1) + n // 2
    return result, float(u.reshape(-1, 6)[centre, 2])


@pytest.mark.parametrize("n", [8, 16])
def test_owner_case_c1_completes_near_legacy_oracle(n):
    result, w_centre = _centre_deflection(n, 1)
    assert result.status == "completed" and abs(float(result.load_factor) - 1.0) <= 1e-12
    assert abs(w_centre - LEGACY_C1[n]) <= 0.05 * LEGACY_C1[n]


@pytest.mark.skipif(os.environ.get("ANYSOLVER_RUN_SLOW_TESTS") != "1",
                    reason="slow: set ANYSOLVER_RUN_SLOW_TESTS=1")
@pytest.mark.parametrize("n,steps", [(8, 5), (8, 20), (16, 5), (32, 5)])
def test_owner_case_c1_step_and_mesh_sweep(n, steps):
    legacy = {8: 0.038974, 16: 0.036378, 32: 0.035177}
    result, w_centre = _centre_deflection(n, steps)
    assert result.status == "completed"
    assert abs(w_centre - legacy[n]) <= 0.05 * legacy[n]


# ---------------------------------------------------------------------------
# Scope: typed refusal instead of the historical unstable plastic route
# ---------------------------------------------------------------------------

def _hill_material(with_curve=True):
    from anyfesolver.materials import Hill48Yield, OrthotropicMaterial

    shear = 3e7 / np.sqrt(3.0)
    return OrthotropicMaterial(
        name="hill", elastic_modulus_1=150e9, elastic_modulus_2=12e9, elastic_modulus_3=10e9,
        poisson_ratio_12=0.25, poisson_ratio_13=0.2, poisson_ratio_23=0.3,
        shear_modulus_12=5e9, shear_modulus_13=4e9, shear_modulus_23=3.8e9, density=1600.0,
        hill_yield=Hill48Yield(3e7, 3e7, 3e7, shear, shear, shear),
        hardening_curve=_curve() if with_curve else None,
    )


@pytest.mark.parametrize("with_curve", [True, False], ids=["curve", "perfectly_plastic"])
def test_hill48_plastic_q4_is_refused_with_typed_error(with_curve):
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from anyfesolver.fe_core import FEMesh

    mesh = FEMesh()
    for index, (x, y) in enumerate(SQUARE):
        mesh.add_node(index + 1, float(x), float(y), 0.0)
    material = _hill_material(with_curve)
    element = QualifiedE4PLShellElement(1, [1, 2, 3, 4], material.name, thickness=0.003,
                                        material_direction=(1.0, 0.0, 0.0))
    with pytest.raises(AS.UnsupportedQ4PlasticRouteError, match="orthotropic/Hill48"):
        _response(element, type("M", (), {"mesh": mesh})(), material, _field(SQUARE, "stretch", 1e-3), None)


def test_warped_plastic_q4_is_refused_with_typed_error():
    nodes = ((0, 0), (0.125, 0), (0.125, 0.125), (0, 0.125))
    model, element, material = _single(nodes)
    model.mesh.nodes[3].z = 0.01  # warp one corner
    element._nonlinear_geometry_cache = None
    with pytest.raises(AS.UnsupportedQ4PlasticRouteError, match="warped"):
        AS.require_supported(element, model.mesh, material)


def _zero_field_state(element):
    state = element.init_nonlinear_state(5)
    state["initial_membrane_prestrain"] = np.zeros((4, 3))
    return state


def test_initial_fields_one_predicate_for_scalar_and_batch():
    model, load = _plate(2, 0.003, 5e4)
    material = model.get_material("mat")
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    u[6 * 4 + 2] = 0.03
    u += 2e-4 * np.random.default_rng(5).standard_normal(u.size)
    # Inactive (all-zero) field: scalar == reference batch == plan == no field.
    inactive = {eid: _zero_field_state(e) for eid, e in model.mesh.elements.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_none, K_none, _ = _assemble_nonlinear_system(model, u, {}, 5, tangent=True, require_full_coordinates=True)
        f_ref, K_ref, _ = _assemble_nonlinear_system(model, u, inactive, 5, tangent=True, require_full_coordinates=True)
        f_plan, K_plan, _ = NonlinearAssemblyPlan.build(model, 5).assemble(u, inactive, tangent=True)
    F_sc = np.zeros_like(f_ref)
    for eid, element in model.mesh.elements.items():
        dm = np.asarray(element.get_dof_mapping(model.mesh))
        F_sc[dm] += np.asarray(_response(element, model, material, u[dm], inactive[eid])[0])
    scale = np.abs(f_none).max()
    for f in (f_ref, f_plan, F_sc):
        assert np.abs(f - f_none).max() <= 1e-12 * scale
    for K in (K_ref, K_plan):
        assert np.abs(K.toarray() - K_none.toarray()).max() <= 1e-12 * np.abs(K_none.toarray()).max()
    # Active field: refused by the scalar route and by both batch routes.
    active = {eid: _zero_field_state(e) for eid, e in model.mesh.elements.items()}
    active[1]["initial_membrane_prestrain"][0, 0] = 1e-4
    element = model.mesh.elements[1]
    dm = np.asarray(element.get_dof_mapping(model.mesh))
    with pytest.raises(AS.UnsupportedQ4PlasticRouteError, match="active initial"):
        _response(element, model, material, u[dm], active[1])
    with pytest.raises(AS.UnsupportedQ4PlasticRouteError, match="active initial"):
        _assemble_nonlinear_system(model, u, active, 5, tangent=True, require_full_coordinates=True)
    with pytest.raises(AS.UnsupportedQ4PlasticRouteError, match="active initial"):
        NonlinearAssemblyPlan.build(model, 5).assemble(u, active, tangent=True)


# ---------------------------------------------------------------------------
# Reversal: plastic strain back at exactly zero on the plastic branch
# ---------------------------------------------------------------------------

def test_tangent_correction_follows_algorithmic_branch_not_plastic_strain_value(monkeypatch):
    """A point may return to eps_p == 0 with C_t != C (reversed flow).

    The force correction vanishes there but its derivative does not.  The
    return map is wrapped to report eps_p == 0 exactly while keeping its
    algorithmic tangent; the tangent must still carry the plastic term.
    """

    model, element, material = _single(SQUARE)
    cache = element._nonlinear_geometry(model.mesh)
    one = lambda name: np.asarray(cache[name], float)[None]
    offsets = AS.offset_operator(element, model.mesh, material)
    u = _field(SQUARE, "hourglass", 0.01) + _field(SQUARE, "stretch", 2e-3)
    args = (u[None], one("T0"), one("B_m_all"), one("B_b_all"), one("B_d_all"), one("Gw_all"),
            one("detw_all"), one("B_s_all"), one("detw_shear_all"), material.elastic_modulus,
            material.poisson_ratio, material.shear_modulus, element.thickness,
            element.drilling_stabilization, True, material.hardening_curve,
            np.zeros((1, 20, 3)), np.zeros((1, 20)), 5, offsets[None])
    real = AS.plane_stress_return_map
    _f, K_real, ep, *_ = AS.kernel_batch(*args)
    assert np.any(ep != 0.0)

    def zero_plastic_strain(*a, **k):
        sigma, C_t, ep_new, alpha_new = real(*a, **k)
        return sigma, C_t, np.zeros_like(ep_new), alpha_new

    monkeypatch.setattr(AS, "plane_stress_return_map", zero_plastic_strain)
    _f, K_zero, ep0, *_ = AS.kernel_batch(*args)
    assert not np.any(ep0)
    # Same algorithmic tangent -> same plastic tangent term.  With in-plane
    # motion (theta = 0) only the w-w block holds the geometric N_p term,
    # which legitimately vanishes with eps_p.
    keep = np.ones(24, bool)
    keep[2::6] = False
    block = np.ix_(keep, keep)
    assert np.abs(K_zero[0][block] - K_real[0][block]).max() <= 1e-12 * np.abs(K_real[0]).max()


def test_fd_tangent_close_to_reversal_through_zero_plastic_strain():
    model, element, material = _single(SQUARE)
    v = _field(SQUARE, "stretch", 1.0) + _field(SQUARE, "hourglass", 0.2)
    _f, _K, loaded = _response(element, model, material, 2e-3 * v, None)
    assert np.any(loaded["plastic_strain"])
    # Reverse past the elastic range into reverse plastic flow.
    for amplitude in (-0.5e-3, -1.0e-3, -1.5e-3):
        f, K, trial = _response(element, model, material, amplitude * v, loaded)
        assert _fd_error(element, model, material, amplitude * v, loaded, np.asarray(K)) <= 1e-7


# ---------------------------------------------------------------------------
# Runtime authority of the new module
# ---------------------------------------------------------------------------

def test_module_numpy_usage_is_inside_the_frozen_runtime_authority():
    import ast
    import inspect
    from anyfesolver import e4_pl_s3_state

    tree = ast.parse(inspect.getsource(AS))
    root, linalg = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "np":
            root.add(node.attr)
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name) and node.value.value.id == "np"
                and node.value.attr == "linalg"):
            linalg.add(node.attr)
    root.discard("linalg")
    assert root <= set(e4_pl_s3_state._NUMPY_RUNTIME_ROOT_NAMES), root - set(e4_pl_s3_state._NUMPY_RUNTIME_ROOT_NAMES)
    assert linalg <= set(e4_pl_s3_state._NUMPY_RUNTIME_LINALG_NAMES)


def test_module_substitution_before_evaluation_is_refused(monkeypatch):
    model, element, material = _single(SQUARE)
    u = _field(SQUARE, "stretch", 2e-3)
    _response(element, model, material, u, None)  # bind the authority
    original = AS.kernel_batch
    monkeypatch.setattr(AS, "kernel_batch", lambda *a, **k: original(*a, **k))
    with pytest.raises(ValueError, match="authority"):
        _response(element, model, material, u, None)


def test_module_substitution_during_evaluation_is_refused():
    """The swap happens inside the evaluation (from the return map's read of the
    hardening curve), after the entry guard has passed; the refusal must come
    from the in-evaluation authority check and the hook must have run."""

    model, element, material = _single(SQUARE)
    u = _field(SQUARE, "stretch", 2e-3)
    base = material.hardening_curve
    hook = {"armed": False, "calls": 0, "swapped": False}
    original = AS.offset_operator

    class ObservingCurve(type(base)):
        @property
        def sigma_prop(self):
            if hook["armed"]:
                hook["calls"] += 1
                if not hook["swapped"]:
                    hook["swapped"] = True
                    AS.offset_operator = lambda *a, **k: original(*a, **k)
            return self.__dict__["_observed_sigma_prop"]

    observed = ObservingCurve.__new__(ObservingCurve)
    observed.__dict__.update(base.__dict__)
    observed.__dict__["_observed_sigma_prop"] = base.sigma_prop
    material.hardening_curve = observed
    try:
        _response(element, model, material, u, None)  # entry and evaluation pass unarmed
        hook["armed"] = True
        with pytest.raises(ValueError, match="authority changed: .*offset_operator"):
            _response(element, model, material, u, None)
        assert hook["swapped"] and hook["calls"] >= 1
    finally:
        AS.offset_operator = original


def test_preyield_layer_stress_matches_independent_mixed_recovery():
    """Independent oracle for the pre-yield assumed-strain layer stress.

    The qualified element's mixed stress recovery (separate stationary solve
    and field evaluation, ``_recover_planar_mixed_fields``) supplies the hybrid
    membrane strain and curvature at the Gauss points; adding the von-Karman
    strain of the same displacement and multiplying by C must give the layer
    stress the assumed-strain kernel stores.  The compatible legacy element
    differs exactly by the hybrid relief.
    """

    from anyfesolver.plasticity import lobatto_layers

    model, element, material = _single(SQUARE)
    u = (_field(SQUARE, "hourglass", 2e-5) + _field(SQUARE, "stretch", 3e-5)
         + _field(SQUARE, "bend", 2e-4) + _field(SQUARE, "twist", 1e-4))
    _f, _K, trial = _response(element, model, material, u, None)
    assert not np.any(trial["plastic_strain"])
    recovered = element._recover_planar_mixed_fields(model.mesh, u, material, AS._GP_KERNEL)
    cache = element._nonlinear_geometry(model.mesh)
    u_loc = np.asarray(cache["T0"]) @ u
    z, _w = lobatto_layers(5, element.thickness)
    C = plane_stress_elastic_matrix(material.elastic_modulus, material.poisson_ratio)
    expected = []
    for g, Gw in enumerate(np.asarray(cache["Gw_all"])):
        theta = Gw @ u_loc
        membrane = recovered["independent"][g][:3] + np.array(
            [0.5 * theta[0] ** 2, 0.5 * theta[1] ** 2, theta[0] * theta[1]])
        expected.extend((membrane + z[:, None] * recovered["independent"][g][3:6]) @ C.T)
    expected = np.asarray(expected)
    actual = np.asarray(trial["layer_stress"])
    assert np.abs(actual - expected).max() <= 1e-9 * np.abs(expected).max()
    legacy_model, legacy, _ = _single(SQUARE, form="legacy")
    _f, _K, legacy_state = _response(legacy, legacy_model, material, u, None)
    assert np.abs(np.asarray(legacy_state["layer_stress"]) - actual).max() > 1e-3 * np.abs(actual).max()


def _mixed_plate(n, mode):
    """Clamped C1 plate (t = 3 mm, 50 kPa) whose hardening curve never yields."""

    strong = DNVC208MaterialCurve(sigma_prop=3e9, sigma_yield=3.2e9, sigma_yield_2=3.4e9,
                                  eps_p_y1=0.005, eps_p_y2=0.01, K=4e10, n=0.2)
    model = fs.FEModel("mixed")
    model.add_material("curve", 210e9, 0.3, density=7850.0, yield_stress=3e9, hardening_curve=strong)
    model.add_material("plain", 210e9, 0.3, density=7850.0)
    for j in range(n + 1):
        for i in range(n + 1):
            model.add_node(1 + j * (n + 1) + i, i / n, j / n, 0.0)
    for j in range(n):
        for i in range(n):
            v = 1 + j * (n + 1) + i
            e = 1 + j * n + i
            name = {"curve": "curve", "plain": "plain", "mixed": "curve" if i < n // 2 else "plain"}[mode]
            model.add_element(e, create_shell_element(e, [v, v + 1, v + n + 2, v + n + 1], name, thickness=0.003))
    edge = [1 + j * (n + 1) + i for j in range(n + 1) for i in range(n + 1) if i in (0, n) or j in (0, n)]
    model.add_boundary_condition(FixedSupport("clamp", edge))
    load = LoadCase("p")
    for e in model.mesh.elements:
        load.add_pressure_load(e, 5e4)
    model.apply_boundary_conditions()
    return model, load


@pytest.mark.parametrize("kinematics", ["von_karman", "corotational"])
def test_single_elastic_law_mixed_mesh_matches_pure_meshes(kinematics):
    """Yield-capable (never yielding) and plain qualified Q4 share one elastic law."""

    answers = {}
    for mode in ("curve", "plain", "mixed"):
        model, load = _mixed_plate(4, mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fs.solve_static_nonlinear(model, load, num_steps=5, max_iterations=40,
                                               kinematics=kinematics)
        assert result.status == "completed"
        assert max(float(np.max(s["alpha"])) for s in result.element_states.values()
                   if isinstance(s, dict) and "alpha" in s) == 0.0
        answers[mode] = np.asarray(result.displacements, float)
    scale = np.abs(answers["plain"]).max()
    for mode in ("curve", "mixed"):
        assert np.abs(answers[mode] - answers["plain"]).max() <= 1e-9 * scale, mode
