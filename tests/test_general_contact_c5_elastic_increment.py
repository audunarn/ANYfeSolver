"""Direct elastic shell arithmetic and its actual contact assembly connection."""
import copy
import inspect
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

from anyfesolver import FEModel
from anyfesolver.elements import create_shell_element
from anyfesolver import e4_pl_element as qualified
from anyfesolver import vectorized_nonlinear as vectorized
from anyfesolver.nonlinear_static import _assemble_nonlinear_system


def case(*, warped=False, yielding=False):
    model = FEModel("contact_elastic_increment")
    material = model.add_material("elastic", 1.0e5, .3,
                                  yield_stress=1.0 if yielding else 0.0)
    for node, xyz in enumerate(((0., 0., 0.), (1., 0., 0.),
                                (1., 1., .1 if warped else 0.),
                                (0., 1., 0.)), 1):
        model.add_node(node, *xyz)
    element = create_shell_element(1, (1, 2, 3, 4), "elastic", thickness=.05)
    model.add_element(1, element)
    model.apply_boundary_conditions()
    return model, element, material


def pose():
    return np.array((.002, -.001, .008, .001, -.003, .0002,
                     -.001, .002, -.004, -.002, .001, -.0003,
                     .001, .003, .006, .003, -.001, .0001,
                     -.002, -.001, -.003, -.001, .002, -.0001))


def batch_zero(element, mesh, material):
    cache = element._nonlinear_geometry(mesh)
    return vectorized.compensated_elastic_batch_zero(
        cache, material.elastic_modulus, material.poisson_ratio,
        material.shear_modulus, element.thickness, element.drilling_stabilization)


def single_law_expected(element, mesh, material, u, state):
    """Independent single-law oracle (owner decision 2026-10-23).

    Qualified Q4 nonlinear elastic response = historical law (inherited
    compatible kernel + Cr u) + the von Karman cross terms of the hybrid
    assumed strain, dE_m' A eps_nl + B_nl' A dE_m u, evaluated here from the
    certified offset operator and the element's own Gauss data.  Warped
    facets (no assumed-strain route) keep the historical law.
    """

    from anyfesolver import _q4_assumed_strain_plasticity as assumed

    correction = element._qualified_linear_correction(mesh, material, 5)
    element._hourglass_stiffness_matrix = None
    old_force, old_tangent, _ = qualified._QUALIFIED_Q4_BASE_NONLINEAR_KERNEL(
        element, mesh, material, u, state, 5, True)
    force = old_force + correction @ u
    tangent = old_tangent + correction
    if not assumed.elastic_route(element, mesh, material):
        return force, tangent
    cache = element._nonlinear_geometry(mesh)
    T0 = np.asarray(cache["T0"])
    Gw = np.asarray(cache["Gw_all"])
    detw = np.asarray(cache["detw_all"])
    dE = np.asarray(assumed.offset_operator(element, mesh, material))[:, :3]
    A = element.thickness * assumed.plane_stress_elastic_matrix(
        material.elastic_modulus, material.poisson_ratio)
    ul = T0 @ np.asarray(u, dtype=float)
    f_loc = np.zeros(24)
    k_loc = np.zeros((24, 24))
    for g in range(detw.size):
        th = Gw[g] @ ul
        eps_nl = np.array((0.5 * th[0] ** 2, 0.5 * th[1] ** 2, th[0] * th[1]))
        b_nl = np.vstack((th[0] * Gw[g][0], th[1] * Gw[g][1], th[0] * Gw[g][1] + th[1] * Gw[g][0]))
        n_off = A @ (dE[g] @ ul)
        f_loc += detw[g] * (dE[g].T @ (A @ eps_nl) + b_nl.T @ n_off)
        cross = dE[g].T @ A @ b_nl
        geo = (n_off[0] * np.outer(Gw[g][0], Gw[g][0]) + n_off[1] * np.outer(Gw[g][1], Gw[g][1])
               + n_off[2] * (np.outer(Gw[g][0], Gw[g][1]) + np.outer(Gw[g][1], Gw[g][0])))
        k_loc += detw[g] * (cross + cross.T + geo)
    return force + T0.T @ f_loc, tangent + T0.T @ k_loc @ T0


@pytest.mark.parametrize("scale", [0., 1., -1.])
def test_scalar_direct_matches_inherited_law_and_components(scale):
    # Retargeted 2026-10-23 (single elastic law): the scalar response equals the
    # historical law plus the von Karman cross terms of the assumed strain,
    # computed independently by single_law_expected (same tolerances).
    model, element, material = case()
    u = scale * pose()
    state = element.init_nonlinear_state(5)
    saved = copy.deepcopy(state)
    expected_force, expected_tangent = single_law_expected(
        element, model.mesh, material, u, state)
    force, tangent, trial, components = element.compute_nonlinear_response(
        model.mesh, material, u, state, 5, True,
        _return_tangent_components=True)
    np.testing.assert_allclose(force, expected_force,
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(tangent, expected_tangent,
                               rtol=2e-12, atol=2e-10)
    np.testing.assert_array_equal(tangent, components["material"] + components["geometric"])
    for key in saved:
        np.testing.assert_array_equal(state[key], saved[key])
    assert "plastic_strain" in trial and "alpha" in trial


def test_direct_tangent_is_force_derivative():
    model, element, material = case()
    u = pose()
    direction = np.linspace(-.7, .9, 24)
    step = 2e-7
    force, tangent, _ = element.compute_nonlinear_response(model.mesh, material, u)
    plus = element.compute_nonlinear_response(model.mesh, material, u + step*direction,
                                              tangent=False)[0]
    minus = element.compute_nonlinear_response(model.mesh, material, u - step*direction,
                                               tangent=False)[0]
    np.testing.assert_allclose((plus-minus)/(2*step), tangent @ direction,
                               rtol=2e-7, atol=2e-7)
    assert np.all(np.isfinite(force))


def test_nonzero_represented_batch_offset_is_retained():
    model, element, material = case()
    element._qualified_linear_correction(model.mesh, material, 5)
    zero = np.zeros(24)
    # Ks0 must use the same reset as the production correction/direct route.
    element._hourglass_stiffness_matrix = None
    _, scalar_zero, _ = qualified._QUALIFIED_Q4_BASE_NONLINEAR_KERNEL(
        element, model.mesh, material, zero, None, 5, True)
    # Deliberately nonzero metadata-independent operator defect proves that
    # admission cannot silently assume the two represented baselines equal.
    defect = np.diag(np.linspace(.125, .375, 24))
    direct = element._elastic_increment_response(
        model.mesh, material, pose(), None, True,
        batch_linear=scalar_zero + defect)
    scalar = element._elastic_increment_response(model.mesh, material, pose(), None, True)
    np.testing.assert_allclose(direct[0]-scalar[0], defect @ pose(), atol=2e-12)
    np.testing.assert_allclose(direct[1]-scalar[1], defect, atol=2e-10)


def test_actual_batch_assembly_uses_single_law_kernel():
    # Retargeted 2026-10-23 (single elastic law): the retired compensated direct
    # elastic route is no longer the batch route for a planar qualified Q4; the
    # actual batch assembly must equal the independent single-law oracle.
    from anyfesolver import _q4_assumed_strain_plasticity as assumed

    model, element, material = case()
    u = pose()
    assert assumed.elastic_route(element, model.mesh, material)
    force, tangent, states = _assemble_nonlinear_system(
        model, u, {}, 5, tangent=True, require_full_coordinates=True)
    expected_force, expected_tangent = single_law_expected(
        element, model.mesh, material, u, None)
    # Measured agreement with the oracle: force 4.4e-16, tangent 1.8e-15 (absolute).
    np.testing.assert_allclose(force, expected_force, rtol=0., atol=1e-15)
    np.testing.assert_allclose(tangent.toarray(), expected_tangent, rtol=0., atol=1e-12)
    assert 1 in states


@pytest.mark.parametrize("kind", ["warped", "yielding", "initial_field", "custom_mapping"])
def test_unsupported_routes_do_not_admit_direct_increment(kind):
    model, element, material = case(warped=kind == "warped", yielding=kind == "yielding")
    state = None
    if kind == "initial_field":
        state = element.init_nonlinear_state(5)
        state["initial_membrane_prestrain"] = np.zeros(3)
    elif kind == "custom_mapping":
        class ForeignState(dict):
            pass
        state = ForeignState(element.init_nonlinear_state(5))
    assert not element._elastic_increment_admitted(model.mesh, material, state)
    assert element._elastic_increment_response(model.mesh, material, pose(), state, True) is None
    # Retargeted 2026-10-23 (single elastic law): the fallback is the single
    # law (historical law for warped facets), checked against the independent
    # oracle; the direct-route admission assertions above are unchanged.
    expected_force, expected_tangent = single_law_expected(
        element, model.mesh, material, pose(), state)
    force, tangent, _ = element.compute_nonlinear_response(
        model.mesh, material, pose(), state, 5, True)
    if kind == "warped":
        # Warped facets keep the historical law: exact equality as before.
        np.testing.assert_array_equal(force, expected_force)
        np.testing.assert_array_equal(tangent, expected_tangent)
    else:
        # Measured agreement: force 4.4e-16, tangent <= 4.5e-13 (absolute).
        np.testing.assert_allclose(force, expected_force, rtol=0., atol=1e-15)
        np.testing.assert_allclose(tangent, expected_tangent, rtol=0., atol=1e-12)


def test_geometry_mutation_recomputes_one_call_preparation():
    model, element, material = case()
    first = element.compute_nonlinear_response(model.mesh, material, pose())[0]
    model.mesh.nodes[2].x = 1.2
    second = element.compute_nonlinear_response(model.mesh, material, pose())[0]
    fresh, fresh_element, fresh_material = case()
    fresh.mesh.nodes[2].x = 1.2
    expected = fresh_element.compute_nonlinear_response(fresh.mesh, fresh_material, pose())[0]
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(second, expected)


def test_increment_dependency_rebinding_refuses(monkeypatch):
    model, element, material = case()
    element.compute_nonlinear_response(model.mesh, material, pose())
    # The qualified authority boundary, not a spy replacing the evaluator,
    # must reject a changed helper binding before producing a force.
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        monkeypatch.setattr(vectorized, "elastic_shell_nonlinear_increment", lambda *a, **k: None)
        element.compute_nonlinear_response(model.mesh, material, pose())


@pytest.mark.parametrize("where", ["admission", "geometry", "stiffness"])
def test_original_snapshot_spans_every_preparation_boundary(where):
    # Exercise the actual response body, substituting only metadata/preparation
    # dependencies. This controls mutation *before* a late snapshot could hide
    # it, without weakening or monkeypatching the qualified production class.
    raw = inspect.getclosurevars(
        qualified.QualifiedE4PLShellElement._elastic_increment_response
    ).nonlocals["method"]
    generation = [0]
    captured = []
    def capture(*args):
        captured.append(generation[0])
        return generation[0]
    def observe(name, result):
        if where == name:
            generation[0] += 1
        return result
    fake = SimpleNamespace(
        _elastic_increment_admitted=lambda *a: observe("admission", True),
        _defining_class_input_snapshot_matches=lambda *a: a[-1] == generation[0],
        _nonlinear_geometry=lambda *a: observe("geometry", {}),
        compute_stiffness_matrix=lambda *a: observe("stiffness", np.eye(24)),
    )
    namespace = dict(raw.__globals__,
        _capture_q4_fast_input_snapshot=capture)
    actual_body = FunctionType(raw.__code__, namespace, raw.__name__,
                               raw.__defaults__, raw.__closure__)
    actual_body.__kwdefaults__ = raw.__kwdefaults__
    with pytest.raises(RuntimeError, match="changed during preparation"):
        actual_body(fake, object(), object(), pose(), None, True)
    assert captured == [0]


def test_actual_vector_observation_mutation_cannot_publish():
    model, element, material = case()
    element.compute_nonlinear_response(model.mesh, material, pose())
    class MutatingVector:
        def __array__(self, dtype=None, copy=None):
            material.elastic_modulus = 2.0e5
            return np.asarray(pose(), dtype=dtype)
    with pytest.raises((RuntimeError, ValueError), match="changed|incompatible"):
        element._elastic_increment_response(
            model.mesh, material, MutatingVector(), None, True)


def test_single_law_cross_terms_are_independent_of_the_offset_operator():
    """Operator-independent check of the single-law cross terms.

    The parasitic strain dE u is taken from the element's mixed stress recovery
    (``_recover_planar_mixed_fields``: its own stationary solve and field
    evaluation, not ``offset_operator``) as independent minus compatible
    membrane strain at the Gauss points. The cross terms are the gradient of
    W(u) = sum_g w_g (dE u)' A eps_nl(u). That gradient is formed here by central
    finite differences and compared with the production response minus the
    historical law (kernel + Cr u). W is cubic, so the FD error is O(h^2);
    the tolerance is set from it.
    """

    model, element, material = case()
    u = pose()
    natural = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)),
               (-1 / np.sqrt(3), 1 / np.sqrt(3)), (1 / np.sqrt(3), 1 / np.sqrt(3))]
    cache = element._nonlinear_geometry(model.mesh)
    T0 = np.asarray(cache["T0"])
    Gw = np.asarray(cache["Gw_all"])
    detw = np.asarray(cache["detw_all"])
    nu = material.poisson_ratio
    A = element.thickness * material.elastic_modulus / (1 - nu ** 2) * np.array(
        ((1, nu, 0), (nu, 1, 0), (0, 0, (1 - nu) / 2)))

    def energy(v):
        recovered = element._recover_planar_mixed_fields(model.mesh, v, material, natural)
        parasitic = recovered["independent"][:, :3] - recovered["compatible"][:, :3]
        local = T0 @ v
        total = 0.0
        for g in range(4):
            th = Gw[g] @ local
            eps_nl = np.array((0.5 * th[0] ** 2, 0.5 * th[1] ** 2, th[0] * th[1]))
            total += detw[g] * parasitic[g] @ (A @ eps_nl)
        return total

    h = 1e-6
    gradient = np.array([(energy(u + h * e) - energy(u - h * e)) / (2 * h) for e in np.eye(24)])
    correction = element._qualified_linear_correction(model.mesh, material, 5)
    element._hourglass_stiffness_matrix = None
    old_force, _old_tangent, _ = qualified._QUALIFIED_Q4_BASE_NONLINEAR_KERNEL(
        element, model.mesh, material, u, None, 5, True)
    force, _tangent, _ = element.compute_nonlinear_response(model.mesh, material, u, None, 5, True)
    cross = np.asarray(force) - (old_force + correction @ u)
    assert np.abs(cross).max() > 1e-6 * np.abs(force).max()  # the cross terms are not trivial here
    np.testing.assert_allclose(cross, gradient, rtol=0., atol=1e-7 * np.abs(cross).max())