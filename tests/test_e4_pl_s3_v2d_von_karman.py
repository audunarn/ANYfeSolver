"""Experimental R1 VK regressions; these do not establish qualification."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from anyfesolver.e4_pl_s3_v2d_state import canonical_json_bytes
from test_e4_pl_s3_v2d_native_state_corotational import _model


MEASUREMENTS = {}


@pytest.fixture(scope="session", autouse=True)
def record_measurements():
    yield
    path = os.environ.get("S3_EVIDENCE_OUT")
    if path:
        import anyfesolver
        MEASUREMENTS["runtime"] = {"python":sys.version, "module":anyfesolver.__file__,
            "PYTHONPATH":os.environ.get("PYTHONPATH"), "OMP_NUM_THREADS":os.environ.get("OMP_NUM_THREADS")}
        Path(path).write_text(json.dumps(MEASUREMENTS, indent=2))


def _response(model, element, u, state=None, **kwargs):
    return element.compute_nonlinear_response(
        model.mesh, model.get_material("steel"), u, state, 5, True,
        von_karman=True, **kwargs,
    )


def test_affine_transverse_slope_has_membrane_strain_and_cubic_force():
    model, element = _model()
    geometry = element._geometry(model.mesh)
    local = np.zeros(18)
    local[2::6] = 0.02 * geometry["local_coordinates"][:, 0]
    u = geometry["local_from_external"].T @ local
    f, _, state = _response(model, element, u)
    np.testing.assert_allclose(state["station_generalized_strain"][:, :3],
                               np.tile([0.0002, 0, 0], (3, 1)), atol=1e-16)
    f2, _, _ = _response(model, element, 2*u)
    assert np.linalg.norm(f2-2*f) > 1
    _, _, old = element.compute_nonlinear_response(model.mesh, model.get_material("steel"), u)
    np.testing.assert_array_equal(old["station_generalized_strain"][:, :3], np.zeros((3, 3)))


@pytest.mark.parametrize("plastic,generalized,offset,polarity", [
    (False, False, 0.0, 1), (True, False, 0.004, -1), (False, True, 0.003, -1),
])
def test_consistent_tangent_initial_fields_history_and_nonmutation(plastic, generalized, offset, polarity):
    model, element = _model(plastic=plastic, generalized=generalized)
    element.reference_surface_offset = offset
    element.director_polarity = polarity
    material = model.get_material("steel")
    virgin = element.init_model_bound_nonlinear_state(model.mesh, material, 5,
        initial_fields={"initial_membrane_stress": np.tile([1e6, 2e6, 3e5], (3,1))})
    u = np.zeros(18)
    u[6] = .008 if plastic else .0001
    u[12] = -.002 if plastic else -.00003
    u[8] = .025
    u[14] = -.013
    u[4] = .002
    _, _, committed = _response(model, element, u, virgin)
    if plastic:
        assert np.max(committed["alpha"]) > 0
    before = canonical_json_bytes(committed)
    trial_u = u.copy()
    trial_u[6] += .0002
    trial_u[8] += .0003
    _, tangent, trial = _response(model, element, trial_u, committed)
    numerical = np.empty((18,18))
    h = 2e-8
    for j in range(18):
        du = np.zeros(18); du[j] = h
        numerical[:,j] = (_response(model, element, trial_u+du, committed)[0]
                          - _response(model, element, trial_u-du, committed)[0])/(2*h)
    error = np.linalg.norm(tangent-numerical)/np.linalg.norm(numerical)
    print(f"FD plastic={plastic} generalized={generalized}: {error:.17g}")
    MEASUREMENTS[f"FD_plastic_{plastic}_section_{generalized}"] = error
    assert error <= 1e-6
    assert canonical_json_bytes(committed) == before
    assert np.array_equal(trial["initial_generalized_resultant"], committed["initial_generalized_resultant"])
    restored = element.deserialize_nonlinear_state(model.mesh, material,
        element.serialize_nonlinear_state(model.mesh, material, committed, 5), 5)
    restored_response = _response(model, element, trial_u, restored)
    np.testing.assert_array_equal(restored_response[1], tangent)
    assert canonical_json_bytes(restored_response[2]) == canonical_json_bytes(trial)


def test_reference_and_optimized_assembly_dispatch_and_recovery():
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system_unchecked
    from anyfesolver.nonlinear_performance import NonlinearAssemblyPlan
    from anyfesolver.recovery import _recover_v2d_committed_state
    model, element = _model()
    model.apply_boundary_conditions()
    u = np.zeros(18); u[8] = .02
    f, k, state = _response(model, element, u)
    for response in (
        _assemble_nonlinear_system_unchecked(model, u, {}, 5, require_full_coordinates=True),
        NonlinearAssemblyPlan.build(model, 5).assemble(u, {}, tangent=True),
    ):
        np.testing.assert_allclose(response[0], f, rtol=1e-14, atol=1e-10)
        np.testing.assert_allclose(response[1].toarray(), k, rtol=1e-14, atol=1e-8)
        np.testing.assert_array_equal(response[2][1]["station_generalized_strain"], state["station_generalized_strain"])
    recovered, reason, _ = _recover_v2d_committed_state(model, 1, element, state,
        displacements=u, return_global=True)
    assert recovered is not None, reason
    np.testing.assert_array_equal(recovered["membrane_resultants"], state["station_generalized_resultant"][:, :3])
    np.testing.assert_array_equal(recovered["_recovery_stress_frame"], element._geometry(model.mesh)["frame"])


def test_public_clamped_n4_plate_membrane_stiffens():
    import anyfesolver as fs
    from anyfesolver.elements import create_shell_element
    from anyfesolver.boundary import FixedSupport, LoadCase
    model = fs.FEModel("experimental-s3-vk-n4")
    model.add_material("steel", 210e9, .3, density=7850.)
    n = 4
    for j in range(n+1):
        for i in range(n+1):
            model.add_node(1+j*(n+1)+i, i/n, j/n, 0.)
    eid = 0
    for j in range(n):
        for i in range(n):
            a = 1+j*(n+1)+i; b,c,d = a+1,a+n+2,a+n+1
            triangles = [[a,b,c],[a,c,d]] if (i+j)%2 == 0 else [[a,b,d],[b,c,d]]
            for nodes in triangles:
                eid += 1
                model.add_element(eid, create_shell_element(eid, nodes, "steel", thickness=.003,
                    reference_normal=[0.,0.,1.]))
    edges = [1+j*(n+1)+i for j in range(n+1) for i in range(n+1) if i in (0,n) or j in (0,n)]
    model.add_boundary_condition(FixedSupport("clamp", edges))
    load = LoadCase("pressure")
    for eid in model.mesh.elements:
        load.add_pressure_load(eid, 50000.)
    model.apply_boundary_conditions()
    result = fs.solve_static_nonlinear(model, load, num_steps=4, max_iterations=25,
                                      kinematics="von_karman")
    assert result.converged
    assert result.load_factor == pytest.approx(1.)
    centre = model.mesh.dof_manager.get_node_dofs(13)[2]
    deflection = abs(result.displacements[centre])
    print(f"n4 VK deflection={deflection:.17g}m")
    MEASUREMENTS["n4_elastic_50kPa_deflection_m"] = float(deflection)
    MEASUREMENTS["n4_displacement_sha256"] = hashlib.sha256(np.asarray(result.displacements).tobytes()).hexdigest()
    # Broad coarse-mesh check around the independent 11.79 mm reference;
    # this is not a mesh-convergence/qualification tolerance.
    assert .010 < deflection < .016
    assert max(np.linalg.norm(s["station_generalized_resultant"][:,:3])
               for s in result.element_states.values()) > 1.
    from anyfesolver.matrix_assembly import assemble_load_vector
    internal = np.zeros_like(result.displacements)
    for eid, element in model.mesh.elements.items():
        dofs = element.get_dof_mapping(model.mesh)
        internal[dofs] += element.compute_nonlinear_response(model.mesh,
            model.get_material("steel"), result.displacements[dofs],
            result.element_states[eid], 5, False, von_karman=True)[0]
    external, _ = assemble_load_vector(model, load)
    fixed = [dof for nid in edges for dof in model.mesh.dof_manager.get_node_dofs(nid)]
    free = np.setdiff1d(np.arange(len(internal)), fixed)
    relative_residual = np.linalg.norm((internal-external)[free])/np.linalg.norm(external[free])
    MEASUREMENTS["n4_free_relative_residual"] = float(relative_residual)
    assert relative_residual < 1e-6


def test_candidate_state_provenance_integrity_and_spectral_refusal():
    from anyfesolver.e4_pl_s3_v2d_state import V2DStateError, seal_v2d_state
    from anyfesolver.element_capabilities import ElementCapabilityError
    from anyfesolver.current_state_tangent import _s3_v2d_committed_current_tangent_components
    model, element = _model()
    material = model.get_material("steel")
    u = np.zeros(18); u[8] = .02
    _, _, state = _response(model, element, u)
    before = canonical_json_bytes(state)
    for mutation in ("missing", "unknown", "tampered"):
        bad = copy.deepcopy(state)
        if mutation == "missing":
            bad.pop("experimental_von_karman_law")
            bad = seal_v2d_state(bad)
        elif mutation == "unknown":
            bad["experimental_von_karman_law"] = "other"
            bad = seal_v2d_state(bad)
        else:
            bad["station_generalized_strain"][0,0] += 1.
        with pytest.raises(V2DStateError):
            _response(model, element, u, bad)
    with pytest.raises(V2DStateError, match="cannot change kinematics"):
        element.compute_nonlinear_response(model.mesh, material, u, state)
    with pytest.raises(ElementCapabilityError, match="eigen decomposition is unavailable"):
        _s3_v2d_committed_current_tangent_components(element, model, material, u, state)
    assert canonical_json_bytes(state) == before


def test_virgin_accepted_state_recovery_and_non_vk_isolation():
    from anyfesolver.corotational import corotational_element_response
    model, element = _model()
    model.apply_boundary_conditions()
    material = model.get_material("steel")
    virgin = element.init_model_bound_nonlinear_state(model.mesh, material, 5)
    before = canonical_json_bytes(virgin)
    u = np.zeros(18); u[8] = .02; u[6] = .001
    linear_k = element.compute_stiffness_matrix(model.mesh, material).copy()
    local = element.compute_nonlinear_response(model.mesh, material, u, virgin)
    cr = corotational_element_response(model, 1, element, u, True,
        committed_state=virgin, tangent_mode="consistent")
    accepted = element.seal_solver_integrated_nonlinear_state(
        model.mesh, material, virgin, 5, u, kinematics="von_karman")
    candidate = _response(model, element, u, virgin)[2]
    assert canonical_json_bytes(accepted) == canonical_json_bytes(candidate)
    np.testing.assert_array_equal(linear_k, element.compute_stiffness_matrix(model.mesh, material))
    local_after = element.compute_nonlinear_response(model.mesh, material, u, virgin)
    cr_after = corotational_element_response(model, 1, element, u, True,
        committed_state=virgin, tangent_mode="consistent")
    for original, after in ((local, local_after), (cr, cr_after)):
        np.testing.assert_array_equal(original[0], after[0])
        np.testing.assert_array_equal(original[1], after[1])
        assert canonical_json_bytes(original[2]) == canonical_json_bytes(after[2])
    assert canonical_json_bytes(virgin) == before


def test_cancelled_trial_discards_candidate_without_mutating_history():
    from anyfesolver.nonlinear_state import NonlinearStateStore, discard_active_state_candidate
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system_unchecked
    model, element = _model(plastic=True)
    model.apply_boundary_conditions()
    virgin = element.init_model_bound_nonlinear_state(model.mesh, model.get_material("steel"), 5)
    before = canonical_json_bytes(virgin)
    store = NonlinearStateStore.pack((), {1:virgin})
    u = np.zeros(18); u[6] = .008; u[8] = .02
    _assemble_nonlinear_system_unchecked(model, u, store, 5, require_full_coordinates=True)
    assert store.has_active_trial
    discard_active_state_candidate(store)
    assert not store.has_active_trial
    assert canonical_json_bytes(store[1]) == before
    assert canonical_json_bytes(virgin) == before


@pytest.mark.parametrize("von_karman", [False, True])
def test_deleted_frozen_history_operator_preserves_explicit_mode(von_karman):
    model, element = _model(plastic=True)
    material = model.get_material("steel")
    u = np.zeros(18); u[6] = .008; u[8] = .02
    _, _, accepted = element.compute_nonlinear_response(
        model.mesh, material, u, None, 5, True, von_karman=von_karman)
    deleted = element.seal_noncurrent_deleted_state(
        model.mesh, material, u, accepted, 5, deletion_step_index=1,
        deletion_load_factor=.25, residual_stiffness_fraction=.1,
        trigger_name="max_equivalent_plastic_strain")
    core = element.restore_noncurrent_deleted_state_for_internal_use(
        model.mesh, material, deleted, 5)
    before = canonical_json_bytes(core)
    u[6] += .0002; u[8] += .001
    expected = element.compute_nonlinear_response(
        model.mesh, material, u, core, 5, True, von_karman=von_karman)
    actual = element.compute_noncurrent_deleted_residual_operator(
        model.mesh, material, u, core, 5, tangent=True, von_karman=von_karman)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert canonical_json_bytes(core) == before


def test_second_production_deleted_assembly_keeps_frozen_vk_history():
    from anyfesolver.nonlinear_performance import _optimized_assemble_nonlinear_system
    from anyfesolver.nonlinear_state import NonlinearStateStore, discard_active_state_candidate
    model, element = _model(plastic=True)
    model.apply_boundary_conditions()
    material = model.get_material("steel")
    u = np.zeros(18); u[6] = .008; u[8] = .02
    _, _, accepted = _response(model, element, u)
    deleted = element.seal_noncurrent_deleted_state(
        model.mesh, material, u, accepted, 5, deletion_step_index=1,
        deletion_load_factor=.25, residual_stiffness_fraction=.1,
        trigger_name="max_equivalent_plastic_strain")
    core = element.restore_noncurrent_deleted_state_for_internal_use(model.mesh, material, deleted, 5)
    before = canonical_json_bytes(core)
    store = NonlinearStateStore.pack((), {1:core})
    for increment in (.0001, .0003):
        trial_u = u.copy(); trial_u[6] += increment; trial_u[8] += 2*increment
        expected = _response(model, element, trial_u, core)
        actual = _optimized_assemble_nonlinear_system(model, trial_u, store, 5,
            tangent=True, deleted_element_ids=(1,), residual_stiffness_fraction=.1,
            kinematics="von_karman")
        np.testing.assert_array_equal(actual[0], .1*expected[0])
        np.testing.assert_allclose(actual[1].toarray(), .1*expected[1], rtol=1e-15, atol=1e-10)
        discard_active_state_candidate(store)
        assert canonical_json_bytes(store[1]) == before
    assert canonical_json_bytes(core) == before


def test_public_cancellation_after_accepted_step_preserves_restart_input():
    import anyfesolver as fs
    from anyfesolver.boundary import FixedSupport, LoadCase
    from anyfesolver.control import CancellationToken, SolveCancelled
    model, element = _model()
    model.add_boundary_condition(FixedSupport("root", [1,3]))
    model.apply_boundary_conditions()
    load = LoadCase("tip"); load.add_nodal_load(2, [0.,0.,100.,0.,0.,0.])
    virgin = element.init_model_bound_nonlinear_state(model.mesh, model.get_material("steel"), 5)
    states = {1:virgin}; before = canonical_json_bytes(virgin)
    token = CancellationToken()
    accepted = []
    def progress(event):
        if event["type"] == "nonlinear_static_step":
            accepted.append(float(event["load_factor"]))
            token.cancel("R1 regression after accepted step")
    with pytest.raises(SolveCancelled, match="R1 regression"):
        fs.solve_static_nonlinear(model, load, num_steps=3,
            kinematics="von_karman", initial_element_states=states,
            cancellation_token=token, progress_callback=progress)
    assert len(accepted) == 1 and 0 < accepted[0] < 1
    assert canonical_json_bytes(states[1]) == before
    restarted = fs.solve_static_nonlinear(model, load, num_steps=3,
        kinematics="von_karman", initial_element_states=states)
    assert restarted.converged and restarted.load_factor == pytest.approx(1.)
    assert canonical_json_bytes(states[1]) == before


def test_yielded_uniform_stretch_and_bending_physical_tangent_psd():
    model, element = _model(plastic=True)
    material = model.get_material("steel")
    geometry = element._geometry(model.mesh)
    transform = geometry["local_from_external"]
    xy = geometry["local_coordinates"]
    local = np.zeros((3,6))
    local[:,:2] = .004*xy[:,:2]
    local[:,2] = .002*xy[:,0]**2+.001*xy[:,1]**2
    local[:,3] = .002*xy[:,1]
    local[:,4] = -.004*xy[:,0]
    u = transform.T @ local.ravel()
    _, tangent, state = _response(model, element, u)
    assert np.min(state["alpha"]) > 0
    # Exclude numerical PL stabilization so it cannot conceal an indefinite
    # physical tangent. Remove the six reference rigid-motion directions by
    # orthogonal projection; also check the unprojected matrix at roundoff.
    physical = transform @ (tangent-element.compute_stiffness_components(model.mesh, material)["pl"]) @ transform.T
    rigid = np.zeros((18,6))
    for i, position in enumerate(xy):
        position3 = np.array([position[0], position[1], 0.])
        rigid[6*i:6*i+3,:3] = np.eye(3)
        for axis in range(3):
            rotation = np.eye(3)[axis]
            rigid[6*i:6*i+3,3+axis] = np.cross(rotation, position3)
            rigid[6*i+3:6*i+6,3+axis] = rotation
    basis = np.linalg.qr(rigid, mode="complete")[0][:,6:]
    reduced = basis.T @ physical @ basis
    eigenvalues = np.linalg.eigvalsh(reduced)
    scale = np.linalg.norm(physical, 2)
    roundoff = 256*np.finfo(float).eps*scale
    assert np.linalg.eigvalsh(physical)[0] >= -roundoff
    assert eigenvalues[0] >= -roundoff
    MEASUREMENTS["yielded_physical_psd"] = {
        "minimum_projected_eigenvalue":float(eigenvalues[0]),
        "minimum_full_eigenvalue":float(np.linalg.eigvalsh(physical)[0]),
        "roundoff_bound":float(roundoff), "minimum_alpha":float(np.min(state["alpha"]))}
