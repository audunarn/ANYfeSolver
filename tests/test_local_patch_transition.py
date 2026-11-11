"""Local-patch (quad+tri) detail-mesh transition tests.

The local patch style subdivides only the cells inside detail windows and
closes the fine/coarse interface with conforming templates, so the rest of the
structure keeps its base mesh.  These tests pin mesh conformity (no hanging
nodes), locality (uniform far field), element quality and the graded fallback.
"""

import math
import multiprocessing
import os
import queue
import time
import traceback
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from anyfesolver.runtime import (
    LightweightFEMConfig,
    build_generated_geometry,
    run_production_fem,
)

FLAT = {
    "geometry": "flat panel",
    "length_m": 4.0,
    "width_m": 3.0,
    "thickness_m": 0.012,
    "has_stiffener": True,
    "has_girder": True,
    "stiffener_spacing_m": 0.75,
    "girder_spacing_m": 2.0,
}
CYLINDER = {
    "geometry": "cylinder",
    "radius_m": 2.0,
    "length_m": 8.0,
    "thickness_m": 0.012,
    "has_stiffener": True,
    "has_girder": True,
    "stiffener_spacing_m": 0.75,
    "girder_spacing_m": 3.5,
}

_MESH_STYLE_WORKER_WALL_SECONDS = 360.0
_MESH_STYLE_CLEANUP_SECONDS = 15.0
_MESH_STYLE_WAVE_ENV = "ANYSOLVER_BURNIN_S3_Q4_STYLE_WAVE"
_MESH_STYLE_RESULT_DEADLINE_ENV = (
    "ANYSOLVER_BURNIN_S3_Q4_STYLE_RESULT_DEADLINE"
)
_MESH_STYLE_CLEANUP_DEADLINE_ENV = (
    "ANYSOLVER_BURNIN_S3_Q4_STYLE_CLEANUP_DEADLINE"
)
_MESH_STYLE_RESULT_KEYS = frozenset({"lf", "uz", "vm_p50"})
_NUMERICAL_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def _mesh_style_worker_entry(
    index: int,
    worker: Callable[[tuple[str, dict[str, Any]]], tuple[str, dict[str, float]]],
    case: tuple[str, dict[str, Any]],
    result_queue: Any,
) -> None:
    """Run one independent mesh style and return only canonical scalars."""

    try:
        inherited_policy = {
            name: os.environ.get(name) for name in _NUMERICAL_THREAD_ENVIRONMENT
        }
        if set(inherited_policy.values()) != {"1"}:
            raise RuntimeError(
                f"mesh-style worker did not inherit the one-thread policy: {inherited_policy}"
            )
        result_queue.put((index, True, worker(case)))
    except BaseException:  # pragma: no cover - exercised by real child failures
        result_queue.put((index, False, traceback.format_exc()))


def _validated_mesh_style_outcome(
    payload: Any,
    expected_label: str,
) -> dict[str, float]:
    if type(payload) is not tuple or len(payload) != 2:
        pytest.fail(f"mesh-style {expected_label!r} returned a malformed tuple")
    label, values = payload
    if type(label) is not str or label != expected_label:
        pytest.fail(f"mesh-style worker order mismatch: {label!r} != {expected_label!r}")
    if type(values) is not dict or set(values) != _MESH_STYLE_RESULT_KEYS:
        pytest.fail(f"mesh-style {expected_label!r} returned malformed value keys")
    validated: dict[str, float] = {}
    for key in sorted(_MESH_STYLE_RESULT_KEYS):
        value = values[key]
        if type(value) is not float or not math.isfinite(value):
            pytest.fail(f"mesh-style {expected_label!r} returned invalid {key}: {value!r}")
        validated[key] = value
    return validated


def _parallel_mesh_style_deadlines() -> tuple[float, float] | None:
    if os.environ.get(_MESH_STYLE_WAVE_ENV) != "1":
        return None
    raw_result = os.environ.get(_MESH_STYLE_RESULT_DEADLINE_ENV)
    raw_cleanup = os.environ.get(_MESH_STYLE_CLEANUP_DEADLINE_ENV)
    if raw_result is None or raw_cleanup is None:
        pytest.fail("mesh-style wave opt-in requires both absolute deadlines")
    try:
        supplied_result = float(raw_result)
        supplied_cleanup = float(raw_cleanup)
    except ValueError:
        pytest.fail("mesh-style wave deadlines must be finite monotonic seconds")
    if not math.isfinite(supplied_result) or not math.isfinite(supplied_cleanup):
        pytest.fail("mesh-style wave deadlines must be finite monotonic seconds")
    if supplied_cleanup <= supplied_result:
        pytest.fail("mesh-style cleanup deadline must follow its result deadline")
    if supplied_cleanup - supplied_result > _MESH_STYLE_CLEANUP_SECONDS:
        pytest.fail("mesh-style cleanup authority exceeds fifteen seconds")
    started = time.monotonic()
    result_deadline = min(
        started + _MESH_STYLE_WORKER_WALL_SECONDS,
        supplied_result,
    )
    if result_deadline <= started:
        pytest.fail("mesh-style result deadline was already exhausted")
    return result_deadline, supplied_cleanup


def _run_mesh_styles(
    worker: Callable[[tuple[str, dict[str, Any]]], tuple[str, dict[str, float]]],
    cases: Sequence[tuple[str, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Run independent style calculations concurrently, with ordered results.

    ``spawn`` gives every style an isolated runtime module and solver state.  A
    common deadline is deliberately shorter than the enclosing 520-second
    functional-shard cutoff, leaving that shard time to serialize its normal
    evidence.  There is no retry: every launched process contributes exactly
    one terminal record or the test fails.
    """

    if not cases:
        return {}
    deadlines = _parallel_mesh_style_deadlines()
    if deadlines is None:
        ordered: dict[str, dict[str, float]] = {}
        for expected_label, overrides in cases:
            ordered[expected_label] = _validated_mesh_style_outcome(
                worker((expected_label, overrides)),
                expected_label,
            )
        return ordered

    labels = tuple(case[0] for case in cases)
    if (
        len(cases) != 3
        or any(type(label) is not str or not label for label in labels)
        or len(set(labels)) != 3
    ):
        pytest.fail("mesh-style wave requires exactly three unique ordered labels")

    result_deadline, supplied_cleanup_deadline = deadlines
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    old_environment = {name: os.environ.get(name) for name in _NUMERICAL_THREAD_ENVIRONMENT}
    processes = []
    messages: dict[int, tuple[bool, Any]] = {}
    collection_complete = False
    pending_error: BaseException | None = None
    pending_traceback = None
    try:
        try:
            for name in _NUMERICAL_THREAD_ENVIRONMENT:
                os.environ[name] = "1"
            for index, case in enumerate(cases):
                process = context.Process(
                    target=_mesh_style_worker_entry,
                    args=(index, worker, case, result_queue),
                    name=f"mesh-style-{index}-{case[0]}",
                )
                process.start()
                processes.append(process)
        finally:
            for name, value in old_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        while len(messages) < len(processes):
            remaining = result_deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                message = result_queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    break
                continue
            if type(message) is not tuple or len(message) != 3:
                pytest.fail("mesh-style worker emitted a malformed result record")
            index, passed, payload = message
            if type(index) is not int or not 0 <= index < len(processes):
                pytest.fail(f"mesh-style worker emitted an invalid index: {index!r}")
            if type(passed) is not bool:
                pytest.fail(f"mesh-style worker {index} emitted a non-boolean status")
            if index in messages:
                pytest.fail(f"mesh-style worker {index} emitted duplicate results")
            if passed:
                _validated_mesh_style_outcome(payload, cases[index][0])
            elif type(payload) is not str or not payload:
                pytest.fail(f"mesh-style worker {index} emitted a malformed failure")
            messages[index] = (passed, payload)
        collection_complete = len(messages) == len(processes)
    except BaseException as exc:
        pending_error = exc
        pending_traceback = exc.__traceback__
    finally:
        cleanup_started = time.monotonic()
        cleanup_hard_deadline = min(
            supplied_cleanup_deadline,
            cleanup_started + _MESH_STYLE_CLEANUP_SECONDS,
        )
        cleanup_soft_deadline = min(cleanup_hard_deadline, cleanup_started + 10.0)
        if not collection_complete:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        while any(process.is_alive() for process in processes):
            remaining = cleanup_soft_deadline - time.monotonic()
            if remaining <= 0.0:
                break
            for process in processes:
                process.join(timeout=min(0.1, max(0.0, remaining)))
        for process in processes:
            if process.is_alive():
                process.kill()
        while any(process.is_alive() for process in processes):
            remaining = cleanup_hard_deadline - time.monotonic()
            if remaining <= 0.0:
                break
            for process in processes:
                process.join(timeout=min(0.1, max(0.0, remaining)))
        for process in processes:
            if process.is_alive():
                process.terminate()
        result_queue.cancel_join_thread()
        result_queue.close()

    if any(process.is_alive() for process in processes):
        # Only the burn-in gate can opt into this path.  Immediate exit lets
        # its enclosing Windows Job close the otherwise unproven child tree.
        os._exit(125)
    exit_codes = tuple(process.exitcode for process in processes)
    for process in processes:
        process.close()
    if pending_error is not None:
        raise pending_error.with_traceback(pending_traceback)

    if len(messages) != len(processes):
        missing = [index for index in range(len(processes)) if index not in messages]
        pytest.fail(f"mesh-style workers did not return before the common deadline: {missing}")

    ordered: dict[str, dict[str, float]] = {}
    failures = []
    for index, (expected_label, _overrides) in enumerate(cases):
        passed, payload = messages[index]
        if exit_codes[index] != 0:
            failures.append(
                (expected_label, f"worker exit code was {exit_codes[index]!r}")
            )
            continue
        if not passed:
            failures.append((expected_label, payload))
            continue
        ordered[expected_label] = _validated_mesh_style_outcome(
            payload,
            expected_label,
        )
    if failures:
        pytest.fail("mesh-style worker failure(s):\n" + "\n".join(f"{label}:\n{error}" for label, error in failures))
    return ordered


def _impact_config(**overrides) -> LightweightFEMConfig:
    values = dict(
        mesh_fidelity="coarse",
        collision_enabled=True,
        collision_adaptive_mesh_enabled=True,
        collision_adaptive_fine_size_m=0.05,
        collision_adaptive_extent_m=0.4,
        collision_radius_m=0.15,
        collision_start_x_m=2.0,
        collision_start_y_m=1.5,
        collision_start_z_m=0.5,
        collision_vector_x=0.0,
        collision_vector_y=0.0,
        collision_vector_z=-1.0,
        detail_transition_style="local patch (quad+tri)",
    )
    values.update(overrides)
    return LightweightFEMConfig(**values)


def _skin_shells(generated: dict) -> list[dict]:
    return [shell for shell in generated["shells"] if "role" not in shell]


def _edge_counts(shells: list[dict]) -> Counter:
    edges: Counter = Counter()
    for shell in shells:
        ids = [int(i) for i in shell["node_ids"]]
        for k in range(len(ids)):
            edges[tuple(sorted((ids[k], ids[(k + 1) % len(ids)])))] += 1
    return edges


def _assert_conforming(generated: dict, outline_check) -> None:
    """Interior skin edges shared by exactly 2 elements; count-1 edges on the outline."""
    nodes = {int(n["id"]): np.asarray([float(c) for c in n["coords"]]) for n in generated["nodes"]}
    edges = _edge_counts(_skin_shells(generated))
    assert not [e for e, c in edges.items() if c > 2], "non-manifold skin edges"
    hanging = [
        e for e, c in edges.items() if c == 1 and not outline_check(nodes[e[0]], nodes[e[1]])
    ]
    assert not hanging, f"{len(hanging)} hanging (non-conforming) edges"


def test_flat_local_patch_is_conforming_and_local() -> None:
    generated = build_generated_geometry(FLAT, _impact_config())

    def on_outline(p, q):
        for axis, span in ((0, 4.0), (1, 3.0)):
            for value in (0.0, span):
                if abs(p[axis] - value) < 1e-9 and abs(q[axis] - value) < 1e-9:
                    return True
        return False

    _assert_conforming(generated, on_outline)
    adaptive = generated["adaptive_mesh"]
    assert adaptive["enabled"] is True
    assert adaptive["transition"] == "local patch (quad+tri)"
    assert adaptive["refined_cells"] > 0
    assert adaptive["tri_count"] > 0
    assert adaptive["beam_splits"] > 0
    # Locality: triangles a small minority (transition ring only).
    skin = _skin_shells(generated)
    tris = sum(1 for s in skin if len(s["node_ids"]) == 3)
    assert tris / len(skin) < 0.2
    # Locality: cells far from the window (panel corners) keep the base size.
    # (The 2:1 balance rings may legitimately reach nearer edges on a coarse
    # base grid, but the extremes must stay untouched.)
    xs = sorted({round(float(n["coords"][0]), 6) for n in generated["nodes"]
                 if abs(float(n["coords"][1])) < 1e-9 and abs(float(n["coords"][2])) < 1e-9})
    dx = np.diff(xs)
    assert dx[0] == pytest.approx(dx.max()), "first corner cell should stay at base size"
    assert dx[-1] == pytest.approx(dx.max()), "last corner cell should stay at base size"
    assert dx.max() > 3.0 * dx.min(), "refined zone should be much finer than the base"


def test_cylinder_local_patch_quality_beats_graded() -> None:
    base = dict(
        mesh_fidelity="medium",
        collision_enabled=True,
        collision_adaptive_mesh_enabled=True,
        collision_adaptive_fine_size_m=0.05,
        collision_adaptive_extent_m=0.75,
        collision_radius_m=0.25,
        collision_start_x_m=-3.01,
        collision_start_y_m=0.0,
        collision_start_z_m=4.0,
        collision_vector_x=1.0,
        collision_vector_y=0.0,
        collision_vector_z=0.0,
    )
    patched = build_generated_geometry(
        CYLINDER, LightweightFEMConfig(**base, detail_transition_style="local patch (quad+tri)")
    )

    def on_end_ring(p, q):
        length = 8.0
        return (abs(p[2]) < 1e-8 and abs(q[2]) < 1e-8) or (
            abs(p[2] - length) < 1e-8 and abs(q[2] - length) < 1e-8
        )

    _assert_conforming(patched, on_end_ring)

    def worst_aspect(generated: dict) -> float:
        nodes = {int(n["id"]): np.asarray([float(c) for c in n["coords"]]) for n in generated["nodes"]}
        worst = 1.0
        for shell in _skin_shells(generated):
            ids = [int(i) for i in shell["node_ids"]]
            if len(ids) != 4:
                continue
            pts = [nodes[i] for i in ids]
            lengths = [np.linalg.norm(pts[(k + 1) % 4] - pts[k]) for k in range(4)]
            if min(lengths) > 0.0:
                worst = max(worst, max(lengths) / min(lengths))
        return worst

    graded = build_generated_geometry(
        CYLINDER, LightweightFEMConfig(**base, detail_transition_style="graded grid")
    )
    assert worst_aspect(patched) < 4.0
    assert worst_aspect(patched) < worst_aspect(graded)

    nodes = {
        int(node["id"]): np.asarray(node["coords"], dtype=float)
        for node in patched["nodes"]
    }
    triangles = [shell for shell in _skin_shells(patched) if len(shell["node_ids"]) == 3]
    assert triangles
    for shell in triangles:
        assert shell["formulation"] == "e4-pl-s3-v2d"
        assert shell["formulation_id"] == "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1"
        assert shell["owner_normal_authority"] == "PHYSICAL_SURFACE_OWNER_NORMAL_V2D_V1"
        points = [nodes[int(node_id)] for node_id in shell["node_ids"]]
        edge_lengths = [
            float(np.linalg.norm(points[(index + 1) % 3] - points[index]))
            for index in range(3)
        ]
        doubled_area = float(
            np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
        )
        angles = []
        for index in range(3):
            first = points[(index - 1) % 3] - points[index]
            second = points[(index + 1) % 3] - points[index]
            cosine = float(first @ second) / float(
                np.linalg.norm(first) * np.linalg.norm(second)
            )
            angles.append(math.degrees(math.acos(min(1.0, max(-1.0, cosine)))))
        area = 0.5 * doubled_area
        shape_metric = 4.0 * math.sqrt(3.0) * area / sum(
            length * length for length in edge_lengths
        )
        assert min(angles) >= 30.0
        assert max(angles) <= 150.0
        assert max(edge_lengths) / min(edge_lengths) <= 4.0
        assert shape_metric >= 0.60


def _cylinder_skin_normal_counts(generated: dict) -> tuple[int, int]:
    """(outward, inward) radial-normal counts over cylinder skin shells."""
    nodes = {int(n["id"]): np.asarray([float(c) for c in n["coords"]]) for n in generated["nodes"]}
    outward = inward = 0
    for shell in _skin_shells(generated):
        pts = [nodes[int(i)] for i in shell["node_ids"]]
        centroid = np.mean(pts, axis=0)
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        radial = np.asarray([centroid[0], centroid[1], 0.0])
        if np.linalg.norm(normal) < 1e-14 or np.linalg.norm(radial) < 1e-9:
            continue
        if float(np.dot(normal, radial)) > 0.0:
            outward += 1
        else:
            inward += 1
    return outward, inward


def test_cylinder_local_patch_window_on_seam_is_conforming() -> None:
    """A refinement window crossing the periodic seam (theta=0) must not slit
    the cylinder open: no duplicate skin nodes and no hanging edges."""
    generated = build_generated_geometry(
        CYLINDER,
        LightweightFEMConfig(
            mesh_fidelity="coarse",
            point_refinement_enabled=True,
            point_refinement_x_m=4.0,   # u = axial
            point_refinement_y_m=0.0,   # v = arc: window straddles the seam
            point_refinement_fine_size_m=0.05,
            point_refinement_extent_m=0.4,
            detail_transition_style="local patch (quad+tri)",
        ),
    )
    assert generated["adaptive_mesh"]["transition"] == "local patch (quad+tri)"
    assert generated["adaptive_mesh"]["refined_cells"] > 0

    def on_end_ring(p, q):
        length = CYLINDER["length_m"]
        return (abs(p[2]) < 1e-8 and abs(q[2]) < 1e-8) or (
            abs(p[2] - length) < 1e-8 and abs(q[2] - length) < 1e-8
        )

    _assert_conforming(generated, on_end_ring)
    # No duplicated skin nodes along the seam (a slit doubles seam nodes).
    nodes = {int(n["id"]): tuple(round(float(c), 7) for c in n["coords"]) for n in generated["nodes"]}
    skin_ids = {int(i) for shell in _skin_shells(generated) for i in shell["node_ids"]}
    coords = [nodes[i] for i in skin_ids]
    assert len(coords) == len(set(coords)), "duplicate skin nodes (seam slit)"


def test_cylinder_local_patch_keeps_normal_winding() -> None:
    """Emitted refinement elements must keep the base mesh's surface normal
    orientation, or pressure loads on the patch act in the wrong direction."""
    generated = build_generated_geometry(
        CYLINDER,
        LightweightFEMConfig(
            mesh_fidelity="coarse",
            point_refinement_enabled=True,
            point_refinement_x_m=4.0,
            point_refinement_y_m=1.0,
            point_refinement_fine_size_m=0.05,
            point_refinement_extent_m=0.4,
            detail_transition_style="local patch (quad+tri)",
        ),
    )
    assert generated["adaptive_mesh"]["refined_cells"] > 0
    outward, inward = _cylinder_skin_normal_counts(generated)
    assert outward > 0
    assert min(outward, inward) == 0, (
        f"mixed skin normals after refinement: {outward} outward vs {inward} inward"
    )


def test_local_patch_falls_back_for_shell_web_members() -> None:
    generated = build_generated_geometry(
        FLAT, _impact_config(member_model="webs as shells, flanges as beams")
    )
    adaptive = generated["adaptive_mesh"]
    # Graded refinement still applies; the patch transition does not.
    assert adaptive.get("transition") != "local patch (quad+tri)"
    assert adaptive["enabled"] is True


def test_local_patch_model_solves_with_collision() -> None:
    # This is the bounded end-to-end collision smoke.  The larger FLAT fixture
    # above pins dense-patch conformity, locality, and quality.  A coarser
    # adaptive patch and short positive approach retain mixed Q4/S3 topology,
    # member splitting, automatic time control, contact, and damage without
    # spending hundreds of transient steps before the sphere reaches the panel.
    config = _impact_config(
        collision_adaptive_fine_size_m=0.35,
        collision_start_z_m=0.18,
        collision_result_interval_s=0.005,
        pressure_pa=100000.0,
        boundary_condition="clamped",
        collision_mass_kg=100.0,
        collision_speed_mps=3.0,
    )
    generated = build_generated_geometry(FLAT, config)
    skin = _skin_shells(generated)
    adaptive = generated["adaptive_mesh"]
    assert adaptive["enabled"] is True
    assert adaptive["transition"] == "local patch (quad+tri)"
    assert adaptive["max_level"] >= 1
    assert adaptive["fine_element_size_m"] < adaptive["coarse_element_size_m"]
    assert adaptive["refined_cells"] > 0
    assert adaptive["tri_count"] > 0
    assert adaptive["beam_splits"] > 0
    assert any(len(shell["node_ids"]) == 3 for shell in skin)
    assert any(len(shell["node_ids"]) == 4 for shell in skin)

    result = run_production_fem(
        FLAT,
        config,
    )
    summary = result.prestress_summary
    assert result.status == "ok"
    assert math.isfinite(result.displacement_max_m)
    assert result.displacement_max_m > 0.0
    assert summary["collision_status"] == "completed"
    assert summary["collision_time_mode"] == "auto"
    assert math.isfinite(float(summary["collision_peak_contact_force_n"]))
    assert float(summary["collision_peak_contact_force_n"]) > 0.0
    assert math.isfinite(float(summary["collision_contact_duration_s"]))
    assert float(summary["collision_contact_duration_s"]) > 0.0
    assert float(summary["collision_saved_steps"]) > 1.0
    assert float(summary["collision_resolved_dt_s"]) > 0.0
    assert float(summary["collision_resolved_total_time_s"]) > float(
        summary["collision_estimated_arrival_time_s"]
    )
    assert any("Local patch transition" in str(d) for d in result.diagnostics)
    assert any("collision transient: completed" in str(d) for d in result.diagnostics)


def test_point_refinement_uses_local_patch_when_selected() -> None:
    generated = build_generated_geometry(
        FLAT,
        LightweightFEMConfig(
            mesh_fidelity="coarse",
            point_refinement_enabled=True,
            point_refinement_x_m=1.0,
            point_refinement_y_m=1.0,
            point_refinement_fine_size_m=0.08,
            point_refinement_extent_m=0.3,
            detail_transition_style="local patch (quad+tri)",
        ),
    )
    adaptive = generated["adaptive_mesh"]
    assert adaptive["enabled"] is True
    assert adaptive["transition"] == "local patch (quad+tri)"
    assert any(s.get("source") == "selected_point" for s in adaptive["sources"])


def test_axial_cylinder_stress_is_mesh_style_invariant() -> None:
    """Release-blocker verification case: unstiffened cylinder r=1 m, L=10 m,
    t=30 mm under 50 MN balanced axial force.  Analytical membrane solution:
    sigma_vm = F/A = 265.3 MPa uniform, max |uz| = sigma L / (2E) = 6.32 mm.

    All mesh styles must reproduce it: an equal per-node end-force split used
    to turn the clustered graded end rings into a global bending moment
    (81-484 MPa), and true-surface node placement used to bulge the local
    patch proud of the chordal base facets (222-447 MPa stress dimple)."""
    from anyfesolver import runtime as fs

    captured = {}
    original = fs._backend_solve_linear

    def wrapper(model, load_case, **kwargs):
        displacements, info = original(model, load_case, **kwargs)
        captured["model"] = model
        captured["disp"] = displacements
        return displacements, info

    cylinder = {
        "geometry": "cylinder",
        "radius_m": 1.0,
        "length_m": 10.0,
        "thickness_m": 0.030,
        "has_stiffener": False,
        "has_girder": False,
    }
    refine = dict(
        point_refinement_enabled=True,
        point_refinement_x_m=5.0,
        point_refinement_y_m=5.694,
        point_refinement_fine_size_m=0.30,
        point_refinement_extent_m=0.30,
        point_refinement_growth_factor=1.35,
    )
    styles = {
        "uniform": {},
        "graded grid": dict(detail_transition_style="graded grid", **refine),
        "local patch (quad+tri)": dict(detail_transition_style="local patch (quad+tri)", **refine),
    }
    analytical_vm = 50.0e6 / (2.0 * math.pi * 1.0 * 0.030)
    analytical_uz = analytical_vm * 10.0 / (2.0 * 210.0e9)

    fs._backend_solve_linear = wrapper
    try:
        for label, overrides in styles.items():
            config = fs.LightweightFEMConfig(
                mesh_fidelity="coarse",
                mesh_size_m=1.0,
                boundary_condition="auto",
                include_end_lids=True,
                axial_force_n=50.0e6,
                pressure_pa=0.0,
                runtime_solver="static only",
                analysis_type="linear",
                num_buckling_modes=0,
                **overrides,
            )
            generated = fs.build_generated_geometry(cylinder, config)
            if label != "uniform":
                assert generated["adaptive_mesh"]["enabled"] is True
            if label == "local patch (quad+tri)":
                adaptive = generated["adaptive_mesh"]
                assert adaptive["transition"] == "local patch (quad+tri)"
                assert adaptive["refined_cells"] > 0
                assert adaptive["tri_count"] > 0
            result = fs.run_production_fem(cylinder, config)
            assert result.status == "ok", (label, result.status)
            assert result.buckling_factors == (), label
            model, disp = captured["model"], captured["disp"]
            stresses = fs._backend_compute_stresses(model, disp)
            vm_values = []
            for element_id, stress in stresses.items():
                element = model.mesh.elements.get(element_id)
                if element is None or not hasattr(element, "thickness"):
                    continue
                values = np.asarray(stress.get("von_mises", ()), dtype=float).reshape(-1)
                if values.size:
                    vm_values.append(float(np.max(values)))
            vm_values = np.asarray(vm_values)
            uz_peak = 0.0
            for node in model.mesh.nodes.values():
                dofs = np.asarray(node.dofs[:3], dtype=np.intp)
                if dofs.size == 3 and int(dofs.max()) < disp.size:
                    uz_peak = max(uz_peak, abs(float(disp[dofs[2]])))
            # Uniform membrane state within the end-lid disturbance band.
            assert vm_values.min() > 0.90 * analytical_vm, (label, vm_values.min())
            assert vm_values.max() < 1.15 * analytical_vm, (label, vm_values.max())
            assert uz_peak == pytest.approx(analytical_uz, rel=0.10), (label, uz_peak)
    finally:
        fs._backend_solve_linear = original


def test_local_patch_buckling_has_no_spurious_flat_facet_modes() -> None:
    """Release-blocker follow-up: subdividing a coarsely faceted cylinder
    chordally made the patch buckle as a FLAT PLATE at a fraction of the true
    shell load factor (spiky single-element modes, LF ~3 vs classical ~7).
    With the curvature-adequate base ring + blended node placement the first
    buckling factor must stay at shell level, above the classical value."""
    from anyfesolver import runtime as fs

    cylinder = {
        "geometry": "cylinder",
        "radius_m": 2.0,
        "length_m": 30.0,
        "thickness_m": 0.030,
        "has_stiffener": False,
        "has_girder": False,
    }
    config = fs.LightweightFEMConfig(
        mesh_fidelity="coarse",
        boundary_condition="auto",
        include_end_lids=True,
        axial_force_n=50.0e6,
        pressure_pa=0.0,
        num_buckling_modes=3,
        detail_transition_style="local patch (quad+tri)",
        point_refinement_enabled=True,
        point_refinement_x_m=10.0,
        point_refinement_y_m=5.0,
        point_refinement_extent_m=1.0,
        point_refinement_growth_factor=1.35,
    )
    result = fs.run_production_fem(cylinder, config)
    assert result.status == "ok"
    factors = tuple(float(value) for value in result.buckling_factors)
    assert factors, "buckling factors expected"
    # Classical axisymmetric axial buckling: sigma_cr = E t / (r sqrt(3(1-nu^2)))
    # = 1.91 GPa -> LF = 7.2 at 265 MPa membrane stress.  Flat-facet plate
    # modes sat at ~3; anything below ~6 signals the spurious softness.
    assert min(factors) > 6.0, factors
    # And the refined model must not report the coarse-mesh over-prediction
    # unchallenged either (coarse gave 33; adequate resolution sits below 20).
    assert min(factors) < 20.0, factors


def test_axial_flat_plate_stress_is_mesh_style_invariant() -> None:
    """Flat-plate counterpart of the cylinder verification case: the curved-
    shell fixes (tributary end loads, surface blend, curvature base floor)
    must leave flat panels exact.  Unstiffened plate L=4 m, W=3 m, t=12 mm
    under 10 MN balanced axial force: sigma = F/(W t) = 277.8 MPa uniform and
    the plate stays exactly planar for every mesh style."""
    from anyfesolver import runtime as fs

    captured = {}
    original = fs._backend_solve_linear

    def wrapper(model, load_case, **kwargs):
        displacements, info = original(model, load_case, **kwargs)
        captured["model"] = model
        captured["disp"] = displacements
        return displacements, info

    flat = {
        "geometry": "flat panel",
        "length_m": 4.0,
        "width_m": 3.0,
        "thickness_m": 0.012,
        "has_stiffener": False,
        "has_girder": False,
    }
    refine = dict(
        point_refinement_enabled=True,
        point_refinement_x_m=2.0,
        point_refinement_y_m=1.5,
        point_refinement_extent_m=0.5,
        point_refinement_growth_factor=1.35,
    )
    styles = {
        "uniform": {},
        "graded grid": dict(detail_transition_style="graded grid", **refine),
        "local patch (quad+tri)": dict(detail_transition_style="local patch (quad+tri)", **refine),
    }
    analytical_vm = 10.0e6 / (3.0 * 0.012)

    fs._backend_solve_linear = wrapper
    try:
        for label, overrides in styles.items():
            config = fs.LightweightFEMConfig(
                mesh_fidelity="coarse",
                boundary_condition="auto",
                axial_force_n=10.0e6,
                pressure_pa=0.0,
                runtime_solver="static only",
                analysis_type="linear",
                num_buckling_modes=0,
                **overrides,
            )
            generated = build_generated_geometry(flat, config)
            planarity = max(abs(float(node["coords"][2])) for node in generated["nodes"])
            assert planarity == 0.0, (label, "flat panel must stay exactly planar")

            result = fs.run_production_fem(flat, config)
            assert result.status == "ok", (label, result.status)
            assert result.buckling_factors == (), label
            model, disp = captured["model"], captured["disp"]
            stresses = fs._backend_compute_stresses(model, disp)
            vm_values = []
            for element_id, stress in stresses.items():
                element = model.mesh.elements.get(element_id)
                if element is None or not hasattr(element, "thickness"):
                    continue
                values = np.asarray(stress.get("von_mises", ()), dtype=float).reshape(-1)
                if values.size:
                    vm_values.append(float(np.max(values)))
            vm_values = np.asarray(vm_values)
            # Uniform membrane state: consistent tributary loading keeps the
            # field exact to solver precision for every mesh style.
            assert vm_values.min() == pytest.approx(analytical_vm, rel=1.0e-3), (label, vm_values.min())
            assert vm_values.max() == pytest.approx(analytical_vm, rel=1.0e-3), (label, vm_values.max())
    finally:
        fs._backend_solve_linear = original


def _stiffened_cylinder_style_outcome(
    case: tuple[str, dict[str, Any]],
) -> tuple[str, dict[str, float]]:
    """Construct and solve one unchanged stiffened-cylinder mesh style."""

    from anyfesolver import runtime as fs

    captured = {}
    original = fs._backend_solve_linear

    def wrapper(model, load_case, **kwargs):
        displacements, info = original(model, load_case, **kwargs)
        captured["model"] = model
        captured["disp"] = displacements
        return displacements, info

    cylinder = {
        "geometry": "cylinder",
        "radius_m": 1.0,
        "length_m": 10.0,
        "thickness_m": 0.030,
        "has_stiffener": True,
        "has_girder": True,
        "stiffener_spacing_m": 0.785,
        "girder_spacing_m": 3.0,
    }
    label, overrides = case
    fs._backend_solve_linear = wrapper
    try:
        config = fs.LightweightFEMConfig(
            mesh_fidelity="coarse",
            boundary_condition="auto",
            include_end_lids=True,
            axial_force_n=50.0e6,
            pressure_pa=0.0,
            num_buckling_modes=2,
            **overrides,
        )
        result = fs.run_production_fem(cylinder, config)
        assert result.status == "ok", (label, result.status)
        model, disp = captured["model"], captured["disp"]
        stresses = fs._backend_compute_stresses(model, disp)
        vm_values = []
        for element_id, stress in stresses.items():
            element = model.mesh.elements.get(element_id)
            if element is None or not hasattr(element, "thickness"):
                continue
            values = np.asarray(stress.get("von_mises", ()), dtype=float).reshape(-1)
            if values.size:
                vm_values.append(float(np.max(values)))
        uz_peak = 0.0
        for node in model.mesh.nodes.values():
            dofs = np.asarray(node.dofs[:3], dtype=np.intp)
            if dofs.size == 3 and int(dofs.max()) < disp.size:
                uz_peak = max(uz_peak, abs(float(disp[dofs[2]])))
        return label, {
            "vm_p50": float(np.percentile(np.asarray(vm_values), 50)),
            "uz": uz_peak,
            "lf": min((float(v) for v in result.buckling_factors), default=0.0),
        }
    finally:
        fs._backend_solve_linear = original


def test_stiffened_cylinder_keeps_mesh_style_invariance_with_beams() -> None:
    """Beams must not reintroduce the release-blocker artifacts: stiffener-
    forced ring breaks interact with the tributary end-load weighting, beam
    segments are split inside blended patches, and the curvature floor is
    raised to a stiffener-count multiple.  Statics and buckling must stay
    mesh-style invariant on the stiffened cylinder."""
    refine = dict(
        point_refinement_enabled=True,
        point_refinement_x_m=5.0,
        point_refinement_y_m=5.694,
        point_refinement_extent_m=1.0,
        point_refinement_growth_factor=1.35,
    )
    cases = (
        ("uniform", {}),
        ("graded grid", dict(detail_transition_style="graded grid", **refine)),
        (
            "local patch (quad+tri)",
            dict(detail_transition_style="local patch (quad+tri)", **refine),
        ),
    )
    outcomes = _run_mesh_styles(
        _stiffened_cylinder_style_outcome,
        cases,
    )

    baseline = outcomes["uniform"]
    for label in ("graded grid", "local patch (quad+tri)"):
        other = outcomes[label]
        # Membrane state invariance: median stress and axial shortening agree.
        assert other["vm_p50"] == pytest.approx(baseline["vm_p50"], rel=0.05), (label, outcomes)
        assert other["uz"] == pytest.approx(baseline["uz"], rel=0.03), (label, outcomes)
    # Buckling: refined styles agree with each other and stay at shell level
    # (spurious flat-facet plate modes sat at ~3 vs the correct ~21-27 band).
    graded_lf = outcomes["graded grid"]["lf"]
    patch_lf = outcomes["local patch (quad+tri)"]["lf"]
    assert patch_lf == pytest.approx(graded_lf, rel=0.25), outcomes
    assert patch_lf > 0.5 * baseline["lf"], outcomes
