"""Submodel transfer (docs/SUBMODEL_TRANSFER_CONTRACT.md): interpolation, driven trace,
interface forces and error allowance on small ANYfeSolver-only shell models.

Oracles are closed-form affine and rigid fields and hand-derived multiplier relations;
no result is compared with another run of the same code.
"""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import BoundaryCondition, LoadCase, solve_linear
from anyfesolver.analysis_session import AnalysisSession
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.elements import ShellElement, create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement
from anyfesolver.submodel_transfer import (
    DOF_NAMES,
    SubmodelTransferError,
    TransferPoint,
    apply_driven_trace,
    element_interpolation_weights,
    build_driven_trace,
    estimate_static_error_allowance,
    force_allowance,
    interface_forces,
    interpolate_nodal_field,
    interpolation_allowance,
    interface_resultant,
    rotation_rows,
    update_driven_trace_values,
)

E, NU, T = 210.0e9, 0.3, 0.01


def plate(n=4, *, triangles=False, membrane=True):
    """Plate [0,1]^2 in the xy-plane; node (i, j) at (i/n, j/n)."""

    model = FEModel("submodel-transfer")
    model.add_material("steel", E, NU, density=7850.0)
    grid = {}
    node = 1
    for j in range(n + 1):
        for i in range(n + 1):
            model.add_node(node, i / n, j / n, 0.0)
            grid[i, j] = node
            node += 1
    element = 1
    for j in range(n):
        for i in range(n):
            a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
            if triangles:
                for ids in ((a, b, c), (a, c, d)):
                    model.add_element(element, NativeParityE4PLS3V2DShellElement(
                        element, ids, "steel", thickness=T, reference_normal=(0.0, 0.0, 1.0)))
                    element += 1
            else:
                model.add_element(element, create_shell_element(element, [a, b, c, d], "steel", thickness=T))
                element += 1
    if membrane:
        model.add_boundary_condition(BoundaryCondition(
            name="membrane", node_ids=sorted(grid.values()), dof_constraints={"uz": 0.0, "rx": 0.0, "ry": 0.0}))
    return model, grid


def coords(model, node):
    return np.asarray(model.mesh.get_node(node).coords(), dtype=float)


def nodal_vector(model, field):
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in model.mesh.nodes:
        u[model.mesh.dof_manager.get_node_dofs(node_id)] = field(coords(model, node_id))
    return u


def affine(x, omega=2.0e-4):
    return np.array([1.0e-4 * x[0] + (2.5e-5 - omega) * x[1], (2.5e-5 + omega) * x[0] - 3.0e-5 * x[1],
                     0.0, 0.0, 0.0, omega])


A_RIGID = np.array([1.0e-3, -2.0e-3, 3.0e-3])
OM_RIGID = np.array([1.0e-3, -5.0e-4, 2.0e-3])


def rigid(x):
    return np.concatenate([A_RIGID + np.cross(OM_RIGID, x), OM_RIGID])


def node_values(model, u, node):
    return np.asarray(u, dtype=float)[model.mesh.dof_manager.get_node_dofs(node)]


# ----------------------------------------------------------------------
# interpolation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("triangles", [False, True])
@pytest.mark.parametrize("field", [affine, rigid])
def test_interpolation_reproduces_representable_fields(triangles, field):
    model, _grid = plate(2, triangles=triangles, membrane=False)
    u = nodal_vector(model, field)
    rng = np.random.default_rng(7)
    points = []
    for k, element_id in enumerate(sorted(model.mesh.elements)):
        if triangles:
            r, s = rng.uniform(0.05, 0.45, size=2)
            natural = (float(r), float(s))
        else:
            natural = tuple(float(v) for v in rng.uniform(-0.95, 0.95, size=2))
        element = model.mesh.elements[element_id]
        weights = element_interpolation_weights(element, natural)
        x_p = weights @ np.array([coords(model, n) for n in element.node_ids])
        # Off the surface by 1 cm: exact only through the rigid transport rule.
        offset = np.array([0.0, 0.0, 0.01]) if field is rigid else np.zeros(3)
        points.append(TransferPoint(1000 + k, element_id, natural, tuple(x_p + offset), tuple(weights)))
    values = interpolate_nodal_field(model, u, points)
    for point in points:
        expected = field(np.asarray(point.target_point))
        got = values[point.target_node].values
        assert np.max(np.abs(got - expected)) <= 1.0e-15 * max(np.max(np.abs(expected)), 1.0)


def test_interpolation_refuses_convention_mismatch_and_unadmitted_elements():
    model, _grid = plate(1, membrane=False)
    u = nodal_vector(model, rigid)
    wrong = TransferPoint(99, 1, (0.2, -0.3), (0.5, 0.5, 0.0), (0.25, 0.25, 0.25, 0.25))
    with pytest.raises(SubmodelTransferError, match="TRANSFER_CONVENTION_MISMATCH"):
        interpolate_nodal_field(model, u, [wrong])

    legacy = FEModel("legacy")
    legacy.add_material("steel", E, NU, density=7850.0)
    for node, xyz in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0)], start=1):
        legacy.add_node(node, *xyz)
    legacy.add_element(1, ShellElement(1, [1, 2, 3], "steel", thickness=T))
    with pytest.raises(SubmodelTransferError, match="TRANSFER_ELEMENT_NOT_ADMITTED"):
        interpolate_nodal_field(legacy, np.zeros(18), [TransferPoint(9, 1, (0.2, 0.2), (0.2, 0.2, 0.0))])


# ----------------------------------------------------------------------
# contract section 8.1: slave on the cut, one master on the cut, one interior
# ----------------------------------------------------------------------
def g_field(x):
    return np.array([1.0e-4 * x[0] + 2.0e-5 * x[1] + 1.0e-4, -3.0e-5 * x[1] + 5.0e-5 * x[0], 0.0, 0.0, 0.0, 1.5e-5])


def line_loaded(model, grid, n, q=(2.0e5, 3.0e4)):
    case = LoadCase("line")
    for j in range(n + 1):
        share = (0.5 if j in (0, n) else 1.0) / n
        case.add_nodal_load(grid[n, j], np.array([q[0] * share, q[1] * share, 0.0]))
    return case


def cut_nodes(grid, n):
    return sorted(grid[0, j] for j in range(n + 1))


def test_contract_mpc_example_trace_and_multipliers():
    n = 4
    model, grid = plate(n)
    case = line_loaded(model, grid, n)
    dm = model.mesh.dof_manager
    s, m1, m2 = grid[0, 4], grid[0, 2], grid[1, 4]
    sx, m1x, m2x = (dm.get_node_dofs(node)[0] for node in (s, m1, m2))
    model.add_constraint_equation(terms=((sx, 1.0), (m1x, -0.5), (m2x, -0.5)), rhs=0.0,
                                  source_id="internal", dependent_dof=sx)
    cut = cut_nodes(grid, n)
    values = {node: g_field(coords(model, node)) for node in cut}
    trace = build_driven_trace(model, values, policy="drive_all")
    # uz, rx, ry rows of cut nodes are implied by the membrane support: redundant.
    redundant = {(trace.rows[i].node, trace.rows[i].label) for i in trace.redundant}
    assert redundant == {(node, name) for node in cut for name in ("uz", "rx", "ry")}
    apply_driven_trace(model, trace)
    u, info = solve_linear(model, case)
    assert node_values(model, u, s)[0] == pytest.approx(values[s][0], abs=1e-18)
    assert node_values(model, u, m1)[0] == pytest.approx(values[m1][0], abs=1e-18)
    assert node_values(model, u, m2)[0] == pytest.approx(2 * values[s][0] - values[m1][0], abs=1e-18)

    forces = interface_forces(model, u, case, trace)
    from anyfesolver.matrix_assembly import assemble_load_vector
    with AnalysisSession(model) as session:
        k = session.stiffness_plan().matrix
    f = np.asarray(assemble_load_vector(model, case)[0])
    r = k @ u - f
    assert forces.node_forces[m1][0] == pytest.approx(r[m1x] - r[m2x], rel=1e-9)
    assert forces.node_forces[s][0] == pytest.approx(2 * r[m2x] + r[sx], rel=1e-9)
    total = sum(forces.node_forces[node][:2] for node in cut)
    assert np.max(np.abs(total + np.array([2.0e5, 3.0e4]))) < 1.0e-8 * 2.0e5
    assert forces.equilibrium_residual < 1.0e-8 * 2.0e5


# ----------------------------------------------------------------------
# contract section 8.2: eccentric coupling on the cut
# ----------------------------------------------------------------------
def coupled(model, grid, shell_node, ecc):
    beam = max(model.mesh.nodes) + 1
    model.add_node(beam, *(coords(model, shell_node) + ecc))
    element_id = max(model.mesh.elements) + 1
    model.add_element(element_id, InterpolatedBeamShellMPCElement(element_id, beam, [shell_node], np.array([1.0]), ecc))
    return beam


def transported(six, ecc):
    out = six.copy()
    out[:3] = out[:3] + np.cross(six[3:], ecc)
    return out


def test_coupled_beam_node_on_the_cut_is_redundant_not_prescribed():
    n = 4
    model, grid = plate(n)
    case = line_loaded(model, grid, n)
    ecc = np.array([0.0, 0.0, 0.05])
    shell = grid[0, 2]
    beam = coupled(model, grid, shell, ecc)
    values = {node: g_field(coords(model, node)) for node in cut_nodes(grid, n)}
    values[beam] = transported(values[shell], ecc)
    trace = build_driven_trace(model, values, policy="drive_all")
    assert {trace.rows[i].node for i in trace.redundant} >= {beam}
    assert all(trace.rows[i].node != beam for i in trace.kept)
    apply_driven_trace(model, trace)
    u, _ = solve_linear(model, case)
    assert np.max(np.abs(node_values(model, u, beam) - values[beam])) < 1.0e-18

    model2, grid2 = plate(n)
    beam2 = coupled(model2, grid2, grid2[0, 2], ecc)
    bad = {node: g_field(coords(model2, node)) for node in cut_nodes(grid2, n)}
    bad[beam2] = transported(bad[grid2[0, 2]], ecc) + np.array([1.0e-6, 0, 0, 0, 0, 0])
    with pytest.raises(SubmodelTransferError, match="TRACE_INCONSISTENT"):
        build_driven_trace(model2, bad, policy="drive_all")


# ----------------------------------------------------------------------
# contract sections 8.3 and 8.4: redundant and inconsistent model equations
# ----------------------------------------------------------------------
@pytest.mark.parametrize("consistent", [True, False])
def test_equality_of_two_cut_dofs_redundant_or_inconsistent(consistent):
    n = 4
    model, grid = plate(n)
    dm = model.mesh.dof_manager
    s, m = grid[0, 4], grid[0, 3]
    model.add_constraint_equation(terms=((dm.get_node_dofs(s)[0], 1.0), (dm.get_node_dofs(m)[0], -1.0)),
                                  rhs=0.0, source_id="equal", dependent_dof=dm.get_node_dofs(s)[0])

    def field(x):
        six = g_field(x)
        if consistent:
            six[0] = 1.0e-4 * x[0] + 1.0e-4  # ux independent of y, so u_s = u_m holds
        return six

    values = {node: field(coords(model, node)) for node in cut_nodes(grid, n)}
    if consistent:
        trace = build_driven_trace(model, values, policy="drive_all")
        assert any(trace.rows[i].node == s and trace.rows[i].label == "ux" for i in trace.redundant)
    else:
        with pytest.raises(SubmodelTransferError, match="TRACE_INCONSISTENT"):
            build_driven_trace(model, values, policy="drive_all")


# ----------------------------------------------------------------------
# junction rotation rule (contract section 4)
# ----------------------------------------------------------------------
def test_junction_rotation_rule():
    dofs = (3, 4, 5)
    coplanar = rotation_rows(1, dofs, [(0, 0, 1), (0, 0, -1)])
    assert len(coplanar) == 2 and all(row.direction is not None and abs(row.direction[2]) < 1e-12 for row in coplanar)
    corner = rotation_rows(1, dofs, [(0, 0, 1), (1, 0, 0)])
    assert [row.label for row in corner] == ["rx", "ry", "rz"]
    tilt = 1.0e-5
    with pytest.raises(SubmodelTransferError, match="TRACE_JUNCTION_ILL_CONDITIONED"):
        rotation_rows(1, dofs, [(0, 0, 1), (np.sin(tilt), 0, np.cos(tilt))])


# ----------------------------------------------------------------------
# A2-exact through the solve, both policies, Q4 and S3 V2D
# ----------------------------------------------------------------------
def boundary(grid, n):
    return sorted({grid[i, j] for (i, j) in grid if i in (0, n) or j in (0, n)})


@pytest.mark.parametrize("triangles", [False, True])
@pytest.mark.parametrize("policy", ["drive_all", "release_normal_rotation"])
def test_affine_field_through_the_solve_within_allowance(triangles, policy):
    n = 4
    model, grid = plate(n, triangles=triangles, membrane=False)
    driven = boundary(grid, n)
    values = {node: affine(coords(model, node)) for node in driven}
    normals = {node: [(0.0, 0.0, 1.0)] for node in driven}
    trace = build_driven_trace(model, values, policy=policy, normals=normals)
    apply_driven_trace(model, trace)
    case = LoadCase("none")
    u, info = solve_linear(model, case)
    allowance = estimate_static_error_allowance(model, u, case)
    assert not allowance.inconclusive
    exact = nodal_vector(model, affine)
    assert np.all(np.abs(u - exact) <= allowance.displacement + 1.0e-300)


def test_many_cases_refresh_values_without_refactorization():
    n = 4
    model, grid = plate(n, membrane=False)
    driven = boundary(grid, n)
    first = {node: affine(coords(model, node), 2.0e-4) for node in driven}
    trace = apply_driven_trace(model, build_driven_trace(model, first, policy="drive_all"))
    case = LoadCase("none")
    session = AnalysisSession(model)
    u1, _ = solve_linear(model, case, session=session)
    second = {node: rigid(coords(model, node)) for node in driven}
    update_driven_trace_values(model, trace, second)
    u2, _ = solve_linear(model, case, session=session)
    assert np.max(np.abs(u1 - nodal_vector(model, lambda x: affine(x, 2.0e-4)))) < 1.0e-15
    assert np.max(np.abs(u2 - nodal_vector(model, rigid))) < 1.0e-14
    counters = session.diagnostics()["counters"]
    assert counters["constraint_value_refreshes"] == 1
    assert counters["stiffness_builds"] == 1 and counters["static_operator_builds"] == 1
    session.close()


def test_zero_load_rigid_interface_forces_within_force_allowance():
    n = 4
    model, grid = plate(n, membrane=False)
    driven = boundary(grid, n)
    trace = apply_driven_trace(model, build_driven_trace(model, {node: rigid(coords(model, node)) for node in driven}, policy="drive_all"))
    case = LoadCase("none")
    u, _ = solve_linear(model, case)
    forces = interface_forces(model, u, case, trace)
    allowance = estimate_static_error_allowance(model, u, case)
    floor = force_allowance(model, u, case, allowance.displacement)
    assert np.all(floor > 0.0)
    dm = model.mesh.dof_manager
    for node, six in forces.node_forces.items():
        assert np.all(np.abs(six) <= floor[dm.get_node_dofs(node)])


# ----------------------------------------------------------------------
# review round 1 additions
# ----------------------------------------------------------------------
def test_multi_term_equation_and_its_update():
    """Slave on the cut with two interior masters: the trace row becomes a 2-term equation."""

    n = 4
    model, grid = plate(n)
    case = line_loaded(model, grid, n)
    dm = model.mesh.dof_manager
    s, m2, m3 = grid[0, 2], grid[1, 1], grid[1, 3]
    sx, m2x, m3x = (dm.get_node_dofs(node)[0] for node in (s, m2, m3))
    model.add_constraint_equation(terms=((sx, 1.0), (m2x, -0.5), (m3x, -0.5)), rhs=0.0,
                                  source_id="internal", dependent_dof=sx)
    cut = cut_nodes(grid, n)
    values = {node: g_field(coords(model, node)) for node in cut}
    trace = apply_driven_trace(model, build_driven_trace(model, values, policy="drive_all"))
    assert len(trace.applied["equations"]) == 1
    u, _ = solve_linear(model, case)
    assert node_values(model, u, s)[0] == pytest.approx(values[s][0], abs=1e-18)
    assert 0.5 * (u[m2x] + u[m3x]) == pytest.approx(values[s][0], abs=1e-18)
    shifted = {node: v + np.array([2.0e-5, 0, 0, 0, 0, 0]) for node, v in values.items()}
    update_driven_trace_values(model, trace, shifted)
    u2, _ = solve_linear(model, case)
    assert node_values(model, u2, s)[0] == pytest.approx(shifted[s][0], abs=1e-18)
    forces = interface_forces(model, u2, case, trace)
    total = sum(forces.node_forces[node][:2] for node in cut)
    assert np.max(np.abs(total + np.array([2.0e5, 3.0e4]))) < 1.0e-8 * 2.0e5


def test_refused_update_is_atomic():
    n = 4
    model, grid = plate(n, membrane=False)
    driven = boundary(grid, n)
    values = {node: affine(coords(model, node)) for node in driven}
    trace = apply_driven_trace(model, build_driven_trace(model, values, policy="drive_all"))
    before_bc = [dict(bc.dof_constraints) for bc in model.boundary_conditions]
    before_values = trace.values.copy()
    bad = dict(values)
    bad[driven[0]] = np.array([np.nan, 0, 0, 0, 0, 0])
    with pytest.raises(SubmodelTransferError, match="TRACE_VALUES"):
        update_driven_trace_values(model, trace, bad)
    assert [dict(bc.dof_constraints) for bc in model.boundary_conditions] == before_bc
    assert np.array_equal(trace.values, before_values)


def test_trace_is_bound_to_its_model_and_retained_constraints():
    n = 2
    model, grid = plate(n, membrane=False)
    other, _ = plate(n, membrane=False)
    values = {node: affine(coords(model, node)) for node in boundary(grid, n)}
    trace = build_driven_trace(model, values, policy="drive_all")
    with pytest.raises(SubmodelTransferError, match="TRACE_MODEL_MISMATCH"):
        apply_driven_trace(other, trace)
    model.add_boundary_condition(BoundaryCondition(name="late", node_ids=[grid[1, 1]], dof_constraints={"uz": 0.0}))
    with pytest.raises(SubmodelTransferError, match="TRACE_MODEL_CHANGED"):
        apply_driven_trace(model, trace)


def test_nearly_dependent_rows_refuse():
    n = 4
    model, grid = plate(n)
    dm = model.mesh.dof_manager
    s, m1, m2 = grid[0, 4], grid[0, 3], grid[1, 3]
    sx, m1x, m2x = (dm.get_node_dofs(node)[0] for node in (s, m1, m2))
    model.add_constraint_equation(terms=((sx, 1.0), (m1x, -1.0), (m2x, -1.0e-10)), rhs=0.0,
                                  source_id="near", dependent_dof=sx)
    values = {node: g_field(coords(model, node)) for node in cut_nodes(grid, n)}
    with pytest.raises(SubmodelTransferError, match="TRACE_RANK_AMBIGUOUS"):
        build_driven_trace(model, values, policy="drive_all")


def test_retained_support_reactions_close_the_free_body():
    n = 4
    model, grid = plate(n)
    case = line_loaded(model, grid, n)
    model.add_boundary_condition(BoundaryCondition(name="pin", node_ids=[grid[2, 2]], dof_constraints={"ux": 0.0, "uy": 0.0}))
    cut = cut_nodes(grid, n)
    trace = apply_driven_trace(model, build_driven_trace(
        model, {node: g_field(coords(model, node)) for node in cut}, policy="drive_all"))
    u, _ = solve_linear(model, case)
    forces = interface_forces(model, u, case, trace)
    assert abs(forces.retained_support_reactions[grid[2, 2]][0]) > 1.0e3  # the pin really reacts
    total = sum(forces.node_forces[node][:2] for node in cut) + sum(
        r[:2] for r in forces.retained_support_reactions.values())
    assert np.max(np.abs(total + np.array([2.0e5, 3.0e4]))) < 1.0e-8 * 2.0e5
    resultant = interface_resultant(model, forces.node_forces, (0.0, 0.5, 0.0))
    assert resultant.shape == (6,) and np.all(np.isfinite(resultant))


def test_equilibrium_is_enforced():
    n = 4
    model, grid = plate(n)
    case = line_loaded(model, grid, n)
    cut = cut_nodes(grid, n)
    trace = apply_driven_trace(model, build_driven_trace(
        model, {node: g_field(coords(model, node)) for node in cut}, policy="drive_all"))
    u, _ = solve_linear(model, case)
    wrong = u.copy()
    wrong[model.mesh.dof_manager.get_node_dofs(grid[2, 2])[0]] += 1.0e-6
    with pytest.raises(SubmodelTransferError, match="FORCE_UNCERTIFIED"):
        interface_forces(model, wrong, case, trace)
    trusted = estimate_static_error_allowance(model, u, case).displacement
    with pytest.raises(SubmodelTransferError, match="FORCE_EQUILIBRIUM"):
        interface_forces(model, wrong, case, trace, displacement_allowance=trusted)


def test_fully_prescribed_model_has_zero_allowance():
    model, grid = plate(1, membrane=False)
    values = {node: rigid(coords(model, node)) for node in grid.values()}
    apply_driven_trace(model, build_driven_trace(model, values, policy="drive_all"))
    case = LoadCase("none")
    u, _ = solve_linear(model, case)
    allowance = estimate_static_error_allowance(model, u, case)
    assert np.all(allowance.displacement == 0.0) and not allowance.inconclusive


def test_interpolation_allowance_covers_an_in_plane_correspondence_residual():
    model, _grid = plate(2, membrane=False)
    u = nodal_vector(model, affine)
    element = model.mesh.elements[1]
    natural = (0.1, -0.2)
    weights = element_interpolation_weights(element, natural)
    x_p = weights @ np.array([coords(model, n) for n in element.node_ids])
    d = 1.0e-3
    target = x_p + np.array([d, 0.0, 0.0])
    point = TransferPoint(77, 1, natural, tuple(target), tuple(weights), inverse_residual=d)
    value = interpolate_nodal_field(model, u, [point])[77]
    error = np.abs(value.values - affine(target))
    allowance = interpolation_allowance(value, model, u)
    assert np.max(error) > 0.0 and np.all(error <= allowance)
    with pytest.raises(SubmodelTransferError, match="TRANSFER_CONVENTION_MISMATCH"):
        interpolate_nodal_field(model, u, [TransferPoint(78, 1, natural, tuple(x_p), (np.nan,) * 4)])


# ----------------------------------------------------------------------
# review round 2 additions
# ----------------------------------------------------------------------
def test_scaled_rows_are_not_discarded():
    """Reduced rows [1e-10, 0] and [1, 1e-5] are independent: both must be enforced."""

    n = 4
    model, grid = plate(n)
    dm = model.mesh.dof_manager
    s1, s2, a, b = grid[0, 1], grid[0, 3], grid[2, 1], grid[2, 3]
    s1x, s2x, ax, bx = (dm.get_node_dofs(node)[0] for node in (s1, s2, a, b))
    model.add_constraint_equation(terms=((s1x, 1.0), (ax, -1.0e-10)), rhs=0.0, source_id="tiny", dependent_dof=s1x)
    model.add_constraint_equation(terms=((s2x, 1.0), (ax, -1.0), (bx, -1.0e-5)), rhs=0.0, source_id="mix", dependent_dof=s2x)
    big_a, big_b = 2.0e-4, 3.0e-2
    values = {node: g_field(coords(model, node)) for node in cut_nodes(grid, n)}
    values[s1] = values[s1].copy(); values[s1][0] = 1.0e-10 * big_a
    values[s2] = values[s2].copy(); values[s2][0] = big_a + 1.0e-5 * big_b
    trace = build_driven_trace(model, values, policy="drive_all")
    kept = {(trace.rows[i].node, trace.rows[i].label) for i in trace.kept}
    assert (s1, "ux") in kept and (s2, "ux") in kept
    apply_driven_trace(model, trace)
    u, _ = solve_linear(model, LoadCase("none"))
    assert u[ax] == pytest.approx(big_a, rel=1e-9)
    assert u[bx] == pytest.approx(big_b, rel=1e-6)


def test_support_coupled_through_mpc_gets_the_right_reaction():
    """u_m = 0 (support), u_s = (u_m + u_n)/2 (MPC), u_s = g (trace): reaction at m is r_m - r_n."""

    n = 4
    model, grid = plate(n)
    case = line_loaded(model, grid, n)
    dm = model.mesh.dof_manager
    s, m, nn = grid[0, 2], grid[1, 1], grid[1, 3]
    sx, mx, nx = (dm.get_node_dofs(node)[0] for node in (s, m, nn))
    model.add_boundary_condition(BoundaryCondition(name="hold_m", node_ids=[m], dof_constraints={"ux": 0.0}))
    model.add_constraint_equation(terms=((sx, 1.0), (mx, -0.5), (nx, -0.5)), rhs=0.0, source_id="avg", dependent_dof=sx)
    cut = cut_nodes(grid, n)
    trace = apply_driven_trace(model, build_driven_trace(
        model, {node: g_field(coords(model, node)) for node in cut}, policy="drive_all"))
    u, _ = solve_linear(model, case)
    forces = interface_forces(model, u, case, trace)
    from anyfesolver.matrix_assembly import assemble_load_vector
    with AnalysisSession(model) as session:
        k = session.stiffness_plan().matrix
    r = k @ u - np.asarray(assemble_load_vector(model, case)[0])
    assert forces.retained_support_reactions[m][0] == pytest.approx(r[mx] - r[nx], rel=1e-9)
    assert forces.node_forces[s][0] == pytest.approx(r[sx] + 2 * r[nx], rel=1e-9)
    total = sum(forces.node_forces[node][:2] for node in cut) + sum(
        rr[:2] for rr in forces.retained_support_reactions.values())
    assert np.max(np.abs(total + np.array([2.0e5, 3.0e4]))) < 1.0e-8 * 2.0e5


def thin_triangle(h):
    model = FEModel("thin")
    model.add_material("steel", E, NU, density=7850.0)
    for node, xyz in enumerate([(0, 0, 0), (1, 0, 0), (0.5, h, 0)], start=1):
        model.add_node(node, *xyz)
    model.add_element(1, NativeParityE4PLS3V2DShellElement(1, (1, 2, 3), "steel", thickness=T, reference_normal=(0, 0, 1)))
    return model


def test_gradient_term_uses_the_element_jacobian():
    h, rho = 1.0e-3, 1.0e-5
    model = thin_triangle(h)
    u = nodal_vector(model, lambda x: np.array([x[1], 0, 0, 0, 0, 0]))
    natural = (0.3, 0.3)
    weights = element_interpolation_weights(model.mesh.elements[1], natural)
    x_p = weights @ np.array([coords(model, n) for n in (1, 2, 3)])
    target = x_p + np.array([0.0, rho, 0.0])
    value = interpolate_nodal_field(model, u, [TransferPoint(5, 1, natural, tuple(target), inverse_residual=rho)])[5]
    error = abs(value.values[0] - target[1])
    allowance = interpolation_allowance(value, model, u)
    assert error == pytest.approx(rho, rel=1e-6) and allowance[0] >= error


def test_alternative_hits_enter_the_allowance():
    model, grid = plate(2, membrane=False)
    field = np.zeros(model.mesh.dof_manager.total_dofs)
    far = grid[2, 1]
    field[model.mesh.dof_manager.get_node_dofs(far)[0]] = 1.0e-3
    left, right = 3, 4  # bottom-left is element 1; use the upper row: 3 (x<0.5), 4 (x>0.5)
    x_t = np.array([0.5 - 1.0e-4, 0.75, 0.0])
    point = TransferPoint(9, left, (1.0 - 4.0e-4, 0.0), tuple(x_t), alternatives=((right, (-1.0 - 4.0e-4, 0.0)),))
    value = interpolate_nodal_field(model, field, [point])[9]
    assert value.alternative_spread[0] > 0.0
    assert interpolation_allowance(value, model, field)[0] >= value.alternative_spread[0]


def test_binding_is_checked_after_application():
    n = 4
    model, grid = plate(n)
    cut = cut_nodes(grid, n)
    trace = apply_driven_trace(model, build_driven_trace(
        model, {node: g_field(coords(model, node)) for node in cut}, policy="drive_all"))
    values = {node: g_field(coords(model, node)) for node in cut}
    model.boundary_conditions[0].dof_constraints["uz"] = 1.0e-3  # retained membrane support changed
    with pytest.raises(SubmodelTransferError, match="TRACE_MODEL_CHANGED"):
        update_driven_trace_values(model, trace, values)


def test_unequal_but_allowance_consistent_redundant_values_are_accepted_end_to_end():
    n = 4
    model, grid = plate(n)
    case = line_loaded(model, grid, n)
    dm = model.mesh.dof_manager
    s, m = grid[0, 4], grid[0, 3]
    model.add_constraint_equation(terms=((dm.get_node_dofs(s)[0], 1.0), (dm.get_node_dofs(m)[0], -1.0)),
                                  rhs=0.0, source_id="equal", dependent_dof=dm.get_node_dofs(s)[0])
    cut = cut_nodes(grid, n)
    values = {node: g_field(coords(model, node)) for node in cut}
    values[s] = values[s].copy(); values[s][0] = values[m][0] + 1.5e-6
    allowance = {node: np.array([1.0e-6, 0, 0, 0, 0, 0]) for node in cut}
    trace = apply_driven_trace(model, build_driven_trace(model, values, policy="drive_all", allowance=allowance))
    u, _ = solve_linear(model, case)
    forces = interface_forces(model, u, case, trace)
    assert forces.trace_residual_ratio <= 1.0


def test_perturbed_fully_prescribed_input_is_refused():
    model, grid = plate(1, membrane=False)
    values = {node: rigid(coords(model, node)) for node in grid.values()}
    apply_driven_trace(model, build_driven_trace(model, values, policy="drive_all"))
    case = LoadCase("none")
    u, _ = solve_linear(model, case)
    u = np.asarray(u).copy()
    u[0] += 1.0e-10
    with pytest.raises(SubmodelTransferError, match="ALLOWANCE_RECONSTRUCTION"):
        estimate_static_error_allowance(model, u, case)


def test_unadmitted_target_shell_and_invalid_controls_refuse():
    legacy = FEModel("legacy-target")
    legacy.add_material("steel", E, NU, density=7850.0)
    for node, xyz in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0)], start=1):
        legacy.add_node(node, *xyz)
    legacy.add_element(1, ShellElement(1, [1, 2, 3], "steel", thickness=T))
    with pytest.raises(SubmodelTransferError, match="TRANSFER_ELEMENT_NOT_ADMITTED"):
        build_driven_trace(legacy, {1: np.zeros(6)}, policy="drive_all")
    model, _grid = plate(1, membrane=False)
    u = nodal_vector(model, rigid)
    with pytest.raises(SubmodelTransferError, match="CONTROL_INVALID"):
        interpolate_nodal_field(model, u, [], weight_tolerance=np.nan)
    with pytest.raises(SubmodelTransferError, match="CONTROL_INVALID"):
        estimate_static_error_allowance(model, u, LoadCase("none"), inconclusive_threshold=np.nan)


# ----------------------------------------------------------------------
# review round 3 additions
# ----------------------------------------------------------------------
def pinned_case(as_equation, direction=None):
    n = 4
    model, grid = plate(n)
    case = line_loaded(model, grid, n)
    dm = model.mesh.dof_manager
    pin = grid[2, 2]
    dofs = dm.get_node_dofs(pin)
    if direction is not None:
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        model.add_constraint_equation(terms=((dofs[0], d[0]), (dofs[1], d[1])), rhs=0.0, source_id="dir", dependent_dof=dofs[0])
    elif as_equation:
        model.add_constraint_equation(terms=((dofs[0], 1.0),), rhs=0.0, source_id="pin_x", dependent_dof=dofs[0])
        model.add_constraint_equation(terms=((dofs[1], 1.0),), rhs=0.0, source_id="pin_y", dependent_dof=dofs[1])
    else:
        model.add_boundary_condition(BoundaryCondition(name="pin", node_ids=[pin], dof_constraints={"ux": 0.0, "uy": 0.0}))
    cut = cut_nodes(grid, n)
    trace = apply_driven_trace(model, build_driven_trace(
        model, {node: g_field(coords(model, node)) for node in cut}, policy="drive_all"))
    u, _ = solve_linear(model, case)
    forces = interface_forces(model, u, case, trace)
    total = sum(forces.node_forces[node][:2] for node in cut) + sum(
        r[:2] for r in forces.retained_support_reactions.values())
    return pin, forces, total


@pytest.mark.parametrize("as_equation", [False, True])
def test_restraint_written_as_support_or_equation_gives_the_same_reaction(as_equation):
    pin, forces, total = pinned_case(as_equation)
    _pin, reference, _total = pinned_case(False)
    assert np.allclose(forces.retained_support_reactions[pin][:2], reference.retained_support_reactions[pin][:2], rtol=1e-9)
    assert np.max(np.abs(total + np.array([2.0e5, 3.0e4]))) < 1.0e-8 * 2.0e5


def test_directional_restraint_equation_is_external():
    pin, forces, total = pinned_case(False, direction=(1.0, 1.0))
    assert np.linalg.norm(forces.retained_support_reactions[pin][:2]) > 1.0e3
    assert np.max(np.abs(total + np.array([2.0e5, 3.0e4]))) < 1.0e-8 * 2.0e5


def test_constant_rotation_transport_term_enters_the_allowance():
    model, _grid = plate(2, membrane=False)
    u = nodal_vector(model, lambda x: np.array([0, 0, 0, 0, 0, 1.0e-3]))
    element = model.mesh.elements[1]
    natural = (0.0, 0.0)
    weights = element_interpolation_weights(element, natural)
    x_p = weights @ np.array([coords(model, n) for n in element.node_ids])
    rho = 1.0e-5
    value = interpolate_nodal_field(model, u, [TransferPoint(6, 1, natural, tuple(x_p + [rho, 0, 0]), inverse_residual=rho)])[6]
    assert abs(value.values[1]) == pytest.approx(1.0e-8, rel=1e-9)  # transport of a field whose translations are 0
    assert interpolation_allowance(value, model, u)[1] >= 1.0e-8


def test_alternative_uses_its_own_uncertainty_and_value():
    model, grid = plate(2, membrane=False)
    dm = model.mesh.dof_manager
    u = np.zeros(dm.total_dofs)
    far = grid[2, 2]  # a node of the right element only
    u[dm.get_node_dofs(far)[0]] = 1.0e-3
    left, right = 3, 4
    x_t = np.array([0.5 - 1.0e-4, 0.75, 0.0])
    alt_natural = (-1.0 - 4.0e-4, 0.0)
    point = TransferPoint(9, left, (1.0 - 4.0e-4, 0.0), tuple(x_t), alternatives=((right, alt_natural),))
    value = interpolate_nodal_field(model, u, [point])[9]
    weights = element_interpolation_weights(model.mesh.elements[right], alt_natural)
    expected = sum(w * u[dm.get_node_dofs(n)[0]] for w, n in zip(weights, model.mesh.elements[right].node_ids))
    assert value.alternatives[0].values[0] == pytest.approx(expected, rel=1e-12)
    # Equal computed values but uncertainty only on the neighbour's own node.
    zero = np.zeros(dm.total_dofs)
    du = np.zeros(dm.total_dofs)
    du[dm.get_node_dofs(far)[0]] = 1.0e-6
    value0 = interpolate_nodal_field(model, zero, [point])[9]
    assert value0.alternative_spread[0] == 0.0
    w_far = weights[list(model.mesh.elements[right].node_ids).index(far)]
    assert interpolation_allowance(value0, model, zero, du)[0] >= abs(w_far) * 1.0e-6


def test_coefficients_the_audit_would_prune_are_refused():
    """Equilibrating u_s = 1e3 u_a + 5e-12 u_b leaves a 5e-15 coefficient the audit drops."""

    n = 4
    model, grid = plate(n)
    dm = model.mesh.dof_manager
    s, a, b = grid[0, 2], grid[1, 2], grid[2, 2]
    sx, ax, bx = (dm.get_node_dofs(node)[0] for node in (s, a, b))
    model.add_constraint_equation(terms=((sx, 1.0), (ax, -1.0e3), (bx, -5.0e-12)), rhs=0.0,
                                  source_id="s", dependent_dof=sx)
    trace = build_driven_trace(model, {node: g_field(coords(model, node)) for node in cut_nodes(grid, n)}, policy="drive_all")
    before = (len(model.boundary_conditions), len(model.constraint_equations))
    with pytest.raises(SubmodelTransferError, match="TRACE_AUDIT_PRUNING"):
        apply_driven_trace(model, trace)
    assert (len(model.boundary_conditions), len(model.constraint_equations)) == before


@pytest.mark.parametrize("factor", [1.0, 1.0e16])
def test_rescaled_equation_gives_the_same_forces(factor):
    n = 4
    model, grid = plate(n)
    case = line_loaded(model, grid, n)
    dm = model.mesh.dof_manager
    s, m2, m3 = grid[0, 2], grid[1, 1], grid[1, 3]
    sx, m2x, m3x = (dm.get_node_dofs(node)[0] for node in (s, m2, m3))
    model.add_constraint_equation(terms=((sx, factor), (m2x, -0.5 * factor), (m3x, -0.5 * factor)), rhs=0.0,
                                  source_id="avg", dependent_dof=sx)
    cut = cut_nodes(grid, n)
    trace = apply_driven_trace(model, build_driven_trace(
        model, {node: g_field(coords(model, node)) for node in cut}, policy="drive_all"))
    u, _ = solve_linear(model, case)
    forces = interface_forces(model, u, case, trace)
    total = sum(forces.node_forces[node][:2] for node in cut)
    assert np.max(np.abs(total + np.array([2.0e5, 3.0e4]))) < 1.0e-8 * 2.0e5


def test_junction_threshold_order_is_enforced():
    with pytest.raises(SubmodelTransferError, match="CONTROL_INVALID"):
        rotation_rows(1, (3, 4, 5), [(0, 0, 1)], rank3=2.0, rank2=1.0)


# ----------------------------------------------------------------------
# review round 4 additions
# ----------------------------------------------------------------------
import anyfesolver.submodel_transfer as transfer_module


def free_body(model, grid, n, forces, case):
    loads = {node: np.concatenate([np.asarray(v, float)[:3], np.zeros(3)]) if len(v) == 3 else np.asarray(v, float)
             for node, v in case.nodal_loads.items()}
    about = (0.5, 0.5, 0.0)
    total = (interface_resultant(model, forces.node_forces, about)
             + interface_resultant(model, forces.retained_support_reactions, about)
             + interface_resultant(model, loads, about))
    return total


def test_nearly_rigid_invariant_restraint_is_external_and_the_free_body_closes():
    n = 4
    model, grid = plate(n)
    case = line_loaded(model, grid, n)
    dm = model.mesh.dof_manager
    a, b, c = grid[1, 2], grid[2, 2], grid[3, 2]
    ax, bx, cx = (dm.get_node_dofs(node)[0] for node in (a, b, c))
    model.add_constraint_equation(terms=((ax, 1.0), (bx, -1.0), (cx, 1.0e-11)), rhs=0.0, source_id="near", dependent_dof=ax)
    cut = cut_nodes(grid, n)
    trace = apply_driven_trace(model, build_driven_trace(
        model, {node: g_field(coords(model, node)) for node in cut}, policy="drive_all"))
    u, _ = solve_linear(model, case)
    forces = interface_forces(model, u, case, trace)
    total = free_body(model, grid, n, forces, case)
    assert np.max(np.abs(total)) < 1.0e-8 * 2.0e5


def test_alternative_is_transported_through_its_own_offset():
    """Two perpendicular quads share the x-axis edge; the selected hit has zero offset."""

    model = FEModel("junction")
    model.add_material("steel", E, NU, density=7850.0)
    for node, xyz in enumerate([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 1), (0, 0, 1)], start=1):
        model.add_node(node, *xyz)
    model.add_element(1, create_shell_element(1, [1, 2, 3, 4], "steel", thickness=T))  # xy-plane
    model.add_element(2, create_shell_element(2, [1, 2, 5, 6], "steel", thickness=T))  # xz-plane
    zero = np.zeros(model.mesh.dof_manager.total_dofs)
    du = np.zeros_like(zero)
    for node in (1, 2):  # rotational uncertainty about x on the shared edge only
        du[model.mesh.dof_manager.get_node_dofs(node)[3]] = 1.0e-4
    d = 1.0e-3
    target = (0.5, d, 0.0)  # inside element 1, distance d from element 2's plane
    point = TransferPoint(9, 1, (0.0, -1.0 + 2.0 * d), target, alternatives=((2, (0.0, -1.0)),))
    value = interpolate_nodal_field(model, zero, [point])[9]
    assert np.allclose(value.offset, 0.0, atol=1e-15)
    assert np.allclose(value.alternatives[0].offset, (0.0, d, 0.0), atol=1e-15)
    allowance = interpolation_allowance(value, model, zero, du)
    # The alternative's rotation uncertainty about x moves its translation by about d * 1e-4.
    assert np.linalg.norm(allowance[:3]) >= 0.9 * d * 1.0e-4


def two_master_model():
    n = 4
    model, grid = plate(n)
    dm = model.mesh.dof_manager
    s, m2, m3 = grid[0, 2], grid[1, 1], grid[1, 3]
    sx, m2x, m3x = (dm.get_node_dofs(node)[0] for node in (s, m2, m3))
    model.add_constraint_equation(terms=((sx, 1.0), (m2x, -0.5), (m3x, -0.5)), rhs=0.0, source_id="avg", dependent_dof=sx)
    values = {node: g_field(coords(model, node)) for node in cut_nodes(grid, n)}
    return model, build_driven_trace(model, values, policy="drive_all")


def snapshot(model):
    return (list(model.boundary_conditions), list(model.constraint_equations),
            set(model.mesh.dof_manager._constrained_dofs))


def test_apply_rolls_back_on_any_failure_during_addition(monkeypatch):
    model, trace = two_master_model()
    before = snapshot(model)

    def broken(*_args, **_kwargs):
        raise RuntimeError("injected failure while adding an equation")

    monkeypatch.setattr(model, "add_constraint_equation", broken)
    with pytest.raises(RuntimeError, match="injected"):
        apply_driven_trace(model, trace)
    assert snapshot(model) == before and not trace.applied


def test_apply_rolls_back_when_the_constraint_map_fails(monkeypatch):
    model, trace = two_master_model()
    before = snapshot(model)

    def broken(_model):
        raise RuntimeError("injected failure in the constraint map")

    monkeypatch.setattr(transfer_module, "_constraint_map", broken)
    with pytest.raises(RuntimeError, match="injected"):
        apply_driven_trace(model, trace)
    assert snapshot(model) == before and not trace.applied


def test_verification_refuses_a_surviving_tiny_dependence(monkeypatch):
    model, trace = two_master_model()
    before = snapshot(model)
    real = transfer_module._constraint_map
    target_dof = trace.rows[trace.kept[0]].b[0][0]

    def leaky(m):
        t, c, independent = real(m)
        t = t.tolil()
        t[target_dof, 0] = t[target_dof, 0] + 1.0e-15
        return t.tocsr(), c, independent

    monkeypatch.setattr(transfer_module, "_constraint_map", leaky)
    with pytest.raises(SubmodelTransferError, match="TRACE_AUDIT_CHANGED"):
        apply_driven_trace(model, trace)
    assert snapshot(model) == before


def test_rescaled_equation_keeps_every_node_force():
    results = []
    for factor in (1.0, 1.0e16):
        n = 4
        model, grid = plate(n)
        case = line_loaded(model, grid, n)
        dm = model.mesh.dof_manager
        s, m2, m3 = grid[0, 2], grid[1, 1], grid[1, 3]
        sx, m2x, m3x = (dm.get_node_dofs(node)[0] for node in (s, m2, m3))
        model.add_constraint_equation(terms=((sx, factor), (m2x, -0.5 * factor), (m3x, -0.5 * factor)), rhs=0.0,
                                      source_id="avg", dependent_dof=sx)
        cut = cut_nodes(grid, n)
        trace = apply_driven_trace(model, build_driven_trace(
            model, {node: g_field(coords(model, node)) for node in cut}, policy="drive_all"))
        u, _ = solve_linear(model, case)
        results.append(interface_forces(model, u, case, trace).node_forces)
    for node, six in results[0].items():
        assert np.allclose(results[1][node], six, rtol=1e-9, atol=1e-6)


def test_rigid_mode_classification_is_per_mode_and_conservative():
    model, grid = plate(4)
    dm = model.mesh.dof_manager
    rigid = transfer_module._rigid_modes(model)
    a, b, c = (dm.get_node_dofs(grid[i, 2])[0] for i in (1, 2, 3))
    assert not transfer_module._is_external({a: 1.0, b: -1.0}, rigid)  # relative motion: internal
    assert transfer_module._is_external({a: 1.0, b: -1.0, c: 1.0e-11}, rigid)  # net 1e-11 on x-translation
    assert transfer_module._is_external({a: 1.0}, rigid)  # grounded restraint



def test_alternative_uses_its_own_inverse_residual():
    model, _grid = plate(2, membrane=False)
    u = nodal_vector(model, lambda x: np.array([1.0e-3 * x[0], 0, 0, 0, 0, 0]))
    x_t = (0.5, 0.75, 0.0)  # on the edge shared by elements 3 (left) and 4 (right)
    rho_alt = 1.0e-4
    point = TransferPoint(9, 3, (1.0, 0.0), x_t, inverse_residual=0.0, alternatives=((4, (-1.0, 0.0), rho_alt),))
    value = interpolate_nodal_field(model, u, [point])[9]
    assert value.alternatives[0].inverse_residual == rho_alt
    # Independent value of the alternative's gradient term: rho_alt * |d ux / dx|.
    assert interpolation_allowance(value, model, u)[0] >= rho_alt * 1.0e-3
    with pytest.raises(SubmodelTransferError, match="TRANSFER_ALTERNATIVE"):
        interpolate_nodal_field(model, u, [TransferPoint(9, 3, (1.0, 0.0), x_t, alternatives=((4,),))])
