"""One-way shell -> solid submodel (docs/SOLID_SUBMODEL.md).

Oracles are closed-form: rigid and affine fields that both the shell and the P1/P2 solid
reproduce exactly (so the solid must return them), hand-derived MIN3 / bilinear values at a
centroid, and global equilibrium of the signed interface resultants.  No result is compared
with another run of the same code.  Needs the optional ANYsolid scikit-fem backend.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import pytest

pytest.importorskip("skfem")
anysolid_backend = pytest.importorskip("anysolid.backend")
from anysolid.mesh import SolidMesh  # noqa: E402

from anysolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement  # noqa: E402
from anysolver.elements import create_shell_element  # noqa: E402
from anysolver.fe_core import FEModel  # noqa: E402
from anysolver import BoundaryCondition  # noqa: E402
from anysolver.solid_submodel import (  # noqa: E402
    ATTRIBUTION_RULE,
    SolidSubmodelCancelled,
    SolidSubmodelError,
    evaluate_shell_displacement,
    solve_shell_driven_solid,
)
from anysolver.submodel_transfer import element_interpolation_weights  # noqa: E402

E, NU, T = 210.0e9, 0.3, 0.01
LZ = 0.2
ALL = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
LATERAL_AND_BOTTOM = ["xmin", "xmax", "ymin", "ymax", "zmin"]


# ----------------------------------------------------------------------
# fixtures: shell plate [0,1]^2 at z=0, solid box [0,1]^2 x [-LZ/2, LZ/2]
# ----------------------------------------------------------------------
def shell_plate(n=2, *, triangles=False, membrane=False):
    model = FEModel("solid-submodel")
    model.add_material("steel", E, NU, density=7850.0)
    grid, node = {}, 1
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
    return model


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


def zero(_x):
    return np.zeros(6)


def make_locate(model):
    """Exact inverse of the (affine) element maps of the plate, by the elements' own weights."""

    records = []
    for element_id, element in sorted(model.mesh.elements.items()):
        X = np.array([coords(model, n) for n in element.node_ids])
        p0 = element_interpolation_weights(element, (0.0, 0.0)) @ X
        pa = element_interpolation_weights(element, (1.0, 0.0)) @ X
        pb = element_interpolation_weights(element, (0.0, 1.0)) @ X
        records.append((element_id, len(element.node_ids) == 3, p0, np.column_stack([pa - p0, pb - p0])[:2]))

    def locate(x):
        hits = []
        for element_id, triangle, p0, A in records:
            natural = np.linalg.solve(A, (x - p0)[:2])
            if triangle:
                inside = natural.min() >= -1e-9 and natural.sum() <= 1.0 + 1e-9
            else:
                inside = np.abs(natural).max() <= 1.0 + 1e-9
            if inside:
                hits.append((element_id, tuple(float(v) for v in natural)))
        return hits

    return locate


def box_mesh(n=2, nz=2, *, node_groups=None):
    xs = np.linspace(0.0, 1.0, n + 1)
    zs = np.linspace(-LZ / 2, LZ / 2, nz + 1)
    index = lambda i, j, k: (k * (n + 1) + j) * (n + 1) + i  # noqa: E731
    points = np.array([(xs[i], xs[j], zs[k]) for k in range(nz + 1) for j in range(n + 1) for i in range(n + 1)])
    cells = []
    for k, j, i in itertools.product(range(nz), range(n), range(n)):
        for permutation in itertools.permutations(range(3)):
            current, ids = [i, j, k], [index(i, j, k)]
            for axis in permutation:
                current[axis] += 1
                ids.append(index(*current))
            volume = np.linalg.det(points[ids[1:]] - points[ids[0]])
            if volume < 0.0:
                ids[2], ids[3] = ids[3], ids[2]
            cells.append(ids)
    base = SolidMesh(points, cells)
    groups = {}
    for name, axis, value in (("xmin", 0, 0.0), ("xmax", 0, 1.0), ("ymin", 1, 0.0), ("ymax", 1, 1.0),
                              ("zmin", 2, -LZ / 2), ("zmax", 2, LZ / 2)):
        mask = np.array([np.allclose(points[f][:, axis], value) for f in base.boundary_facets])
        groups[name] = base.boundary_ids[mask]
    return SolidMesh(points, cells, boundary_facets=base.boundary_facets, boundary_ids=base.boundary_ids,
                     boundary_groups=groups, node_groups=node_groups)


@functools.lru_cache(maxsize=None)
def solid(order):
    return anysolid_backend.prepare(box_mesh(), {"elastic_modulus": E, "poisson_ratio": NU}, order=order)


@functools.lru_cache(maxsize=None)
def solid_with_corner_node():
    return anysolid_backend.prepare(box_mesh(node_groups={"pt": [0]}),
                                    {"elastic_modulus": E, "poisson_ratio": NU}, order=1)


class Proxy:
    """Composition wrapper (PreparedSolid is immutable) overriding selected attributes."""

    def __init__(self, base, **override):
        self._base = base
        self._override = override

    def __getattr__(self, name):
        if name in self.__dict__.get("_override", {}):
            return self._override[name]
        return getattr(self._base, name)


def drive(order, field, groups=ALL, *, triangles=False, skin=None, cancel=None, prepared=None):
    shell = shell_plate(2, triangles=triangles)
    return solve_shell_driven_solid(
        shell, nodal_vector(shell, field), prepared or solid(order), cut_groups=groups,
        locate=make_locate(shell), skin_tractions=skin, cancel=cancel)


def code_of(call):
    with pytest.raises(SolidSubmodelError) as info:
        call()
    return info.value.code


# ----------------------------------------------------------------------
# independent rigid / affine P1 and P2
# ----------------------------------------------------------------------
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("triangles", [False, True])
def test_rigid_shell_field_is_reproduced_with_zero_strain_and_interface_resultants(order, triangles):
    result = drive(order, rigid, triangles=triangles)
    prepared = solid(order)
    expected = np.array([(A_RIGID + np.cross(OM_RIGID, x))[c] for x, c in
                         zip(prepared.coefficient_coordinates, prepared.coefficient_components)])
    assert np.max(np.abs(result.coefficients - expected)) <= 1.0e-12
    assert np.max(np.abs(result.recovery.strain)) <= 1.0e-12
    scale = E * 1.0e-3
    assert np.linalg.norm(result.interface_force) <= 1.0e-9 * scale
    assert np.linalg.norm(result.interface_moment) <= 1.0e-9 * scale
    cert = result.diagnostics["certification"]
    assert cert["certified"] and cert["scaled_backward_error"] <= 1.0e-10
    expected_nodal = np.array([A_RIGID + np.cross(OM_RIGID, x) for x in prepared.mesh.coordinates])
    assert result.nodal_displacements.shape == (len(prepared.mesh.node_ids), 3)
    assert np.max(np.abs(result.nodal_displacements - expected_nodal)) <= 1.0e-12


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("triangles", [False, True])
def test_affine_shell_field_gives_uniform_strain_and_stress(order, triangles):
    result = drive(order, affine, triangles=triangles)
    prepared = solid(order)
    expected_u = np.array([affine(x)[c] for x, c in
                           zip(prepared.coefficient_coordinates, prepared.coefficient_components)])
    assert np.max(np.abs(result.coefficients - expected_u)) <= 1.0e-12
    eps = np.array([[1.0e-4, 2.5e-5, 0.0], [2.5e-5, -3.0e-5, 0.0], [0.0, 0.0, 0.0]])
    assert np.max(np.abs(result.recovery.strain - eps)) <= 1.0e-12
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    sigma = lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
    assert np.max(np.abs(result.recovery.stress - sigma)) <= 1.0e-8 * np.max(np.abs(sigma))
    # A closed, uniformly stressed body: the interface resultant is zero but the faces carry load.
    assert np.linalg.norm(result.interface_force) <= 1.0e-8 * np.max(np.abs(sigma))
    assert result.group_resultants["xmax"].force[0] > 0.0 > result.group_resultants["xmin"].force[0]


@pytest.mark.parametrize("order", [1, 2])
def test_every_cut_dof_including_p2_midsides_is_driven_and_shared_dofs_are_deduplicated(order):
    result = drive(order, affine)
    prepared = solid(order)
    union = np.unique(np.concatenate([prepared.boundary_dofs(g) for g in ALL]))
    assert np.array_equal(result.prescribed_dofs, union)
    total_listed = sum(len(prepared.boundary_dofs(g)) for g in ALL)
    constraints = result.diagnostics["constraints"]
    assert constraints["duplicate_dof_count"] == total_listed - union.size > 0
    assert constraints["prescribed_dofs"] == union.size
    if order == 2:
        vertices = len(prepared.mesh.coordinates)
        assert prepared.ndof > 3 * vertices  # midside coefficients exist and are all in the union
    assert result.attribution_rule == ATTRIBUTION_RULE
    assert result.shared_attribution
    for dof, owners in result.shared_attribution.items():
        assert len(owners) >= 2 and set(owners) <= set(ALL) and dof in union
        assert all(dof in prepared.boundary_dofs(g) for g in owners)
    force = sum(g.force for g in result.group_resultants.values())
    moment = sum(g.moment for g in result.group_resultants.values())
    scale = np.max(np.abs(result.reactions))
    assert np.max(np.abs(force - result.interface_force)) <= 1.0e-9 * scale
    assert np.max(np.abs(moment - result.interface_moment)) <= 1.0e-9 * scale
    assert sum(g.shared_dof_count for g in result.group_resultants.values()) > 0


# ----------------------------------------------------------------------
# signed loads
# ----------------------------------------------------------------------
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("traction", [(0.0, 0.0, -1.0e6), (2.0e5, -3.0e5, 5.0e5)])
def test_skin_traction_is_balanced_by_the_signed_interface_resultants(order, traction):
    t = np.asarray(traction)
    result = drive(order, zero, LATERAL_AND_BOTTOM, skin={"zmax": t})
    centroid = np.array([0.5, 0.5, LZ / 2])  # unit-area top face
    assert np.max(np.abs(result.skin_force - t)) <= 1.0e-9 * np.linalg.norm(t)
    assert np.max(np.abs(result.interface_force + t)) <= 1.0e-8 * np.linalg.norm(t)
    moment = np.cross(centroid, t)
    assert np.max(np.abs(result.skin_moment - moment)) <= 1.0e-9 * np.linalg.norm(moment)
    assert np.max(np.abs(result.interface_moment + moment)) <= 1.0e-8 * np.linalg.norm(moment)
    # Reversing the load reverses the field and every resultant (linearity, sign).
    reverse = drive(order, zero, LATERAL_AND_BOTTOM, skin={"zmax": -t})
    assert np.max(np.abs(reverse.coefficients + result.coefficients)) <= 1.0e-9 * np.max(np.abs(result.coefficients))
    assert np.max(np.abs(reverse.interface_force + result.interface_force)) <= 1.0e-8 * np.linalg.norm(t)
    assert result.diagnostics["residual"]["force_balance_ratio"] <= 1.0e-6


def _top_centre(prepared):
    centre = [i for i, x in enumerate(prepared.mesh.coordinates) if np.allclose(x, (0.5, 0.5, LZ / 2))]
    assert len(centre) == 1
    return centre[0]


def test_pressure_like_traction_has_physical_sign_work_balance_and_reversal():
    # Evidence solver-solid-01 (kept): the former assertion |ux| < 1e-6 |uz| at the top centre failed
    # on this Kuhn mesh (ux ~ 8.04e-8 vs |uz| ~ 6.70e-7).  Kuhn tetrahedra are not mirror-symmetric
    # about x = 0.5, so a zero lateral displacement is not guaranteed; that assertion was invalid.
    prepared = solid(1)
    load = np.array([0.0, 0.0, -1.0e6])
    result = drive(1, zero, LATERAL_AND_BOTTOM, skin={"zmax": load})
    centre = _top_centre(prepared)
    assert result.nodal_displacements[centre, 2] < 0.0  # moves in the loaded direction
    top = [i for i, x in enumerate(prepared.mesh.coordinates) if np.isclose(x[2], LZ / 2)]
    assert np.mean(result.nodal_displacements[top, 2]) < 0.0
    # External work f.q equals twice the strain energy q.K.q (zero prescribed values) and is positive.
    f = np.asarray(prepared.traction("zmax", load))
    q = result.coefficients
    work = float(f @ q)
    energy2 = float(q @ (prepared.assemble_linear() @ q))
    assert work > 0.0 and energy2 > 0.0
    assert work == pytest.approx(energy2, rel=1.0e-6)
    # Balance from the total applied load (unit-area face), independent of the solver's own sums.
    assert np.max(np.abs(result.skin_force - load)) <= 1.0e-9 * np.linalg.norm(load)
    summed = np.array([result.reactions[prepared.coefficient_components[result.prescribed_dofs] == c].sum()
                       for c in range(3)])
    assert np.max(np.abs(summed + load)) <= 1.0e-8 * np.linalg.norm(load)
    # Reversal flips the displacement, the work keeps its sign.
    reverse = drive(1, zero, LATERAL_AND_BOTTOM, skin={"zmax": -load})
    assert reverse.nodal_displacements[centre, 2] > 0.0
    assert np.max(np.abs(reverse.coefficients + q)) <= 1.0e-9 * np.max(np.abs(q))
    assert float(-f @ reverse.coefficients) == pytest.approx(work, rel=1.0e-6)


def test_vertical_load_is_symmetric_under_the_x_y_swap_of_the_kuhn_mesh():
    # The Kuhn subdivision (all six axis permutations per cell) is invariant under x <-> y, so the
    # top-centre lateral displacement must have equal x and y components.  (It is not mirror
    # symmetric under x -> 1 - x, hence no zero-ux claim.)
    prepared = solid(1)
    result = drive(1, zero, LATERAL_AND_BOTTOM, skin={"zmax": (0.0, 0.0, -1.0e6)})
    ux, uy, uz = result.nodal_displacements[_top_centre(prepared)]
    assert abs(ux - uy) <= 1.0e-5 * abs(uz)


# ----------------------------------------------------------------------
# bending / rotation correction (hand-derived centroid values)
# ----------------------------------------------------------------------
AMP = 1.0e-3


def single(element_factory, nodes, rows):
    model = FEModel("single")
    model.add_material("steel", E, NU, density=7850.0)
    for node, xyz in enumerate(nodes, start=1):
        model.add_node(node, *xyz)
    model.add_element(1, element_factory(tuple(range(1, len(nodes) + 1))))
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node, six in enumerate(rows, start=1):
        u[model.mesh.dof_manager.get_node_dofs(node)] = AMP * np.asarray(six, dtype=float)
    return model, u


def triangle(rows):
    return single(
        lambda ids: NativeParityE4PLS3V2DShellElement(1, ids, "steel", thickness=T, reference_normal=(0.0, 0.0, 1.0)),
        [(0, 0, 0), (1, 0, 0), (0, 1, 0)], rows)


def test_v2d_min3_twist_correction_at_the_centroid():
    # w = x*y: nodal w = 0, rotations (theta_x, theta_y) = (w_y, -w_x).  Exact w(1/3,1/3) = 1/9.
    model, u = triangle([(0,) * 6, (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, -1, 0)])
    value = evaluate_shell_displacement(model, u, [(1, (1 / 3, 1 / 3))], (1 / 3, 1 / 3, 0.0))
    assert value.value[2] == pytest.approx(AMP / 9, rel=1e-12)
    # The interpolated nodal w is 0: the MIN3 term replaces it by 1/9.
    assert value.transverse_correction == pytest.approx(AMP / 9, rel=1e-12)
    assert np.max(np.abs(value.value[:2])) <= 1e-18


def test_v2d_min3_curvature_correction_at_the_centroid():
    # w = x^2: nodal w = (0, 1, 0), theta_y = -w_x = (0, -2, 0).  Exact 1/9; interpolated nodal w 1/3.
    model, u = triangle([(0,) * 6, (0, 0, 1, 0, -2, 0), (0,) * 6])
    value = evaluate_shell_displacement(model, u, [(1, (1 / 3, 1 / 3))], (1 / 3, 1 / 3, 0.0))
    assert value.value[2] == pytest.approx(AMP / 9, rel=1e-12)
    assert value.transverse_correction == pytest.approx(-2 * AMP / 9, rel=1e-12)


def test_v2d_transverse_term_leaves_the_normal_offset_rotation_transport():
    # Constant nodal rotation about x (zero translations), target 5 mm above the reference
    # point: the lever arm to every node is (x - X_i), so the displacement is theta x (x - p)
    # = theta x (0, 0, 0.005) and the MIN3 correction (sum l_i = 0) vanishes.
    model, u = triangle([(0, 0, 0, 1, 0, 0)] * 3)
    value = evaluate_shell_displacement(model, u, [(1, (0.2, 0.3))], (0.2, 0.3, 0.005))
    expected = AMP * np.cross([1.0, 0.0, 0.0], [0.0, 0.0, 0.005])
    assert np.max(np.abs(value.value - expected)) <= 1e-15
    assert value.transverse_correction == pytest.approx(0.0, abs=1e-18)
    assert value.normal_offset == pytest.approx(0.005)


def quad(rows):
    return single(
        lambda ids: create_shell_element(1, list(ids), "steel", thickness=T),
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], rows)


def reference_point(model):
    element = model.mesh.elements[1]
    natural = (0.0, 0.0) if len(element.node_ids) == 4 else (1 / 3, 1 / 3)
    X = np.array([coords(model, n) for n in element.node_ids])
    return natural, element_interpolation_weights(element, natural) @ X


def test_q4_varying_drilling_rotation_does_not_displace_the_midsurface_centre():
    # theta_z = 1 only at the origin node: the centre translation and the centre in-plane
    # displacement are exactly zero (interpolated translation 0, lever arm 0).
    model, u = quad([(0, 0, 0, 0, 0, 1), (0,) * 6, (0,) * 6, (0,) * 6])
    natural, centre = reference_point(model)
    value = evaluate_shell_displacement(model, u, [(1, natural)], centre)
    assert np.array_equal(value.value[:2], np.zeros(2))
    assert value.normal_offset == 0.0


def test_v2d_varying_drilling_rotation_does_not_displace_the_midsurface_centroid():
    model, u = triangle([(0, 0, 0, 0, 0, 1), (0,) * 6, (0,) * 6])
    natural, centroid = reference_point(model)
    value = evaluate_shell_displacement(model, u, [(1, natural)], centroid)
    assert np.array_equal(value.value[:2], np.zeros(2))


def test_q4_signed_face_offset_with_varying_bending():
    # Mean translation (1,1,1) and mean rotation (1,1,1) (nodal values 4x on three nodes).
    rows = [(4, 0, 0, 4, 0, 0), (0, 4, 0, 0, 4, 0), (0, 0, 4, 0, 0, 4), (0,) * 6]
    model, u = quad(rows)
    natural, centre = reference_point(model)
    for h in (0.005, -0.005, 0.0):
        value = evaluate_shell_displacement(model, u, [(1, natural)], centre + [0.0, 0.0, h])
        # u = u_ref + theta_bar x (0, 0, h) = (1 + h, 1 - h, 1)
        assert np.max(np.abs(value.value - AMP * np.array([1.0 + h, 1.0 - h, 1.0]))) <= 1e-15
        assert value.normal_offset == pytest.approx(h)


def test_v2d_signed_face_offset_with_varying_bending():
    # Mean in-plane translation (1,1) and mean rotation (1,1,1); the normal part is the MIN3 w.
    rows = [(3, 0, 0, 3, 0, 0), (0, 3, 0, 0, 3, 0), (0, 0, 0, 0, 0, 3)]
    model, u = triangle(rows)
    natural, centroid = reference_point(model)
    mid = evaluate_shell_displacement(model, u, [(1, natural)], centroid)
    for h in (0.005, -0.005):
        value = evaluate_shell_displacement(model, u, [(1, natural)], centroid + [0.0, 0.0, h])
        assert np.max(np.abs(value.value[:2] - AMP * np.array([1.0 + h, 1.0 - h]))) <= 1e-15
        assert value.value[2] == pytest.approx(mid.value[2], abs=1e-18)  # (theta x e_z) has no z part
        assert value.normal_offset == pytest.approx(h)
    assert np.max(np.abs(mid.value[:2] - AMP)) <= 1e-15


def test_q4_represents_twist_exactly_without_correction():
    # Unit square, w = x*y: centre value 1/4 through the nodal transport alone.
    model, u = single(
        lambda ids: create_shell_element(1, list(ids), "steel", thickness=T),
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        [(0,) * 6, (0, 0, 0, 1, 0, 0), (0, 0, 1, 1, -1, 0), (0, 0, 0, 0, -1, 0)])
    value = evaluate_shell_displacement(model, u, [(1, (0.0, 0.0))], (0.5, 0.5, 0.0))
    assert value.value[2] == pytest.approx(AMP / 4, rel=1e-12)
    assert value.transverse_correction == 0.0


# ----------------------------------------------------------------------
# refusals: source, locate, prepared, solve
# ----------------------------------------------------------------------
def test_source_refusals():
    plate = shell_plate(1)
    u = nodal_vector(plate, rigid)
    bad = u.copy()
    bad[0] = np.nan
    hits = [(1, (0.0, 0.0))]
    target = (0.5, 0.5, 0.0)
    assert code_of(lambda: evaluate_shell_displacement(plate, bad, hits, target)) == "SOURCE_INVALID"
    assert code_of(lambda: evaluate_shell_displacement(plate, u[:-1], hits, target)) == "SOURCE_INVALID"
    assert code_of(lambda: evaluate_shell_displacement(object(), u, hits, target)) == "SOURCE_INVALID"
    assert code_of(lambda: evaluate_shell_displacement(plate, u, [(99, (0.0, 0.0))], target)) == "SOURCE_ELEMENT_MISSING"

    legacy = FEModel("legacy")
    legacy.add_material("steel", E, NU, density=7850.0)
    for node, xyz in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0)], start=1):
        legacy.add_node(node, *xyz)
    legacy.add_element(1, create_shell_element(1, [1, 2, 3], "steel", thickness=T, formulation="legacy-s3"))
    assert code_of(lambda: evaluate_shell_displacement(
        legacy, np.zeros(18), [(1, (0.2, 0.2))], (0.2, 0.2, 0.0))) == "SOURCE_ELEMENT_NOT_ADMITTED"

    warped = FEModel("warped")
    warped.add_material("steel", E, NU, density=7850.0)
    for node, xyz in enumerate([(0, 0, 0), (1, 0, 0), (1, 1, 0.3), (0, 1, 0)], start=1):
        warped.add_node(node, *xyz)
    warped.add_element(1, create_shell_element(1, [1, 2, 3, 4], "steel", thickness=T))
    assert code_of(lambda: evaluate_shell_displacement(
        warped, np.zeros(24), [(1, (0.0, 0.0))], (0.5, 0.5, 0.0))) == "SOURCE_ELEMENT_INVALID"


def test_inconsistently_constrained_source_is_refused():
    plate = shell_plate(1, membrane=True)  # uz = rx = ry = 0 supports
    code = code_of(lambda: evaluate_shell_displacement(
        plate, nodal_vector(plate, rigid), [(1, (0.0, 0.0))], (0.5, 0.5, 0.0)))
    assert code == "SOURCE_CONSTRAINT_INCONSISTENT"
    # A field that honours the supports is accepted.
    ok = evaluate_shell_displacement(plate, nodal_vector(plate, affine), [(1, (0.0, 0.0))], (0.5, 0.5, 0.0))
    assert np.all(np.isfinite(ok.value))


def test_locate_hits_must_agree_and_lie_inside():
    plate = shell_plate(2)
    rng = np.random.default_rng(3)
    u = rng.normal(scale=1.0e-3, size=plate.mesh.dof_manager.total_dofs)
    target = (0.25, 0.25, 0.0)
    # Conforming Q4s share both translations and rotations along their entire edge.
    shared = evaluate_shell_displacement(plate, u, [(1, (1., 0.)), (2, (-1., 0.))], (.5, .25, 0.))
    assert shared.hits == 2 and shared.hit_spread <= 1e-15
    # An overlapping, disconnected sheet can have a different physical displacement.
    duplicate = shell_plate(1)
    for node in range(1, 5):
        duplicate.add_node(node + 4, *coords(duplicate, node))
    duplicate.add_element(2, create_shell_element(2, [5, 6, 8, 7], "steel", thickness=T))
    separate = np.zeros(duplicate.mesh.dof_manager.total_dofs)
    for node in range(5, 9):
        separate[duplicate.mesh.dof_manager.get_node_dofs(node)[0]] = 0.01
    assert code_of(lambda: evaluate_shell_displacement(
        duplicate, separate, [(1, (0., 0.)), (2, (0., 0.))], (.5, .5, 0.))) == "LOCATE_HITS_DISAGREE"
    assert code_of(lambda: evaluate_shell_displacement(
        plate, u, [(1, (1.5, 0.0))], target)) == "LOCATE_NATURAL_OUTSIDE"
    # Conforming hits on a shared edge of a smooth field agree and are reported.
    smooth = nodal_vector(plate, rigid)
    both = evaluate_shell_displacement(plate, smooth, [(1, (1.0, 0.0)), (2, (-1.0, 0.0))], (0.5, 0.25, 0.0))
    assert both.hits == 2 and both.hit_spread <= 1.0e-15


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _tilted(triangle_element, shift=(0.0, 0.0, 0.0)):
    """Unit facet rotated out of the xy plane (and shifted); returns model, u, unit normal, centre."""

    R = _rotation((1.0, 2.0, 0.5), 0.7)
    normal = R @ np.array([0.0, 0.0, 1.0])
    shift = np.asarray(shift, dtype=float)
    local = [(0, 0, 0), (1, 0, 0), (0, 1, 0)] if triangle_element else [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    nodes = [tuple(R @ np.asarray(p, dtype=float) + shift) for p in local]
    if triangle_element:
        # Use the effective facet built from the stored world coordinates. At a large
        # origin, subtracting rounded points perturbs the nominal analytic director.
        normal = np.cross(np.subtract(nodes[1], nodes[0]), np.subtract(nodes[2], nodes[0]))
        normal /= np.linalg.norm(normal)
        factory = lambda ids: NativeParityE4PLS3V2DShellElement(  # noqa: E731
            1, ids, "steel", thickness=T, reference_normal=tuple(normal))
    else:
        factory = lambda ids: create_shell_element(1, list(ids), "steel", thickness=T)  # noqa: E731
    rng = np.random.default_rng(11)
    model, u = single(factory, nodes, rng.normal(size=(len(nodes), 6)))
    return model, u, normal


@pytest.mark.parametrize("triangle_element", [False, True])
@pytest.mark.parametrize("shift", [(0.0, 0.0, 0.0), (1.0e6, -2.0e6, 3.0e6)])
def test_tilted_large_coordinate_signed_normal_offset_is_admitted_and_tangential_forgery_refused(
        triangle_element, shift):
    model, u, normal = _tilted(triangle_element, shift)
    natural, centre = reference_point(model)
    tangent = np.cross(normal, [0.0, 0.0, 1.0])
    tangent /= np.linalg.norm(tangent)
    # Any signed normal offset (thickness-like or large) stays admissible and finite.
    for h in (0.0, 0.005, -0.005, 0.3, -2.0):
        value = evaluate_shell_displacement(model, u, [(1, natural)], centre + h * normal)
        assert np.all(np.isfinite(value.value))
        assert value.normal_offset == pytest.approx(h, abs=1.0e-6)
    # A tangential displacement of the target far above roundoff is a forged pairing.
    for sign in (1.0, -1.0):
        for lateral in (1.0e-3, 2.0):
            with pytest.raises(SolidSubmodelError) as info:
                evaluate_shell_displacement(
                    model, u, [(1, natural)], centre + 0.005 * normal + sign * lateral * tangent)
            assert info.value.code == "LOCATE_TANGENTIAL_MISMATCH"
            assert info.value.details["tangential_residual"] == pytest.approx(lateral, rel=1.0e-3)
            assert info.value.details["allowed"] < 1.0e-6 * max(1.0, abs(shift[0]) / 1.0e6)


def test_tangential_forgery_is_refused_for_primary_and_shared_hits_through_the_public_evaluator():
    plate = shell_plate(2)
    u = nodal_vector(plate, rigid)
    # Valid natural point of element 2, but 0.5 away tangentially from the target.
    assert code_of(lambda: evaluate_shell_displacement(
        plate, u, [(2, (0.0, 0.0))], (0.25, 0.25, 0.0))) == "LOCATE_TANGENTIAL_MISMATCH"
    # Target 2 m from a perfectly valid natural point of element 1.
    assert code_of(lambda: evaluate_shell_displacement(
        plate, u, [(1, (0.0, 0.0))], (2.25, 0.25, 0.0))) == "LOCATE_TANGENTIAL_MISMATCH"
    # Honest primary hit, forged shared (second / third) hit: the rigid field would make the
    # values agree, so only the geometric check can refuse it.
    honest = [(1, (1.0, 0.0)), (2, (-1.0, 0.0))]
    target = (0.5, 0.25, 0.0)
    assert evaluate_shell_displacement(plate, u, honest, target).hits == 2
    assert code_of(lambda: evaluate_shell_displacement(
        plate, u, honest + [(4, (0.0, 0.0))], target)) == "LOCATE_TANGENTIAL_MISMATCH"
    assert code_of(lambda: evaluate_shell_displacement(
        plate, u, [honest[0], (2, (0.0, 0.0))], target)) == "LOCATE_TANGENTIAL_MISMATCH"
    # The refusal carries the diagnostic.
    with pytest.raises(SolidSubmodelError) as info:
        evaluate_shell_displacement(plate, u, [(2, (0.0, 0.0))], (0.25, 0.25, 0.0))
    assert info.value.details["element_id"] == 2
    assert info.value.details["tangential_residual"] == pytest.approx(0.5)
    assert info.value.details["allowed"] < 1.0e-6
    # Large signed normal offsets over the honest correspondence remain admissible.
    for h in (0.005, -0.005, 0.5):
        assert evaluate_shell_displacement(plate, u, [(1, (0.0, 0.0))], (0.25, 0.25, h)).normal_offset \
            == pytest.approx(h)


def test_forged_locator_is_refused_by_the_solve():
    shell = shell_plate(2)
    u = nodal_vector(shell, rigid)
    honest = make_locate(shell)

    def forged(x):
        hits = list(honest(x))
        if np.allclose(x, (0.0, 0.0, 0.0)):
            return [(2, (0.0, 0.0))]  # valid natural point, 0.75 m away tangentially
        return hits

    assert code_of(lambda: solve_shell_driven_solid(
        shell, u, solid(1), cut_groups=ALL, locate=forged)) == "LOCATE_TANGENTIAL_MISMATCH"


def test_locate_failures():
    shell = shell_plate(2)
    u = nodal_vector(shell, zero)
    run = lambda locate: solve_shell_driven_solid(  # noqa: E731
        shell, u, solid(1), cut_groups=["xmin"], locate=locate)
    assert code_of(lambda: run(lambda x: [])) == "LOCATE_MISS"
    assert code_of(lambda: run(lambda x: 7)) == "LOCATE_INVALID"
    assert code_of(lambda: run(lambda x: [(True, (0.0, 0.0))])) == "LOCATE_INVALID"
    assert code_of(lambda: run(lambda x: [(1, (np.nan, 0.0))])) == "LOCATE_INVALID"

    def boom(x):
        raise RuntimeError("host failure")

    assert code_of(lambda: run(boom)) == "LOCATE_FAILED"


def test_prepared_and_argument_refusals():
    shell = shell_plate(2)
    u = nodal_vector(shell, zero)
    locate = make_locate(shell)
    run = lambda prepared, **kw: solve_shell_driven_solid(  # noqa: E731
        shell, u, prepared, cut_groups=kw.pop("cut_groups", ["xmin"]), locate=locate, **kw)
    prepared = solid(1)
    assert code_of(lambda: run(object())) == "PREPARED_INVALID"
    bad_components = prepared.coefficient_components.copy()
    bad_components[0] = 5
    assert code_of(lambda: run(Proxy(prepared, coefficient_components=bad_components))) == "PREPARED_INVALID"
    assert code_of(lambda: run(prepared, cut_groups=["nope"])) == "CUT_GROUP_INVALID"
    assert code_of(lambda: run(prepared, cut_groups=[])) == "CUT_GROUP_INVALID"
    assert code_of(lambda: run(prepared, cut_groups="xmin")) == "CUT_GROUP_INVALID"
    assert code_of(lambda: run(prepared, cut_groups=["xmin", "xmin"])) == "CUT_GROUP_INVALID"
    assert code_of(lambda: run(prepared, skin_tractions={"xmin": (0, 0, 1)})) == "SKIN_INVALID"  # also a cut group
    assert code_of(lambda: run(prepared, skin_tractions={"nope": (0, 0, 1)})) == "SKIN_INVALID"
    assert code_of(lambda: run(prepared, skin_tractions={"zmax": (0, 0, np.inf)})) == "SKIN_INVALID"
    assert code_of(lambda: run(prepared, skin_tractions={"zmax": (0, 0)})) == "SKIN_INVALID"
    assert code_of(lambda: run(prepared, cancel=42)) == "CANCEL_INVALID"
    assert code_of(lambda: solve_shell_driven_solid(
        shell, u, prepared, cut_groups=["xmin"], locate="not callable")) == "LOCATE_INVALID"


def test_singular_and_nonfinite_operators_are_refused():
    # A single driven vertex leaves rigid rotations free.
    prepared = solid_with_corner_node()
    shell = shell_plate(2)
    code = code_of(lambda: solve_shell_driven_solid(
        shell, nodal_vector(shell, zero), prepared, cut_groups=["pt"], locate=make_locate(shell)))
    assert code == "UNDERCONSTRAINED"

    base = solid(1)
    interior = next(i for i, x in enumerate(base.coefficient_coordinates) if np.allclose(x, (0.5, 0.5, 0.0)))
    K = base.assemble_linear().tolil()
    K[interior, :] = 0.0
    K[:, interior] = 0.0
    assert code_of(lambda: drive(1, zero, prepared=Proxy(base, assemble_linear=lambda: K.tocsr()))) == "SOLVE_SINGULAR"

    nan_K = base.assemble_linear()
    nan_K.data[0] = np.nan
    assert code_of(lambda: drive(1, zero, prepared=Proxy(base, assemble_linear=lambda: nan_K))) \
        == "PREPARED_OPERATOR_INVALID"


def test_cancellation_returns_no_result():
    with pytest.raises(SolidSubmodelCancelled):
        drive(1, rigid, cancel=lambda: True)

    class Flag:
        def __init__(self):
            self.value = False

        def is_set(self):
            return self.value

    flag = Flag()
    flag.value = True
    with pytest.raises(SolidSubmodelCancelled):
        drive(1, rigid, cancel=flag)

    calls = {"n": 0}

    def late():
        calls["n"] += 1
        return calls["n"] > 25  # after the cut groups are resolved, while evaluating the shell field

    with pytest.raises(SolidSubmodelCancelled) as info:
        drive(1, rigid, cancel=late)
    assert info.value.code == "CANCELLED" and calls["n"] > 25
    # An un-triggered probe changes nothing.
    result = drive(1, rigid, cancel=lambda: False)
    assert result.diagnostics["certification"]["certified"]
