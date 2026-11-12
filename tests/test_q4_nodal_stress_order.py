"""Q4 nodal stress recovery must pair rows with the stations they came from.

The qualified planar Q4 returns stress rows in its stiffness-station order
(``[--, +-, ++, -+]``) and reports that order in ``natural_point_coordinates``,
while ``element.gauss_points`` is tensor ordered (``[--, +-, -+, ++]``).
The first group exercises the production consumer helper with manufactured,
asymmetric fields.  The second group drives ``recover_nodal_stresses`` on a
one-element production model: the global displacement vector of a one-element
model has exactly the element DOF count, so the model-level caller must gather
it into connectivity order itself.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.results import _operator_for_returned_rows, recover_nodal_stresses

G = 1.0 / np.sqrt(3.0)
STIFFNESS_ORDER = np.array([[-G, -G], [G, -G], [G, G], [-G, G]])
TENSOR_ORDER = np.array([[-G, -G], [G, -G], [-G, G], [G, G]])
NODES = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
KEYS = ("global_xx_top", "global_xx_bot")


def _field(points, scale=1.0):
    r, s = points[:, 0], points[:, 1]
    return scale * (2.0 + 3.0 * r - 5.0 * s + 7.0 * r * s)


def _element():
    return SimpleNamespace(element_id=1, num_nodes=4, gauss_points=TENSOR_ORDER)


def _stresses(points, declared=True):
    out = {"global_xx_top": _field(points), "global_xx_bot": _field(points, -2.0)}
    if declared:
        out["natural_point_coordinates"] = points.copy()
    return out


def test_declared_stiffness_order_recovers_exact_nodal_field():
    stresses = _stresses(STIFFNESS_ORDER)
    operator = _operator_for_returned_rows(_element(), stresses, KEYS)
    np.testing.assert_allclose(operator @ stresses["global_xx_top"], _field(NODES), rtol=0, atol=1e-13)
    np.testing.assert_allclose(operator @ stresses["global_xx_bot"], _field(NODES, -2.0), rtol=0, atol=1e-13)


def test_legacy_gauss_point_layout_unchanged_without_declared_coordinates():
    stresses = _stresses(TENSOR_ORDER, declared=False)
    operator = _operator_for_returned_rows(_element(), stresses, KEYS)
    np.testing.assert_allclose(operator @ stresses["global_xx_top"], _field(NODES), rtol=0, atol=1e-13)


def test_old_pairing_would_have_been_wrong():
    rows = _field(STIFFNESS_ORDER)
    wrong = _operator_for_returned_rows(_element(), _stresses(TENSOR_ORDER, declared=False), KEYS) @ rows
    assert not np.allclose(wrong, _field(NODES), atol=1e-6)


@pytest.mark.parametrize("shift", range(4))
def test_cyclic_station_permutation_gives_same_nodal_values(shift):
    points = np.roll(STIFFNESS_ORDER, shift, axis=0)
    stresses = _stresses(points)
    operator = _operator_for_returned_rows(_element(), stresses, KEYS)
    np.testing.assert_allclose(operator @ stresses["global_xx_top"], _field(NODES), rtol=0, atol=1e-13)


def test_inconsistent_rows_are_refused():
    element = _element()
    bad_shape = _stresses(STIFFNESS_ORDER)
    bad_shape["natural_point_coordinates"] = STIFFNESS_ORDER[:3]
    with pytest.raises(ValueError, match="inconsistent"):
        _operator_for_returned_rows(element, bad_shape, KEYS)
    duplicate = _stresses(STIFFNESS_ORDER)
    duplicate["natural_point_coordinates"] = np.vstack((STIFFNESS_ORDER[:3], STIFFNESS_ORDER[:1]))
    with pytest.raises(ValueError, match="duplicate"):
        _operator_for_returned_rows(element, duplicate, KEYS)
    ragged = _stresses(STIFFNESS_ORDER)
    ragged["global_xx_bot"] = ragged["global_xx_bot"][:3]
    with pytest.raises(ValueError, match="row counts"):
        _operator_for_returned_rows(element, ragged, KEYS)
    nonfinite = _stresses(STIFFNESS_ORDER)
    nonfinite["natural_point_coordinates"][0, 0] = np.nan
    with pytest.raises(ValueError, match="inconsistent"):
        _operator_for_returned_rows(element, nonfinite, KEYS)


# ---------------------------------------------------------------------------
# One-element production model through recover_nodal_stresses
# ---------------------------------------------------------------------------

E_MOD = 210.0e9
NU = 0.3
SKEW_COORDS = np.array(
    [[0.0, 0.0, 0.0], [1.35, 0.08, 0.0], [1.08, 0.95, 0.0], [-0.18, 1.12, 0.0]]
)
CYCLIC_CONNECTIVITIES = [[1, 2, 3, 4], [4, 1, 2, 3], [3, 4, 1, 2], [2, 3, 4, 1]]


def _rotation():
    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return rz(np.radians(30.0)) @ ry(np.radians(-25.0)) @ rx(np.radians(40.0))


def _one_element_model(connectivity, coords):
    model = FEModel("one_element")
    model.add_material("steel", E_MOD, NU)
    for node_id, coord in enumerate(coords, start=1):
        model.add_node(node_id, *(float(v) for v in coord))
    model.add_element(1, create_shell_element(1, list(connectivity), "steel", thickness=0.01))
    return model


def _physical_field(model, coords, rotation):
    """Asymmetric in-plane strain plus an oblique rigid motion, in global axes.

    Returns the global displacement vector (indexed by each node's own DOFs,
    independent of element connectivity) and the exact global stress tensor.
    """
    a, b, c = 1.2e-4, -0.4e-4, 0.35e-4
    local_strain = np.array([[a, c, 0.0], [c, b, 0.0], [0.0, 0.0, 0.0]])
    strain = rotation @ local_strain @ rotation.T
    normal = rotation[:, 2]
    projector = np.eye(3) - np.outer(normal, normal)
    omega = np.array([2.0e-4, -1.5e-4, 3.0e-4])
    translation = np.array([1.0e-4, -2.0e-4, 0.5e-4])

    dofs = {node_id: model.mesh.get_node(node_id).dofs for node_id in range(1, 5)}
    u = np.zeros(1 + max(max(d) for d in dofs.values()))
    for node_id, coord in enumerate(coords, start=1):
        x = np.asarray(coord, dtype=float)
        base = list(dofs[node_id])
        u[base[:3]] = translation + strain @ x + np.cross(omega, x)
        u[base[3:6]] = omega

    shear_modulus = E_MOD / (2.0 * (1.0 + NU))
    lam = E_MOD * NU / (1.0 - NU**2)
    stress = 2.0 * shear_modulus * strain + lam * np.trace(strain) * projector
    return u, stress


_COMPONENT_INDEX = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}


@pytest.mark.parametrize("connectivity", CYCLIC_CONNECTIVITIES)
def test_one_element_recovery_matches_physical_stress_for_cyclic_connectivity(connectivity):
    rotation = _rotation()
    coords = SKEW_COORDS @ rotation.T
    model = _one_element_model(connectivity, coords)
    u, expected = _physical_field(model, coords, rotation)
    scale = E_MOD * 1.2e-4  # stress scale of the imposed strain, Pa

    recovered = recover_nodal_stresses(model, u)

    assert recovered["skipped_element_ids"] == []
    assert sorted(recovered["nodal"]) == [1, 2, 3, 4]
    for node_id, values in recovered["nodal"].items():
        for surface in ("top", "bot"):
            for component, index in _COMPONENT_INDEX.items():
                assert values[f"global_{component}_{surface}"] == pytest.approx(
                    expected[index], rel=0.0, abs=1.0e-8 * scale
                ), (connectivity, node_id, surface, component)


def test_one_element_recovery_is_invariant_under_cyclic_connectivity():
    rotation = _rotation()
    coords = SKEW_COORDS @ rotation.T
    scale = E_MOD * 1.2e-4
    results = []
    for connectivity in CYCLIC_CONNECTIVITIES:
        model = _one_element_model(connectivity, coords)
        u, _ = _physical_field(model, coords, rotation)
        results.append(recover_nodal_stresses(model, u)["nodal"])
    for other in results[1:]:
        for node_id, reference in results[0].items():
            for key, value in reference.items():
                assert other[node_id][key] == pytest.approx(value, rel=0.0, abs=1.0e-8 * scale)


def test_raw_compute_stresses_keeps_element_local_convention_for_full_length_vector():
    """A raw element-length vector is element-ordered input, as before."""
    rotation = _rotation()
    coords = SKEW_COORDS @ rotation.T
    scale = E_MOD * 1.2e-4

    identity = _one_element_model([1, 2, 3, 4], coords)
    u, expected = _physical_field(identity, coords, rotation)
    material = identity.get_material("steel")
    raw = identity.mesh.get_element(1).compute_stresses(identity.mesh, u, material, return_global=True)
    assert np.max(np.abs(raw["global_xx_top"] - expected[0, 0])) < 1.0e-8 * scale

    permuted = _one_element_model([4, 1, 2, 3], coords)
    element = permuted.mesh.get_element(1)
    node_order_raw = element.compute_stresses(permuted.mesh, u, material, return_global=True)
    gather = np.asarray(element.get_dof_mapping(permuted.mesh), dtype=int)
    element_order = element.compute_stresses(permuted.mesh, u[gather], material, return_global=True)
    # Node-ordered input to the raw API is interpreted as element-local, hence wrong...
    assert np.max(np.abs(node_order_raw["global_xx_top"] - expected[0, 0])) > 1.0e-3 * scale
    # ...while explicitly gathered element-order input is physical.
    assert np.max(np.abs(element_order["global_xx_top"] - expected[0, 0])) < 1.0e-8 * scale


def test_recover_nodal_stresses_rejects_malformed_global_vector():
    rotation = _rotation()
    coords = SKEW_COORDS @ rotation.T
    model = _one_element_model([4, 1, 2, 3], coords)
    u, _ = _physical_field(model, coords, rotation)
    with pytest.raises(ValueError):
        recover_nodal_stresses(model, u.reshape(-1, 1))
    with pytest.raises((ValueError, IndexError)):
        recover_nodal_stresses(model, u[:-6])


def test_collinear_q4_stations_cannot_recover_a_bilinear_stress():
    points = np.array([[-.9, 0.], [-.3, 0.], [.3, 0.], [.9, 0.]])
    with pytest.raises(ValueError, match="span the recovery polynomial"):
        _operator_for_returned_rows(_element(), {"natural_point_coordinates": points,
                                                **{key: _field(points) for key in KEYS}}, KEYS)
