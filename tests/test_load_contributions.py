"""OD3: the shared load traversal ``LoadCase._load_contributions`` is a bit-identical refactor.

The reference below is the nodal/element/pressure loop of ``_get_load_vector_core`` at 0dea1cc0f,
copied verbatim; the refactored kernel must give the same bytes on every case, and the coupled
ledger's view (``report_skips``) must name every component the native loop leaves out.
"""

import numpy as np
import pytest

from anyfesolver import FEModel, LoadCase
from anyfesolver.elements import create_shell_element
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement


def _plate(n=3, *, triangles=False):
    model = FEModel("load-contributions")
    model.add_material("steel", 210e9, 0.3, density=7850.0)
    grid, node = {}, 1
    for j in range(n + 1):
        for i in range(n + 1):
            model.add_node(node, 0.37 * i / n, 0.29 * j / n, 0.0 if triangles else 0.011 * i * j / n ** 2)
            grid[i, j] = node
            node += 1
    element = 1
    for j in range(n):
        for i in range(n):
            a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
            if triangles:
                for ids in ((a, b, c), (a, c, d)):
                    model.add_element(element, NativeParityE4PLS3V2DShellElement(
                        element, ids, "steel", thickness=0.012, reference_normal=(0.0, 0.0, 1.0)))
                    element += 1
            else:
                model.add_element(element, create_shell_element(element, [a, b, c, d], "steel", thickness=0.012))
                element += 1
    return model


class _Activity:
    def load_scales(self, ids):
        return np.array([0.25 + 0.125 * (int(i) % 3) for i in ids])


def _reference_core(case, mesh, dof_manager, displacements, element_activity):
    """0dea1cc0f boundary.py:707-751 (nodal, element and pressure loops), verbatim."""
    total_dofs = dof_manager.total_dofs
    F = np.zeros(total_dofs)

    def activity_scale(element_id):
        if element_activity is None:
            return 1.0
        values = element_activity.load_scales([int(element_id)])
        return float(np.asarray(values, dtype=float).reshape(-1)[0])

    for node_id, load in case.nodal_loads.items():
        node = mesh.get_node(node_id)
        if node:
            for i, dof in enumerate(node.dofs):
                if i < len(load):
                    F[dof] += load[i]
    for element_id, load in case.element_loads.items():
        element = mesh.get_element(element_id)
        if element is None:
            continue
        dof_mapping = element.get_dof_mapping(mesh)
        load = np.asarray(load, dtype=float)
        scale = activity_scale(int(element_id))
        for i, dof in enumerate(dof_mapping):
            if i < len(load):
                F[dof] += scale * load[i]
    for element_id, pressure in case.pressure_loads.items():
        element = mesh.get_element(element_id)
        if element is None or not hasattr(element, "node_ids"):
            continue
        coords = None
        if case.follower_pressure:
            case._reject_strict_flat_s3_v2_follower_pressure(element)
            coords = case._current_element_coordinates(element, mesh, displacements)
        f_elem = case._consistent_pressure_load(element, mesh, pressure, coords)
        scale = activity_scale(int(element_id))
        dof_mapping = element.get_dof_mapping(mesh)
        for i, dof in enumerate(dof_mapping):
            if i < len(f_elem):
                F[dof] += scale * f_elem[i]
    return F


def _core(case, model, displacements=None, activity=None):
    return case._get_load_vector_core(model.mesh, model.mesh.dof_manager, model.get_material, displacements,
                                      activity, qualified_runtime_guard=lambda **_kw: None)


def _cases(model):
    rng = np.random.default_rng(67)
    nodes = sorted(model.mesh.nodes)
    elements = sorted(model.mesh.elements)
    full = LoadCase("full")
    for k, nid in enumerate(nodes):
        full.add_nodal_load(nid, rng.standard_normal(6) * 10.0 ** rng.integers(-3, 4))
    full.nodal_loads[nodes[1]] = [1.0, 2.0]  # short list: only two components
    full.nodal_loads[nodes[2]] = np.arange(8.0)  # longer than the node's DOFs
    full.nodal_loads[10 ** 6] = np.ones(6)  # unknown node
    for eid in elements[::2]:
        full.element_loads[eid] = rng.standard_normal(30)
    full.element_loads[10 ** 6] = np.ones(24)
    full.pressure_loads = {eid: 1.0e3 * (1 + 0.37 * eid) for eid in elements}
    full.pressure_loads[10 ** 6] = 5.0
    pressure = LoadCase("pressure")
    pressure.pressure_loads = {eid: -2.5e4 for eid in reversed(elements)}
    nodal = LoadCase("nodal")
    for nid in nodes[::3]:
        nodal.add_nodal_load(nid, forces=[1e-300, -1e300, 3.0], moments=[0.5, 2.0 ** -1074, -7.0])
    return [full, pressure, nodal, LoadCase("empty")]


@pytest.mark.parametrize("triangles", [False, True], ids=["q4", "s3"])
@pytest.mark.parametrize("activity", [None, _Activity()], ids=["no_activity", "activity"])
def test_load_contributions_bit_identical(triangles, activity):
    model = _plate(triangles=triangles)
    for case in _cases(model):
        expected = _reference_core(case, model.mesh, model.mesh.dof_manager, None, activity)
        actual = _core(case, model, activity=activity)
        assert actual.tobytes() == expected.tobytes(), case.name
        # The traversal is exactly the native sum, in the same order.
        replay = np.zeros(model.mesh.dof_manager.total_dofs)
        for _record, dof, value in case._load_contributions(model.mesh, None, lambda eid: 1.0 if activity is None
                                                            else float(activity.load_scales([eid])[0])):
            replay[dof] += value
        assert replay.tobytes() == expected.tobytes(), case.name


def test_report_skips_names_every_component_the_native_loop_leaves_out():
    model = _plate()
    case = _cases(model)[0]
    items = list(case._load_contributions(model.mesh, None, lambda eid: 1.0, report_skips=True))
    skipped = {record for record, dof, value in items if dof is None}
    nodes = sorted(model.mesh.nodes)
    assert skipped == {
        ("nodal_skipped", nodes[2], "components_past_dofs"),
        ("nodal_skipped", 10 ** 6, "missing_node"),
        ("element_skipped", 10 ** 6, "missing_element"),
        ("pressure_skipped", 10 ** 6, "missing_element"),
    } | {("element_skipped", eid, "length_mismatch") for eid in sorted(model.mesh.elements)[::2]}
    kept = [item for item in items if item[1] is not None]
    assert kept == list(case._load_contributions(model.mesh, None, lambda eid: 1.0))
    assert all(value is None for _record, dof, value in items if dof is None)


def test_native_path_never_reports_skips():
    model = _plate()
    case = _cases(model)[0]
    assert all(dof is not None for _record, dof, _value in case._load_contributions(model.mesh, None,
                                                                                     lambda eid: 1.0))
