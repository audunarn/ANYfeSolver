"""Coupled route S1: C1 formation, section 3.1 provenance, the 3.2 ledger, section 4 codes and the
section 5 basis (docs/SOLID_COUPLING.md). Every test fails on 642c43d64 (solid_coupling.py absent)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('skfem')
pytest.importorskip('anysolid')

import _coupled_fixtures as fx  # noqa: E402
from anyfesolver import solid_coupling as sc  # noqa: E402
from anyfesolver._coupled_certificate import CoupledSolveError  # noqa: E402
from anyfesolver.boundary import LoadCase  # noqa: E402

HOLED = __import__('types').SimpleNamespace(mesh=None)
SIDES = ('side_0', 'side_1', 'side_2', 'side_3')
T = 0.02


def csr(rows):
    return sparse.csr_matrix(np.asarray(rows, dtype=float))


def formation(**overrides):
    """A 1-DOF shell (K = 1, T = 1, u0 = 0) tied to a 1-DOF solid by C = [1] / [1]."""
    base = dict(K=csr([[1.0]]), F=np.array([1.0]), K_s=csr([[1.0]]), f_s=np.array([1.0]), T=csr([[1.0]]),
                u0=np.zeros(1), C_shell=csr([[1.0]]), C_solid=csr([[1.0]]), K_v=csr([[1.0]]), f_v=np.zeros(1))
    base.update(overrides)
    return sc._form_system(**base)


def refused(code, call, *args, **kwargs):
    with pytest.raises(CoupledSolveError) as info:
        call(*args, **kwargs)
    assert info.value.code == code, (info.value.code, info.value.reason, str(info.value))
    return info.value


# --- C1 ---------------------------------------------------------------------------------------------

def test_c1_review_case_half_one_with_subnormal_offset_refuses():
    error = refused('COUPLED_FORMATION_RANGE', formation, K=csr(np.eye(2)), F=np.zeros(2), T=csr([[1.0], [0.0]]),
                    u0=np.array([2.0 ** -1074, 0.0]), C_shell=csr([[0.5, 1.0]]))
    assert error.reason == 'raw_window'


def test_c1_solid_stiffness_window_low_edge():
    refused('COUPLED_FORMATION_RANGE', formation, K_v=csr([[2.0 ** -401]]))
    assert formation(K_v=csr([[2.0 ** -400]])).system.shape == (3, 3)


def test_c1_window_high_edge():
    assert formation(K_v=csr([[2.0 ** 64]])).system.shape == (3, 3)
    refused('COUPLED_FORMATION_RANGE', formation, K_v=csr([[2.0 ** 65]]))


def test_c1_exact_coupling_product_loss_refuses():
    # B_s = 2^-300 * 2^-300 = 2^-600 is exactly nonzero but rounds outside the raw window.
    error = refused('COUPLED_FORMATION_LOSS', formation, C_shell=csr([[2.0 ** -300]]), T=csr([[2.0 ** -300]]))
    assert error.reason == 'formation_loss'
    refused('COUPLED_FORMATION_LOSS', formation, C_shell=csr([[2.0 ** -300]]), u0=np.array([2.0 ** -300]))


def test_c1_offset_is_carried_exactly():
    formed = formation(K=csr(np.eye(2)), F=np.zeros(2), T=csr([[1.0], [0.0]]), u0=np.array([0.0, 0.25]),
                       C_shell=csr([[0.5, 1.0]]))
    assert formed.rhs[-1] == -0.25 and formed.system[2, 0] == 0.5


def test_c1_types_and_dtypes_are_exact():
    refused('COUPLED_FORMATION_RANGE', formation, K_v=sparse.csr_matrix(np.array([[1]], dtype=np.int8)))
    refused('COUPLED_FORMATION_RANGE', formation, K_v=sparse.coo_matrix(np.eye(1)))
    refused('COUPLED_FORMATION_RANGE', formation, f_v=np.zeros(1, dtype=np.float32))
    refused('COUPLED_FORMATION_RANGE', formation, f_v=np.array([np.inf]))


def test_c1_duplicates_are_summed_before_the_window():
    duplicate = sparse.csr_matrix((np.array([2.0 ** -401, 2.0 ** -401]), np.array([0, 0]), np.array([0, 2])),
                                  shape=(1, 1))
    assert formation(K_v=duplicate).system[1, 1] == 2.0 ** -400


# --- fixtures for the provenance, ledger and code tests ----------------------------------------------

def coupled(n=6, hole=(2, 4, 2, 4), nx=4, ny=4, nz=2, skin='end', case=None, mesh_kwargs=None, box=None,
            model_hook=None, order=2):
    model, grid, removed, footprint = fx.plate(n, thickness=T, hole=hole)
    if model_hook:
        model_hook(model, grid, removed)
    mesh = fx.box_mesh(footprint, T, nx, ny, nz, **(mesh_kwargs or {}))
    prepared = fx.prepare(mesh, order=order)
    if case is None:
        case = fx.pressure_case(removed, model) if skin else LoadCase(name='none')
    box = box or sc.BoxFaces((nx, ny, nz), skin, SIDES)
    return sc.solve_shell_solid_coupled(model, prepared, removed_element_ids=removed, box=box, load_case=case)


def test_reference_fixture_is_accepted():
    result = coupled()
    assert result.certificate['accepted'] and result.sizes['N'] == 1515
    assert np.isfinite(result.shell_displacements.values).all()


# --- section 3.1 ------------------------------------------------------------------------------------

def test_skin_one_by_two_presented_as_two_by_one_refuses():
    refused('COUPLED_MESH_PROVENANCE', coupled, hole=(2, 4, 2, 3), nx=4, ny=2,
            box=sc.BoxFaces((2, 4, 2), 'end', SIDES))


def test_missing_triangle_refuses():
    error = refused('COUPLED_MESH_PROVENANCE', coupled, mesh_kwargs={'drop_facet': ('end', 3)})
    assert error.reason == 'facet_count'


def lifted(dz, where='start', inplane=(0.0, 0.0)):
    x = fx.lattice_coordinates((1 / 3, 2 / 3, 1 / 3, 2 / 3), T, 4, 4, 2)
    k = 0 if where == 'start' else 2
    v = (k * 5 + 2) * 5 + 2   # an interior face vertex
    x[v, 2] += dz
    x[v, :2] += inplane
    return x


def test_lift_of_two_to_minus_24_refuses():
    error = refused('COUPLED_MESH_PROVENANCE', coupled, mesh_kwargs={'coordinates': lifted(2.0 ** -24)})
    assert error.reason == 'plane'


def test_graded_lattice_refuses():
    error = refused('COUPLED_MESH_PROVENANCE', coupled, mesh_kwargs={'coordinates': lifted(0.0, inplane=(1e-6, 0.0))})
    assert error.reason == 'lattice'


def test_start_end_swap_refuses():
    refused('COUPLED_MESH_PROVENANCE', coupled, mesh_kwargs={'group_names': {'start': 'end', 'end': 'start'}})


def test_side_order_must_be_counter_clockwise():
    refused('COUPLED_MESH_PROVENANCE', coupled, box=sc.BoxFaces((4, 4, 2), 'end', ('side_0', 'side_3', 'side_2', 'side_1')))


# --- section 3.2 ------------------------------------------------------------------------------------

def nodal_case(model, removed, loads):
    case = fx.pressure_case(removed, model)
    for nid, vector in loads.items():
        case.nodal_loads[nid] = np.asarray(vector, dtype=float)
    return case


def test_dropped_pair_missing_node_refuses():
    model, grid, removed, _ = fx.plate(6, thickness=T)
    case = nodal_case(model, removed, {999: [0, 0, 5.0, 0, 0, 0], grid[1, 1]: [0, 0, -5.0, 0, 0, 0]})
    error = refused('COUPLED_LOAD_UNCOVERED', coupled, case=case)
    assert error.reason == 'nodal_skipped'


def test_dropped_pair_component_past_dofs_refuses():
    model, grid, removed, _ = fx.plate(6, thickness=T)
    case = nodal_case(model, removed, {grid[1, 1]: [0, 0, 5.0, 0, 0, 0, -5.0]})
    refused('COUPLED_LOAD_UNCOVERED', coupled, case=case)


def test_inactive_dof_refuses():
    class Stub:
        nodal_loads = {1: [1.0, 1.0]}     # the case's records (every yielded record must be one)
        pressure_loads: dict = {}

        def _load_contributions(self, mesh, displacements, scale, *, report_skips):
            yield ('nodal', 1, 0), 10, 1.0
    refused('COUPLED_LOAD_UNCOVERED', sc._ledger, HOLED, Stub(), 6, np.zeros(6))


def test_ledger_binding_requires_bitwise_native_sum():
    class Stub:
        nodal_loads = {1: [1.0, 1.0]}     # the case's records (every yielded record must be one)
        pressure_loads: dict = {}

        def _load_contributions(self, mesh, displacements, scale, *, report_skips):
            yield ('nodal', 1, 0), 0, 0.1
            yield ('nodal', 1, 1), 0, 0.2
    assert sc._ledger(HOLED, Stub(), 1, np.array([0.1 + 0.2]))
    refused('COUPLED_LOAD_UNCOVERED', sc._ledger, HOLED, Stub(), 1, np.array([0.3]))


def test_element_activity_refuses():
    class Activity:
        def __init__(self, ids):
            self.element_ids = np.asarray(ids)

        def load_scales(self, ids):
            return np.ones(len(ids))
    error = refused('COUPLED_LOAD_UNSUPPORTED', coupled,
                    model_hook=lambda m, g, r: m.set_element_activity(Activity(list(m.mesh.elements))))
    assert error.reason == 'element_activity'


@pytest.mark.parametrize('change, reason', [
    (lambda c: setattr(c, 'follower_pressure', True), 'follower_pressure'),
    (lambda c: c.element_loads.__setitem__(1, np.zeros(24)), 'element_loads'),
    (lambda c: setattr(c, 'gravity', np.array([0.0, 0.0, -9.81])), 'gravity'),
    (lambda c: c.pressure_loads.__setitem__(1, np.array([1.0, 2.0])), 'pressure_kind'),
])
def test_unsupported_load_route_refuses(change, reason):
    model, grid, removed, _ = fx.plate(6, thickness=T)
    case = fx.pressure_case(removed, model)
    change(case)
    assert refused('COUPLED_LOAD_UNSUPPORTED', coupled, case=case).reason == reason


def test_skin_rules():
    model, grid, removed, _ = fx.plate(6, thickness=T)
    refused('COUPLED_LOAD_UNSUPPORTED', coupled, skin=None, case=fx.pressure_case(removed, model))
    partial = fx.pressure_case(removed, model)
    partial.pressure_loads.pop(removed[0])
    refused('COUPLED_LOAD_UNSUPPORTED', coupled, case=partial)


# --- section 4: one test per code -----------------------------------------------------------------

def test_code_cut_mismatch_hole_on_model_boundary():
    refused('COUPLED_CUT_MISMATCH', coupled, hole=(0, 2, 2, 4))


def test_code_free_node():
    model, grid, removed, _ = fx.plate(6, thickness=T)
    case = nodal_case(model, removed, {grid[3, 3]: [0, 0, 1.0, 0, 0, 0]})   # the hole's interior node
    refused('COUPLED_FREE_NODE', coupled, case=case)


def test_code_target_removed():
    from anyfesolver import BoundaryCondition

    def hook(model, grid, removed):
        model.add_boundary_condition(BoundaryCondition(name='ghost', node_ids=[12345], dof_constraints={'uz': 0.0}))
    refused('COUPLED_TARGET_REMOVED', coupled, model_hook=hook)


def test_code_mechanism_zero_row():
    S = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert refused('COUPLED_MECHANISM', sc._check_zero_rows, S).reason == 'zero_row'


def test_code_mechanism_solid_rigid_mode_free():
    C = sparse.csr_matrix(np.eye(6)[:5])
    assert refused('COUPLED_MECHANISM', sc._check_solid_rigid_rank, C, np.eye(6)).reason == 'solid_rigid_mode_free'
    assert sc._check_solid_rigid_rank(sparse.csr_matrix(np.eye(6)), np.eye(6)) == 6


def test_code_connector_partition():
    error = refused('COUPLED_CONNECTOR_PARTITION', coupled, nx=1, ny=1)
    assert error.reason == 'segment_columns'


def test_code_connector_group():
    refused('COUPLED_CONNECTOR_GROUP', coupled, box=sc.BoxFaces((4, 4, 2), 'end', ('side_0', 'side_1', 'side_2', 'nope')))


def test_code_connector_rank():
    knots = np.linspace(0.0, 1.0, 52)
    stations = np.linspace(0.0, 1.0, 400)
    assert refused('COUPLED_CONNECTOR_RANK', sc._connector_prechecks, knots, stations, 1.0, 1e-10).reason == 'row_limit'
    sc._connector_prechecks(np.linspace(0.0, 1.0, 51), stations, 1.0, 1e-10)


def test_code_connector_invalid(monkeypatch):
    import anysolid.coupling

    def broken(*args, **kwargs):
        raise ValueError('any builder refusal')
    monkeypatch.setattr(anysolid.coupling, 'build_weak_connector', broken)
    refused('COUPLED_CONNECTOR_INVALID', coupled)


def test_knots_every_second_node():
    assert sc._knots(np.arange(5.0)).tolist() == [0.0, 2.0, 4.0]
    assert sc._knots(np.arange(4.0)).tolist() == [0.0, 2.0, 3.0]


# --- section 5 ------------------------------------------------------------------------------------

def test_basis_lever_arm_gives_two_e_z():
    centre = (Fraction(0), Fraction(0), Fraction(0))
    basis = sc._rigid_basis([(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], ['uy', 'rz'], centre)
    force = [1, 1]   # F = e_y at r = e_x, couple e_z
    resultant = [sum(f * row[k] for f, row in zip(force, basis)) for k in range(6)]
    assert resultant == [0, 1, 0, 0, 0, 2]


def test_basis_pure_nodal_couple_enters_the_moment():
    centre = (Fraction(5), Fraction(-3), Fraction(1, 7))
    basis = sc._rigid_basis([(0.3, 0.1, 0.2)] * 3, ['rx', 'ry', 'rz'], centre)
    assert [list(row) for row in basis] == [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]


def test_basis_refuses_unknown_dof_name():
    refused('COUPLED_TARGET_REMOVED', sc._rigid_basis, [(0.0, 0.0, 0.0)], ['wx'], (Fraction(0),) * 3)


def test_side_resultant_matches_virtual_work():
    result = coupled()
    total = sum(s.force.values for s in result.side_resultants)
    assert abs(total[2] - 1000.0 / 9.0) <= 1e-9 * 1000.0 / 9.0   # the skin load reaches the shell
