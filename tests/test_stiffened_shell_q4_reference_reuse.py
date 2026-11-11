"""Reference-only Q4 reuse against the unchanged full constitutive route."""
from dataclasses import replace
from contextlib import ExitStack

import numpy as np
import pytest

from anymaterial import DNVC208MaterialCurve
from anyfesolver import stiffened_shell as ss
from anyfesolver import _stiffened_shell_ge_experimental as ge
from anyfesolver._stiffened_shell_q4_reference import elastic_reference_response
from anyfesolver.corotational import corotational_element_response
from anyfesolver.corotational_performance import (
    corotational_performance_status, reset_corotational_performance_status,
)
from anyfesolver.fe_core import FEMesh
from test_stiffened_shell_ge_experimental import _panel


def _local(case="rectangle", *, j2=False):
    xyz = np.array(((0., 0., 0.), (2., 0., 0.), (2., 1., 0.), (0., 1., 0.)))
    thickness, order = .01, (1, 2, 3, 4)
    if case == "translated":
        xyz += (1e5, -2e5, .3)
    elif case == "thin":
        thickness = .0005
    elif case == "cyclic":
        order = (2, 3, 4, 1)
    elif case == "small_units":
        xyz *= .001
        thickness *= .001
    elif case == "large_units":
        xyz *= 1000.
        thickness *= 1000.
    elif case == "skew":
        xyz = np.array(((0., 0., 0.), (2., .2, 0.), (1.8, 1., 0.), (-.2, .8, 0.)))
    elif case == "rotated":
        c, s = np.cos(.47), np.sin(.47)
        xyz = xyz @ np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.))).T
    elif case == "near_planar":
        xyz[3, 2] = 1e-13
    curve = DNVC208MaterialCurve(80e6, 85e6, 100e6, .002, .02, 280e6, .18) if j2 else None
    panel = ss.ShellPanel((1, 2, 3, 4), tuple(map(tuple, xyz)), (order,),
                         thickness, 210e9, .3, 7850., j2_curve=curve)
    model, _, _, _, pieces = ge._shell_reference(panel)
    eid, element, _, origin = pieces[0]
    return model, eid, element, origin


def _full(model, eid, element, origin):
    return corotational_element_response(model, eid, element, np.zeros(24), True,
        committed_state=origin, num_layers=3, tangent_mode="consistent")


@pytest.mark.parametrize("case", (
    "rectangle", "translated", "thin", "cyclic", "small_units", "large_units",
))
def test_cold_reference_matches_full_work_operator_and_rigid_modes(case):
    # Separate cold models avoid hiding preparation or cache differences.
    model, eid, element, origin = _local(case)
    oracle = _full(*_local(case))
    response = elastic_reference_response(model, eid, element, origin)
    assert response is not None
    force, tangent, candidate = response
    xyz = np.array([model.mesh.get_node(node).coords() for node in element.node_ids])
    length = np.linalg.norm(np.ptp(xyz, axis=0))
    scale = np.tile((length, length, length, 1., 1., 1.), 4)
    expected = scale[:, None]*oracle[1]*scale[None, :]
    actual = scale[:, None]*tangent*scale[None, :]
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) <= 1e-10
    assert np.linalg.norm(tangent-oracle[1])/np.linalg.norm(oracle[1]) <= 1e-10
    np.testing.assert_array_equal(force, oracle[0])
    basis = np.zeros((24, 6))
    for i, arm in enumerate(xyz-xyz.mean(axis=0)):
        basis[6*i:6*i+3, :3] = np.eye(3)
        for j, axis in enumerate(np.eye(3)):
            basis[6*i:6*i+3, 3+j] = np.cross(axis, arm)
            basis[6*i+3:6*i+6, 3+j] = axis
    assert np.linalg.norm(tangent@basis)/(np.linalg.norm(tangent)*np.linalg.norm(basis)) <= 1e-9
    for name in origin:
        np.testing.assert_array_equal(candidate[name], oracle[2][name])
        assert not np.shares_memory(candidate[name], origin[name])
        candidate[name].flat[0] = 1.
        assert not np.any(origin[name])
    # Returned tangent is owned independently of the guarded public view.
    view = element.compute_stiffness_matrix(model.mesh, model.get_material("shell"))
    assert not np.shares_memory(tangent, view)
    tangent[:] = 0.
    assert np.any(view)


@pytest.mark.parametrize("case", ("rectangle", "skew", "rotated"))
def test_j2_and_nonexact_zero_chart_keep_full_response(case):
    assert elastic_reference_response(*_local(case, j2=True)) is None
    if case != "rectangle":
        assert elastic_reference_response(*_local(case)) is None


@pytest.mark.parametrize("change", ("plastic", "alpha", "nan", "shape", "extra"))
def test_nonvirgin_or_unknown_state_is_not_reused(change):
    model, eid, element, origin = _local()
    if change == "plastic":
        origin["plastic_strain"][0, 0] = 1e-20
    elif change == "alpha":
        origin["alpha"][0] = 1e-20
    elif change == "nan":
        origin["alpha"][0] = np.nan
    elif change == "shape":
        origin["alpha"] = np.zeros(1)
    else:
        origin["other_history"] = np.zeros(1)
    before = {name: value.copy() for name, value in origin.items()}
    assert elastic_reference_response(model, eid, element, origin) is None
    for name in origin:
        np.testing.assert_array_equal(origin[name], before[name])


def test_reference_rotation_diagnostics_match_full_route():
    reset_corotational_performance_status()
    _full(*_local())
    baseline = corotational_performance_status()
    reset_corotational_performance_status()
    assert elastic_reference_response(*_local()) is not None
    candidate = corotational_performance_status()
    for key in ("force_block_rotations", "tangent_block_rotations", "dense_consistent_rotations"):
        assert baseline[key] == candidate[key]
    assert candidate["force_block_rotations"] == candidate["dense_consistent_rotations"] == 1


def test_warped_geometry_keeps_full_response():
    model, eid, element, origin = _local()
    model.mesh.set_node_coordinates(4, 0., 1., .01)
    assert elastic_reference_response(model, eid, element, origin) is None


def test_admitted_near_planar_geometry_uses_full_owner_response(monkeypatch):
    assert elastic_reference_response(*_local("near_planar")) is None
    definition = _panel()
    xyz = list(definition.shell.coordinates_m)
    xyz[4] = (*xyz[4][:2], 1e-13)
    definition = replace(definition, shell=replace(definition.shell, coordinates_m=tuple(xyz)))
    original = ge.corotational_element_response
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(ge, "corotational_element_response", observed)
    owner = ss.create_analysis(definition=definition)
    try:
        assert len(calls) == 2
        result = owner.solve_linear((ss.NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),))
        assert result.status == "completed"
    finally:
        owner.close()


def test_owner_dispatch_reuses_elastic_but_evaluates_j2(monkeypatch):
    original = ge.corotational_element_response
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(ge, "corotational_element_response", observed)
    definition = _panel()
    owner = ss.create_analysis(definition=definition)
    owner.close()
    assert calls == []
    curve = DNVC208MaterialCurve(80e6, 85e6, 100e6, .002, .02, 280e6, .18)
    owner = ss.create_analysis(definition=replace(definition,
        shell=replace(definition.shell, j2_curve=curve)))
    owner.close()
    assert len(calls) == 2


def test_warm_reuse_keeps_revision_and_provider_guards(monkeypatch):
    model, eid, element, origin = _local()
    first = elastic_reference_response(model, eid, element, origin)
    components = element._qualified_components
    model.mesh.bump_revision("material")
    assert element._qualified_components is None
    second = elastic_reference_response(model, eid, element, origin)
    assert element._qualified_components is not components
    np.testing.assert_array_equal(first[1], second[1])
    original = FEMesh.get_node
    reached = []

    def changed(mesh, node_id):
        reached.append(node_id)
        return original(mesh, node_id)

    with monkeypatch.context() as patcher:
        patcher.setattr(FEMesh, "get_node", changed)
        with pytest.raises(ValueError, match="builtin mesh class authority changed"):
            elastic_reference_response(model, eid, element, origin)
    assert reached == []
    np.testing.assert_array_equal(
        elastic_reference_response(model, eid, element, origin)[1], first[1])


def test_owner_finite_history_and_restart_match_forced_full_compilation(monkeypatch):
    with ExitStack() as cleanup:
        _check_owner_history(monkeypatch, cleanup)


def _check_owner_history(monkeypatch, cleanup):
    definition = _panel()
    direct = ss.create_analysis(definition=definition)
    cleanup.callback(direct.close)
    with monkeypatch.context() as patcher:
        patcher.setattr(ge, "elastic_reference_response", lambda *args: None)
        oracle = ss.create_analysis(definition=definition)
    cleanup.callback(oracle.close)
    actions = (ss.NodalAction("beam", 105, (10., 3., 100., 0., 0., 0.)),)
    accepted = direct.solve(actions, (.5, 1.))
    expected = oracle.solve(actions, (.5, 1.))
    assert accepted.accepted_factors == expected.accepted_factors == (.5, 1.)
    for field in ("shell_displacement", "beam_positions", "beam_frames",
                  "cell_frames", "cell_resultants", "multipliers", "support_reactions"):
        a, b = getattr(accepted, field), getattr(expected, field)
        assert np.linalg.norm(a-b) <= 1e-10*max(np.linalg.norm(b), 1e-8)
    saved = direct.checkpoint()
    resumed = ss.create_analysis(definition=definition)
    cleanup.callback(resumed.close)
    resumed.restore(saved)
    assert resumed.checkpoint() == saved
    with monkeypatch.context() as patcher:
        patcher.setattr(ge, "elastic_reference_response", lambda *args: None)
        oracle_restart = ss.create_analysis(definition=definition)
    cleanup.callback(oracle_restart.close)
    oracle_restart.restore(saved)
    continued = direct.solve(actions, (1.5,))
    restarted = resumed.solve(actions, (1.5,))
    expected_continued = oracle.solve(actions, (1.5,))
    old_restart = oracle_restart.solve(actions, (1.5,))
    for field in ("shell_displacement", "beam_positions", "beam_frames",
                  "cell_frames", "cell_resultants", "multipliers", "support_reactions"):
        np.testing.assert_array_equal(getattr(continued, field), getattr(restarted, field))
        a = getattr(continued, field)
        for b in (getattr(expected_continued, field), getattr(old_restart, field)):
            assert np.linalg.norm(a-b) <= 1e-10*max(np.linalg.norm(b), 1e-8)
    assert direct.checkpoint() == resumed.checkpoint()


@pytest.mark.parametrize("policy", ("algorithmic", "mixed_s3"))
def test_other_shell_routes_do_not_use_reference_reuse(monkeypatch, policy):
    definition = _panel()
    if policy == "algorithmic":
        from anyfesolver._stiffened_shell_q4_algorithmic_pullback import POLICY
        definition = replace(definition, q4_pullback_policy=POLICY)
    else:
        definition = replace(definition, shell=replace(definition.shell,
            q4_elements=(), s3_elements=((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5))))

    def forbidden(*args):
        raise AssertionError("other shell route reached historical Q4 reuse")

    monkeypatch.setattr(ge, "elastic_reference_response", forbidden)
    owner = ss.create_analysis(definition=definition)
    result = owner.solve_linear((ss.NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),))
    assert result.status == "completed"
