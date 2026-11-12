"""Bounded nonlinear Q4/S3 two-body response with independent contact work."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np


def test_clipping_signature_distinguishes_equal_vertex_counts_with_different_edges():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tests" / "fixtures"))
    from d1_current_skin import overlap_incidence, _clip

    source = np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
    lower_left = np.array(((0., 0.), (0.2, 0.), (0., 0.2)))
    upper_right = np.array(((1., 1.), (0.8, 1.), (1., 0.8)))
    first = _clip(source, lower_left)
    second = _clip(source, upper_right)
    assert len(first) == len(second) == 3
    assert (overlap_incidence(source, lower_left, first)
            == overlap_incidence(source, lower_left, np.roll(first, 1, axis=0)))
    assert (overlap_incidence(source, lower_left, first)
            != overlap_incidence(source, upper_right, second))


def test_saved_slide_matches_independent_current_contact_kkt(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(root / "src"), str(root / "packages/results/src"), str(root),
    ))
    completed = subprocess.run(
        [sys.executable, str(root / "tests" / "fixtures"
                                   / "d1_nonlinear_kkt.py")],
        cwd=root, env=environment, capture_output=True, text=True,
        check=False, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    (tmp_path / "d1_kkt_comparison.json").write_text(completed.stdout, encoding="utf-8")
    result = json.loads(completed.stdout)
    assert result["status"] == "development_independent_contact_nonlinear_kkt"
    assert result["active_pressure_modes"] == [3, 8]
    assert result["clipping_incidence_signature"]
    assert result["final_scaled_merit"] <= 1e-9
    assert result["minimum_inactive_gap_m"] >= -1e-9
    assert min(row["minimum_active_force_n"] for row in result["trace"]) > 0
    assert result["maximum_displacement_difference_m_or_rad"] <= 1e-9
    derivative = result["gradient_step_sensitivity"]
    assert derivative["directions"] == 30
    assert derivative["resolved_columns_at_norm_1e_minus_4"] >= 10
    assert derivative["maximum_half_vs_nominal_resolved_relative_difference"] <= 1e-5
    assert derivative["maximum_double_vs_nominal_resolved_relative_difference"] <= 1e-5
    assert derivative["maximum_half_vs_nominal_absolute_difference"] <= 1e-8
    assert derivative["maximum_double_vs_nominal_absolute_difference"] <= 1e-8
    for name in (
        "relative_pressure_l2_difference",
        "relative_nodal_force_l2_difference",
        "relative_integrated_force_difference",
        "control_reaction_relative_l2_difference",
    ):
        assert result[name] <= 1e-6, (name, result[name])


def test_active_contact_work_is_exactly_independent_of_control_dofs():
    """The control-reaction contact route is bitwise zero, for every FD step.

    Locks the KKT control-reaction diagnosis: the active pressure modes'
    source elements are disjoint from the control nodes' elements and the
    control nodes are absent from the target skin, so the independent
    finite-difference control gradient and the production contact force at
    the control dofs are exactly zero. The control reaction is a purely
    structural cancelled sum; its independent-vs-production comparison is
    limited by arithmetic-realization rounding of that sum, not by contact
    work or finite-difference error.
    """
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(root / "src"), str(root / "packages/results/src"), str(root),
    ))
    completed = subprocess.run(
        [sys.executable, str(root / "tests" / "fixtures"
                                   / "d1_control_reaction.py")],
        cwd=root, env=environment, capture_output=True, text=True,
        check=False, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "development_control_reaction_route_probe"
    assert result["active_pressure_modes"] == [3, 8]
    disjointness = result["disjointness"]
    assert disjointness["active_mode_nodes"] == [4, 9]
    assert disjointness["control_nodes"] == [1, 6]
    assert disjointness["active_elements_disjoint_from_control_elements"] is True
    assert disjointness["control_nodes_on_target_skin"] == [False, False]
    zeros = [[0.0, 0.0], [0.0, 0.0]]
    for label, key in (("half", "half_step"), ("nominal", "nominal"),
                       ("double", "double_step")):
        gradient = result["independent_control_gradient_"+key]
        active_gradient = [gradient[index]
                           for index in result["active_pressure_modes"]]
        assert active_gradient == zeros, (label, gradient)
        assert result["active_rows_bitwise_invariance"][label] is True
    assert result["production_contact_force_control_n"] == [0.0, 0.0]
    active_rows = [result["production_gap_gradient_control_columns"][index]
                   for index in result["active_pressure_modes"]]
    assert active_rows == zeros
    assert result["external_control_n"] == [0.0, 0.0]


@pytest.mark.parametrize("script", ["tests/fixtures/d1_nonlinear_kkt.py", "docs/d1_review_support/d1_two_body_displacement_compare.py"])
def test_oracle_source_identity_does_not_attribute_an_enclosing_checkout(monkeypatch,tmp_path,script):
    # Extract this metadata-only helper without running the nonlinear fixture.
    # Its negative cases protect exported/installed-source provenance.
    import ast
    source=Path(__file__).resolve().parents[1]/script
    tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_checkout_commit')
    namespace=dict(Path=Path,subprocess=subprocess)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    project=tmp_path/'source-export';project.mkdir()
    calls=[]
    def git_output(args,**kwargs):
        calls.append(args)
        assert kwargs['cwd']==project
        return str(tmp_path) if '--show-toplevel' in args else 'must-not-be-used'
    monkeypatch.setattr(subprocess,'check_output',git_output)
    assert namespace['_checkout_commit'](project) is None
    assert len(calls)==1
    monkeypatch.setattr(subprocess,'check_output',lambda args,**kw:
        str(project) if '--show-toplevel' in args else 'own-checkout-sha')
    assert namespace['_checkout_commit'](project)=='own-checkout-sha'
    def unavailable(*args,**kw):raise subprocess.CalledProcessError(128,args[0])
    monkeypatch.setattr(subprocess,'check_output',unavailable)
    assert namespace['_checkout_commit'](project) is None
