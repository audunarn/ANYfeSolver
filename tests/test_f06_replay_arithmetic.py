"""Focused regression tests for the F06 cursor-3 beam_nodal_moment replay arithmetic.

The paired field is the full KKT beam nodal residual
``f_int + f_constraint - parameter*external`` sliced to the moment rows of the
retained beam nodes.  The frozen fixture admits no nodal couples, so the
external moment term is exactly zero and each moment row is the exactly
rounded two-term sum of the beam internal moment and, at the joint-attached
node only, the joint port moment.

These tests pin the frozen pilot metric arithmetic and document the saved
cross-host counterexample for the unchanged scientific predicate.  They do
not adjudicate and do not fabricate a pass: the saved Linux replay exceeds
the unchanged 1e-9 normalized paired threshold and is asserted to do so,
while the saved Windows replay passes it.  The saved arrays show that the
entire paired delta is the exact image of one-ulp joint-multiplier and
shell-chart state differences between the two frozen Newton controllers, so
no compensated-summation change inside the residual assembly can alter this
field without changing its representation.

The approved P1 route adds a private deterministic fixed-order binary64
partial-pivot LU reference backend, wired only into the F06 prototype,
control and checkpoint-replay Newton step through the private
``CoupledShellBeamAnalysis._linear_solve`` hook whose default remains
``numpy.linalg.solve``.  The tests below cover that backend directly:
exact small oracles (nonsymmetric, indefinite, mixed magnitudes), the
first-max pivot tie, owned-operand immutability, singular, nonfinite,
invalid, nonfinite-intermediate and complex refusal, backward error
against LAPACK, bitwise determinism, and source-bound checks that the
public default backend is unchanged.
"""

import ast
import os
from pathlib import Path
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "f06_v4_numba"))

import numpy as np
import pytest

from scripts.probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4 import (
    PILOT_FLOORS, REFERENCE_LINEAR_SOLVE_ID, _metric, _reference_linear_solve,
)

PAIRED_THRESHOLD = 1.e-9
MOMENT_FLOOR = 1.e-10

SAVED = {
    "windows": {
        "left_values": [
            [-5.89534931288993e-18, 5.583641188300348e-17, -2.1982199047143602e-17],
            [-4.445228907190568e-18, 5.860112742284151e-17, -2.599374708533997e-17],
            [-1.3755815063409838e-18, 1.791644090032296e-17, -9.161508357502512e-18],
            [-2.473336205982557e-19, 2.6020852139652106e-18, -1.836367429647323e-18],
            [-9.720054805043348e-28, 1.434717680734783e-27, -3.1808603404355785e-24]],
        "right_values": [
            [-5.8817967857338616e-18, 5.57822017743792e-17, -2.1982199047143602e-17],
            [-4.445228907190568e-18, 5.860112742284151e-17, -2.599374708533997e-17],
            [-1.3755815063409838e-18, 1.791644090032296e-17, -9.161508357502512e-18],
            [-2.473336205982557e-19, 2.6020852139652106e-18, -1.836367429647323e-18],
            [-9.719384734894466e-28, 1.5636626162341055e-27, -3.966363301441106e-24]],
        "multipliers_control": [2.531287203924595e-07, 7.490059356189578e-05,
            0.0001500494485535558, -0.00011145159050814541, 0.00014980255581351835,
            0.000278638733997296],
        "multipliers_rebased": [2.531287203924603e-07, 7.490059356189577e-05,
            0.0001500494485535558, -0.00011145159050814541, 0.00014980255581351835,
            0.00027863873399729605],
        "left_norm": 9.045569509078425e-17,
        "right_norm": 9.042135979454056e-17,
        "absolute_difference": 5.587850096404445e-20,
        "normalized_difference": 5.587850096404444e-10,
    },
    "linux": {
        "left_values": [
            [-5.89534931288993e-18, 5.57822017743792e-17, -2.1982199047143602e-17],
            [-4.458781434346637e-18, 5.860112742284151e-17, -2.599374708533997e-17],
            [-1.3721933745519665e-18, 1.791644090032296e-17, -9.161508357502512e-18],
            [-2.439454888092385e-19, 2.588532686809142e-18, -1.836367429647323e-18],
            [-9.720058792444408e-28, 1.457518354072388e-27, -3.465890842975338e-24]],
        "right_values": [
            [-5.8817967857338616e-18, 5.567378155713065e-17, -2.1982199047143602e-17],
            [-4.458781434346637e-18, 5.860112742284151e-17, -2.599374708533997e-17],
            [-1.3721933745519665e-18, 1.791644090032296e-17, -9.161508357502512e-18],
            [-2.439454888092385e-19, 2.588532686809142e-18, -1.836367429647323e-18],
            [-9.716291801499502e-28, 1.6775481132536715e-27, -7.513459604535892e-24]],
        "multipliers_control": [2.531287203924595e-07, 7.490059356189578e-05,
            0.0001500494485535558, -0.00011145159050814541, 0.00014980255581351835,
            0.000278638733997296],
        "multipliers_rebased": [2.531287203924604e-07, 7.490059356189577e-05,
            0.0001500494485535558, -0.00011145159050814541, 0.00014980255581351835,
            0.00027863873399729605],
        "left_norm": 9.042245997923402e-17,
        "right_norm": 9.035473189975967e-17,
        "absolute_difference": 1.092639671480112e-19,
        "normalized_difference": 1.092639671480112e-09,
    },
}


def _two_sum(a, b):
    high = a+b
    v = high-a
    return high, (a-(high-v))+(b-v)


def test_frozen_pilot_metric_reproduces_saved_scalars():
    assert PILOT_FLOORS["moment"] == (MOMENT_FLOOR, "moment")
    for host, row in SAVED.items():
        metric = _metric(row["left_values"], row["right_values"], "moment")
        assert metric["left_norm"] == row["left_norm"], host
        assert metric["right_norm"] == row["right_norm"], host
        assert metric["absolute_difference"] == row["absolute_difference"], host
        assert metric["normalized_difference"] == row["normalized_difference"], host
        assert metric["normalization_scale"] == MOMENT_FLOOR, host


def test_saved_cross_host_verdicts_under_unchanged_threshold():
    # The threshold and floor are unchanged; the saved Windows replay passes
    # and the saved Linux replay fails.  This documents the counterexample;
    # it is not an acceptance dataset.
    assert SAVED["windows"]["normalized_difference"] <= PAIRED_THRESHOLD
    assert SAVED["linux"]["normalized_difference"] > PAIRED_THRESHOLD
    assert SAVED["windows"]["normalized_difference"] == 5.587850096404444e-10
    assert SAVED["linux"]["normalized_difference"] == 1.092639671480112e-09


def test_paired_delta_is_exact_multiplier_ulp_image():
    for host, row in SAVED.items():
        left = np.array(row["left_values"])
        right = np.array(row["right_values"])
        delta = left-right
        # Interior beam rows carry no joint slot and no external couple; they
        # are bitwise identical, so the beam internal assembly is common.
        assert np.array_equal(left[1:4], right[1:4]), host
        lm_delta = (np.array(row["multipliers_control"])
            -np.array(row["multipliers_rebased"]))
        # The attached-node moment difference is the exact image of the
        # one-ulp joint-multiplier differences of the two Newton paths.
        assert lm_delta[1] == 2.**-66, host
        assert lm_delta[5] == -2.**-64, host
        assert delta[0, 0] == -lm_delta[1], host
        if host == "windows":
            assert delta[0, 1] == -lm_delta[5], host
        else:
            # Linux adds one host-dependent shell-chart ulp in the same
            # component on top of the deterministic multiplier image.
            assert delta[0, 1] == -2.*lm_delta[5], host
        # The loaded-tip row contributes five orders below the gate.
        assert float(np.linalg.norm(delta[4])) < 1.e-23, host


def test_slot_addition_is_already_exactly_rounded():
    # The moment rows are assembled as one two-term IEEE addition
    # (beam internal + joint port) minus an exactly zero external couple.
    # A compensated two-term sum rounds back to the same binary64 value, so
    # compensated slot assembly cannot change this field.
    row = SAVED["windows"]
    beam_internal = 1.7e-4
    joint_port = -beam_internal+row["left_values"][0][1]
    plain = beam_internal+joint_port
    high, low = _two_sum(beam_internal, joint_port)
    assert high+low == plain
    rng = np.random.default_rng(1701)
    for _ in range(1000):
        a = float(rng.uniform(-1., 1.))*10.**float(rng.integers(-6, 0))
        b = -a*(1.+float(rng.uniform(-1.e-12, 1.e-12)))
        high, low = _two_sum(a, b)
        assert high+low == a+b


# ---------------------------------------------------------------------------
# Private P1 reference backend: deterministic fixed-order binary64 LU.

STATE_SOURCE = Path(__file__).resolve().parents[1]/"src/anyfesolver/_ge_beam3_shell_joint_state.py"
PROBE_SOURCE = Path(__file__).resolve().parents[1]/"scripts/probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4.py"


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def test_reference_solve_exact_small_oracles():
    # Hand-verified bitwise oracle: nonsymmetric, indefinite (det < 0), and
    # every elimination and substitution operation is exactly rounded here.
    a = np.array([[16., .25], [.015625, -16.]])
    x = np.array([.125, 64.])
    b = a@x
    assert b[0] == 18. and b[1] == -1023.998046875
    np.testing.assert_array_equal(_reference_linear_solve(a, b), x)
    # Nonsymmetric indefinite integer oracle that needs row pivoting.
    a = np.array([[0., 1., 2.], [4., 5., 6.], [7., 8., 10.]])
    x = np.array([1., -2., 3.])
    b = a@x
    assert np.array_equal(b, [4., 12., 21.])
    np.testing.assert_allclose(_reference_linear_solve(a, b), x,
        rtol=1.e-12, atol=1.e-12)
    # Mixed magnitudes: exact power-of-two row scaling of the same system.
    scaled = a*np.array([2.**-30, 2.**30, 1.])[:, None]
    np.testing.assert_allclose(_reference_linear_solve(scaled, scaled@x), x,
        rtol=1.e-12, atol=1.e-12)


def test_reference_solve_first_max_pivot_tie():
    # Equal-magnitude pivot candidates: the first row must win.  Only the
    # first-max tie choice reproduces the exact dyadic solution.
    a = np.array([[1., 3.], [1., 3.+2.**-51]])
    b = np.array([4., 4.])
    np.testing.assert_array_equal(_reference_linear_solve(a, b),
        np.array([4., 0.]))


def test_reference_solve_owned_operands_are_immutable():
    rng = np.random.default_rng(20261012)
    a = rng.standard_normal((40, 40))
    a[np.arange(40), np.arange(40)] += 40.
    b = rng.standard_normal(40)
    a_copy, b_copy = a.copy(), b.copy()
    x = _reference_linear_solve(a, b)
    assert np.array_equal(a, a_copy) and np.array_equal(b, b_copy)
    assert not np.shares_memory(x, a) and not np.shares_memory(x, b)
    # The fixed-order elementwise arithmetic is invariant to operand layout.
    np.testing.assert_array_equal(_reference_linear_solve(np.asfortranarray(a), b), x)
    big = np.zeros((40, 80))
    big[:, ::2] = a
    np.testing.assert_array_equal(_reference_linear_solve(big[:, ::2], b), x)


def test_reference_solve_refuses_singular_systems():
    cases = (np.array([[1., 1.], [2., 2.]]),
             np.array([[0., 1.], [0., 2.]]),
             np.array([[1., 1., 1.], [1., 1., 1.], [0., 0., 1.]]))
    for a in cases:
        b = np.ones(a.shape[0])
        before = a.copy()
        with pytest.raises(ValueError) as raised:
            _reference_linear_solve(a, b)
        assert "singular" in str(raised.value), a.tolist()
        assert np.array_equal(a, before)


def test_reference_solve_refuses_invalid_and_nonfinite_operands():
    cases = (
        (np.array([[np.nan, 0.], [0., 1.]]), np.ones(2), "nonfinite"),
        (np.array([[np.inf, 0.], [0., 1.]]), np.ones(2), "nonfinite"),
        (np.eye(2), np.array([np.nan, 1.]), "nonfinite"),
        (np.eye(2), np.array([1., np.inf]), "nonfinite"),
        (np.eye(2)[:1, :], np.ones(1), "square"),
        (np.zeros((2, 2, 2)), np.ones((2, 2)), "square"),
        (np.zeros((0, 0)), np.zeros(0), "square"),
        (np.eye(2), np.ones(3), "shape"),
        (np.eye(2), np.ones((2, 2, 2)), "shape"),
        (np.eye(2), np.ones((3, 1)), "shape"),
    )
    for matrix, rhs, expected in cases:
        with pytest.raises(ValueError) as raised:
            _reference_linear_solve(matrix, rhs)
        assert expected in str(raised.value)


def test_reference_solve_refuses_nonfinite_intermediate_updates():
    # Finite operands whose Schur complement overflows binary64 during
    # elimination: the unchecked algorithm returns the finite but silently
    # wrong [1., -0.]; the reference solve must refuse (exact, no tolerance).
    a = np.array([[1.e308, -1.e308], [1.e308, 1.e308]])
    b = np.array([1.e308, 5.e307])
    with pytest.raises(ValueError) as raised:
        _reference_linear_solve(a, b)
    assert "nonfinite intermediate elimination update" in str(raised.value)
    # Finite operands whose back-substitution division overflows binary64.
    a = np.array([[1.e-154, 1.e-154], [1.e-154, -1.e-154]])
    b = np.array([1.e308, 0.])
    with pytest.raises(ValueError) as raised:
        _reference_linear_solve(a, b)
    assert "nonfinite intermediate substitution update" in str(raised.value)


def test_reference_solve_refuses_complex_operands():
    # Complex input is refused before the real cast; imaginary parts are
    # never silently discarded (ndarray and plain Python complex alike).
    cases = ((np.array([[1.+2j, 0.], [0., 1.]]), np.ones(2)),
             (np.eye(2), np.array([1.+2j, 1.])),
             ([[1.+2j, 0.], [0., 1.]], np.ones(2)),
             (np.eye(2), [1., 1.+2j]))
    for matrix, rhs in cases:
        with pytest.raises(ValueError) as raised:
            _reference_linear_solve(matrix, rhs)
        assert "complex" in str(raised.value)


def test_reference_solve_backward_error_and_lapack_agreement():
    rng = np.random.default_rng(4242)
    n = 48
    a = rng.standard_normal((n, n))
    a[np.arange(n), np.arange(n)] += 6.*np.abs(a).sum(axis=1)+1.
    a[7, 7] = -a[7, 7]  # indefinite while staying strictly diagonally dominant
    columns = 2.**rng.integers(-30, 31, n)
    a = a*columns[None, :]
    x_true = rng.standard_normal(n)/columns
    b = a@x_true
    x = _reference_linear_solve(a, b)
    backward = float(np.linalg.norm(a@x-b))
    assert backward <= 1.e-12*float(np.linalg.norm(a))*float(np.linalg.norm(x))
    lapack = np.linalg.solve(a, b)
    assert float(np.linalg.norm(x-lapack)) <= 1.e-12*float(np.linalg.norm(lapack))


def test_reference_solve_is_deterministic_across_repeats_and_rhs_layouts():
    rng = np.random.default_rng(606)
    a = rng.standard_normal((30, 30))
    a[np.arange(30), np.arange(30)] += 8.
    b = rng.standard_normal((30, 3))
    block = _reference_linear_solve(a, b)
    assert block.shape == (30, 3)
    np.testing.assert_array_equal(_reference_linear_solve(a, b), block)
    for column in range(3):
        np.testing.assert_array_equal(_reference_linear_solve(a, b[:, column]),
            block[:, column])


def test_coupled_analysis_default_hook_signature_is_unchanged():
    from anyfesolver._ge_beam3_shell_joint_state import CoupledShellBeamAnalysis
    assert CoupledShellBeamAnalysis.__init__.__kwdefaults__.get(
        "_linear_solve") is None
    assert isinstance(REFERENCE_LINEAR_SOLVE_ID, str) and REFERENCE_LINEAR_SOLVE_ID
    assert "PARTIAL_PIVOT_LU_BINARY64" in REFERENCE_LINEAR_SOLVE_ID


def test_state_step_routes_through_the_private_hook():
    tree = ast.parse(STATE_SOURCE.read_bytes().decode("ascii"))
    owner = next(node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CoupledShellBeamAnalysis")
    init = next(node for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    names = [argument.arg for argument in init.args.kwonlyargs]
    assert "_linear_solve" in names
    default = dict(zip(names, init.args.kw_defaults))["_linear_solve"]
    assert isinstance(default, ast.Constant) and default.value is None
    assigns = [node for node in ast.walk(init) if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_linear_solve"]
    assert len(assigns) == 1
    assert _dotted(assigns[0].value) == "_linear_solve"
    step = next(node for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_step")
    selections = [node for node in ast.walk(step) if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "solve"]
    assert len(selections) == 1
    selection = selections[0].value
    assert isinstance(selection, ast.IfExp)
    assert ast.unparse(selection.test) == "self._linear_solve is None"
    assert _dotted(selection.body) == "np.linalg.solve"
    assert _dotted(selection.orelse) == "self._linear_solve"
    calls = [node for node in ast.walk(step) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "solve"]
    assert len(calls) == 1


def test_probe_wires_the_reference_backend_and_records_provenance():
    tree = ast.parse(PROBE_SOURCE.read_bytes().decode("utf-8"))
    functions = {node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)}
    solver = functions["_reference_linear_solve"]
    body = ast.unparse(solver)
    assert "np.argmax" in body and "np.abs" in body
    assert "np.linalg" not in body and ".dot(" not in body
    assert not any(isinstance(node, ast.MatMult) for node in ast.walk(solver))
    prototype = next(node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OwnedRebasePrototype")
    calls = [node for node in ast.walk(prototype) if isinstance(node, ast.Call)
        and _dotted(node.func) == "CoupledShellBeamAnalysis"]
    assert len(calls) == 1
    keywords = {keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[0].keywords}
    assert keywords.get("_linear_solve") == "_reference_linear_solve"
    text = PROBE_SOURCE.read_bytes().decode("utf-8")
    # The unchanged V1 payload carries no backend key: the strict adjudicator
    # allows only effective_environment as an extra field.  The backend stays
    # bound by the explicit owner-factory selection above and by the full
    # probe source_sha256, which hashes this file and the ID constant in it.
    assert "linear_solve_backend" not in text
    assert "scripts/probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4.py" in ast.unparse(
        functions["_pilot_sources"])
    assert "source_sha256=_pilot_sources()" in ast.unparse(functions["measure_pilot"])
