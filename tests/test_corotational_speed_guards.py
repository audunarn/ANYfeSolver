"""Regression tests for the corotational shell speed work (2026-10-23).

Two classes of overhead were removed without changing any value:

* post-observation authority rechecks after reads that provably run no
  foreign code (plain instance-dict attributes, exact floats, exact numeric
  ndarrays).  Every observation that can run foreign code is still rechecked.
* redundant Python finiteness screens in front of the compiled contraction
  kernels (the kernels refuse nonfinite input themselves and the reference
  exception is still raised on refusal).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from anymaterial import IsotropicMaterial

from anyfesolver import _compensated_contractions as cc
from anyfesolver import elements as el
from anyfesolver.corotational import rotation_matrix_from_vector


class _Recorder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


class _Plain:
    def __init__(self) -> None:
        self.value = 1.5


class _DictProperty:
    """``__dict__`` is a property: reading it runs foreign code."""

    def __init__(self) -> None:
        self.value = 1.0

    @property
    def __dict__(self):  # type: ignore[override]
        return {"value": 2.0}


class _SpoofedGetattribute:
    """Class access shows object.__getattribute__, instance access another one."""

    class _Spoof:
        def __get__(self, instance, owner=None):
            if instance is None:
                return object.__getattribute__
            return lambda name: 3.0

    __getattribute__ = _Spoof()  # type: ignore[assignment]

    def __init__(self) -> None:
        object.__setattr__(self, "value", 1.0)


class _EqKey:
    """Non-str key with the hash of ``value``; its __eq__ is foreign code."""

    def __init__(self, log) -> None:
        self.log = log

    def __hash__(self) -> int:
        return hash("value")

    def __eq__(self, other) -> bool:
        self.log.append("eq")
        return False


def _owner_with_eq_key(log):
    owner = _Plain()
    owner.__dict__[_EqKey(log)] = 0.0
    return owner


@pytest.mark.parametrize(
    "make",
    [
        _Plain,
        _DictProperty,
        _SpoofedGetattribute,
        lambda: _owner_with_eq_key([]),
    ],
    ids=["plain", "dict-property", "spoofed-getattribute", "eq-key-collision"],
)
def test_attribute_reads_are_always_rechecked(make):
    """Every external attribute read is followed by a recheck (no purity shortcut)."""

    recorder = _Recorder()
    el._guarded_observe_attribute(make(), "value", post_observation=recorder)
    assert recorder.calls == 1


def test_guarded_observe_attribute_still_propagates_guard_failures():
    class Hostile:
        @property
        def value(self) -> float:
            return 1.0

    def failing() -> None:
        raise RuntimeError("authority changed")

    with pytest.raises(RuntimeError, match="authority changed"):
        el._guarded_observe_attribute(Hostile(), "value", post_observation=failing)


def test_guarded_float_and_array_recheck_when_conversion_can_run_foreign_code():
    class Floaty:
        def __float__(self) -> float:
            return 2.0

    recorder = _Recorder()
    assert el._guarded_float(2.5, post_observation=recorder) == 2.5
    assert recorder.calls == 0
    assert el._guarded_float(Floaty(), post_observation=recorder) == 2.0
    assert recorder.calls == 1

    recorder = _Recorder()
    array = np.arange(4.0)
    owned = el._guarded_array(array, post_observation=recorder)
    assert recorder.calls == 0 and owned is not array and np.array_equal(owned, array)
    recorder = _Recorder()
    el._guarded_array(np.array([1.0, 2.0], dtype=object), post_observation=recorder)
    assert recorder.calls == 2
    recorder = _Recorder()
    el._guarded_array([1.0, 2.0], post_observation=recorder)
    assert recorder.calls == 2


def test_isotropic_material_fingerprint_rechecks_after_every_read():
    material = IsotropicMaterial(name="steel", elastic_modulus=210e9, poisson_ratio=0.3)
    recorder = _Recorder()
    first = el._shell_elastic_material_cache_fingerprint(material, recorder)
    assert recorder.calls >= 4  # symmetry + three attribute reads; exact floats skip
    assert first == el._shell_elastic_material_cache_fingerprint(material, None)


@pytest.mark.parametrize("use_native", [True, False])
def test_contractions_refuse_nonfinite_and_overflow_like_the_reference(monkeypatch, use_native):
    if use_native and not cc._USE_NATIVE:
        pytest.skip("compiled contraction kernels unavailable")
    monkeypatch.setattr(cc, "_USE_NATIVE", use_native)
    good = np.arange(6.0).reshape(2, 3)
    vec = np.array([1.0, 2.0, 3.0])
    for bad in (math.nan, math.inf, -math.inf):
        left = good.copy()
        left[1, 2] = bad
        with pytest.raises(ValueError, match="finite"):
            cc.product(left, vec)
        right = vec.copy()
        right[0] = bad
        with pytest.raises(ValueError, match="finite"):
            cc.product(good, right)
        with pytest.raises(ValueError, match="finite"):
            cc.sum_arrays((np.array([1.0, bad]), np.array([1.0, 2.0])))
    huge = np.array([[1.0e200, 1.0]])
    with pytest.raises(OverflowError):
        cc.product(huge, np.array([1.0e200, 1.0]))
    with pytest.raises(OverflowError):
        cc.sum_arrays((np.array([1.7e308]), np.array([1.7e308])))
    with pytest.raises(ValueError, match="shape"):
        cc.sum_arrays((np.zeros(2), np.zeros(3)))


def test_native_and_numpy_contractions_are_bitwise_identical(monkeypatch):
    if not cc._USE_NATIVE:
        pytest.skip("compiled contraction kernels unavailable")
    rng = np.random.default_rng(20261023)
    for _ in range(40):
        a = rng.standard_normal((5, 7)) * 2.0 ** rng.integers(-30, 30, (5, 7))
        b = rng.standard_normal((7, 4)) * 2.0 ** rng.integers(-30, 30, (7, 4))
        v = rng.standard_normal(7)
        terms = [rng.standard_normal((3, 3)) * 1.0e6 for _ in range(4)]
        terms[1] = -terms[0] * (1.0 + 2.0 ** -40)
        results = []
        for use in (True, False):
            monkeypatch.setattr(cc, "_USE_NATIVE", use)
            results.append(
                (
                    cc.product(a, b).tobytes(),
                    cc.product(a, v).tobytes(),
                    cc.sum_arrays(terms).tobytes(),
                )
            )
        assert results[0] == results[1]


def test_rotation_vector_branches_are_the_same_rotation_but_not_equivariant():
    """Why output rotation DOFs are not wrapped into |theta| <= pi.

    ``theta`` and ``theta - 2*pi*theta/|theta|`` are the same rotation, but the
    solver updates rotation DOFs additively.  The two representations then
    diverge after any increment, so a restart/continuation from a wrapped
    state would not continue the same solution.
    """

    theta = np.array([0.4, -5.9, 2.2])
    wrapped = theta - 2.0 * math.pi * theta / np.linalg.norm(theta)
    assert np.abs(rotation_matrix_from_vector(theta) - rotation_matrix_from_vector(wrapped)).max() < 1.0e-14
    delta = np.array([0.03, -0.02, 0.05])
    gap = np.abs(
        rotation_matrix_from_vector(theta + delta) - rotation_matrix_from_vector(wrapped + delta)
    ).max()
    assert gap > 1.0e-3


def _simply_supported_plate(n: int, thickness: float, pressure: float):
    import anyfesolver as fs
    from anyfesolver.boundary import BoundaryCondition, LoadCase
    from anyfesolver.elements import create_shell_element

    model = fs.FEModel("plate")
    model.add_material("mat", 210e9, 0.3, density=7850.0)
    for j in range(n + 1):
        for i in range(n + 1):
            model.add_node(1 + j * (n + 1) + i, i / n, j / n, 0.0)
    for j in range(n):
        for i in range(n):
            v = 1 + j * (n + 1) + i
            e = 1 + j * n + i
            model.add_element(
                e,
                create_shell_element(e, [v, v + 1, v + n + 2, v + n + 1], "mat", thickness=thickness),
            )
    edge = [
        1 + j * (n + 1) + i
        for j in range(n + 1)
        for i in range(n + 1)
        if i in (0, n) or j in (0, n)
    ]
    model.add_boundary_condition(BoundaryCondition("ss", edge, {"ux": 0.0, "uy": 0.0, "uz": 0.0}))
    load = LoadCase("p")
    for element in model.mesh.elements:
        load.add_pressure_load(element, pressure)
    model.apply_boundary_conditions()
    return model, load


def test_one_step_and_five_step_corotational_plate_agree_as_rotations():
    """Rotation DOF values may differ by multiples of 2*pi; the rotations do not."""

    import anyfesolver as fs

    model, load = _simply_supported_plate(4, 0.001, 5.0e4)
    states = {}
    for steps in (1, 5):
        result = fs.solve_static_nonlinear(
            model, load, num_steps=steps, max_iterations=25, kinematics="corotational"
        )
        assert result.converged
        states[steps] = np.asarray(result.displacements).reshape(-1, 6)
    one, five = states[1], states[5]
    assert np.abs(one[:, :3] - five[:, :3]).max() < 1.0e-8
    gap = max(
        np.abs(rotation_matrix_from_vector(a) - rotation_matrix_from_vector(b)).max()
        for a, b in zip(one[:, 3:], five[:, 3:])
    )
    assert gap < 1.0e-6
