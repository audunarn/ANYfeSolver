"""Bounded experimental descriptor controls; independent kinetic/QZ oracle separate."""
from dataclasses import replace
import json

import numpy as np
import pytest

from anyfesolver import b3_ge, stiffened_shell as public
from anyfesolver.control import CancellationToken
from anyfesolver import _stiffened_shell_ge_reference_modal as modal


def _definition(inertia=None):
    e, nu, rho, width, depth = 210e9, .3, 7850., .1, .1
    g = e/(2*(1+nu)); area = width*depth
    iy, iz = width*depth**3/12., depth*width**3/12.
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*area, g*.8*area, g*.8*area,
                 g*(iy+iz), e*iy, e*iz)), np.eye(6), 1e20, 1e9)
    # Explicit rectangular physical section about its native reference axis,
    # with an off-axis mass center. No stiffness-derived mass.
    if inertia is None:
        m = rho*area; c = np.array((0., .03, -.02))
        cross = np.array(((0., -c[2], c[1]), (c[2], 0., -c[0]),
                          (-c[1], c[0], 0.)))
        central = rho*np.diag((iy+iz, iy, iz))
        inertia = np.block([[m*np.eye(3), -m*cross],
                            [m*cross, central-m*cross@cross]])
    beam = b3_ge.define_beam("b3-ge", 17, (101, 102, 103),
        np.array(((0., 0., .1), (.5, 0., .1), (1., 0., .1))),
        np.tile(np.eye(3), (3, 1, 1)), section, inertia)
    shell = public.ShellPanel((1, 2, 3, 4),
        ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)),
        ((1, 2, 3, 4),), .02, e, nu, rho)
    return public.LinearDefinition(public.B3_GE_E4_PL_STIFFENED_SHELL_V1,
        shell, (beam,), (public.PointConnection(1, 101), public.PointConnection(2, 103)),
        tuple(public.FixedCoordinate("shell", node, axis)
              for node in (1, 4) for axis in range(6)))


def _owner(definition=None):
    return public.create_analysis(definition=definition or _definition())


def _pristine(owner, identity, capabilities, operators):
    assert owner._identity == identity
    assert json.dumps(owner.capabilities, sort_keys=True) == capabilities
    assert owner._generation == 0 and owner._accepted is None
    assert owner._finite_driver is None and owner._linear_factor is None
    assert not owner._writing
    for current, saved in zip((owner._K, owner._C, owner._A), operators):
        for name in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(current, name), getattr(saved, name))


def _snapshot(owner):
    return (owner._identity, json.dumps(owner.capabilities, sort_keys=True),
            tuple(a.copy() for a in (owner._K, owner._C, owner._A)))


def test_modal_operation_owns_modes_reuses_mass_and_preserves_static_owner():
    owner = _owner(); control = _owner()
    try:
        before = _snapshot(owner)
        assert owner.reference_modal_capabilities["eligible"]
        result = owner.modes(public.GEReferenceModesRequest(3, 1.))
        assert type(result) is public.GEReferenceModesResult
        assert result.status == "completed", result.failure_reason
        assert result.factorization_count == 1 and result.solve_count > 0
        assert len(result.full_coordinates) == len(result.multiplier_modes) == 3
        assert max(result.equation_residuals) <= 1e-8
        assert max(result.constraint_residuals) <= 1e-9
        assert max(result.physical_equation_residuals) <= 1e-8
        mass = owner._reference_modal_mass
        for values in (mass[0].data, mass[0].indices, mass[0].indptr,
                       *result.full_coordinates, *result.multiplier_modes):
            assert not values.flags.writeable
            with pytest.raises(ValueError): values.setflags(write=True)
        again = owner.modes(public.GEReferenceModesRequest(2, 2.))
        assert again.status == "completed", again.failure_reason
        assert owner._reference_modal_mass is mass
        assert again.mass_identity == result.mass_identity
        _pristine(owner, *before)
        load = (public.NodalAction("beam", 103, (0., 0., 10., 0., 0., 0.)),)
        a, b = owner.solve_linear(load), control.solve_linear(load)
        assert a.status == b.status == "completed"
        assert owner.checkpoint() == control.checkpoint()
        assert not owner.reference_modal_capabilities["eligible"]
        with pytest.raises(ValueError, match="fresh virgin"):
            owner.modes(public.GEReferenceModesRequest())
    finally:
        owner.close(); control.close()


@pytest.mark.parametrize("stage", ["start", "beam_mass", "iteration", "publish"])
def test_cancelled_modal_operation_never_publishes_an_accepted_static_state(stage):
    owner = _owner()
    class AtStage(CancellationToken):
        def raise_if_cancelled(self, current=""):
            if current == "ge_reference_modes."+stage: self.cancel("test boundary")
            super().raise_if_cancelled(current)
    try:
        before = _snapshot(owner)
        result = owner.modes(public.GEReferenceModesRequest(), cancellation_token=AtStage())
        assert result.status == "cancelled"
        assert not result.full_coordinates
        _pristine(owner, *before)
    finally: owner.close()


@pytest.mark.parametrize("error,structured", [
    (RuntimeError("Sparse solve produced NaN/Inf values"), True),
    (RuntimeError("programming error"), False), (MemoryError("allocation"), False)])
def test_backend_errors_are_not_disguised_as_numerical_failures(monkeypatch, error, structured):
    owner = _owner()
    class Broken:
        status = "ok"; factorization_count = 1; solve_count = 0
        def solve(self, _): raise error
    monkeypatch.setattr(modal, "factorize_cached", lambda *_: Broken())
    try:
        before = _snapshot(owner)
        if structured:
            result = owner.modes(public.GEReferenceModesRequest())
            assert result.status == "failed" and str(error) in result.failure_reason
        else:
            with pytest.raises(type(error)) as caught:
                owner.modes(public.GEReferenceModesRequest())
            assert caught.value is error
        _pristine(owner, *before)
    finally: owner.close()


def test_callback_reentry_is_rejected_without_closing_or_mutating_owner():
    owner = _owner()
    class Reentrant(CancellationToken):
        called = False
        def raise_if_cancelled(self, current=""):
            if not self.called:
                self.called = True
                with pytest.raises(RuntimeError, match="reentrant"):
                    owner.close()
            super().raise_if_cancelled(current)
    try:
        before = _snapshot(owner)
        result = owner.modes(public.GEReferenceModesRequest(1), cancellation_token=Reentrant())
        assert result.status == "completed", result.failure_reason
        _pristine(owner, *before)
    finally: owner.close()


@pytest.mark.parametrize("inertia", [np.diag((1., 1., 1., 3., 1., 1.)),
                                      np.diag((1., 2., 1., 1., 1., 1.)),
                                      np.eye(6)+.1*(np.eye(6, k=3)+np.eye(6, k=-3))])
def test_new_physical_spatial_admission_does_not_change_legacy_capture(inertia):
    owner = _owner(_definition(inertia))
    try:
        before = _snapshot(owner)
        assert not owner.reference_modal_capabilities["eligible"]
        with pytest.raises(ValueError, match="physical spatial inertia"):
            owner.modes(public.GEReferenceModesRequest())
        _pristine(owner, *before)
    finally: owner.close()


def test_partial_physical_supports_are_outside_initial_scope():
    owner = _owner(replace(_definition(), fixed=(public.FixedCoordinate("shell", 1, 0),)))
    try:
        assert not owner.reference_modal_capabilities["eligible"]
        with pytest.raises(ValueError, match="six physical rigid fields"):
            owner.modes(public.GEReferenceModesRequest())
    finally: owner.close()


@pytest.mark.parametrize("separately_supported", [False, True])
def test_disconnected_native_component_rejected_before_any_eigensolve(monkeypatch, separately_supported):
    definition = _definition()
    element, inertia = definition.beams[0].instantiate()
    isolated = b3_ge.define_beam("b3-ge", 18, (201, 202, 203),
        np.array(((2., 2., .1), (2.5, 2., .1), (3., 2., .1))),
        np.tile(np.eye(3), (3, 1, 1)), element.section, inertia)
    fixed = definition.fixed
    if separately_supported:
        fixed += tuple(public.FixedCoordinate("beam", 201, axis) for axis in range(6))
    owner = _owner(replace(definition, beams=definition.beams+(isolated,), fixed=fixed))
    def forbidden(*args, **kwargs):
        raise AssertionError("eigensolve must not run for disconnected native component")
    monkeypatch.setattr(modal, "eigs", forbidden)
    try:
        before = _snapshot(owner)
        descriptor = owner.reference_modal_capabilities
        assert not descriptor["eligible"] and "connected" in descriptor["scope"]
        with pytest.raises(ValueError, match="every native beam component linked"):
            owner.modes(public.GEReferenceModesRequest())
        assert owner._reference_modal_mass is None
        _pristine(owner, *before)
    finally: owner.close()


@pytest.mark.parametrize("kind", ["scalar_mass", "parallel_axis", "central_second"])
def test_finite_spd_inertia_overflow_fails_with_representability_error(kind):
    # These finite, bitwise symmetric SPD kinetic tensors remain capturable by
    # the legacy definition. New physical admission must stop before passing
    # overflowed intermediates into SVD/eigen evaluation.
    if kind == "scalar_mass":
        inertia = np.eye(6)*1e308
    elif kind == "parallel_axis":
        h = np.array((0., .1, 0.))
        cross = np.array(((0., -h[2], h[1]), (h[2], 0., -h[0]),
                          (-h[1], h[0], 0.)))
        inertia = np.block([[1e-200*np.eye(3), -cross],
                            [cross, 1e308*np.eye(3)]])
    else:
        inertia = np.diag((1., 1., 1., 1e308, 1e308, 1e308))
    definition = _definition(inertia)
    _, captured = definition.beams[0].instantiate()
    np.testing.assert_array_equal(captured, inertia)
    with pytest.raises(ValueError, match="not representable"):
        modal._physical_spatial_inertia(captured)


@pytest.mark.parametrize("kwargs", [{"count": True}, {"count": 0},
    {"shift_hz": float("nan")}, {"shift_hz": 0.}, {"max_iterations": False}])
def test_strict_modal_request(kwargs):
    with pytest.raises(ValueError): public.GEReferenceModesRequest(**kwargs)


def test_typed_request_and_closed_owner_fail_explicitly():
    owner = _owner()
    try:
        with pytest.raises(ValueError, match="typed"):
            owner.modes({"count": 3})
    finally: owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.modes(public.GEReferenceModesRequest())
