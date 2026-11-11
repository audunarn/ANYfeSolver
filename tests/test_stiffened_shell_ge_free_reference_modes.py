"""Explicit free-free lifecycle/admission; root owns independent complete QZ oracle."""
from dataclasses import replace
import numpy as np
import pytest

from anyfesolver import stiffened_shell as public
from anyfesolver import _stiffened_shell_ge_reference_modal as modal
from anyfesolver.control import CancellationToken
from test_stiffened_shell_ge_reference_modal import _definition


def _free_definition():
    return replace(_definition(), fixed=(),
                   modal_boundary_policy=public.GE_FREE_FREE_MODAL_ONLY_V1)


def _pristine(owner):
    assert owner._generation == 0 and owner._accepted is None
    assert owner._finite_driver is None and owner._linear_factor is None
    assert not owner._writing and not owner._closed


def test_explicit_free_policy_and_default_no_support_rejection():
    with pytest.raises(ValueError, match="supports"):
        public.create_analysis(definition=replace(_definition(), fixed=()))
    with pytest.raises(ValueError, match="boundary policy"):
        public.create_analysis(definition=replace(_definition(), modal_boundary_policy="free-free"))
    with pytest.raises(ValueError, match="empty supports"):
        public.create_analysis(definition=replace(_definition(),
            modal_boundary_policy=public.GE_FREE_FREE_MODAL_ONLY_V1))


def test_free_owner_keeps_six_rigid_fields_and_full_flexible_descriptor():
    owner = public.create_analysis(definition=_free_definition())
    try:
        matrices = tuple(m.copy() for m in (owner._K, owner._C, owner._A))
        assert owner.preflight()["modal_boundary_policy"] == public.GE_FREE_FREE_MODAL_ONLY_V1
        assert not owner.capabilities["accepted_state_writes"]
        result = owner.modes(public.GEReferenceModesRequest(3, 1.))
        assert type(result) is public.GEFreeReferenceModesResult
        assert result.status == "completed", result.failure_reason
        assert result.spectral_shift_rad2_s2 < 0.
        assert len(result.rigid_coordinates) == len(result.rigid_multipliers) == 6
        assert len(result.flexible.full_coordinates) == 3
        assert result.flexible.factorization_count == 1
        assert max(result.rigid_equation_residuals) <= 1e-8
        assert max(result.rigid_constraint_residuals) <= 1e-9
        assert result.rigid_flexible_mass_overlap <= 1e-7
        mass = modal._assemble_mass(owner)[0]
        rigid = np.column_stack(result.rigid_coordinates)
        flexible = np.column_stack(result.flexible.full_coordinates)
        np.testing.assert_allclose(rigid.T@mass@rigid, np.eye(6), rtol=0., atol=1e-7)
        np.testing.assert_allclose(rigid.T@mass@flexible, 0., rtol=0., atol=1e-7)
        for q in result.canonical_rigid_coordinates:
            assert q.shape == (owner._K.shape[0],)
            for _, slots, _ in owner._beam_parts:
                np.testing.assert_array_equal(q[owner._shell_count+slots[24:]], 0.)
        for a in (*result.rigid_coordinates, result.rigid_kinetic_gram):
            assert not a.flags.writeable
            with pytest.raises(ValueError): a.setflags(write=True)
        for a in result.rigid_multipliers: np.testing.assert_array_equal(a, 0.)
        for actual, before in zip((owner._K, owner._C, owner._A), matrices):
            for field in ("data", "indices", "indptr"):
                np.testing.assert_array_equal(getattr(actual, field), getattr(before, field))
        _pristine(owner)
    finally: owner.close()


@pytest.mark.parametrize("operation", ["static", "finite", "rebase", "restore"])
def test_modal_only_owner_rejects_accepted_state_writes_atomically(operation):
    owner = public.create_analysis(definition=_free_definition())
    try:
        identity = owner._identity
        with pytest.raises(ValueError, match="modal-only"):
            if operation == "static": owner.solve_linear(())
            elif operation == "finite": owner.solve((), (1.,))
            elif operation == "rebase": owner.rebase_rotations(0)
            else: owner.restore(b"invalid")
        assert owner._identity == identity
        _pristine(owner)
    finally: owner.close()


@pytest.mark.parametrize("stage", ["start", "rigid_field", "iteration", "publish"])
def test_free_modal_cancellation_never_publishes_partial_rigid_or_flexible_modes(stage):
    class AtStage(CancellationToken):
        def raise_if_cancelled(self, current=""):
            if current == "ge_reference_modes."+stage: self.cancel("free boundary")
            super().raise_if_cancelled(current)
    owner = public.create_analysis(definition=_free_definition())
    try:
        result = owner.modes(public.GEReferenceModesRequest(), cancellation_token=AtStage())
        assert result.status == "cancelled" and result.flexible.status == "cancelled"
        assert not result.rigid_coordinates and not result.flexible.full_coordinates
        _pristine(owner)
    finally: owner.close()


def test_additional_mechanism_is_not_hidden_by_the_rigid_projection():
    owner = public.create_analysis(definition=_free_definition())
    try:
        # Independent synthetic obstruction: removing all physical stiffness
        # leaves >6 physical mechanisms. Do not change product or supports.
        owner._K = owner._K*0.
        from scipy import sparse
        owner._A = sparse.bmat([[owner._K, owner._C.T], [owner._C, None]], format="csc")
        result = owner.modes(public.GEReferenceModesRequest(1))
        assert result.status == "failed"
        assert not result.rigid_coordinates and not result.flexible.full_coordinates
        assert ("mechanism" in result.failure_reason or "factorization" in result.failure_reason
                or "work scale" in result.failure_reason)
        _pristine(owner)
    finally: owner.close()


def test_rigid_field_mismatch_is_explicit_failure():
    owner = public.create_analysis(definition=_free_definition())
    try:
        # Finite independent wrong-law row: it must not be projected away.
        owner._C = owner._C.copy(); owner._C.data = owner._C.data.copy()
        owner._C[0, 0] += 1.
        result = owner.modes(public.GEReferenceModesRequest(1))
        assert result.status == "failed" and "rigid field" in result.failure_reason
        assert not result.rigid_coordinates
        _pristine(owner)
    finally: owner.close()
