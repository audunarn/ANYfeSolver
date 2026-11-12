"""Owned schema sampling amortizes audits without broadening live observations."""
import sys
import numpy as np
import pytest
from anyfesolver import RecoveryConfig, PreparedStaticCapabilityError
from anyfesolver import prepared_static_envelopes as envelopes
from anyfesolver import prepared_static_recovery as recovery
from test_prepared_static_evaluator import _shell_model


def test_schema_scope_uses_two_authority_scans_and_same_native_schema():
    model = _shell_model()
    model.apply_boundary_conditions()
    config = RecoveryConfig(components=('von_mises',), include_reactions=True)
    calls = []
    code = recovery.require_native_source_authority.__code__
    previous = sys.getprofile()
    def observe(frame, event, arg):
        if event == 'call' and frame.f_code is code:
            calls.append(1)
    try:
        sys.setprofile(observe)
        plan = envelopes.compile_envelope_plan(model, config)
    finally:
        sys.setprofile(previous)
    assert len(plan.elements) == 2
    assert len(calls) == 2
    for element in plan.elements:
        record = envelopes._numeric_fields(recovery._recover_owned_local(
            element.recipe, np.zeros(element.dof_mapping.size)), ('von_mises',))
        assert element.schema == tuple((name, tuple(value.shape)) for name, value in record.items())
    assert recovery._OWNED_RECOVERY_SCOPE.get() is None


@pytest.mark.parametrize('target', ['_capture_owned_local_recovery', '_recover_owned_local',
                                  '_numeric_fields', 'owned_recovery_scope'])
def test_schema_helper_replacement_refused_before_receiving_private_data(monkeypatch, target):
    model = _shell_model()
    model.apply_boundary_conditions()
    seen = []
    monkeypatch.setattr(envelopes, target, lambda *args: seen.append(args))
    with pytest.raises(PreparedStaticCapabilityError, match='envelope schema helper authority'):
        envelopes.compile_envelope_plan(model, RecoveryConfig(components=('von_mises',)))
    assert seen == []
    assert recovery._OWNED_RECOVERY_SCOPE.get() is None


def test_unknown_schema_component_refuses_and_revokes_scope():
    model = _shell_model()
    model.apply_boundary_conditions()
    with pytest.raises(PreparedStaticCapabilityError, match='component unavailable'):
        envelopes.compile_envelope_plan(model, RecoveryConfig(components=('not_a_stress',)))
    assert recovery._OWNED_RECOVERY_SCOPE.get() is None


def test_schema_source_mutation_refused_before_capture(monkeypatch):
    from anyfesolver.elements import BeamElement
    model = _shell_model()
    model.apply_boundary_conditions()
    seen = []
    monkeypatch.setattr(BeamElement, 'compute_mass_matrix', lambda *args: seen.append(args))
    with pytest.raises(PreparedStaticCapabilityError):
        envelopes.compile_envelope_plan(model, RecoveryConfig(components=('von_mises',)))
    assert seen == []
    assert recovery._OWNED_RECOVERY_SCOPE.get() is None


@pytest.mark.parametrize('name', ['principal_stresses', 'shell_surface_tensor'])
def test_schema_principal_helper_inplace_code_mutation_refused(monkeypatch, name):
    from anyfesolver import prepared_static_measures as measures
    model = _shell_model()
    model.apply_boundary_conditions()
    def foreign(*args):
        raise AssertionError('foreign helper received private recovery data')
    monkeypatch.setattr(getattr(measures, name), '__code__', foreign.__code__)
    with pytest.raises(PreparedStaticCapabilityError, match='envelope schema helper authority'):
        envelopes.compile_envelope_plan(model, RecoveryConfig(components=('principal_stresses_top',)))
    assert recovery._OWNED_RECOVERY_SCOPE.get() is None
