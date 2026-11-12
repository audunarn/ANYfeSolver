"""M7a verification scope: operation counts and pre-entry refusal."""
import sys
import pytest
from test_prepared_static_evaluator import _shell_model
from anyfesolver import PreparedStaticCapabilityError
from anyfesolver import prepared_static_loads as loads
from anyfesolver import prepared_static_recovery as recovery


def test_pressure_verification_uses_one_owned_authority_span():
    model = _shell_model()
    model.apply_boundary_conditions()
    maps = loads.compile_canonical_load_maps(model, families=('pressure',))
    assert len(maps.pressure_recipes) == 2
    calls = []
    code = loads.require_native_source_authority.__code__
    def observe(frame, event, arg):
        if event == 'call' and frame.f_code is code:
            calls.append(1)
    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        verification = loads._verify_maps_against_oracle(model, maps)
    finally:
        sys.setprofile(previous)
    assert len(calls) == 2  # one entry and one exit, independent of pressure count
    assert dict(verification)['pressure'] <= 1e-11
    assert recovery._OWNED_RECOVERY_SCOPE.get() is None


@pytest.mark.parametrize('target', ['evaluate', 'scope', 'record', 'recipe'])
def test_verification_refuses_replaced_code_before_trusted_private_observation(monkeypatch, target):
    model = _shell_model()
    model.apply_boundary_conditions()
    maps = loads.compile_canonical_load_maps(model, families=('pressure',))
    seen = []
    if target == 'evaluate':
        monkeypatch.setattr(loads, 'evaluate_load_vector', lambda *args: seen.append(args))
    elif target == 'scope':
        monkeypatch.setattr(recovery, 'owned_recovery_scope', lambda *args: seen.append(args))
    elif target == 'recipe':
        from dataclasses import replace
        class Foreign:
            @property
            def execution_owner(self):
                seen.append('foreign observation')
                return ()
        maps = replace(maps, pressure_recipes={1: Foreign()})
    else:
        monkeypatch.setattr(loads.CanonicalLoadMaps, 'family_enabled', lambda *args: seen.append(args) or True)
        # Verification first consults family flags before the protected span;
        # that existing public observation is outside the enlarged trusted scope.
    with pytest.raises(PreparedStaticCapabilityError, match='pressure verification'):
        loads._verify_maps_against_oracle(model, maps)
    if target != 'record':
        assert seen == []
    assert recovery._OWNED_RECOVERY_SCOPE.get() is None


def test_owned_scope_postcheck_and_primary_error_precedence(monkeypatch):
    from anyfesolver.elements import BeamElement
    model = _shell_model()
    model.apply_boundary_conditions()
    maps = loads.compile_canonical_load_maps(model, families=('pressure',))
    with pytest.raises(ValueError, match='primary') as caught:
        with recovery._OWNED_SCOPE_FACTORY(tuple(maps.pressure_recipes.values())):
            monkeypatch.setattr(BeamElement, 'compute_mass_matrix', lambda *args: None)
            raise ValueError('primary')
    assert any('output authority also failed' in note for note in caught.value.__notes__)
    assert recovery._OWNED_RECOVERY_SCOPE.get() is None


@pytest.mark.parametrize('target', ['ingest_static_case', '_static_case_probe', 'StaticCase'])
def test_verification_refuses_foreign_numeric_ingress_before_invocation(monkeypatch, target):
    model = _shell_model()
    model.apply_boundary_conditions()
    maps = loads.compile_canonical_load_maps(model, families=('pressure',))
    seen = []
    class ForeignNumber:
        def __int__(self):
            seen.append(('int', recovery._OWNED_RECOVERY_SCOPE.get()))
            return next(iter(maps.pressure_recipes))
        def __float__(self):
            seen.append(('float', recovery._OWNED_RECOVERY_SCOPE.get()))
            return 1.0
    foreign = ForeignNumber()
    packet = loads.OwnedCaseLoads(None, (), (),
        ((foreign, ('scalar', foreign)),), None, (), False)
    def replaced(*args):
        seen.append(('ingress', recovery._OWNED_RECOVERY_SCOPE.get()))
        return ('probe', 0.0, packet)
    if target == 'StaticCase':
        monkeypatch.setattr(loads._pressure_static_batch, target, replaced)
    else:
        monkeypatch.setattr(loads, target, replaced)
    with pytest.raises(PreparedStaticCapabilityError, match='pressure verification .* authority'):
        loads._verify_maps_against_oracle(model, maps)
    assert seen == []
    assert recovery._OWNED_RECOVERY_SCOPE.get() is None
