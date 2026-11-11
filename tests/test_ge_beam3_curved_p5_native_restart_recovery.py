"""Private P5 restart/recovery through the actual native solver envelope."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes, _state_records, NonlinearCheckpointError
from docs.reference_cases.ge_beam3_curved_p5_native_material_probe import canonical, seal, NativeMaterialError
from docs.reference_cases import ge_beam3_curved_p5_native_state_codec as codec
from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import rotation
from test_ge_beam3_curved_p5_native_driver_probe import problem
from test_ge_beam3_curved_p5_algebra_probe import assert_scaled_close


def solve(model, load, **kwargs):
    return solve_static_nonlinear(model, load, max_iterations=10, tolerance=1e-10,
                                 num_layers=1, min_step_fraction=1., **kwargs)


@pytest.fixture(scope='module')
def completed_paths():
    model, element, load=problem()
    continuous=solve(model, load, num_steps=2, emit_restart_checkpoint=True)
    assert continuous.status=='completed'
    first_model, _, first_load=problem()
    first=solve(first_model, first_load, num_steps=1, max_load_factor=.5, emit_restart_checkpoint=True)
    assert first.status=='completed'
    restart_model, restart_element, restart_load=problem()
    raw=canonical_checkpoint_json_bytes(first.restart_checkpoint)
    resumed=solve(restart_model, restart_load, num_steps=2, max_load_factor=1.,
                  restart_checkpoint=raw, emit_restart_checkpoint=True)
    assert resumed.status=='completed'
    return model, element, continuous, first, restart_model, restart_element, resumed


def test_native_checkpoint_continuation_is_byte_identical_to_uninterrupted(completed_paths):
    model, element, continuous, first, restart_model, restart_element, resumed=completed_paths
    assert np.array_equal(resumed.displacements, continuous.displacements)
    assert canonical(resumed.element_states)==canonical(continuous.element_states)
    assert canonical_checkpoint_json_bytes(resumed.restart_checkpoint)==canonical_checkpoint_json_bytes(continuous.restart_checkpoint)
    a=element.recover_native_fields(model.mesh, continuous.element_states[1], expected_committed_total_u=continuous.displacements)
    b=restart_element.recover_native_fields(restart_model.mesh, resumed.element_states[1], expected_committed_total_u=resumed.displacements)
    assert canonical(a)==canonical(b)


def test_fresh_process_checkpoint_continuation_matches(completed_paths, tmp_path):
    _, _, continuous, first, _, _, _=completed_paths
    source=tmp_path/'checkpoint.json';output=tmp_path/'resumed.json'
    source.write_bytes(canonical_checkpoint_json_bytes(first.restart_checkpoint))
    script='''from pathlib import Path
import sys
from test_ge_beam3_curved_p5_native_restart_recovery import problem, solve
from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes
model, element, load=problem()
result=solve(model, load, num_steps=2, max_load_factor=1., restart_checkpoint=Path(sys.argv[1]).read_bytes(), emit_restart_checkpoint=True)
assert result.status=='completed'
with Path(sys.argv[2]).open('xb') as stream: stream.write(canonical_checkpoint_json_bytes(result.restart_checkpoint))
'''
    env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parent)+os.pathsep+env.get('PYTHONPATH', '')
    child=subprocess.run([sys.executable, '-B', '-c', script, str(source), str(output)],
                         env=env, capture_output=True, text=True, timeout=60)
    assert child.returncode==0, child.stdout+child.stderr
    assert output.read_bytes()==canonical_checkpoint_json_bytes(continuous.restart_checkpoint)


def test_typed_codec_preserves_signed_zero_and_exact_classes():
    model, element, _=problem()
    state=element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)
    # A structural codec check only; arbitrary altered records still require
    # accepted-origin mechanics validation before the solver accepts them.
    state['committed_total_u'][0]=-0.
    envelope=codec.encode(state, order=8);decoded=codec.decode(envelope, order=8)
    assert canonical(decoded)==canonical(state)
    assert np.signbit(decoded['committed_total_u'][0])
    assert type(decoded['material_state']['response']) is type(state['material_state']['response'])
    assert type(decoded['material_state']['origins']) is tuple


@pytest.mark.parametrize('change', ['duplicate', 'nonfinite', 'noncanonical', 'deep', 'large', 'unknown', 'integer_float', 'station_count'])
def test_malformed_typed_payload_rejected(change):
    model, element, _=problem();state=element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)
    envelope=codec.encode(state, order=8)
    if change=='duplicate': envelope['payload']='{"driver_schema":"a","driver_schema":"b"}\n'
    elif change=='nonfinite': envelope['payload']='{"bad":NaN}\n'
    elif change=='noncanonical': envelope['payload']=' '+envelope['payload']
    elif change=='deep': envelope['payload']='['*33+'0'+']'*33
    elif change=='large': envelope['payload']=' '* (codec.MAX_BYTES+1)
    else:
        record=json.loads(envelope['payload'])
        if change=='unknown': record['extra']=1
        elif change=='integer_float': record['material_state']['epoch']=0.0
        else: record['material_state']['origins'].pop()
        envelope['payload']=canonical(record).decode('ascii')
    with pytest.raises(ValueError): codec.decode(envelope, order=8)


def test_resealed_mechanical_mutation_is_rejected_after_decode(completed_paths):
    model, element, complete, _, _, _, _=completed_paths
    state=deepcopy(complete.element_states[1]);inner=state['material_state']
    inner['response']=replace(inner['response'], potential=inner['response'].potential+.01)
    state['material_state']=seal(inner);state=seal(state)
    envelope=codec.encode(state, order=8)
    with pytest.raises(NativeMaterialError, match='replay'):
        element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, envelope, 1)


def test_serializer_is_mandatory_for_opted_in_native_state(monkeypatch):
    model, element, _=problem();state=element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)
    monkeypatch.setattr(element, 'serialize_native_material_state', None)
    with pytest.raises(NonlinearCheckpointError, match='serializer'):
        _state_records({1: state}, _guard_model=model)


def test_recovery_is_read_only_work_conjugate_and_reports_no_fabricated_fibres(completed_paths):
    model, element, complete, _, _, _, _=completed_paths
    state=complete.element_states[1];before=canonical(state)
    fields=element.recover_native_fields(model.mesh, state, expected_committed_total_u=complete.displacements)
    assert fields['station_ids']==tuple((c, i) for c in (0, 1) for i in range(8))
    assert fields['strains'].shape==fields['resultants'].shape==(16, 6)
    assert fields['fibre_stress_status']=='SECTION_DOES_NOT_SUPPLY_FIBRE_STRESSES'
    assert not fields['production_qualified']
    direction=np.sin(np.arange(6)+.2)
    for i, frame in enumerate(fields['current_frames']):
        global_direction=np.r_[frame@direction[:3], frame@direction[3:]]
        assert abs(fields['global_resultants'][i]@global_direction-fields['resultants'][i]@direction)<=1e-11
    for value in fields.values():
        if isinstance(value, np.ndarray): assert not value.flags.writeable
    assert canonical(state)==before


def test_recovery_transforms_objectively_under_common_rigid_motion(completed_paths):
    model, element, complete, _, _, _, _=completed_paths
    state=complete.element_states[1];inner=state['material_state']
    vector=np.array([.2, -.1, .3]);observer=rotation(vector);shift=np.array([.3, -.2, .1])
    positions=inner['committed_positions']@observer.T+shift
    operators=observer@inner['committed_nodal_rotation_matrices']
    total=inner['committed_total_u'].reshape(3, 6).copy()
    total[:, :3]=positions-element.core.reference.coordinates;total[:, 3:]+=vector
    response=element.core._solve(positions, operators, inner['origins'])
    transformed=element._wrap(element.core._state(inner['epoch'], total.ravel(), positions,
                                                operators, inner['origins'], response))
    first=element.recover_native_fields(model.mesh, state);second=element.recover_native_fields(model.mesh, transformed)
    assert_scaled_close(second['strains'], first['strains'])
    assert_scaled_close(second['resultants'], first['resultants'])
    assert_scaled_close(second['current_frames'], observer@first['current_frames'])
    assert_scaled_close(second['current_positions'], first['current_positions']@observer.T+shift)
    assert_scaled_close(second['global_resultants'][:, :3], first['global_resultants'][:, :3]@observer.T)
    assert_scaled_close(second['global_resultants'][:, 3:], first['global_resultants'][:, 3:]@observer.T)
