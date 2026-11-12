"""SOURCE_ONLY_UNRUN: common-origin full native BL5 arc directional oracle.

One mixed/P1 loaded fixture and one uncommitted arc candidate. This is neither
a limit-point test nor authority to run a new numerical campaign.
"""
import json
import hashlib
import os
from pathlib import Path
from dataclasses import fields, is_dataclass
import numpy as np
import pytest
from scipy import sparse

from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_arc_trial import correct, path_row
from anyfesolver._stiffened_shell_ge_finite_experimental import _fibre_recovery
from anyfesolver._stiffened_shell_beam_history import arrays as beam_arrays
from anyfesolver._stiffened_shell_q4_material_event import _origin_values
from anyfesolver.plasticity import _jit_consistency_residual_scalar
from examples.stiffened_shell_native_fibre_bl5_mixed_j2_displacement import definition
from examples.stiffened_shell_native_fibre_bl5_displacement import ACTIONS, CONTROL, OPTIONS, TARGETS
from examples.stiffened_shell_native_material_bl5_arc import arc_control


def _loaded_seed(owner, tmp_path):
    """Optional authentic gen3 input; strict restore remains the admission gate."""
    source = os.environ.get('ANY_NATIVE_BL5_ARC_JACOBIAN_FIXTURE')
    expected = os.environ.get('ANY_NATIVE_BL5_ARC_JACOBIAN_FIXTURE_SHA256')
    attempt = dict(source=source or 'fresh loaded fixture', expected_sha256=expected, phase='requested')
    diagnostic = tmp_path/'seed-attempt.json'
    diagnostic.write_text(json.dumps(attempt, indent=2))
    assert (source is None) == (expected is None), 'fixture path and expected SHA256 must be supplied together'
    if source is None:
        owner.solve(ACTIONS, (.001,), **OPTIONS)
        owner.solve(ACTIONS, TARGETS[:2], displacement_control=CONTROL, **OPTIONS)
        provenance = dict(source='fresh loaded fixture')
    else:
        # Root adopted this exact existing matched mixed/P1 checkpoint. Neither
        # an arbitrary caller digest nor a fabricated replacement is admitted.
        raw = Path(source).read_bytes()
        attempt.update(phase='input read', actual_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        diagnostic.write_text(json.dumps(attempt, indent=2))
        assert expected == '70b162593ee8939aea0755fa20a5d84956da58a435879518aa5e26ad0573f8dc'
        assert len(raw) == 46984 and hashlib.sha256(raw).hexdigest() == expected
        copied = tmp_path/'loaded-seed-fixture.bin'
        copied.write_bytes(raw)
        owned = copied.read_bytes()
        assert owned == raw and hashlib.sha256(owned).hexdigest() == expected
        loaded = owner.restore(owned)
        attempt.update(phase='strict restore returned', status=loaded.status,
                       generation=loaded.generation, target=loaded.target,
                       restored_sha256=hashlib.sha256(owner.checkpoint()).hexdigest())
        diagnostic.write_text(json.dumps(attempt, indent=2))
        assert loaded.status == 'completed' and loaded.generation == 3
        assert loaded.target == TARGETS[1]
        assert owner.checkpoint() == owned, 'strict model/pose/material restore changed bytes'
        provenance = dict(source=str(Path(source)), sha256=expected, bytes=len(raw))
    loaded = owner.recover()
    attempt.update(phase='public recovery returned', status=loaded.status,
                   generation=loaded.generation, target=loaded.target)
    diagnostic.write_text(json.dumps(attempt, indent=2))
    assert loaded.status == 'completed' and loaded.generation == 3
    assert loaded.target == TARGETS[1]
    return loaded, provenance


def _json_record(value):
    """Serialize the complete actual public record, including failure diagnostics."""
    if is_dataclass(value):
        return {field.name: _json_record(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_json_record(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_record(item) for key, item in value.items()}
    return value


def _inventory(owner, driver, state, shell, beam, evaluation, trace=None):
    """Read actual material recovery and fixed-origin shell switch diagnostics.

    The existing scalar consistency kernel at zero multiplier supplies the
    elastic-trial drive; this does not reconstruct an equilibrium or tangent.
    Its returned-stress arithmetic differs slightly from the initial batch
    switch arithmetic, so require agreement with actual candidate alpha growth.
    Exact switch equality or a hardening knot fails; it is never skipped.
    """
    labels = []
    margins = []
    knots = []
    trace = (lambda event: None) if trace is None else trace
    captures = driver._q4_boundary_trial
    trace(dict(stage='capture binding', capture_present=captures is not None))
    assert captures is not None, 'missing fixed-origin shell material capture'
    trace(dict(stage='capture identity', state_matches=captures[0] is state,
               origins_match=captures[1] is shell, capture_count=len(captures[3])))
    assert captures[0] is state and captures[1] is shell
    assert len(captures[3]) == len(owner._shell_parts)
    for part, origin, candidate, (slots, capture) in zip(
            owner._shell_parts, shell, evaluation[4], captures[3], strict=True):
        eid, element, actual_slots, _ = part
        trace(dict(stage='shell origin binding', element_id=int(eid),
                   slots_match=tuple(actual_slots) == slots, origin_matches=capture.origin is origin,
                   origin_values_match=capture.origin_values == _origin_values(origin)))
        assert tuple(actual_slots) == slots
        assert capture.origin is origin
        assert capture.origin_values == _origin_values(origin)
        material = owner._shell_model.get_material(element.material_name)
        curve = material.hardening_curve
        parameters = tuple(float(getattr(curve, name)) for name in (
            'sigma_prop', 'sigma_yield', 'sigma_yield_2', 'eps_p_y1',
            'eps_p_y2', 'K', 'n', '_power_offset'))
        alpha = np.asarray(candidate['alpha']).reshape(-1)
        previous = np.asarray(origin['alpha']).reshape(-1)
        stress = (np.asarray(capture.strain)-np.asarray(origin['plastic_strain']).reshape(-1, 3)) @ np.asarray(capture.elastic_matrix).T
        family = 'S3' if len(slots) == 18 else 'Q4'
        for station, (sigma, old, new) in enumerate(zip(stress, previous, alpha, strict=True)):
            value, flow = _jit_consistency_residual_scalar(
                0., sigma[0]+sigma[1], sigma[0]-sigma[1], sigma[2],
                float(old), material.elastic_modulus/(3*(1-material.poisson_ratio)),
                material.elastic_modulus/(2*(1+material.poisson_ratio)), *parameters)
            margin = float(value-capture.tolerance*max(flow*flow, 1.))
            plastic = bool(new > old)
            band = int(np.searchsorted((curve.eps_p_y1, curve.eps_p_y2), new))
            distances = (float(new-curve.eps_p_y1), float(new-curve.eps_p_y2))
            trace(dict(stage='shell station', family=family, element_id=int(eid), station=station,
                       previous_alpha=float(old), candidate_alpha=float(new), plastic=plastic,
                       scalar_drive=float(value), flow=float(flow), switch_margin=margin,
                       hardening_band=band, hardening_distances=distances))
            assert np.isfinite(margin) and margin != 0., 'shell switch: no smooth-branch claim'
            assert plastic == (margin > 0.), 'shell scalar/candidate branch disagreement'
            if not plastic:
                assert new == old
            assert all(value != 0. for value in distances), 'shell hardening knot'
            labels.append((family, int(eid), station, 'PLASTIC' if plastic else 'ELASTIC', band))
            margins.append(margin)
            knots.extend(distances)
    fibres, _ = _fibre_recovery(owner, state, beam, evaluation[5])
    for station in fibres:
        for fibre in station.fibres:
            trace(dict(stage='physical fibre', element_id=station.element_id, cell=station.cell,
                       station=station.station, fibre=_json_record(fibre)))
            assert fibre.branch not in ('YIELD_BOUNDARY', 'HARDENING_KNOT')
            assert all(np.isfinite(fibre.algorithmic_modulus))
            labels.append(('FIBRE', station.element_id, station.cell, station.station,
                           fibre.fibre_id, fibre.branch))
    generalized = driver._generalized_records(state, beam, evaluation[5], include_native=True)
    for member in generalized:
        for station in member.stations:
            trace(dict(stage='generalized station', element_id=member.element_id,
                       recovery=_json_record(station)))
            assert station.branch in ('ELASTIC', 'PLASTIC'), 'generalized yield boundary'
            labels.append(('GENERALIZED', member.element_id, station.cell, station.station, station.branch))
    return tuple(labels), np.asarray(margins), np.asarray(knots)


def test_common_accepted_origins_full_arc_residual_direction(request, tmp_path):
    diagnostic = dict(status='PARTIAL_UNVERIFIED', stage='owner construction', evaluations={}, stencils=[])
    artifact = tmp_path/'same-origin-arc-jacobian.json'
    event_log = tmp_path/'same-origin-arc-material-events.jsonl'
    diagnostic['material_event_log'] = event_log.name
    def persist():
        pending = artifact.with_suffix('.json.pending')
        pending.write_text(json.dumps(_json_record(diagnostic), indent=2))
        pending.replace(artifact)
    persist()
    owner = ss.create_analysis(definition=definition('mixed', 'biaxial', 'mixed', 'continuous_p1_coarse'))
    request.addfinalizer(owner.close)
    loaded, fixture_provenance = _loaded_seed(owner, tmp_path)
    diagnostic.update(stage='loaded seed', fixture=fixture_provenance,
                      loaded_generation=loaded.generation, loaded_status=loaded.status)
    persist()
    assert any(np.any(row.alpha > 0.) for row in loaded.shell_s3_material)
    driver = owner._finite_driver
    saved = owner.checkpoint()
    parent = driver._state
    shell, beam = driver._shell_origins, driver._beam_origins
    histories = dict(beam_arrays(owner._beam_parts, beam, 'beam_origin'))
    diagnostic.update(stage='accepted origins', accepted_checkpoint_sha256=hashlib.sha256(saved).hexdigest(),
                      model_identity=owner._identity,
                      shell_origins=[_origin_values(origin) for origin in shell], beam_origins=histories)
    persist()
    assert np.any(histories['beam_origin_fibres'][:, 2:] > 0.)
    control = arc_control(owner)
    diagnostic.update(stage='arc correction', control=dict(radius=control.radius,
        orientation=control.orientation, max_iterations=control.max_iterations,
        max_backtracks=control.max_backtracks, metric_weights=control.metric.weights,
        metric_length=control.metric.length))
    persist()
    trial = correct(driver, load_direction=driver._load(ACTIONS), metric=control.metric,
        radius=control.radius, predictor_dof=CONTROL.dof(owner), orientation=control.orientation,
        max_iterations=control.max_iterations, max_backtracks=control.max_backtracks)
    diagnostic.update(stage='arc candidate returned', candidate=_json_record(trial.state),
        amplitude=trial.amplitude, base_load=trial.base_load, load_direction=trial.load_direction,
        normal_translations=trial.normal_translations, normal_spins=trial.normal_spins,
        predecessor_matches=trial.predecessor_state is parent,
        shell_origins_match=trial.shell_predecessor is shell, beam_origins_match=trial.beam_predecessor is beam)
    persist()
    assert trial.predecessor_state is parent
    assert trial.shell_predecessor is shell and trial.beam_predecessor is beam
    state = trial.state
    normal = (trial.normal_translations, trial.normal_spins)
    n = len(state.q)
    size = n+len(state.multipliers)
    nodal = owner._shell_count+6*len(owner._beam_ids)
    direction = np.zeros(size)
    # Explicit component units: translation m, rotation rad, retained force N
    # and moment N*m; multiplier units follow the existing BL5 row work law.
    for index in owner._free:
        if index < nodal:
            direction[index] = np.sin(index+1.)*(1e-6 if index % 6 < 3 else 1e-3)
    for member in range(len(owner._beam_parts)):
        start = nodal+24*member
        direction[start:start+6] = 1e-3*np.sin(np.arange(6)+member+1.)
        direction[start+6:start+24] = np.sin(np.arange(18)+member+1.)
    direction[n:] = np.cos(np.arange(size-n)+1.)
    amplitude_direction = .003
    load_column = np.r_[trial.load_direction, np.zeros(size-n)]
    diagnostic.update(stage='direction defined', direction=direction,
        amplitude_direction=amplitude_direction, load_column=load_column,
        material_criterion=1e-5, norm='all-row unweighted L2; denominator max(norm(expected),1)',
        path_criteria=dict(relative=1e-8, absolute=1e-10))
    persist()

    def evaluate(point, amplitude, label):
        packet = dict(stage='assembly requested', amplitude=amplitude, state=_json_record(point),
                      material_event_log=event_log.name)
        diagnostic['evaluations'][label] = packet
        diagnostic['stage'] = label
        persist()
        evaluation = driver._assemble(point, 0., shell, beam,
            applied_load=trial.base_load+amplitude*trial.load_direction)
        value, row = path_row(driver, point, parent, control.metric, normal)
        packet.update(stage='assembled', residual=evaluation[0], matrix_shape=list(evaluation[1].shape),
                      matrix_nnz=evaluation[1].nnz, path_value=value-trial.radius, path_row=row)
        persist()
        def trace(event):
            # Station-local actual values only; phase label binds them to the
            # already persisted state/residual without copying global arrays.
            with event_log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(_json_record(dict(evaluation=label, event=event)))+'\n')
        inventory = _inventory(owner, driver, point, shell, beam, evaluation, trace=trace)
        packet.update(stage='inventory passed', branches=inventory[0], shell_switch_margins=inventory[1],
                      hardening_distances=inventory[2])
        persist()
        return np.r_[evaluation[0], value-trial.radius], evaluation[1], row, inventory

    residual, matrix, row, center = evaluate(state, trial.amplitude, 'center')
    assert matrix.shape == (size, size)
    assert any(label[-1] == 'PLASTIC' or (label[0] in ('Q4', 'S3') and label[-2] == 'PLASTIC')
               for label in center[0]), 'chosen arc trial has no active plastic material branch'
    np.testing.assert_array_equal(row[nodal:], np.zeros(size-nodal))
    border = sparse.bmat([[matrix, sparse.csc_matrix(-load_column[:, None])],
                          [sparse.csc_matrix(row[None, :]), sparse.csc_matrix((1, 1))]], format='csc')
    expected = border @ np.r_[direction, amplitude_direction]
    material_expected = matrix @ direction
    diagnostic.update(border_shape=list(border.shape), expected=expected, material_expected=material_expected)
    persist()
    errors = []
    inventories = []
    for h in (1e-5, 1e-6):
        plus_state = driver._advance(state, direction, h)
        minus_state = driver._advance(state, direction, -h)
        stencil = dict(h=h, stage='updates made', plus_state=_json_record(plus_state),
                       minus_state=_json_record(minus_state))
        diagnostic['stencils'].append(stencil)
        persist()
        assert np.any(plus_state.cells != state.cells) and np.any(plus_state.resultants != state.resultants)
        assert np.any(plus_state.frames != state.frames) and np.any(plus_state.multipliers != state.multipliers)
        plus, _, _, upper = evaluate(plus_state, trial.amplitude+h*amplitude_direction, f'plus-{h}')
        minus, _, _, lower = evaluate(minus_state, trial.amplitude-h*amplitude_direction, f'minus-{h}')
        difference = (plus-minus)/(2*h)
        equilibrium_error = float(np.linalg.norm(difference[:-1]-expected[:-1])/max(np.linalg.norm(expected[:-1]), 1.))
        # Removing the exact dead-load column prevents it masking J-direction error.
        material_error = float(np.linalg.norm(difference[:-1]+load_column*amplitude_direction-material_expected)/max(np.linalg.norm(material_expected), 1.))
        stencil.update(stage='stencil available, criteria unverified', derivative=difference,
            equilibrium_error=equilibrium_error, material_error=material_error,
            path_error=float(abs(difference[-1]-expected[-1])), branches_match=upper[0] == center[0] == lower[0],
            margins=[dict(center=center[slot], plus=upper[slot], minus=lower[slot],
                movement=np.maximum(abs(upper[slot]-center[slot]), abs(lower[slot]-center[slot])))
                     for slot in (1, 2)])
        persist()
        assert upper[0] == center[0] == lower[0], 'central stencil changes actual material branch'
        for slot in (1, 2):
            movement = np.maximum(abs(upper[slot]-center[slot]), abs(lower[slot]-center[slot]))
            assert np.all(abs(center[slot]) > movement), 'stencil reaches shell switch/knot margin'
        assert equilibrium_error <= 1e-5 and material_error <= 1e-5
        assert difference[-1] == pytest.approx(float(row @ direction), rel=1e-8, abs=1e-10)
        stencil['stage'] = 'stencil criteria passed; ownership/checkpoint still unverified'
        persist()
        errors.append(dict(h=h, equilibrium=equilibrium_error, material=material_error,
                          path=float(abs(difference[-1]-expected[-1]))))
        inventories.append(dict(h=h, plus=upper[0], minus=lower[0],
                                shell_switch_margins=[upper[1].tolist(), lower[1].tolist()]))
    diagnostic.update(stage='final ownership/checkpoint assertions', driver_matches=owner._finite_driver is driver,
                      state_matches=driver._state is parent, shell_origins_match=driver._shell_origins is shell,
                      beam_origins_match=driver._beam_origins is beam,
                      final_checkpoint_sha256=hashlib.sha256(owner.checkpoint()).hexdigest())
    persist()
    assert owner._finite_driver is driver and driver._state is parent
    assert driver._shell_origins is shell and driver._beam_origins is beam
    assert owner.checkpoint() == saved
    diagnostic.update(status='EXECUTED_PASS_DEVELOPMENT_DIRECTIONAL_CHECK', stage='all criteria passed')
    persist()
    (tmp_path/'same-origin-arc-jacobian.json').write_text(json.dumps(_json_record(dict(
        status='EXECUTED_PASS_DEVELOPMENT_DIRECTIONAL_CHECK', errors=errors,
        diagnostic=diagnostic,
        fixture=fixture_provenance,
        center_branches=center[0], stencil=inventories, direction=direction.tolist(),
        amplitude_direction=amplitude_direction, shape=list(border.shape),
        material_criterion=1e-5, norm='all-row unweighted L2; denominator max(norm(expected),1)',
        path_criteria=dict(relative=1e-8, absolute=1e-10),
        production_qualified=False)), indent=2))


def test_authentic_old_loaded_checkpoint_current_owner_recovery(request, tmp_path):
    """Old fixed-displacement layout compatibility, never a Jacobian substitute."""
    if os.environ.get('ANY_NATIVE_BL5_ARC_JACOBIAN_FIXTURE') is None:
        pytest.skip('local loaded BL5 checkpoint evidence is unavailable; '
                    'set ANY_NATIVE_BL5_ARC_JACOBIAN_FIXTURE to exercise recovery')
    owner = ss.create_analysis(definition=definition('mixed', 'biaxial', 'mixed', 'continuous_p1_coarse'))
    request.addfinalizer(owner.close)
    loaded, provenance = _loaded_seed(owner, tmp_path)
    public = _json_record(loaded)
    # Preserve the actual record even when a later finite/content assertion fails.
    (tmp_path/'restored-public-recovery.json').write_text(json.dumps(public, indent=2))
    json.dumps(public, allow_nan=False)
    assert loaded.support_reactions.size and loaded.applied_dead_load.size
    assert loaded.cell_frames.size and loaded.cell_resultants.size and loaded.multipliers.size
    assert loaded.beam_fibre_material and loaded.beam_generalized_material
    assert loaded.shell_s3_material and loaded.shell_q4_material
    assert any(np.any(row.alpha > 0.) for row in loaded.shell_s3_material)
    physical = {row.element_id for row in loaded.beam_fibre_material}
    generalized = {row.element_id for row in loaded.beam_generalized_material}
    assert physical and generalized and physical.isdisjoint(generalized)
    from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot
    raw = owner.checkpoint()
    _, payload = decode_snapshot(owner, raw)
    for name, value in (('frames', loaded.beam_frames), ('cells', loaded.cell_frames),
                        ('resultants', loaded.cell_resultants), ('multipliers', loaded.multipliers),
                        ('applied_load', loaded.applied_dead_load),
                        ('shell_anchors', loaded.shell_rotation_anchors)):
        np.testing.assert_array_equal(value, payload[name])
    np.testing.assert_array_equal(loaded.shell_chart_coordinates.reshape(-1),
                                  payload['q'][:owner._shell_count])
    accepted = dict(beam_arrays(owner._beam_parts, owner._finite_driver._beam_origins, 'beam'))
    assert np.any(accepted['beam_fibres'][:, 2:] > 0.)
    for name, value in accepted.items():
        np.testing.assert_array_equal(value, payload[name])
    for record, alpha, plastic in zip(loaded.shell_q4_material, payload['shell_alpha'],
                                     payload['shell_plastic'], strict=True):
        np.testing.assert_array_equal(record.alpha, alpha)
        np.testing.assert_array_equal(record.plastic_strain, plastic)
    q4_count = len(owner._definition.shell.q4_elements)
    s3_parts = owner._shell_parts[q4_count:]
    assert len(payload['s3_states']) == 2*len(s3_parts)
    material = owner._shell_model.get_material('shell')
    # Native layout stores accepted S3 blobs first, predecessor blobs second.
    # Use the element's actual model-bound codec; never invent a blob schema.
    for part, record, blob in zip(s3_parts, loaded.shell_s3_material,
                                  payload['s3_states'][:len(s3_parts)], strict=True):
        eid, element, _, _ = part
        assert record.element_id == eid
        original = element.deserialize_nonlinear_state(owner._shell_model.mesh, material, blob, 3)
        np.testing.assert_array_equal(record.alpha, original['alpha'])
        np.testing.assert_array_equal(record.plastic_strain, original['plastic_strain'])
    assert loaded.free_equilibrium_ratio <= 1e-8 and loaded.constraint_max <= 1e-9
    assert owner.checkpoint() == (tmp_path/'loaded-seed-fixture.bin').read_bytes()
    (tmp_path/'old-checkpoint-compatibility.json').write_text(json.dumps(dict(
        status='EXECUTED_OLD_CHECKPOINT_COMPATIBILITY_ONLY', fixture=provenance,
        generation=loaded.generation, target=loaded.target, restored_status=loaded.status,
        physical_member_ids=sorted(physical), generalized_member_ids=sorted(generalized),
        public_recovery='restored-public-recovery.json', checkpoint_bytes_equal=True,
        s3_accepted_blob_material_arrays_equal=True,
        full_arc_jacobian_executed=False, production_qualified=False), indent=2))
