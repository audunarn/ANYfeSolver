"""SOURCE_ONLY_UNRUN tests; no execution/collection authority is granted.

Public numerical tests below need their own bounded execution authorization.
No turning, stability, capacity or qualification threshold is invented here.
"""
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import numpy as np
import pytest

from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver.control import CancellationToken
from examples import stiffened_shell_native_capacity as capacity


def action_vector(actions, order):
    index = {row:i for i,row in enumerate(order)}
    values = np.zeros((len(order),6))
    for action in actions:
        values[index[action.domain,action.node]] += action.wrench
    return values


@pytest.mark.parametrize('beams',('fibre','mixed'))
def test_definition_and_public_admission_preserve_two_native_lines(beams):
    model = capacity.definition(beams=beams)
    assert len(model.shell.q4_elements) == 16
    assert len(model.shell.node_ids) == 27
    assert len(model.beams) == 8 and len(model.lines) == 2
    assert not model.points and not model.shell.s3_elements
    assert model.bl5_material_policy == ss.BL5_NATIVE_FIBRE_MATERIAL_V1
    assert model.q4_pullback_policy == ss.Q4_ALGORITHMIC_PULLBACK_V1
    assert model.shell_reference_policy == 'flat' and model.shell.j2_curve is None
    assert {(f.domain,f.node,f.component) for f in model.fixed} == {
        ('shell',node,axis) for node in (1,10,19) for axis in range(6)}
    assert [line.beam_halfcells for line in model.lines] == [
        tuple((line*4+member,half) for member in range(4) for half in range(2))
        for line in range(2)]
    assert all(len(line.shell_nodes)-1 == len(line.beam_halfcells) == 8
               and line.basis == 'continuous_p1_coarse' for line in model.lines)
    specimen = capacity.Specimen()
    assert all(line.expected_beam_arm == (0.,0.,-specimen.stiffener_height_m/2)
               for line in model.lines)
    # BL5 binds shell top; EI and centroid loads use midsurface-to-axis distance.
    shell_top_z = specimen.skin_m/2
    assert shell_top_z-specimen.eccentricity_m == pytest.approx(
        model.lines[0].expected_beam_arm[2])
    sections = [definition.instantiate()[0].section for definition in model.beams]
    assert sum(type(s) is b3_ge.PhysicalFibreSection for s in sections) == (8 if beams == 'fibre' else 4)
    assert sum(type(s) is b3_ge.EllipsoidalGeneralizedSection for s in sections) == (0 if beams == 'fibre' else 4)
    iy = specimen.stiffener_width_m*specimen.stiffener_height_m**3/12
    iz = specimen.stiffener_height_m*specimen.stiffener_width_m**3/12
    a = specimen.beam_area_m2
    for member in model.beams:
        np.testing.assert_allclose(member.instantiate()[1],
            specimen.density_kg_m3*np.diag((a,a,a,iy+iz,iy,iz)),rtol=1e-14)
    owner = ss.create_analysis(definition=model)
    try:
        preflight = owner.preflight()
        assert preflight['retained_dofs'] > 0
        assert len(preflight['arc_metric_node_order']) == 45
        assert not preflight['production_qualified']
        assert preflight['experimental_line_samples'] > 0
    finally:
        owner.close()


def test_composite_neutral_axis_correction_and_reference_load_wrench():
    specimen = capacity.Specimen()
    estimate = specimen.estimate()
    preload, probe = capacity.load_patterns(specimen,1234.,17.)
    shell, beam = capacity.reference_nodes(specimen)
    order = tuple(('shell',n) for n in shell)+tuple(('beam',n) for n in beam)
    positions = np.array(list(shell.values())+list(beam.values()))
    wrench = capacity.common_origin_wrench(positions,action_vector(preload,order))
    np.testing.assert_allclose(wrench,(-1234.,0.,0.,0.,
        -1234.*estimate['centroid_z_m'],1234.*specimen.width_m/2),rtol=1e-14,atol=1e-12)
    # Moving the reference origin to the loaded centroid removes its moment.
    shifted = positions - (specimen.length_m,specimen.width_m/2,estimate['centroid_z_m'])
    np.testing.assert_allclose(capacity.common_origin_wrench(shifted,
        action_vector(preload,order))[3:],0.,atol=1e-12)
    ei_naive = (specimen.young_pa*specimen.width_m*specimen.skin_m**3/12 +
        2*specimen.young_pa*(specimen.stiffener_width_m*specimen.stiffener_height_m**3/12 +
                             specimen.beam_area_m2*specimen.eccentricity_m**2))
    assert 0. < estimate['ei_effective_nm2'] < ei_naive
    np.testing.assert_allclose(action_vector(probe,order)[:,2].sum(),17.)


def test_generalized_force_comparator_and_fibre_stress_are_independent():
    s = capacity.Specimen(fibre_yield_pa=7e19,generalized_yield_force_n=3e15)
    sections = [member.instantiate()[0].section for member in capacity.definition(s).beams]
    for section in sections:
        if type(section) is b3_ge.EllipsoidalGeneralizedSection:
            assert section.yield_force == s.generalized_yield_force_n
            np.testing.assert_array_equal(section.metric,
                np.diag((1.,1.,1.,*(1/s.stiffener_height_m**2,)*3)))
            # Equal forces and moments scaled by h have equal q in newtons.
            force = np.array((1.,0.,0.,0.,0.,0.))
            moment = np.array((0.,0.,0.,s.stiffener_height_m,0.,0.))
            assert force@section.metric@force == pytest.approx(moment@section.metric@moment)
        else:
            assert all(f.curve.flow_stress[0] == s.fibre_yield_pa for f in section.fibres)
    for key in ('fibre_yield_pa','generalized_yield_force_n'):
        for value in (0.,-1.,float('nan'),float('inf')):
            with pytest.raises(ValueError): capacity.Specimen(**{key:value})


def test_opposite_probe_keeps_compression_and_full_seed_load():
    s = capacity.Specimen(); shell, beam = capacity.reference_nodes(s)
    order = tuple(('shell',n) for n in shell)+tuple(('beam',n) for n in beam)
    preload, positive = capacity.load_patterns(s,1200.,20.)
    seed = capacity.combine((1.,preload),(1.,positive))
    direction = capacity.combine((-1.,positive))
    for amplitude in (-.2,0.,1.,2.):
        actual = action_vector(seed,order)+amplitude*action_vector(direction,order)
        expected = action_vector(preload,order)+(1-amplitude)*action_vector(positive,order)
        np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-14)
        assert actual[:,0].sum() == pytest.approx(-1200.)
        assert np.count_nonzero(actual[:,3:]) == 0


def test_metric_quadrature_is_geometric_positive_and_unit_aware():
    specimen = capacity.Specimen(); shell, beam = capacity.reference_nodes(specimen)
    order = tuple(('shell',n) for n in shell)+tuple(('beam',n) for n in beam)
    metric = capacity.metric_for(order,specimen)
    assert metric.shell_nodes == 27 and len(metric.weights) == 45
    assert np.all(metric.weights > 0.)
    assert sum(metric.weights[:27]) == pytest.approx(.5)
    assert sum(metric.weights[27:]) == pytest.approx(.5)
    # Mathematical length-unit covariance, not an SI solver invocation in mm.
    scaled = replace(specimen,length_m=1000*specimen.length_m,
        width_m=1000*specimen.width_m,skin_m=1000*specimen.skin_m,
        stiffener_width_m=1000*specimen.stiffener_width_m,
        stiffener_height_m=1000*specimen.stiffener_height_m)
    other = capacity.metric_for(order,scaled)
    np.testing.assert_array_equal(metric.weights,other.weights)
    u = np.tile((.001,-.002,.003),(45,1)); frames = np.tile(np.eye(3),(45,1,1))
    assert metric.evaluate(u,frames,frames)[0] == pytest.approx(
        other.evaluate(1000*u,frames,frames)[0],rel=1e-14)
    for bad in (order[:-1],order[:-1]+(order[0],),order[::-1]):
        with pytest.raises(ValueError): capacity.metric_for(bad,specimen)


def test_bounded_stop_and_turning_report_does_not_assert_capacity():
    for count in (0,1,65,True):
        with pytest.raises(ValueError): capacity.workflow(arc_steps=count)
    with pytest.raises(ValueError): capacity.Specimen(nx=4)
    with pytest.raises(ValueError): capacity.definition(beams='generalized')
    observations = [dict(amplitude=a,generation=i) for i,a in enumerate((0.,1.,2.))]
    assert capacity.turning_observations(observations) == []
    observations[-1]['amplitude'] = .5
    assert capacity.turning_observations(observations) == [dict(
        generations=[0,1,2],kind='amplitude_direction_change_unvalidated')]


def test_full_seed_origin_catches_first_arc_turn_and_excludes_partial_seed():
    origin = dict(amplitude=None,generation=5,state=dict(status='completed',load_factor=1.),
        affine_origin_diagnosis=dict(amplitude=0.,provenance='completed_factor_one_declared_seed'))
    arc = [dict(amplitude=.1,generation=6),dict(amplitude=.05,generation=7)]
    assert capacity.turning_observations([origin]+arc) == [dict(
        generations=[5,6,7],kind='amplitude_direction_change_unvalidated')]
    assert origin['amplitude'] is None and origin['state']['load_factor'] == 1.
    partial = {**origin,'state':dict(status='completed',load_factor=.75)}
    assert capacity.turning_observations([partial]+arc) == []
    unbound = dict(amplitude=None,generation=5,state=dict(status='completed',load_factor=1.))
    assert capacity.turning_observations([unbound]+arc) == []


def test_streamed_and_retained_first_turn_diagnosis_match():
    retained, streamed = [], []
    provenance = 'declared_seed_actions_times_accepted_public_load_factor'
    for factor in (.75,1.):
        observation = dict(amplitude=None,generation=4 if factor == .75 else 5,
            state=dict(status='completed',load_factor=factor),applied_load_provenance=provenance)
        capacity._bind_seed_origin(observation,'seed')
        capacity._record_observation(retained,observation,streamed.append)
    for generation,amplitude in ((6,.1),(7,.05)):
        capacity._record_observation(retained,dict(generation=generation,amplitude=amplitude),streamed.append)
    assert 'affine_origin_diagnosis' not in streamed[0]
    assert streamed[1]['amplitude'] is None and streamed[1]['state']['load_factor'] == 1.
    expected = [dict(generations=[5,6,7],kind='amplitude_direction_change_unvalidated')]
    assert capacity.turning_observations(streamed) == expected
    assert capacity.turning_observations(retained) == expected


def test_nonzero_material_history_or_update_invalidates_elastic_screen():
    # Isolated observation-contract data, not a fabricated solver endpoint.
    fibre_ids = (2,4,6,8)
    records = [dict(element_id=i,fibres=[dict(plastic_increment=(0.,0.),branch='ELASTIC')])
               for i in fibre_ids]
    station = dict(accumulated=(0.,0.),plastic_increment=(0.,0.),
                   plastic_strain=[0.]*6,plastic_strain_low=[0.]*6,branch='ELASTIC')
    generalized = [dict(element_id=i,stations=[dict(station)]) for i in (1,3,5,7)]
    result = SimpleNamespace(beam_fibre_material=records,beam_generalized_material=generalized,
                             beam_station_accumulated=((0.,),)*8)
    virgin = capacity.material_activity(result)
    assert not virgin['any_material_history_or_update']
    assert virgin['elastic_fold_interpretation_eligible']
    generalized[0]['stations'][0]['accumulated'] = (0.,1e-100)
    active = capacity.material_activity(result)
    assert active['generalized_history_accumulated']
    summary = capacity.initial_elastic_screen([dict(material_activity=active)])
    assert not summary['elastic_fold_interpretation_eligible']
    assert not summary['capacity_or_stability_proved']
    generalized[0]['stations'][0]['accumulated'] = (0.,0.)
    result.beam_station_accumulated = ((0.,),(1e-100,))+((0.,),)*6
    assert capacity.material_activity(result)['fibre_history_accumulated']
    result.beam_station_accumulated = ((0.,),)*8
    records[0]['fibres'][0]['plastic_increment'] = (0.,1e-100)
    assert capacity.material_activity(result)['fibre_update_nonzero']


@pytest.mark.parametrize('error',(ValueError,TypeError,RuntimeError,MemoryError))
def test_programming_and_admission_errors_propagate_without_fabricated_state(monkeypatch,error):
    def refuse(**kwargs): raise error('deliberate construction refusal')
    monkeypatch.setattr(ss,'create_analysis',refuse)
    with pytest.raises(error,match='deliberate construction refusal'):
        capacity.workflow(arc_steps=2)


def test_observation_stream_owns_copy_and_propagates_writer_error():
    observations = []
    original = dict(generation=7,state=dict(beam_positions=[[1.,2.,3.]]))
    def mutate(delivered): delivered['state']['beam_positions'][0][0] = 999.
    capacity._record_observation(observations,original,mutate)
    original['state']['beam_positions'][0][0] = 333.
    assert observations[0]['state']['beam_positions'][0][0] == 1.
    def fail(delivered):
        delivered['state']['beam_positions'][0][0] = 888.
        raise OSError('observation stream unavailable')
    with pytest.raises(OSError,match='observation stream unavailable'):
        capacity._record_observation(observations,observations[0],fail)
    assert observations[0] == observations[1]
    assert observations[1]['state']['beam_positions'][0][0] == 1.


def test_ordinary_seed_load_reconstruction_preserves_retained_rows():
    s = capacity.Specimen(); shell,beam = capacity.reference_nodes(s)
    order = tuple(('shell',n) for n in shell)+tuple(('beam',n) for n in beam)
    preload,probe = capacity.load_patterns(s,1000.,20.)
    seed = capacity.combine((1.,preload),(1.,probe))
    vector = capacity.declared_load_vector(seed,order,6*45+48)
    np.testing.assert_array_equal(vector[:6*45].reshape(45,6),action_vector(seed,order))
    assert not np.any(vector[6*45:])


def test_only_typed_solver_failure_is_terminal_report(monkeypatch):
    def refuse(**kwargs): raise ss.ArcSelectionFailure('retained attempts',())
    monkeypatch.setattr(ss,'create_analysis',refuse)
    report = capacity.workflow(arc_steps=2)
    assert report['status'] == 'typed_failure'
    assert report['failure']['type'] == 'ArcSelectionFailure'
    assert report['failure']['attempts'] == []
    assert not report['observations'] and not report['checkpoint_inventory']


def assert_prefix_inventory(report):
    for row in report['checkpoint_inventory']:
        assert len(row['checkpoint_bytes']) == row['size_bytes']
        assert sha256(row['checkpoint_bytes']).hexdigest() == row['sha256']
    for observation in report['observations']:
        state = observation['state']
        for field in ('cell_frames','cell_resultants','multipliers','beam_frames',
                      'beam_positions','shell_frames','shell_chart_coordinates',
                      'shell_rotation_anchors','applied_dead_load','support_reactions',
                      'beam_station_accumulated','beam_fibre_material',
                      'beam_generalized_material','beam_shell_line_work'):
            assert field in state
        assert len(observation['physical_positions_m']) == 45
        assert len(observation['physical_applied_rows']) == 45
        assert np.shape(observation['shell_physical_frames']) == (27,3,3)
        assert np.shape(observation['shell_chart_coordinates']) == (27,6)


def test_public_candidate_restart_or_honest_bounded_failure():
    """Future numerical diagnostic; failure cannot be counted as route acceptance."""
    report = capacity.workflow(arc_steps=2)
    assert report['status'] in ('bounded_no_turning_observed','turning_candidate_unvalidated',
                               'seed_failed','typed_failure')
    assert not report['capacity_claim'] and not report['production_qualified']
    assert_prefix_inventory(report)
    if report['status'] in ('bounded_no_turning_observed','turning_candidate_unvalidated'):
        assert report['restart_checks'] == [dict(prefix_bytes_equal=True,continued_bytes_equal=True,
            prefix_recovery_equal=True,continued_recovery_equal=True)]
        assert len(report['observations']) == 7
        assert report['observations'][-1]['state']['beam_fibre_material']
        assert report['observations'][-1]['state']['beam_generalized_material']
    else:
        assert report.get('failure') or report.get('seed_status')


def test_public_accepted_arc_cancellation_preserves_restartable_endpoint():
    """Future diagnostic: if seed fails, cancellation integration remains unproved."""
    token = CancellationToken()
    observed = []
    def cancel(result):
        observed.append(result.generation)
        # Solver accepted observers own copies. Mutation must not alter owner
        # recovery, saved observation, checkpoint or the restored endpoint.
        try:
            result.beam_positions[0,0] += 100.
        except ValueError:
            pass  # A read-only callback copy is also safe ownership.
        token.cancel('retain the first accepted arc endpoint')
    report = capacity.workflow(arc_steps=2,cancellation_token=token,on_accepted=cancel)
    assert_prefix_inventory(report)
    if observed:
        assert len(observed) == 1 and report['status'] == 'cancelled'
        assert report['observations'][-1]['generation'] == observed[0]
        assert report['checkpoint_inventory'][-1]['generation'] == observed[0]
        raw = report['checkpoint_inventory'][-1]['checkpoint_bytes']
        fresh = ss.create_analysis(definition=capacity.definition())
        try:
            with pytest.raises(ValueError): fresh.restore(raw[:-1])
            restored = fresh.restore(raw)
            assert restored.status == 'accepted_prefix'
            assert fresh.checkpoint() == raw
            np.testing.assert_array_equal(fresh.recover().beam_positions,
                report['observations'][-1]['state']['beam_positions'])
        finally:
            fresh.close()
    else:
        assert report['status'] in ('seed_failed','typed_failure')
        assert report.get('failure') or report.get('seed_status')
