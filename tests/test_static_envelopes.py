import numpy as np
import pytest

from anyfesolver.static_envelopes import EnvelopeReducer


def test_signed_extrema_ties_and_repeated_case_labels():
    reducer = EnvelopeReducer()
    for index, value in [(3, 8.), (0, -8.), (1, 8.), (2, -4.)]:
        reducer.update({('stress', 12, 'top', 'xx'): value}, case_id='same',
                       input_index=index, time=index * .1)
    extrema = reducer.extrema[('stress', 12, 'top', 'xx')]
    assert extrema.minimum.value == -8.
    assert extrema.maximum.input_index == 1
    assert extrema.absolute_maximum.input_index == 0
    assert extrema.absolute_maximum.value == -8.
    assert extrema.absolute_maximum.time == 0.
    assert not reducer.complete
    reducer.finish()
    assert reducer.complete


def test_derived_quantity_keeps_governing_physical_state():
    reducer = EnvelopeReducer()
    cases = [(10., 0., 0.), (6., -6., 2.)]
    for i, (sx, sy, tau) in enumerate(cases):
        vm = np.sqrt(sx*sx - sx*sy + sy*sy + 3*tau*tau)
        reducer.update({'vm': vm}, case_id=f'case-{i}', input_index=i,
                       associated={'vm': {'xx': sx, 'yy': sy, 'xy': tau}})
    governing = reducer.extrema['vm'].maximum
    assert governing.case_id == 'case-1'
    assert dict(governing.associated) == {'xx': 6., 'yy': -6., 'xy': 2.}
    with pytest.raises(TypeError):
        governing.associated['xx'] = 10.


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_failure_is_atomic_and_incomplete(bad):
    reducer = EnvelopeReducer()
    reducer.update({'a': 1., 'b': 2.}, case_id=0, input_index=0)
    reducer.finish()
    with pytest.raises(ValueError, match='non-finite'):
        reducer.update({'a': 10., 'b': bad}, case_id=1, input_index=1)
    assert reducer.extrema['a'].maximum.value == 1.
    assert reducer.case_count == 1
    assert not reducer.complete


def test_selection_cannot_grow_with_case_count_and_inputs_are_copied():
    reducer = EnvelopeReducer()
    associated = {'v': {'xx': 2.}}
    reducer.update({'v': 1.}, case_id='a', input_index=0, associated=associated)
    associated['v']['xx'] = 99.
    with pytest.raises(ValueError, match='selection changed'):
        reducer.update({'v': 2., 'new': 4.}, case_id='b', input_index=1)
    for i in range(1, 100):
        reducer.update({'v': i}, case_id='a', input_index=i)
    assert len(reducer.extrema) == 1
    assert reducer.extrema['v'].minimum.associated['xx'] == 2.
    assert reducer.case_count == 100


def test_empty_and_invalid_identity():
    reducer = EnvelopeReducer()
    reducer.finish()
    assert reducer.complete and not reducer.extrema
    with pytest.raises(ValueError, match='input_index'):
        reducer.update({}, case_id='x', input_index=-1)
    with pytest.raises(ValueError, match='time'):
        reducer.update({}, case_id='x', input_index=0, time=np.nan)


def test_block_ingestion_preserves_stations_and_zero_reaction_cases():
    from types import SimpleNamespace
    def block(index, scalar):
        return SimpleNamespace(
            output_dofs=[6], displacements=np.array([[scalar]]),
            input_indices=[index], case_ids=['repeated'], times=[index*.1],
            case_results=[{'status':'converged','reaction_node_ids':(1,),
                'element_stresses':{9:{'top':np.array([scalar,2*scalar]),'von_mises':abs(scalar)}},
                'constraint_forces':{'support_reactions':{} if scalar==0 else {1:np.array([scalar,0,0,0,0,0])},
                    'mpc_slave_forces':{},'mpc_master_equivalent_forces':{}}}])
    reducer=EnvelopeReducer()
    reducer.update_batch(block(0,0.))
    reducer.update_batch(block(1,-3.))
    selected=reducer.extrema[('stress',9,'top',1)].absolute_maximum
    assert selected.value==-6. and selected.input_index==1
    assert selected.associated[('top',0)]==-3.
    assert reducer.extrema[('support_reactions',1,0)].maximum.input_index==0
    assert reducer.extrema[('support_reactions',1,0)].minimum.value==-3.
    bad=block(2,4.)
    bad.output_dofs=[6,7]
    with pytest.raises(ValueError,match='dimensions'):
        reducer.update_batch(bad)
    assert reducer.case_count==2


def test_update_batch_rolls_back_entire_block_and_stays_incomplete():
    from types import SimpleNamespace
    def case(status='converged', stress=1.):
        return {'status': status,
                'element_stresses': {9: {'top': np.array([stress, 2.])}},
                'constraint_forces': {}}
    def block(results):
        return SimpleNamespace(output_dofs=[6],
            displacements=np.zeros((1, len(results))),
            input_indices=list(range(len(results))),
            case_ids=['a']*len(results), times=[0.]*len(results),
            case_results=results)
    reducer = EnvelopeReducer()
    reducer.update_batch(block([case(stress=1.), case(stress=2.)]))
    reducer.finish()
    assert reducer.case_count == 2 and reducer.complete
    with pytest.raises(ValueError, match='uncertified'):
        reducer.update_batch(block([case(stress=5.), case(status='failed')]))
    assert reducer.case_count == 2
    assert reducer.extrema[('stress', 9, 'top', 0)].maximum.value == 2.
    assert not reducer.complete


def test_input_indices_must_be_integral_nonbool_without_truncation():
    from types import SimpleNamespace
    def block(indices, value=0.):
        return SimpleNamespace(output_dofs=[6],
            displacements=np.full((1, len(indices)), value),
            input_indices=indices, case_ids=['a']*len(indices),
            times=[0.]*len(indices),
            case_results=[{'status': 'converged', 'constraint_forces': {}}
                          for _ in indices])
    reducer = EnvelopeReducer()
    with pytest.raises(ValueError, match='input_index'):
        reducer.update_batch(block([2.9]))
    with pytest.raises(ValueError, match='input_index'):
        reducer.update_batch(block([True]))
    with pytest.raises(ValueError, match='input_index'):
        reducer.update({}, case_id='x', input_index=2.9)
    assert reducer.case_count == 0 and not reducer.extrema
    reducer.update_batch(block([np.int64(4)], value=-1.))
    reducer.update_batch(block([2.0], value=3.))
    assert reducer.case_count == 2
    assert reducer.extrema[('displacement', 6)].minimum.input_index == 4
    assert reducer.extrema[('displacement', 6)].maximum.input_index == 2


def test_complex_stresses_rejected_and_bool_metadata_ignored():
    from types import SimpleNamespace
    def block(stresses):
        return SimpleNamespace(output_dofs=[], displacements=np.zeros((0, 1)),
            input_indices=[0], case_ids=['a'], times=[0.],
            case_results=[{'status': 'converged', 'element_stresses': stresses,
                           'constraint_forces': {}}])
    reducer = EnvelopeReducer()
    reducer.update_batch(block({9: {'top': np.array([1., 2.]), 'flag': True}}))
    assert set(reducer.extrema) == {('stress', 9, 'top', 0),
                                    ('stress', 9, 'top', 1)}
    with pytest.raises(ValueError, match='complex'):
        reducer.update_batch(block({9: {'top': np.array([1. + 1j])}}))
    with pytest.raises(ValueError, match='complex'):
        reducer.update_batch(block({9: {'top': 1 + 2j}}))
    assert reducer.case_count == 1


def test_reaction_entries_must_be_six_component_real_vectors():
    from types import SimpleNamespace
    def block(forces):
        return SimpleNamespace(output_dofs=[], displacements=np.zeros((0, 1)),
            input_indices=[0], case_ids=['a'], times=[0.],
            case_results=[{'status': 'converged', 'reaction_node_ids': (1,),
                           'constraint_forces': forces}])
    reducer = EnvelopeReducer()
    reducer.update_batch(block({'support_reactions': {}}))
    assert reducer.extrema[('support_reactions', 1, 3)].maximum.value == 0.
    with pytest.raises(ValueError, match='six-component'):
        reducer.update_batch(block({'support_reactions':
                                    {1: np.array([1., 0, 0])}}))
    with pytest.raises(ValueError, match='six-component'):
        reducer.update_batch(block({'support_reactions':
                                    {1: np.array([1j]*6)}}))
    assert reducer.case_count == 1


def test_reaction_selection_ignores_forces_outside_selected_nodes():
    from types import SimpleNamespace
    block = SimpleNamespace(output_dofs=[], displacements=np.zeros((0, 1)),
        input_indices=[0], case_ids=['a'], times=[0.],
        case_results=[{'status': 'converged', 'reaction_node_ids': (1,),
            'constraint_forces': {'support_reactions': {
                1: np.array([1., 0, 0, 0, 0, 0]),
                2: np.array([9., 0, 0, 0, 0, 0])}}}])
    reducer = EnvelopeReducer()
    reducer.update_batch(block)
    assert set(reducer.extrema) == {('support_reactions', 1, c)
                                    for c in range(6)}
    assert reducer.extrema[('support_reactions', 1, 0)].maximum.value == 1.
