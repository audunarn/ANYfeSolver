"""Connected finite-state assembly tests; no production qualification."""

import numpy as np
import pytest

from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from docs.reference_cases import ge_beam3_curved_p5_assembly_history_probe as assembly
from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import rotation, T6
from docs.reference_cases.ge_beam3_curved_p5_continuum_probe import parabolic_references, discrete_tip_compliance
from docs.reference_cases.ge_beam3_curved_p5_nonlinear_mixed_probe import NonlinearMixedBeamProbe
from docs.reference_cases.ge_beam3_curved_p5_section_probe import DirectedHardeningSectionProbe
from test_ge_beam3_curved_p5_history_path_probe import NonlinearCantileverHistoryProbe, digest
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law, section, A


def model(count=2, *, order=8, laws=None, refs=None, maps=None):
    refs = parabolic_references(.4, count) if refs is None else refs
    maps = tuple((2*i,2*i+1,2*i+2) for i in range(count)) if maps is None else maps
    return assembly.NonlinearAssemblyHistoryProbe(refs, maps, [law() for _ in refs] if laws is None else laws, order=order)


def forces(count=2):
    result = np.zeros((2*count+1, 3))
    result[-1] = [.1, -.3, .2]
    return result


@pytest.fixture(scope='module')
def loaded():
    made = model()
    return made, made.trial(.1*forces())


@pytest.fixture(scope='module')
def cycle():
    made = model()
    results = []
    for amplitude in (.1, .2, .1, 0., -.1, 0.):
        before = digest(made.committed)
        trial = made.trial(amplitude*forces())
        assert digest(made.committed) == before
        made.commit(trial)
        assert digest(made.replay()) == digest(trial.response)
        results.append(trial)
    return made, results


def test_whole_model_cycle_and_element_owned_histories(cycle):
    made, results = cycle
    assert made.committed.epoch == 6
    assert len(made.committed.histories) == 2
    assert all(len(h) == 16 for h in made.committed.histories)
    assert np.linalg.norm(made.committed.positions[-1]-made._coordinates[-1]) > .1
    assert [sum(s.response.plastic_active for e in t.response.elements for s in e.stations)
            for t in results] == [32, 32, 0, 0, 0, 0]
    for previous, current in zip(results, results[1:]):
        histories = tuple(tuple(s.response.history for s in e.stations) for e in previous.response.elements)
        assert current.origins == histories
        for origin, response in zip(histories, current.response.elements):
            for old, station in zip(origin, response.stations):
                assert station.response.origin == old
                assert station.response.history.accumulated >= old.accumulated
                assert station.response.dissipation_increment >= 0
    for trial in results:
        assert trial.residual_norm <= 1e-11
        assert trial.iterations <= 16 and trial.mixed_evaluations <= 512
        assert np.array_equal(trial.positions[0], made._coordinates[0])
        assert np.array_equal(trial.rotations[0], np.eye(3))


def test_shared_node_action_reaction_and_global_current_moment_balance(cycle):
    made, results = cycle
    for trial in results:
        first, second = [e.residual.reshape(3,6) for e in trial.response.elements]
        assert np.linalg.norm(first[2]+second[0]) <= 1e-11
        total = trial.response.residual.reshape(5,6)
        assert np.linalg.norm(total[:, :3].sum(axis=0)) <= 1e-11
        assert np.linalg.norm((total[:,3:]+np.cross(trial.positions,total[:,:3])).sum(axis=0)) <= 1e-11
        assert np.linalg.norm(total[1:,:3]-trial.forces[1:]) <= 1e-11
        assert np.linalg.norm(total[1:,3:]) <= 1e-11
        external = np.column_stack((trial.forces,np.zeros((5,3))))
        reactions = total-external
        assert np.linalg.norm(reactions[:,:3].sum(axis=0)+external[:,:3].sum(axis=0)) <= 1e-11
        moment = reactions[:,3:]+np.cross(trial.positions,reactions[:,:3]+external[:,:3])
        assert np.linalg.norm(moment.sum(axis=0)) <= 1e-11


def test_manual_scatter_work_and_linear_tip_compliance():
    made = model(laws=[law(1000.),law(1000.)])
    state = made.committed
    response = made.response_at(state.positions,state.rotations)
    tangent = np.zeros((30,30))
    residual = np.zeros(30)
    for row, element in zip(((0,1,2),(2,3,4)), response.elements):
        for i, node_i in enumerate(row):
            residual[6*node_i:6*node_i+6] += element.residual[6*i:6*i+6]
            for j, node_j in enumerate(row):
                tangent[6*node_i:6*node_i+6,6*node_j:6*node_j+6] += element.tangent[6*i:6*i+6,6*j:6*j+6]
    assert np.array_equal(tangent,response.tangent)
    assert np.array_equal(residual,response.residual)
    direction = np.sin(np.arange(30))/9
    work = sum(direction[d] @ e.tangent @ direction[d] for d,e in zip(made._dofs,response.elements))
    assert abs(work-direction @ tangent @ direction) <= 1e-11
    loads = np.zeros((24,6));loads[-6:]=np.eye(6)
    compliance = np.linalg.solve(tangent[6:,6:],loads)[-6:]
    expected = discrete_tip_compliance(parabolic_references(.4,2),section())
    assert np.linalg.norm(compliance-expected) <= 1e-11*np.linalg.norm(expected)


def test_one_element_matches_existing_history_driver():
    ref = parabolic_references(.4,1)[0]
    old = NonlinearCantileverHistoryProbe(ref,law(),order=8)
    new = model(1)
    for amplitude in (.1,.2,0.):
        a,b = old.trial(amplitude*forces(1)),new.trial(amplitude*forces(1))
        assert np.linalg.norm(a.positions-b.positions) <= 1e-11
        assert np.linalg.norm(a.frames-b.rotations @ ref.nodal_triads) <= 1e-11
        assert np.linalg.norm(a.response.tangent-b.response.tangent) <= 1e-11*np.linalg.norm(a.response.tangent)
        old.commit(a);new.commit(b)
        for h,k in zip(old.committed.histories,new.committed.histories[0]):
            assert abs(h.plastic_coordinate-k.plastic_coordinate) <= 1e-11
            assert abs(h.accumulated-k.accumulated) <= 1e-11


def test_global_condensed_tangent_in_common_increment_chart(loaded):
    made, trial = loaded
    direction = np.sin(np.arange(30)+.2)/8
    step = 1e-6
    gradients, energies = [], []
    for sign in (1.,-1.):
        delta = (sign*step*direction).reshape(5,6)
        x = trial.positions+delta[:,:3]
        u = np.array([rotation(d[3:]) @ r for d,r in zip(delta,trial.rotations)])
        solved = made.response_at(x,u)
        gradient = np.zeros(30)
        potential = 0.
        for ref,row,dofs,section_law,old,new in zip(made._references,made._maps,made._dofs,
                made._sections,trial.response.elements,solved.elements):
            local = NonlinearMixedBeamProbe(ref,section_law,order=8)
            check = local.evaluate(trial.positions[row],trial.rotations[row] @ ref.nodal_triads,
                                   new.local_rotations,new.moments,increment=np.r_[delta[row].ravel(),np.zeros(18)])
            assert tuple(s.response.plastic_active for s in old.stations) == tuple(s.response.plastic_active for s in check.stations)
            gradient[dofs] += check.residual[:18]
            potential += check.potential
        gradients.append(gradient);energies.append(potential)
    assert np.linalg.norm((gradients[0]-gradients[1])/(2*step)-trial.response.tangent @ direction) <= 1e-7
    assert abs((energies[0]-energies[1])/(2*step)-trial.response.residual @ direction) <= 1e-7
    assert np.linalg.norm(trial.response.tangent-trial.response.tangent.T) <= 1e-11*np.linalg.norm(trial.response.tangent)


def test_finite_objectivity_with_distinct_reference_rolls(loaded):
    _, trial = loaded
    refs = list(parabolic_references(.4,2))
    refs[1] = CurvedBeam3ReferenceGeometry(refs[1].coordinates, refs[1].nodal_triads @ rotation([.37,0.,0.]))
    made = model(refs=refs)
    original = made.response_at(trial.positions,trial.rotations)
    assert not np.array_equal(refs[0].nodal_triads[-1],refs[1].nodal_triads[0])
    g, c = rotation([2.1,-1.8,2.5]),np.array([2.,-3.,4.])
    moved = made.response_at(trial.positions @ g.T+c,g @ trial.rotations)
    transform = np.kron(np.eye(10),g)
    assert abs(original.potential-moved.potential) <= 1e-11
    assert np.linalg.norm(moved.residual-transform @ original.residual) <= 1e-11
    assert np.linalg.norm(moved.tangent-transform @ original.tangent @ transform.T) <= 1e-11*np.linalg.norm(original.tangent)
    # The shared spatial rotation does not erase either element's material roll.
    for ref,row,law_i,response in zip(refs,made._maps,made._sections,original.elements):
        expected = NonlinearMixedBeamProbe(ref,law_i,order=8).solve(trial.positions[row],trial.rotations[row] @ ref.nodal_triads)
        assert digest(expected) == digest(response)


def test_reversed_element_map_preserves_global_response(loaded):
    made,trial = loaded
    refs = parabolic_references(.4,2)
    reversed_law = DirectedHardeningSectionProbe(T6 @ section() @ T6.T,T6 @ A,.02,.4)
    other = model(refs=[refs[0],refs[1].reversed()],maps=[(0,1,2),(4,3,2)],laws=[law(),reversed_law])
    response = other.response_at(trial.positions,trial.rotations)
    assert abs(response.potential-trial.response.potential) <= 1e-11
    assert np.linalg.norm(response.residual-trial.response.residual) <= 1e-11
    assert np.linalg.norm(response.tangent-trial.response.tangent) <= 1e-11*np.linalg.norm(response.tangent)


def test_late_element_failure_keeps_whole_checkpoint_atomic(monkeypatch):
    made = model()
    made.commit(made.trial(.1*forces()))
    before,replay = digest(made.committed),digest(made.replay())
    trial = made.trial(.2*forces())
    original = made._reconstruct_element
    calls = []
    def fail_last(index, value):
        calls.append(index)
        if index == 1:
            raise ValueError('injected last-element validation failure')
        return original(index,value)
    with monkeypatch.context() as context:
        context.setattr(made,'_reconstruct_element',fail_last)
        with pytest.raises(assembly.AssemblyTransactionError,match='validation'):
            made.commit(trial)
    assert calls == [0,1]
    assert digest(made.committed) == before
    assert digest(made.replay()) == replay
    made.discard(trial)


def test_late_station_failure_and_mutation_cannot_partially_commit(monkeypatch):
    made = model()
    before = digest(made.committed)
    trial = made.trial(.1*forces())
    original,calls = assembly._history,[]
    def fail_last(history):
        calls.append(1)
        if len(calls) == 32:
            raise ValueError('injected last station failure')
        return original(history)
    with monkeypatch.context() as context:
        context.setattr(assembly,'_history',fail_last)
        with pytest.raises(assembly.AssemblyTransactionError):
            made.commit(trial)
    assert len(calls) == 32 and digest(made.committed) == before
    last = trial.response.elements[-1].stations[-1].response.resultants
    last.setflags(write=True);last[0] += .1
    with pytest.raises(assembly.AssemblyTransactionError,match='altered'):
        made.commit(trial)
    assert digest(made.committed) == before


def test_budgets_stale_foreign_and_discard_leave_state_unchanged():
    made = model()
    before = digest(made.committed)
    for budget in (0,1,3):
        with pytest.raises(assembly.AssemblyPathError,match='budget'):
            made.trial(.1*forces(),max_mixed_evaluations=budget)
        assert digest(made.committed) == before
    with pytest.raises(assembly.AssemblyPathError,match='budget'):
        made.trial(.1*forces(),max_iterations=0)
    for kwargs in ({'max_iterations':17},{'max_mixed_evaluations':513},{'max_iterations':True}):
        with pytest.raises(ValueError,match='bounded'):
            made.trial(forces(),**kwargs)
    first = made.trial(np.zeros((5,3)))
    other = made.trial(np.zeros((5,3)))
    for owner,value in ((made,first),(model(),other)):
        with pytest.raises(assembly.AssemblyTransactionError):
            owner.commit(value)
    made.discard(other)
    with pytest.raises(assembly.AssemblyTransactionError):
        made.commit(other)
    assert digest(made.committed) == before


def test_commit_replay_without_newton_and_defensive_state(monkeypatch):
    made = model()
    trial = made.trial(.1*forces())
    def forbidden(*args,**kwargs):
        raise AssertionError('commit/replay must not solve')
    monkeypatch.setattr(NonlinearMixedBeamProbe,'solve',forbidden)
    made.commit(trial)
    before = digest(made.committed)
    assert digest(made.replay()) == digest(trial.response)
    copied = made.committed
    copied.positions[:] = 99.
    trial.positions.setflags(write=True);trial.positions[:] = 99.
    assert digest(made.committed) == before
    assert not np.any(made.replay().residual == 99.)


@pytest.mark.parametrize('maps', [[(0,0,2),(2,3,4)],[(0,1,2),(3,4,5)],
                                   [(0,1,2),(2,3,5)],[(0,1,2),(2,1,0)],[(False,1,2),(2,3,4)]])
def test_invalid_connectivity_is_rejected(maps):
    with pytest.raises(ValueError):
        model(maps=maps)


def test_inconsistent_shared_geometry_and_unsupported_sizes_rejected():
    refs = parabolic_references(.4,2)
    shifted = refs[1].rigidly_transformed(np.eye(3),[.1,0.,0.])
    with pytest.raises(ValueError,match='shared'):
        model(refs=[refs[0],shifted])
    for clamps in ((),(0,0),(9,),(True,)):
        with pytest.raises(ValueError):
            assembly.NonlinearAssemblyHistoryProbe(refs,[(0,1,2),(2,3,4)],[law(),law()],fixed_nodes=clamps)
    with pytest.raises(ValueError):
        assembly.NonlinearAssemblyHistoryProbe([refs[0]]*9,[(0,1,2)]*9,[law()]*9)


def test_default_48_stations_per_element():
    made = model(order=24)
    trial = made.trial(.1*forces())
    made.commit(trial)
    assert [len(h) for h in made.committed.histories] == [48,48]
    assert digest(made.replay()) == digest(trial.response)


def test_closed_piecewise_quadratic_ring_has_six_rigid_modes():
    diagonal = np.sqrt(.5)
    nodes = np.array([[1.,0.,0.],[diagonal,diagonal,0.],[0.,1.,0.],[-diagonal,diagonal,0.],
                      [-1.,0.,0.],[-diagonal,-diagonal,0.],[0.,-1.,0.],[diagonal,-diagonal,0.]])
    maps = [(0,1,2),(2,3,4),(4,5,6),(6,7,0)]
    refs = []
    for row in maps:
        coordinates = nodes[list(row)]
        frames = []
        for xi in (-1.,0.,1.):
            tangent = np.array([xi-.5,-2*xi,xi+.5]) @ coordinates
            tangent /= np.linalg.norm(tangent)
            normal = np.array([0.,0.,1.])
            frames.append(np.column_stack((tangent,normal,np.cross(tangent,normal))))
        refs.append(CurvedBeam3ReferenceGeometry(coordinates,frames))
    made = assembly.NonlinearAssemblyHistoryProbe(refs,maps,[law(1000.)]*4,order=8)
    original = made.response_at(nodes,made.committed.rotations)
    spectrum = np.linalg.eigvalsh(original.tangent)
    assert np.max(np.abs(spectrum[:6])) <= 1e-11*spectrum[-1]
    assert spectrum[6] > 1e-5*spectrum[-1]
    rigid = np.zeros((48,6))
    for i,x in enumerate(nodes):
        rigid[6*i:6*i+3,:3] = np.eye(3)
        rigid[6*i:6*i+3,3:] = np.column_stack([np.cross(axis,x) for axis in np.eye(3)])
        rigid[6*i+3:6*i+6,3:] = np.eye(3)
    assert np.linalg.norm(original.tangent @ rigid) <= 1e-11*np.linalg.norm(original.tangent)
    g = rotation([2.1,-1.8,2.5])
    moved = made.response_at(nodes @ g.T+[2.,-3.,4.],np.tile(g,(8,1,1)))
    assert abs(moved.potential) <= 1e-11
    assert np.linalg.norm(moved.residual) <= 1e-11
    transform = np.kron(np.eye(16),g)
    assert np.linalg.norm(moved.tangent-transform @ original.tangent @ transform.T) <= 1e-11*np.linalg.norm(original.tangent)


def test_per_element_sections_and_rotation_invariant_residual_norm():
    second = DirectedHardeningSectionProbe(2*section(),A,.03,.7)
    made = model(laws=[law(),second])
    state = made.committed
    response = made.response_at(state.positions,state.rotations)
    expected = NonlinearMixedBeamProbe(made._references[1],second,order=8).solve(
        state.positions[2:],made._references[1].nodal_triads)
    assert digest(response.elements[1]) == digest(expected)
    residual = np.cos(np.arange(30)).reshape(5,6)
    force = forces()*100
    g = rotation([2.1,-1.8,2.5])
    moved = np.column_stack((residual[:,:3] @ g.T,residual[:,3:] @ g.T))
    assert abs(made._norm(residual.ravel(),force)-made._norm(moved.ravel(),force @ g.T)) <= 1e-11
