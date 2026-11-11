"""Actual simultaneous histories in two parallel native stiffeners, not G3 acceptance."""
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation
from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver import _stiffened_shell_ge_finite_experimental as finite
from anyfesolver._stiffened_shell_beam_history import arrays as history_arrays
import pytest
from examples.stiffened_shell_coupled_plastic_history import (
    definition, programme, activity, OPTIONS, CONTROL, ENDPOINT_FORCES_N,
)
from test_stiffened_shell_staged_material_history import _capture, _canonical, _material_close


def _pairs(rows):
    return np.array([math.fsum(row) for row in rows])


def _checks(owner, state, previous):
    result = state['result']
    assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
    assert result.load_position == result.accepted_factors[-1]
    assert result.load_programme_complete == (result.load_position == 4.)
    current = state['beam']['accepted_fibres']
    before = np.zeros_like(current) if previous is None else previous['beam']['accepted_fibres']
    assert current.shape == before.shape and current.shape[1] == 4
    z, p = _pairs(current[:,:2]), _pairs(current[:,2:])
    zp, pp = _pairs(before[:,:2]), _pairs(before[:,2:])
    dz = np.array([math.fsum((*a[:2],*-b[:2])) for a,b in zip(current,before)])
    dp = np.array([math.fsum((*a[2:],*-b[2:])) for a,b in zip(current,before)])
    budget = 64.*np.finfo(float).eps*np.maximum(np.abs(p),np.abs(pp))
    assert np.isfinite(current).all() and np.all(dp >= -budget)
    assert np.all(p >= np.abs(z)-64.*np.finfo(float).eps*np.abs(p))
    np.testing.assert_allclose(dp,np.abs(dz),rtol=1e-12,atol=float(np.max(budget)))
    assert np.count_nonzero(state['beam']['accepted_plastic']) == 0
    assert np.count_nonzero(state['beam']['accepted_accumulated']) == 0
    generalized_rows=[i for i,(e,_,_) in enumerate(owner._beam_parts)
                      if type(e.section) is not b3_ge.PhysicalFibreSection]
    assert generalized_rows and np.any(result.cell_resultants[generalized_rows] != 0.)
    assert {r.element_id for r in result.beam_fibre_material} == {1,2}
    keys=[(r.element_id,r.cell,r.station,f.fibre_id) for r in result.beam_fibre_material for f in r.fibres]
    assert len(keys) == len(current) and len(set(keys)) == len(keys)
    expected_keys=[(e.element_id,cell,index,f.fibre_id)
        for e,_,_ in owner._beam_parts if type(e.section) is b3_ge.PhysicalFibreSection
        for cell,index,xi,measure,frame in e.operator.stations for f in e.section.fibres]
    assert keys == expected_keys
    increments = np.array([math.fsum(f.plastic_increment) for r in result.beam_fibre_material for f in r.fibres])
    np.testing.assert_allclose(increments,dz,rtol=1e-12,atol=float(np.max(budget)))
    expected_d = {1:[],2:[]}
    for station in result.beam_fibre_material:
        for fibre in station.fibres:
            # Declared physical A=.0025m² and yield100kPa for fibre0.
            # Other fibres have1e20Pa; no duplicated elastic background energy.
            y0 = 1e5 if fibre.fibre_id == '0' else 1e20
            expected_d[station.element_id].append(station.measure*.0025*y0*abs(math.fsum(fibre.plastic_increment)))
    energy_ids=[e.element_id for e in result.beam_fibre_energy]
    assert len(energy_ids)==len(set(energy_ids))==2 and set(energy_ids)=={1,2}
    for energy in result.beam_fibre_energy:
        for field in ('stored_energy_j','previous_hardening_energy_j',
                      'incremental_dissipation_j','incremental_material_potential_j'):
            pair=getattr(energy,field)
            assert len(pair)==2 and all(math.isfinite(value) for value in pair)
        actual=math.fsum(energy.incremental_dissipation_j)
        expected=math.fsum(expected_d[energy.element_id])
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=0.)
    if previous is not None:
        for new,old in zip(result.shell_q4_material,previous['result'].shell_q4_material):
            work = new.layer_stress*(new.plastic_strain-old.plastic_strain)
            assert np.all(work.sum(axis=1) >= -64.*np.finfo(float).eps*np.abs(work).sum(axis=1))
    positions = np.vstack((np.asarray(owner._definition.shell.coordinates_m)+
        state['q'][:owner._shell_count].reshape(-1,6)[:,:3],result.beam_positions))
    ground = result.support_reactions[:6*len(positions)].reshape(-1,6)
    knots = (0.,)+ENDPOINT_FORCES_N
    j = min(int(result.load_position),3);fraction = result.load_position-j
    force = (1.-fraction)*knots[j]+fraction*knots[j+1]
    load = np.zeros_like(result.applied_dead_load)
    external_moment = np.zeros(3)
    for node in (105,205):
        index=owner._beam_index[node]
        load[owner._shell_count+6*index+2]=force
        external_moment += np.cross(result.beam_positions[index],(0.,0.,force))
    np.testing.assert_array_equal(result.applied_dead_load,load)
    force_error=ground[:,:3].sum(axis=0)+np.array((0.,0.,2.*force))
    moment_error=(ground[:,3:]+np.cross(positions,ground[:,:3])).sum(axis=0)+external_moment
    balance=dict(force_N=force_error.tolist(),moment_Nm=moment_error.tolist(),
                 force_ratio=float(np.linalg.norm(force_error)/3000.),
                 moment_ratio=float(np.linalg.norm(moment_error)/6000.))
    (state['path']/'balance.json').write_bytes(_canonical(balance))
    assert balance['force_ratio'] <= 1e-8 and balance['moment_ratio'] <= 1e-8
    return dict(accumulated_max=float(np.max(p)),signed_plastic= z.tolist(),
                actual_plastic_increment=dz.tolist(),actual_accumulation_increment=dp.tolist(),
                keys=keys,balance=balance)


def _compare(a,b,model):
    x,y=a['result'],b['result']
    assert x.load_position == y.load_position and x.accepted_factors == y.accepted_factors
    np.testing.assert_array_equal(x.applied_dead_load,y.applied_dead_load)
    np.testing.assert_allclose(x.beam_positions/2.,y.beam_positions/2.,rtol=0.,atol=1e-9)
    np.testing.assert_allclose(a['q'][:len(model.shell.node_ids)*6].reshape(-1,6)[:,:3]/2.,
                               b['q'][:len(model.shell.node_ids)*6].reshape(-1,6)[:,:3]/2.,rtol=0.,atol=1e-9)
    for field in ('shell_frames',):
        np.testing.assert_allclose(a[field],b[field],rtol=0.,atol=1e-9)
    for field in ('beam_frames','cell_frames'):
        np.testing.assert_allclose(getattr(x,field),getattr(y,field),rtol=0.,atol=1e-9)
    retained=np.array((3000.,)*6+(6000.,)*12)
    np.testing.assert_allclose(x.cell_resultants/retained,y.cell_resultants/retained,rtol=0.,atol=1e-9)
    wrench=np.array((3000.,)*3+(6000.,)*3)
    for field in ('multipliers','support_reactions'):
        np.testing.assert_allclose(getattr(x,field).reshape(-1,6)/wrench,
                                   getattr(y,field).reshape(-1,6)/wrench,rtol=0.,atol=1e-9)
    _material_close(a['shell'],b['shell'],model)
    _material_close(a['beam'],b['beam'],model)
    for field in ('shell_q4_material','beam_fibre_material','beam_fibre_energy'):
        assert len(getattr(x,field)) == len(getattr(y,field))
        for left,right in zip(getattr(x,field),getattr(y,field)):
            _material_close(left,right,model)


def test_two_stiffener_simultaneous_history_restart_rebase_and_atomicity(request, tmp_path, record_property, monkeypatch):
    model = definition()
    owner = ss.create_analysis(definition=model)
    request.addfinalizer(owner.close)
    loads = programme(owner)
    history = []
    states = []
    def observer(target,label,captures):
        def accepted(result):
            previous = captures[-1] if captures else None
            state = _capture(target,result,tmp_path/(label+'-'+str(len(captures))))
            details = _checks(target,state,previous)
            row = activity(result,previous['result'] if previous else None)
            row.update(details)
            row['shell_stress_abs_max_Pa'] = max(float(np.max(np.abs(r.layer_stress))) for r in result.shell_q4_material)
            captures.append(state)
            if captures is states:
                history.append(row)
                (tmp_path/'activity.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
        return accepted
    accepted=observer(owner,'control',states)
    (tmp_path/'programme.json').write_bytes(loads.serialize())
    record_property('raw_state_directory',str(tmp_path))
    result = owner.solve(loads,(.5,1.),increment_control=CONTROL,on_accepted=accepted,**OPTIONS)
    assert result.status == 'completed'
    assert result.load_position == 1. and not result.load_programme_complete
    assert max(r['equilibrium'] for r in history) <= 1e-8
    assert max(r['gap'] for r in history) <= 1e-9
    assert any(r['simultaneous'] for r in history), 'no actual accepted increment advanced both shell and fibres'
    prefix = states[-1]
    resumed=ss.create_analysis(definition=model);request.addfinalizer(resumed.close)
    resumed.restore(prefix['checkpoint']);assert resumed.checkpoint()==prefix['checkpoint']
    resumed_states=[_capture(resumed,resumed.recover(),tmp_path/'restored-prefix')]
    assert resumed_states[0]['exact_material']==prefix['exact_material']
    _compare(prefix,resumed_states[0],model)
    for target in (2.,3.):
        owner.solve(loads,(target,),increment_control=CONTROL,on_accepted=accepted,**OPTIONS)
    reverse = states[-1];saved = reverse['checkpoint'];count=len(states)
    accepted_before = owner.recover()
    forward = {(r.element_id,r.cell,r.station,f.fibre_id):math.fsum(f.plastic_increment)
               for r in prefix['result'].beam_fibre_material for f in r.fibres}
    assert any(forward[(r.element_id,r.cell,r.station,f.fibre_id)]*math.fsum(f.plastic_increment)<0.
               and math.fsum(f.stress)*math.fsum(f.plastic_increment)>0.
               for r in reverse['result'].beam_fibre_material for f in r.fibres), 'no actual reversed plastic flow/stress'
    driver=owner._finite_driver;original=driver._assemble
    def reject_yielded(*args,**kwargs):
        values=original(*args,**kwargs)
        candidate=dict(history_arrays(owner._beam_parts,values[-1],'accepted'))
        old=reverse['beam']['accepted_fibres']
        delta=np.array([math.fsum((*a[2:],*-b[2:])) for a,b in zip(candidate['accepted_fibres'],old)])
        shell=np.asarray([row['alpha'] for row in values[-2]])
        p,old_p=_pairs(candidate['accepted_fibres'][:,2:]),_pairs(old[:,2:])
        shell_old=np.asarray(reverse['result'].shell_layer_alpha)
        if (np.any(delta>64.*np.finfo(float).eps*np.maximum(abs(p),abs(old_p))) and
                np.any(shell-shell_old>64.*np.finfo(float).eps*np.maximum(abs(shell),abs(shell_old)))):
            (tmp_path/'rejected-candidate-material.json').write_bytes(_canonical((values[-2],values[-1])))
            raise RuntimeError('injected after simultaneous yielded trial')
        return values
    with monkeypatch.context() as patch:
        patch.setattr(driver,'_assemble',reject_yielded)
        with pytest.raises(RuntimeError,match='injected after simultaneous yielded trial'):
            owner.solve(loads,(4.,),increment_control=CONTROL,on_accepted=accepted,**OPTIONS)
    assert owner.checkpoint()==saved and owner.recover() is accepted_before and len(states)==count
    with pytest.raises(ss.AdaptiveIncrementExhausted) as numerical:
        owner.solve(loads,(4.,),increment_control=ss.AdaptiveIncrementControl(1,2),
                    max_iterations=1,max_backtracks=0,on_accepted=accepted)
    attempts=numerical.value.attempts
    (tmp_path/'numerical-rejection.json').write_bytes(_canonical(attempts))
    assert len(attempts)==2 and all(row.status=='rejected' for row in attempts)
    assert [row.factor for row in attempts]==[4.,3.5]
    assert owner.checkpoint()==saved and owner.recover() is accepted_before and len(states)==count
    token=CancellationToken();safe=finite.cancellation_safe_point
    recovered=finite._fibre_recovery;cancelled_candidates=[]
    def capture_recovery(target,trial,predecessors,proposed):
        records,energies=recovered(target,trial,predecessors,proposed)
        assert any(math.fsum(e.incremental_dissipation_j)>0. for e in energies)
        candidate=dict(history_arrays(target._beam_parts,proposed,'candidate'))
        np.savez_compressed(tmp_path/'cancelled-candidate.npz',q=trial.q,frames=trial.frames,
                            cells=trial.cells,resultants=trial.resultants,**candidate)
        cancelled_candidates.append((records,energies))
        return records,energies
    def cancel(value,stage):
        if stage=='stiffened-shell.finite.material-publish':token.cancel('after physical fibre recovery')
        return safe(value,stage)
    with monkeypatch.context() as patch:
        patch.setattr(finite,'cancellation_safe_point',cancel)
        patch.setattr(finite,'_fibre_recovery',capture_recovery)
        with pytest.raises(SolveCancelled,match='after physical fibre recovery'):
            owner.solve(loads,(4.,),increment_control=CONTROL,cancellation_token=token,on_accepted=accepted,**OPTIONS)
    assert owner.checkpoint()==saved and owner.recover() is accepted_before and len(states)==count
    assert len(cancelled_candidates)==1
    chart=ss.create_analysis(definition=model);request.addfinalizer(chart.close)
    chart.restore(saved);before=_capture(chart,chart.recover(),tmp_path/'chart-before')
    chart.rebase_rotations(expected_generation=chart.recover().generation)
    after=_capture(chart,chart.recover(),tmp_path/'chart-after')
    assert np.any(before['q'][:chart._shell_count].reshape(-1,6)[:,3:] != 0.)
    np.testing.assert_array_equal(after['q'][:chart._shell_count].reshape(-1,6)[:,3:],0.)
    assert after['result'].generation==before['result'].generation+1
    assert after['result'].load_position==before['result'].load_position
    assert before['exact_material']==after['exact_material'];_compare(before,after,model)
    chart_restored=ss.create_analysis(definition=model);request.addfinalizer(chart_restored.close)
    chart_restored.restore(after['checkpoint']);assert chart_restored.checkpoint()==after['checkpoint']
    owner.solve(loads,(4.,),increment_control=CONTROL,on_accepted=accepted,**OPTIONS)
    # Compare consecutive physical left-increment rotations at the same
    # shell nodes, using accepted poses rather than additive chart vectors.
    # Both updates must end in increments with simultaneous shell/fibre yield.
    noncommuting=[]
    for i in range(1,len(states)-1):
        if not (history[i]['simultaneous'] and history[i+1]['simultaneous']):
            continue
        before,mid,following=(states[j]['shell_frames'] for j in (i-1,i,i+1))
        first=Rotation.from_matrix(mid@np.swapaxes(before,1,2)).as_rotvec()
        second=Rotation.from_matrix(following@np.swapaxes(mid,1,2)).as_rotvec()
        norms=np.linalg.norm(first,axis=1)*np.linalg.norm(second,axis=1)
        bracket=np.linalg.norm(np.cross(first,second),axis=1)
        sine=np.divide(bracket,norms,out=np.zeros_like(bracket),where=norms>1e-10)
        noncommuting.extend(zip(bracket.tolist(),sine.tolist()))
    assert noncommuting, 'no adjacent simultaneous-yield physical rotation updates'
    qualifying=[row for row in noncommuting if row[0]>1e-12 and row[1]>1e-4]
    assert qualifying, 'no single node has a resolved noncommuting physical path'
    (tmp_path/'noncommuting-chart.json').write_text(json.dumps(dict(
        max_bracket_rad2=max(row[0] for row in noncommuting),
        max_axis_sine=max(row[1] for row in noncommuting),
        qualifying_nodes=len(qualifying)),indent=2)+'\n',encoding='utf-8')
    for target in (2.,3.,4.):
        start=len(resumed_states)
        resumed.solve(loads,(target,),increment_control=CONTROL,on_accepted=observer(resumed,'resumed',resumed_states),**OPTIONS)
        expected=[s for s in states if target-1.<s['result'].load_position<=target]
        actual=resumed_states[start:]
        assert len(actual)==len(expected)
        for a,b in zip(expected,actual):_compare(a,b,model)
    for target,label in ((chart,'rebased'),(chart_restored,'rebased-restored')):
        captures=[after]
        target.solve(loads,(4.,),increment_control=CONTROL,on_accepted=observer(target,label,captures),**OPTIONS)
        expected=[s for s in states if 3.<s['result'].load_position<=4.]
        assert len(captures[1:])==len(expected)
        for a,b in zip(expected,captures[1:]):_compare(a,b,model)
    assert owner.recover().load_position==4. and owner.recover().load_programme_complete
    assert owner.checkpoint()==resumed.checkpoint()
