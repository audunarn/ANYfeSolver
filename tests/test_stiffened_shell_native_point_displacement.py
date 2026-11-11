"""Native point displacement histories, experimental fixed-target admission."""
from dataclasses import replace
from math import fsum
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._stiffened_shell_displacement_checkpoint import (
    decode_snapshot,encode_snapshot,identities)
from anyfesolver._stiffened_shell_beam_history import arrays
from anyfesolver._stiffened_shell_q4_algorithmic_pullback import HISTORICAL_POLICY
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_ge_physical_fibre_finite import _yielding_definition


ACTIONS=(ss.NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
DIRECTION=(ss.NodalAction('beam',105,(0.,0.,100.,0.,0.,0.)),)
OPTIONS=dict(max_iterations=20,max_backtracks=12)
CONTROL=ss.DisplacementControl('beam',105,2)


def model(family):
    value=_panel() if family=='generalized' else _yielding_definition(mixed=family=='mixed')
    return replace(value,q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)


def owner(request,definition):
    result=ss.create_analysis(definition=definition)
    request.addfinalizer(result.close)
    return result


def native_state(control):
    return dict(arrays(control._beam_parts,control._finite_driver._beam_origins,'beam'))


def compare(a,b):
    assert a.history==b.history and a.amplitude==b.amplitude
    assert a.beam_station_accumulated==b.beam_station_accumulated
    for name in ('shell_displacement','beam_positions','beam_frames','cell_frames',
                 'cell_resultants','multipliers','support_reactions','applied_dead_load'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    for left,right in zip(a.beam_fibre_material,b.beam_fibre_material,strict=True):
        assert left.fibres==right.fibres
        np.testing.assert_array_equal(left.strain,right.strain)
    assert a.beam_fibre_energy==b.beam_fibre_energy


@pytest.mark.parametrize('family',('generalized','fibre','mixed'))
def test_native_fixed_displacement_cancel_restore_reload(request,monkeypatch,tmp_path,family):
    definition=model(family)
    control=owner(request,definition);fresh=owner(request,definition)
    assert control.capabilities['displacement_control']=='experimental_native_point_material_fixed_translation_v1'
    seed=control.solve(ACTIONS,(.5,1.),**OPTIONS)
    dof=control._shell_count+6*control._beam_index[105]+2
    initial=control._finite_driver._state.q[dof]
    targets=(initial*.99,initial*1.01)
    if family!='generalized':
        assert any(fsum(row.incremental_dissipation_j)>0. for row in seed.beam_fibre_energy)
    token=CancellationToken()
    observed=[]
    def committed(result):
        assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
        assert result.target==targets[0]
        assert control.recover().generation==result.generation
        assert bool(result.beam_generalized_material)==(family!='fibre')
        assert bool(result.beam_fibre_material)==(family!='generalized')
        assert not np.shares_memory(result.applied_dead_load,control.recover().applied_dead_load)
        result.applied_dead_load[0]=123.
        assert control.recover().applied_dead_load[0]==0.
        result.applied_dead_load[0]=0.
        assert all(fsum(row.incremental_dissipation_j)>=0. for row in result.beam_fibre_energy)
        observed.append(result);token.cancel('accepted displacement prefix')
    kw=dict(displacement_control=CONTROL,**OPTIONS)
    with pytest.raises(SolveCancelled):
        control.solve(DIRECTION,targets,**kw,cancellation_token=token,on_accepted=committed)
    saved=control.checkpoint();prefix=control.recover()
    (tmp_path/(family+'-prefix.checkpoint')).write_bytes(saved)
    assert observed and len(prefix.history)==1
    m,a=decode_snapshot(control,saved)
    assert m['policy']=='B3_GE_NATIVE_POINT_MATERIAL_DISPLACEMENT_V1_DEV'
    assert all(not row.flags.writeable for row in a.values())
    restored=fresh.restore(saved)
    assert fresh.checkpoint()==saved;compare(prefix,restored)
    for name,value in native_state(control).items():
        np.testing.assert_array_equal(value,native_state(fresh)[name])
    cancelled=CancellationToken();cancelled.cancel('before another target')
    with pytest.raises(SolveCancelled):
        control.solve(DIRECTION,(targets[1],),**kw,cancellation_token=cancelled)
    assert control.checkpoint()==saved
    import anyfesolver._stiffened_shell_displacement_corrector as corrector
    with monkeypatch.context() as patch:
        patch.setattr(corrector,'bordered_correction',lambda matrix,*args:(np.zeros(matrix.shape[0]),0.))
        with pytest.raises(ss.DisplacementCorrectorFailure):
            control.solve(DIRECTION,(targets[1],),**kw)
    assert control.checkpoint()==saved
    with pytest.raises(ValueError,match='match restart'):
        control.solve(ACTIONS,(targets[1],),**kw)
    with pytest.raises(ValueError,match='generation'):
        control.rebase_rotations(prefix.generation-1)
    assert control.checkpoint()==saved
    metadata={k:m[k] for k in ('generation','seed_generation','seed_sha256','control_dof')}
    fields=['applied_load','frames','beam_accumulated' if family=='generalized' else 'beam_fibres',
            'beam_origin_accumulated' if family=='generalized' else 'beam_origin_fibres']
    untouched=owner(request,definition)
    for name in fields:
        changed={key:value.copy() for key,value in a.items()}
        changed[name].flat[0]+=1e-5
        with pytest.raises(ValueError):
            bad=encode_snapshot(control,arrays=changed,**metadata)
            untouched.restore(bad)
        assert untouched._generation==0 and untouched._finite_driver is None
        assert fresh.checkpoint()==saved
    from anyfesolver._stiffened_shell_displacement_control import publish
    with pytest.raises(ValueError,match='stale displacement publication'):
        publish(control,saved,expected_driver=control._finite_driver,
                expected_generation=control._generation+1)
    assert control.checkpoint()==saved
    with pytest.raises(ValueError,match='generation/history'):
        encode_snapshot(control,arrays=dict(a),**metadata,chart_events=1)
    final=control.solve(DIRECTION,(targets[1],),**kw)
    resumed=fresh.solve(DIRECTION,(targets[1],),**kw)
    assert control.checkpoint()==fresh.checkpoint();compare(final,resumed)
    assert final.history[-1][0]==targets[1] and not final.production_qualified
    if family!='generalized':
        assert max(max(row) for row in final.beam_station_accumulated)>0.
    positions=np.vstack((np.asarray(definition.shell.coordinates_m)+
                         final.shell_displacement.reshape(-1,6)[:,:3],final.beam_positions))
    nodal=6*len(positions)
    balance=(final.support_reactions[:nodal]+final.applied_dead_load[:nodal]).reshape(-1,6)
    assert np.linalg.norm(balance[:,:3].sum(axis=0))/1000.<=1e-8
    assert np.linalg.norm((balance[:,3:]+np.cross(positions,balance[:,:3])).sum(axis=0))/2000.<=1e-8
    (tmp_path/(family+'-completed.checkpoint')).write_bytes(control.checkpoint())
    if family=='generalized':
        force=owner(request,definition)
        equivalent=(ss.NodalAction('beam',105,(0.,0.,float(final.applied_dead_load[dof]),0.,0.,0.)),)
        oracle=force.solve(equivalent,(1.,),**OPTIONS)
        assert np.linalg.norm(final.beam_positions-oracle.beam_positions)/control._reference_length<=1e-8
        assert np.linalg.norm(final.cell_resultants-oracle.cell_resultants)/1000.<=1e-8


def test_native_displacement_scope_guards_are_atomic(request):
    control=owner(request,model('mixed'))
    with pytest.raises(ValueError,match='accepted force-controlled seed'):
        control.solve(DIRECTION,(1e-5,),displacement_control=CONTROL)
    control.solve(ACTIONS,(.5,),**OPTIONS)
    saved=control.checkpoint();dof=control._shell_count+6*control._beam_index[105]+2
    target=control._finite_driver._state.q[dof]*1.01
    with pytest.raises(ValueError,match='branch event'):
        control.solve(DIRECTION,(target,),displacement_control=replace(CONTROL,trial_policy=ss.GENERALIZED_BRANCH_EVENT_TRIAL))
    with pytest.raises(ValueError,match='branch event'):
        control.solve(DIRECTION,(target,),displacement_control=replace(CONTROL,trial_policy=ss.GENERALIZED_BRANCH_EVENT_TRIAL),increment_control=ss.AdaptiveIncrementControl())
    assert control.checkpoint()==saved
    historical=owner(request,replace(model('mixed'),q4_pullback_policy=HISTORICAL_POLICY))
    assert 'displacement_control' not in historical.capabilities
    with pytest.raises(ValueError,match='displacement control required'):
        historical.solve(DIRECTION,(target,),displacement_control=CONTROL)
    assert historical._generation==0
    staged=owner(request,model('mixed'))
    programme=ss.StagedDeadLoadProgramme(staged._identity,(ACTIONS,))
    staged.solve(programme,(.5,),**OPTIONS)
    staged_saved=staged.checkpoint()
    with pytest.raises(ValueError,match='ordinary unrebased force seed'):
        staged.solve(DIRECTION,(target,),displacement_control=CONTROL)
    assert staged.checkpoint()==staged_saved
