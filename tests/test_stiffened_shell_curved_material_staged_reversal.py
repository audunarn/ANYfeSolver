"""Bounded actual signed material history, not a local strain-only reversal."""
from dataclasses import replace
import json
import numpy as np
from anyfesolver import stiffened_shell as ss
from test_stiffened_shell_curved_material import curved_material_panel,make_owner,_actions
from test_stiffened_shell_curved_material_staged import OPTIONS
from test_stiffened_shell_curved_material_displacement import material_equal


def test_curved_material_staged_actual_force_reversal(request,tmp_path):
    d=replace(curved_material_panel(),retained_scale_policy='native_force_moment_roles_v1')
    owner=make_owner(request,d)
    plan=ss.StagedDeadLoadProgramme(owner.preflight()['identity'],(
        _actions(),(ss.NodalAction('beam',105,(0.,0.,-1000.,0.,0.,0.)),),_actions()))
    seed=owner.solve(plan,(.5,1.),**OPTIONS)
    raw=owner.checkpoint();(tmp_path/'positive-prefix.bin').write_bytes(raw)
    fresh=make_owner(request,d);fresh.restore(raw)
    control=ss.AdaptiveIncrementControl(4,64)
    result=owner.solve(plan,(2.,),**OPTIONS,increment_control=control)
    repeated=fresh.solve(plan,(2.,),**OPTIONS,increment_control=control)
    assert owner.checkpoint()==fresh.checkpoint();material_equal(result,repeated)
    assert result.load_position==2. and not result.load_programme_complete
    assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
    load=owner._finite_driver._programme.load_at(2.)
    np.testing.assert_array_equal(result.applied_dead_load,load)
    dof=owner._shell_count+6*owner._beam_index[105]+2
    assert load[dof]==-1000.
    assert np.all(np.asarray(result.shell_layer_alpha)>=np.asarray(seed.shell_layer_alpha))
    assert np.all(np.asarray(result.beam_station_accumulated)>=np.asarray(seed.beam_station_accumulated))
    assert np.any(np.asarray(result.beam_station_accumulated)>np.asarray(seed.beam_station_accumulated))
    negative=result
    result=owner.solve(plan,(3.,),**OPTIONS,increment_control=control)
    repeated=fresh.solve(plan,(3.,),**OPTIONS,increment_control=control)
    assert owner.checkpoint()==fresh.checkpoint();material_equal(result,repeated)
    assert result.load_position==3. and result.load_programme_complete
    assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
    np.testing.assert_array_equal(result.applied_dead_load,owner._finite_driver._programme.load_at(3.))
    assert result.applied_dead_load[dof]==1000.
    assert np.all(np.asarray(result.shell_layer_alpha)>=np.asarray(negative.shell_layer_alpha))
    assert np.all(np.asarray(result.beam_station_accumulated)>=np.asarray(negative.beam_station_accumulated))
    restored=make_owner(request,d);again=restored.restore(owner.checkpoint())
    assert restored.checkpoint()==owner.checkpoint();material_equal(result,again)
    (tmp_path/'reversal.json').write_text(json.dumps(dict(
        accepted_positions=result.accepted_factors,equilibrium=result.free_equilibrium_ratio,
        constraint=result.constraint_max,loads=(1000.,-1000.,1000.),exact_restart=True,
        shell_alpha=result.shell_layer_alpha,beam_accumulated=result.beam_station_accumulated,
        attempts=len(result.increment_attempts)),indent=2),encoding='utf-8')
