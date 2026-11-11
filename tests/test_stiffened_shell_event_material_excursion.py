"""Frozen local material excursion; not full signed reversal qualification."""
from pathlib import Path
from hashlib import sha256
from decimal import Decimal,localcontext
from math import fsum
import numpy as np
from anyfesolver import stiffened_shell as ss
from test_stiffened_shell_bl5_coupled_material import coupled
from test_stiffened_shell_bl5_material import make
from test_stiffened_shell_bl5_finite import _action

SEED_SHA256="0be9beeb00fe12297e49ac326b1ca61ec403650be20d5d518f1de6d1b5181673"


def fields(result):
    stations=[row for element in result.beam_generalized_material for row in element.stations]
    values={name:np.array([getattr(row,name) for row in stations]) for name in (
        "local_strain","local_strain_low","local_resultants","local_resultants_low",
        "plastic_strain","plastic_strain_low","accumulated","stored_energy_per_length",
        "dissipation_per_length","plastic_increment","reference_frame","current_frame","measure")}
    values.update(shell_alpha=np.array(result.shell_layer_alpha),applied_load=result.applied_dead_load,
                  beam_positions=result.beam_positions)
    return values


def test_local_unload_reload_with_material_restart(request,tmp_path):
    seed=(Path(__file__).parent/'data/stiffened_shell_event_excursion/force-seed.bin').read_bytes()
    assert sha256(seed).hexdigest()==SEED_SHA256
    owner=make(request,coupled());owner.restore(seed)
    options=dict(displacement_control=ss.DisplacementControl("beam",105,2,
        trial_policy=ss.GENERALIZED_BRANCH_EVENT_TRIAL),max_iterations=20,max_backtracks=12)
    restored=None;previous=None;loads=[]
    for index,target in enumerate((-66e-6,-68e-6,-66e-6,-68e-6)):
        result=owner.solve(_action(-15.625),(target,),**options)
        assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
        current=fields(result);raw=owner.checkpoint()
        (tmp_path/f'step-{index}.bin').write_bytes(raw)
        np.savez(tmp_path/f'fields-{index}.npz',**current)
        assert all(fsum(pair)>=0. for pair in current['dissipation_per_length'])
        if previous is not None:
            with localcontext() as context:
                context.prec=80
                delta=[sum(map(Decimal.from_float,a))-sum(map(Decimal.from_float,b))
                       for a,b in zip(current['accumulated'],previous['accumulated'])]
            floor=64*np.finfo(float).eps*max(np.max(abs(current['accumulated'])),
                np.max(abs(previous['accumulated'])),np.finfo(float).tiny)
            assert min(delta)>=-Decimal.from_float(float(floor))
            floor=64*np.finfo(float).eps*max(np.max(abs(current['shell_alpha'])),
                np.max(abs(previous['shell_alpha'])),np.finfo(float).tiny)
            assert np.min(current['shell_alpha']-previous['shell_alpha'])>=-floor
        if restored is not None:
            repeated=restored.solve(_action(-15.625),(target,),**options)
            assert restored.checkpoint()==raw
            replayed=fields(repeated)
            for name in current:np.testing.assert_array_equal(current[name],replayed[name])
            np.savez(tmp_path/f'restored-{index}.npz',**replayed)
        if index==1:
            restored=make(request,coupled());restored.restore(raw)
            assert restored.checkpoint()==raw
        loads.append(float(np.linalg.norm(result.applied_dead_load)))
        previous=current
    assert loads[2]<loads[1] and loads[3]>loads[2]
