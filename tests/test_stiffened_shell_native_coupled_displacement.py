"""Coupled Q4 J2/native fibre fixed-target history; not aggregate qualification."""
from math import fsum
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot,encode_snapshot
from examples.stiffened_shell_coupled_plastic_history import definition,OPTIONS
from test_stiffened_shell_native_point_displacement import owner,compare,native_state


def test_coupled_native_displacement_material_prefix_restart(request,monkeypatch,tmp_path):
    model=definition()
    analysis=owner(request,model);restored=owner(request,model)
    assert analysis.capabilities['displacement_control']=='experimental_native_point_coupled_j2_fixed_translation_v1'
    loads=tuple(ss.NodalAction('beam',node,(0.,0.,1500.,0.,0.,0.)) for node in (105,205))
    seed=analysis.solve(loads,(.25,.5,.75,1.),**OPTIONS)
    assert seed.shell_q4_material and seed.beam_fibre_material and seed.beam_fibre_energy
    assert max(max(row) for row in seed.shell_layer_alpha)>0.
    assert max(max(row) for row in seed.beam_station_accumulated)>0.
    assert any(fsum(row.incremental_dissipation_j)>0. for row in seed.beam_fibre_energy)
    dof=analysis._shell_count+6*analysis._beam_index[105]+2
    initial=analysis._finite_driver._state.q[dof]
    targets=(initial*.99,initial*1.01)
    direction=tuple(ss.NodalAction('beam',node,(0.,0.,150.,0.,0.,0.)) for node in (105,205))
    control=ss.DisplacementControl('beam',105,2)
    kw=dict(displacement_control=control,**OPTIONS)
    token=CancellationToken();prefixes=[]
    def observe(result):
        assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
        assert result.shell_q4_material and result.beam_fibre_material and result.beam_generalized_material
        assert result.shell_layer_alpha==analysis.recover().shell_layer_alpha
        assert all(fsum(row.incremental_dissipation_j)>=0. for row in result.beam_fibre_energy)
        prefixes.append(result);token.cancel('accepted coupled material prefix')
    with pytest.raises(SolveCancelled):
        analysis.solve(direction,targets,**kw,cancellation_token=token,on_accepted=observe)
    raw=analysis.checkpoint();prefix=analysis.recover()
    assert prefixes and len(prefix.history)==1
    (tmp_path/'coupled-prefix.checkpoint').write_bytes(raw)
    m,a=decode_snapshot(analysis,raw)
    assert m['policy']=='B3_GE_NATIVE_POINT_COUPLED_J2_DISPLACEMENT_V1_DEV'
    assert a['shell_alpha'].shape==(2,12) and np.max(a['shell_alpha'])>0.
    replay=restored.restore(raw)
    assert restored.checkpoint()==raw;compare(prefix,replay)
    for left,right in zip(prefix.shell_q4_material,replay.shell_q4_material,strict=True):
        for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
            np.testing.assert_array_equal(getattr(left,name),getattr(right,name))
    for name,value in native_state(analysis).items():
        np.testing.assert_array_equal(value,native_state(restored)[name])
    untouched=owner(request,model)
    metadata={k:m[k] for k in ('generation','seed_generation','seed_sha256','control_dof')}
    for name in ('shell_alpha','shell_plastic','shell_origin_alpha','shell_origin_plastic'):
        changed={key:value.copy() for key,value in a.items()}
        changed[name].flat[0]+=1e-5
        bad=encode_snapshot(analysis,arrays=changed,**metadata)
        with pytest.raises(ValueError):untouched.restore(bad)
        assert untouched._generation==0 and untouched._finite_driver is None
    import anyfesolver._stiffened_shell_displacement_corrector as corrector
    with monkeypatch.context() as patch:
        patch.setattr(corrector,'bordered_correction',lambda matrix,*args:(np.zeros(matrix.shape[0]),0.))
        with pytest.raises(ss.DisplacementCorrectorFailure):
            analysis.solve(direction,(targets[1],),**kw)
    assert analysis.checkpoint()==raw
    final=analysis.solve(direction,(targets[1],),**kw)
    other=restored.solve(direction,(targets[1],),**kw)
    assert analysis.checkpoint()==restored.checkpoint();compare(final,other)
    for before,after in zip(prefix.shell_q4_material,final.shell_q4_material,strict=True):
        assert np.all(after.alpha>=before.alpha-64*np.finfo(float).eps*np.maximum(abs(after.alpha),abs(before.alpha)))
    assert any(np.any(after.alpha>before.alpha) for before,after in zip(prefix.shell_q4_material,final.shell_q4_material))
    assert any(fsum(row.incremental_dissipation_j)>0. for row in final.beam_fibre_energy)
    assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
    positions=np.vstack((np.asarray(model.shell.coordinates_m)+
        final.shell_displacement.reshape(-1,6)[:,:3],final.beam_positions))
    n=6*len(positions)
    net=(final.support_reactions[:n]+final.applied_dead_load[:n]).reshape(-1,6)
    assert np.linalg.norm(net[:,:3].sum(axis=0))/3000.<=1e-8
    assert np.linalg.norm((net[:,3:]+np.cross(positions,net[:,:3])).sum(axis=0))/6000.<=1e-8
    (tmp_path/'coupled-completed.checkpoint').write_bytes(analysis.checkpoint())
