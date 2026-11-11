"""Native same-predecessor chart events, atomic replay and continued histories."""
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot,encode_snapshot,identities
from anyfesolver._stiffened_shell_displacement_adaptive import decode
from test_stiffened_shell_native_point_displacement import owner,OPTIONS,native_state
from test_stiffened_shell_native_generalized_adaptive_history import exact
from test_stiffened_shell_q4_pullback_owner import _definition
from examples.stiffened_shell_coupled_s3_adaptive_history import definition as s3_definition


@pytest.mark.parametrize('family',('q4-generalized','s3-fibre'))
def test_native_material_chart_rebase_atomic_continue(request,monkeypatch,family):
    model=(_definition(plastic=True,mixed=False) if family=='q4-generalized' else s3_definition())
    fixture=('native_generalized_q4_j2_history/unloaded-v1.bin' if family=='q4-generalized' else 'native_s3_fibre_history/unloaded-v1.bin')
    raw=(Path(__file__).parent/'data'/fixture).read_bytes()
    analysis=owner(request,model);resumed=owner(request,model);baseline=owner(request,model);bad=owner(request,model)
    previous=analysis.restore(raw);baseline.restore(raw)
    assert analysis.capabilities['displacement_chart_rebase']=='experimental_native_point_material_chart_v1'
    context,_=decode(analysis,raw);targets=context.state.targets
    inner=analysis._displacement_snapshot[0];metadata,arrays=decode_snapshot(analysis,inner)
    material=native_state(analysis);driver=analysis._finite_driver
    with pytest.raises(ValueError,match='generation'):analysis.rebase_rotations(previous.generation-1)
    assert analysis.checkpoint()==raw and analysis._finite_driver is driver
    token=CancellationToken();token.cancel('before native chart publication')
    with pytest.raises(SolveCancelled):analysis.rebase_rotations(previous.generation,cancellation_token=token)
    assert analysis.checkpoint()==raw and analysis._finite_driver is driver
    # Cancellation after isolated replay, immediately before locked publication.
    import anyfesolver._stiffened_shell_displacement_control as displacement
    safe=displacement.cancellation_safe_point;token=CancellationToken()
    def cancel_before_publish(value,stage):
        if stage=='stiffened-shell.displacement.publish':token.cancel('prepared native chart')
        return safe(value,stage)
    with monkeypatch.context() as patch:
        patch.setattr(displacement,'cancellation_safe_point',cancel_before_publish)
        with pytest.raises(SolveCancelled):analysis.rebase_rotations(previous.generation,cancellation_token=token)
    assert analysis.checkpoint()==raw and analysis._finite_driver is driver
    for name,value in material.items():np.testing.assert_array_equal(native_state(analysis)[name],value)
    rebased=analysis.rebase_rotations(previous.generation)
    changed=analysis.checkpoint();m,a=decode_snapshot(analysis,analysis._displacement_snapshot[0])
    assert m['policy']==identities(analysis)[2] and m['chart_events']==1
    assert rebased.generation==previous.generation+1
    assert rebased.history==previous.history and rebased.increment_attempts==previous.increment_attempts
    assert rebased.amplitude==previous.amplitude and rebased.target==previous.target
    assert not np.any(rebased.shell_chart_coordinates.reshape(-1,6)[:,3:])
    assert np.array_equal(rebased.shell_frames,previous.shell_frames)
    for name in arrays:
        if name in ('q','shell_anchors'):continue
        if name=='s3_states':assert a[name]==arrays[name]
        else:np.testing.assert_array_equal(a[name],arrays[name])
    for name,value in material.items():np.testing.assert_array_equal(native_state(analysis)[name],value)
    for name in ('beam_positions','beam_frames','cell_frames','cell_resultants','multipliers','applied_dead_load'):
        np.testing.assert_array_equal(getattr(rebased,name),getattr(previous,name))
    np.testing.assert_allclose(rebased.support_reactions,previous.support_reactions,rtol=1e-8,atol=1e-8)
    for old,new in zip(previous.shell_q4_material+previous.shell_s3_material,rebased.shell_q4_material+rebased.shell_s3_material,strict=True):
        for name in ('layer_strain','layer_stress','plastic_strain','alpha'):np.testing.assert_array_equal(getattr(old,name),getattr(new,name))
    with pytest.raises(ValueError,match='no nonzero'):analysis.rebase_rotations(rebased.generation)
    assert analysis.checkpoint()==changed
    # A digest-valid chart with a changed supported physical pose is not admissible.
    forged={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in a.items()}
    forged['shell_anchors'][0]=Rotation.from_rotvec((.01,0.,0.)).as_matrix()
    forged['shell_frames'][0]=forged['shell_anchors'][0]
    keys={k:v for k,v in m.items() if k not in ('policy','model','steps')}
    corrupt=encode_snapshot(analysis,arrays=forged,**keys)
    with pytest.raises(ValueError,match='anchor violates rotational support'):bad.restore(corrupt)
    assert bad._generation==0 and bad._finite_driver is None
    if model.shell.s3_elements:
        forged={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in a.items()}
        forged['shell_origin_anchors'][0]=Rotation.from_rotvec((.01,0.,0.)).as_matrix()
        corrupt=encode_snapshot(analysis,arrays=forged,**keys)
        with pytest.raises(ValueError,match='predecessor anchor violates rotational support'):bad.restore(corrupt)
        assert bad._generation==0 and bad._finite_driver is None
    restored=resumed.restore(changed);assert resumed.checkpoint()==changed;exact(restored,rebased)
    directions=(ss.NodalAction('beam',105,(0.,0.,100.,0.,0.,0.)),) if family=='q4-generalized' else tuple(ss.NodalAction('beam',node,(0.,0.,150.,0.,0.,0.)) for node in (105,205))
    kw=dict(OPTIONS,displacement_control=ss.DisplacementControl('beam',105,2),increment_control=ss.AdaptiveIncrementControl(6,24))
    final=analysis.solve(directions,targets,**kw);again=resumed.solve(directions,targets,**kw)
    exact(final,again);assert analysis.checkpoint()==resumed.checkpoint()
    unrebased=baseline.solve(directions,targets,**kw)
    for name in ('beam_positions','beam_frames','cell_frames','cell_resultants','shell_frames'):
        np.testing.assert_allclose(getattr(final,name),getattr(unrebased,name),rtol=1e-9,atol=1e-12)
    np.testing.assert_allclose(final.support_reactions,unrebased.support_reactions,rtol=1e-8,atol=1e-8)
    for left,right in zip(final.shell_q4_material+final.shell_s3_material,unrebased.shell_q4_material+unrebased.shell_s3_material,strict=True):
        np.testing.assert_allclose(left.alpha,right.alpha,rtol=1e-9,atol=1e-14)
        np.testing.assert_allclose(left.plastic_strain,right.plastic_strain,rtol=1e-9,atol=1e-14)
    assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
    assert not final.production_qualified
