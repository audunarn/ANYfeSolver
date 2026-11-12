"""Generalized/Q4-J2 yielded predecessor, adaptive unload/reload and restart."""
from pathlib import Path
from math import fsum
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from test_stiffened_shell_q4_pullback_owner import _definition
from test_stiffened_shell_native_point_displacement import owner,native_state,OPTIONS,CONTROL,DIRECTION
from test_stiffened_shell_native_generalized_adaptive_history import exact
from anyfesolver._stiffened_shell_displacement_adaptive import decode


def test_generalized_q4_j2_adaptive_reload_exact_restart(request):
    model=_definition(plastic=True,mixed=False)
    first=owner(request,model);second=owner(request,model)
    # Actual accepted elastic unload from the existing1000N(.5,1.) yielded seed.
    raw=(Path(__file__).parent/'data/native_generalized_q4_j2_history/unloaded-v1.bin').read_bytes()
    previous=first.restore(raw);repeated=second.restore(raw);exact(previous,repeated)
    assert first.checkpoint()==second.checkpoint()==raw
    assert all(fsum(s.plastic_increment)==0. for e in previous.beam_generalized_material for s in e.stations)
    context,_=decode(first,raw)
    targets=context.state.targets
    assert not context.state.completed and len(targets)==2
    # Resume exact declared pending programme; do not synthesize a new target.
    kw=dict(OPTIONS,displacement_control=CONTROL,increment_control=ss.AdaptiveIncrementControl(6,24))
    token=CancellationToken();token.cancel('before coupled reload')
    with pytest.raises(SolveCancelled):first.solve(DIRECTION,targets,cancellation_token=token,**kw)
    assert first.checkpoint()==raw
    before=native_state(first)
    current=first.solve(DIRECTION,targets,**kw);again=second.solve(DIRECTION,targets,**kw)
    exact(current,again);assert first.checkpoint()==second.checkpoint()
    after=native_state(first)
    for name in after:np.testing.assert_array_equal(after[name],native_state(second)[name])
    delta=np.array([fsum((*a,*(-b))) for a,b in zip(after['beam_accumulated'],before['beam_accumulated'],strict=True)])
    new=np.array([fsum(r) for r in after['beam_accumulated']]);old=np.array([fsum(r) for r in before['beam_accumulated']])
    budget=64.*np.finfo(float).eps*np.maximum(abs(new),abs(old))
    assert np.all(delta>=-budget) and np.any(delta>budget)
    assert not np.array_equal(before['beam_plastic'],after['beam_plastic'])
    stations=[s for e in current.beam_generalized_material for s in e.stations]
    increments=[fsum(s.plastic_increment) for s in stations];density=[fsum(s.dissipation_per_length) for s in stations]
    assert any(i>0. for i in increments) and all(d>=0. for d in density)
    assert max(abs(d-4000.*i)/max(abs(d),abs(4000.*i),np.finfo(float).tiny) for d,i in zip(density,increments))<=1e-10
    shell_changed=False
    for a,b,c in zip(current.shell_q4_material,previous.shell_q4_material,again.shell_q4_material,strict=True):
        assert a.element_id==b.element_id==c.element_id
        for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
            np.testing.assert_array_equal(getattr(a,name),getattr(c,name))
        allowance=64.*np.finfo(float).eps*np.maximum(abs(a.alpha),abs(b.alpha))
        assert np.all(a.alpha>=b.alpha-allowance)
        shell_changed=shell_changed or bool(np.any(a.alpha>b.alpha+allowance))
    assert shell_changed and any(not np.array_equal(a.plastic_strain,b.plastic_strain) for a,b in zip(current.shell_q4_material,previous.shell_q4_material,strict=True))
    force_index=first._shell_count+6*first._beam_index[105]+2
    expected=np.zeros_like(current.applied_dead_load);expected[force_index]=1000.+100.*current.amplitude
    np.testing.assert_array_equal(current.applied_dead_load,expected)
    positions=np.vstack((np.asarray(model.shell.coordinates_m)+current.shell_displacement.reshape(-1,6)[:,:3],current.beam_positions))
    count=6*len(positions);net=(current.support_reactions[:count]+expected[:count]).reshape(-1,6)
    assert np.linalg.norm(net[:,:3].sum(0))/1000.<=1e-8
    assert np.linalg.norm((net[:,3:]+np.cross(positions,net[:,:3])).sum(0))/2000.<=1e-8
    assert current.free_equilibrium_ratio<=1e-8 and current.constraint_max<=1e-9
    assert not current.production_qualified
