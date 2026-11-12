"""Actual simultaneous S3/fibre plasticity, signed continuation and atomic restart."""
from pathlib import Path
import math
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._stiffened_shell_displacement_adaptive import decode,encode
from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot,encode_snapshot
from examples.stiffened_shell_coupled_s3_adaptive_history import definition,activity,DIRECTION,CONTROL,OPTIONS
from test_stiffened_shell_native_point_displacement import owner
from test_stiffened_shell_native_plastic_s3_displacement import compare


def test_actual_s3_fibre_unload_reverse_reload_restart_and_atomicity(request):
    model=definition()
    analysis=owner(request,model);resumed=owner(request,model);bad=owner(request,model)
    raw=(Path(__file__).parent/'data/native_s3_fibre_history/unloaded-v1.bin').read_bytes()
    prefix=analysis.restore(raw);repeated=resumed.restore(raw);compare(prefix,repeated)
    assert analysis.checkpoint()==resumed.checkpoint()==raw
    assert prefix.shell_s3_material and prefix.beam_fibre_material and prefix.beam_generalized_material
    assert max(max(row.alpha) for row in prefix.shell_s3_material)>0.
    assert math.fsum(math.fsum(row.incremental_dissipation_j) for row in prefix.beam_fibre_energy)>0.
    context,inner=decode(analysis,raw)
    assert not context.state.completed and len(context.state.attempts)==1
    metadata,arrays=decode_snapshot(analysis,inner)
    changed=dict(arrays);changed['beam_fibres']=arrays['beam_fibres'].copy()
    changed['beam_fibres'][0,2]+=1e-6
    corrupt=encode_snapshot(analysis,arrays=changed,
        **{k:v for k,v in metadata.items() if k not in ('policy','model','steps')})
    with pytest.raises(ValueError,match='displacement native material replay mismatch'):
        bad.restore(encode(analysis._identity,context,corrupt,native=True))
    assert bad._generation==0 and bad._finite_driver is None
    kw=dict(OPTIONS,displacement_control=CONTROL,increment_control=context.state.limits)
    cancelled=CancellationToken();cancelled.cancel('before reversal trial')
    with pytest.raises(SolveCancelled):
        analysis.solve(DIRECTION,context.state.targets,**kw,cancellation_token=cancelled)
    assert analysis.checkpoint()==raw and analysis.recover() is prefix
    previous=prefix;rows=[]
    def observe(result):
        nonlocal previous
        assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
        rows.append(activity(result,previous));previous=result
    result=analysis.solve(DIRECTION,context.state.targets,**kw,on_accepted=observe)
    compare(result,resumed.solve(DIRECTION,context.state.targets,**kw))
    assert result.increment_attempts[:len(context.state.attempts)]==context.state.attempts
    assert analysis.checkpoint()==resumed.checkpoint()
    assert result.status=='completed' and not result.production_qualified
    assert any(row['target']<0. and row['simultaneous'] for row in rows)
    assert rows[-1]['target']>0. and rows[-1]['simultaneous']
    assert all(row['last_increment_fibre_dissipation_j']>=0. for row in rows)
    assert np.any(np.concatenate([x.alpha for x in result.shell_s3_material])>
                  np.concatenate([x.alpha for x in prefix.shell_s3_material]))
