"""Actual yielded mixed prefix restoration; the full nominal history still fails."""
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot,encode_snapshot
from test_stiffened_shell_native_plastic_s3_displacement import definition,compare
from test_stiffened_shell_native_point_displacement import owner,DIRECTION,CONTROL,OPTIONS


def test_actual_yielded_mixed_prefix_recovery_restore_and_rejection(request):
    model=definition(True,True)
    analysis=owner(request,model);resumed=owner(request,model);bad=owner(request,model)
    raw=(Path(__file__).parent/'data/native_mixed_j2_displacement/quarter-prefix-v1.bin').read_bytes()
    accepted=analysis.restore(raw);repeated=resumed.restore(raw)
    assert analysis.checkpoint()==resumed.checkpoint()==raw
    compare(accepted,repeated)
    assert len(accepted.shell_q4_material)==1 and len(accepted.shell_s3_material)==2
    assert all(np.max(row.alpha)>0. for row in accepted.shell_q4_material+accepted.shell_s3_material)
    assert accepted.free_equilibrium_ratio<=1e-8 and accepted.constraint_max<=1e-9
    assert not accepted.production_qualified
    metadata,arrays=decode_snapshot(analysis,raw)
    np.testing.assert_array_equal(arrays['shell_origin_q'],np.zeros(analysis._shell_count))
    np.testing.assert_array_equal(accepted.shell_q4_material[0].alpha,arrays['shell_alpha'][0])
    assert np.max(arrays['shell_origin_alpha'])==0.
    changed=dict(arrays);changed['shell_alpha']=arrays['shell_alpha'].copy()
    changed['shell_alpha'][0,0]+=1e-6
    corrupted=encode_snapshot(analysis,arrays=changed,
        **{k:v for k,v in metadata.items() if k not in ('policy','model','steps')})
    with pytest.raises(ValueError,match='material replay mismatch'):bad.restore(corrupted)
    assert bad._generation==0 and bad._finite_driver is None
    token=CancellationToken();token.cancel('yielded mixed accepted prefix')
    with pytest.raises(SolveCancelled):
        analysis.solve(DIRECTION,(accepted.target*2.,),displacement_control=CONTROL,
                       cancellation_token=token,**OPTIONS)
    assert analysis.checkpoint()==raw and analysis.recover() is accepted
