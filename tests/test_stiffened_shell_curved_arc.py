"""Elastic curved point arc lifecycle; not roof capacity qualification."""
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_curved_reference_seam import curved_panel


def test_curved_elastic_arc_cancel_restore_continue(request, tmp_path):
    from anyfesolver._stiffened_shell_arc_checkpoint import decode_snapshot, encode_snapshot
    def make():
        owner=ss.create_analysis(definition=curved_panel())
        request.addfinalizer(owner.close)
        return owner
    owner=make();actions=(ss.NodalAction('beam',105,(0.,0.,-100.,0.,0.,0.)),)
    owner.solve(actions,(.1,))
    info=owner.preflight();count=len(info['arc_metric_node_order'])
    control=ss.ArcLengthControl('beam',105,2,
        ss.PhysicalChordMetric(np.full(count,1/count),info['reference_length_m']),1e-5,-1)
    token=CancellationToken();seen=[]
    def stop(result):
        seen.append(result);token.cancel('accepted first curved arc step')
    with pytest.raises(SolveCancelled):
        owner.continue_arc_length(actions,control,steps=2,cancellation_token=token,on_accepted=stop)
    assert len(seen)==1
    raw=owner.checkpoint();m,a=decode_snapshot(owner,raw)
    assert m['policy']=='B3_GE_CURVED_ELASTIC_POINT_TANGENT_PLANE_V1_DEV'
    assert not {'shell_plastic','shell_alpha','shell_origin_plastic','shell_origin_alpha'}&a.keys()
    assert not seen[0].shell_q4_material and not seen[0].shell_layer_alpha
    assert not np.any(a['beam_plastic']) and not np.any(a['beam_accumulated'])
    restored=make();restored.restore(raw)
    assert restored.checkpoint()==raw
    actual=owner.continue_arc_length(actions,control)
    repeated=restored.continue_arc_length(actions,control)
    assert owner.checkpoint()==restored.checkpoint()
    np.testing.assert_array_equal(actual.cell_resultants,repeated.cell_resultants)
    assert actual.free_equilibrium_ratio<=1e-8 and actual.constraint_max<=1e-9
    force_owner=make()
    force_result=force_owner.solve(actions,(.1+actual.amplitude,))
    np.testing.assert_allclose(actual.shell_displacement,force_result.shell_displacement,
                               rtol=1e-7,atol=1e-11)
    np.testing.assert_allclose(actual.cell_resultants,force_result.cell_resultants,
                               rtol=1e-7,atol=1e-8)
    # A framed nonelastic beam history is outside this elastic route.
    bad_arrays=dict(a);bad_arrays['beam_origin_accumulated']=a['beam_origin_accumulated'].copy()
    bad_arrays['beam_origin_accumulated'][0,0]=1e-4
    poisoned=encode_snapshot(owner,m,bad_arrays)
    fresh=make()
    with pytest.raises(ValueError):fresh.restore(poisoned)
    assert fresh._generation==0
    # Correct framing cannot turn a different mechanical policy into this one.
    bad=dict(m,policy='B3_GE_BL5_PHYSICAL_TANGENT_PLANE_V1_DEV')
    with pytest.raises(ValueError,match='metadata'):encode_snapshot(owner,bad,a)
    with pytest.raises(ValueError,match='arc history'):owner.solve(actions)
    import json
    (tmp_path/'curved-arc.json').write_text(json.dumps(dict(history=actual.history,
        equilibrium=actual.free_equilibrium_ratio,constraint=actual.constraint_max,
        restart_bytes_equal=True),indent=2))
