"""Actual load/yield/unload/modal replay with independent station checks."""
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver import _ge_beam3_retained_generalized_modal as modal
from anyfesolver._ge_beam3_retained_generalized_state import Context,Program
from anyfesolver._ge_beam3_retained_generalized_program import solve
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases.ge_beam3_generalized_ellipsoid_oracle import response as independent
from test_ge_beam3_native_generalized import problem
from test_ge_beam3_native_generalized_restart import make as connected
from test_ge_beam3_native_generalized_modal import inertia
from test_ge_beam3_schur_line_program import save


def make(case):
    return connected('connected-plastic') if case=='connected' else problem(case=='curved',True)[0]


@pytest.mark.parametrize('case',('straight','curved','connected'))
def test_post_plastic_modal(case,tmp_path):
    m=make(case)
    p=Program((.25,.5,1.,.5),DistributedPattern(LinePattern(tuple((eid,.09,-.03,.02) for eid in sorted(m.mesh.elements))),()))
    masses={eid:inertia() for eid in m.mesh.elements}
    def progress(row):print(row,flush=True)
    peak=solve(m,p,stop_after=3,progress=progress)
    save(tmp_path/'peak-checkpoint.json',peak.checkpoint)
    save(tmp_path/'peak-result.json',dict(status=peak.status,failure=peak.failure,completed_targets=peak.completed_targets))
    assert peak.status=='paused',peak.failure
    plastic=sum(sum(h.accumulated)>0 for cell in peak.state.histories for h in cell.stations)
    assert plastic>0
    old=canonical(peak.state)
    with pytest.raises(ValueError,match='boundary|elastic-interior|must not advance'):
        modal.prepare(m,p,peak.checkpoint,masses,expected_sha256=sha256(peak.checkpoint).hexdigest())
    assert canonical(peak.state)==old
    print(dict(stage='peak-yield-rejected',case=case,plastic_stations=plastic),flush=True)
    unloaded=solve(m,p,checkpoint=peak.checkpoint,expected_sha256=sha256(peak.checkpoint).hexdigest(),progress=progress)
    save(tmp_path/'unloaded-checkpoint.json',unloaded.checkpoint)
    save(tmp_path/'unloaded-result.json',dict(status=unloaded.status,failure=unloaded.failure,completed_targets=unloaded.completed_targets))
    assert unloaded.status=='completed',unloaded.failure
    context=Context(m,p)
    state,records=context.restore(unloaded.checkpoint,expected_sha256=sha256(unloaded.checkpoint).hexdigest())
    before=canonical(state);recovery=context.recover(state);errors=[];station_count=0
    for i,(eid,e) in enumerate(context.elements):
        nodes=context.nodes[i];s=state.mechanical;core=e.operator
        response=core.evaluate(s.positions[nodes],s.position_low[nodes],s.nodal_frames[nodes],
            s.cell_rotations[i],s.resultants[i],origin=state.histories[i],check=context.guard)
        assert canonical(response.history)==canonical(state.histories[i])
        for row,h in zip(json.loads(response.material)['stations'],state.histories[i].stations,strict=True):
            assert row['branch']=='ELASTIC'
            strain=np.sum(np.array(row['strain']),axis=0);force=np.sum(np.array(row['resultants']),axis=0)
            expected=independent(dict(elastic=e.section.elastic,metric=e.section.metric,strain=strain,
                plastic=h.plastic,accumulated=h.accumulated,yield_force=e.section.yield_force,hardening=e.section.hardening))
            assert expected['branch']=='ELASTIC' and expected['increment']==0.
            error=float(np.linalg.norm(force-expected['stress'])/max(1.,np.linalg.norm(force)))
            assert error<=1e-11;errors.append(error);station_count+=1
    print(dict(stage='independent-current-material-checked',case=case,stations=station_count),flush=True)
    packet,modes=modal.solve_modes(m,p,unloaded.checkpoint,masses,
        expected_sha256=sha256(unloaded.checkpoint).hexdigest(),bounds=(0.,10000.),num_modes=6)
    other,guard=modal.prepare(m,p,unloaded.checkpoint,masses,expected_sha256=sha256(unloaded.checkpoint).hexdigest())
    assert canonical(packet)==canonical(other) and np.all(modes.eigenvalues>0.)
    guard();assert canonical(state)==before and canonical(context.recover(state))==canonical(recovery)
    assert context.checkpoint(records)==unloaded.checkpoint
    save(tmp_path/'modal.json',dict(packet=packet,modes=modes,case=case,plastic_stations=plastic,
        current_material_station_count=station_count,station_errors=errors,unchanged_history=True,
        peak_yield_rejected=True,production_qualified=False))
    save(tmp_path/'recovery.json',recovery)
