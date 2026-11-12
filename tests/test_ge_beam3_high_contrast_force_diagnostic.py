"""Small actual native force-program diagnostics; failures remain unwaived."""
import json
from time import monotonic
import numpy as np
import pytest
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram,solve_force_program
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam
from test_ge_beam3_curved_contrast_probe import make_curved


@pytest.mark.parametrize('slenderness',[100.,10000.,1000000.])
def test_curved_high_contrast_native_force_program(slenderness,tmp_path,monkeypatch):
    model,_=make_curved(slenderness)
    program=ForceProgram((.5,1.),((5,.05,-.001,0.),))
    original=CenteredStationaryBeam.evaluate; last={}; count=0; start=monotonic()
    with (tmp_path/'local-progress.jsonl').open('xb') as local_log, (tmp_path/'global-progress.jsonl').open('xb') as global_log:
        def observe(self,positions,vertices,rotations,moments,**kwargs):
            nonlocal count
            count+=1
            last.clear();last.update(reference=self.reference.coordinates.copy(),
                positions=np.array(positions,copy=True),position_low=self.position_low,
                vertices=np.array(vertices,copy=True),rotations=np.array(rotations,copy=True),
                moments=np.array(moments,copy=True),section=self.section._elastic.copy())
            try:
                output=original(self,positions,vertices,rotations,moments,**kwargs)
            except Exception as exc:
                local_log.write(canonical(dict(index=count,exception=type(exc).__name__,message=str(exc))))
                local_log.flush();raise
            last.update(residual=output.residual.copy(),hessian=output.hessian.copy())
            local_log.write(canonical(dict(index=count,internal_residual=float(np.linalg.norm(output.residual[18:],np.inf)),
                external_residual=float(np.linalg.norm(output.residual[:18])),potential=output.potential)))
            local_log.flush()
            return output
        def progress(event):
            global_log.write(canonical(event));global_log.flush()
            print(str(slenderness)+' '+event['stage']+' '+str(event['target'])+' '+str(event['iteration']),flush=True)
        monkeypatch.setattr(CenteredStationaryBeam,'evaluate',observe)
        result=solve_force_program(model,program,progress=progress)
    with (tmp_path/'last-local-evaluation.npz').open('xb') as stream:np.savez(stream,**last)
    with (tmp_path/'last-accepted-checkpoint.json').open('xb') as stream:stream.write(result.checkpoint)
    record=dict(slenderness=slenderness,status=result.status,completed_targets=result.completed_targets,
        parameter=result.parameter,failure=result.failure,local_evaluations=count,
        accepted_free_residual=float(np.linalg.norm(result.physical_imbalance[6:])),
        displacements=result.displacements,production_qualified=False)
    with (tmp_path/'diagnostic.json').open('xb') as stream:stream.write(canonical(record))
    print(canonical(dict(**record,elapsed_diagnostic_seconds=monotonic()-start)).decode(),flush=True)
    # Genuine failures are captured before this assertion, not xfailed.
    assert result.status=='completed',result.failure
    assert result.completed_targets==2 and record['accepted_free_residual']<=1e-11
