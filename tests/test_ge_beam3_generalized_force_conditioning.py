"""Actual nonlinear force diagnostic; genuine failures are saved then raised."""
import numpy as np
import pytest
from anyfesolver._ge_beam3_native_generalized_program import solve_distributed_model, model_identity
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
from anyfesolver._ge_beam3_retained_generalized import RetainedGeneralizedOperator
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_generalized_slenderness_diagnostic import make
from test_ge_beam3_schur_line_program import save

CASES = tuple((rho,curved,name) for rho in (100.,10000.,1000000.)
              for curved in (False,True) for name in ('E','GENERAL'))
IDS = tuple(str(rho)+'-'+('curved' if curved else 'straight')+'-'+name
            for rho,curved,name in CASES)

@pytest.mark.parametrize('rho,curved,name',CASES,ids=IDS)
def test_actual_generalized_force_conditioning(rho,curved,name,tmp_path,monkeypatch):
    q=np.eye(3) if name=='E' else rotation([.4,-.3,.2])
    print(dict(stage='initialization',rho=rho,curved=curved,frame=name),flush=True)
    model,virgin,_=make(rho,curved,q);initial=canonical(virgin);binding=model_identity(model)
    e=model.mesh.elements[1]
    force=q@np.array([.005,-.003,.002]);couple=q@np.array([.002,-.001,.003])
    load=DistributedPattern(LinePattern(((1,*map(float,force)),)),((1,*map(float,couple)),))
    save(tmp_path/'inputs.json',dict(rho=rho,curved=curved,frame=q,model_sha256=binding,
        element=e.to_dict(),load=load,virgin=virgin,steps=2,max_iterations=24,
        equilibrium_tolerance=1e-11,production_qualified=False))
    original=RetainedGeneralizedOperator.evaluate;count=0;last={}
    with (tmp_path/'evaluation-progress.jsonl').open('xb') as log:
        def observed(self,*args,**kwargs):
            nonlocal count,last
            count+=1
            last=dict(index=count,positions=args[0],position_low=args[1],nodal_frames=args[2],
                      cell_rotations=args[3],resultants=args[4],origin=kwargs.get('origin'))
            try: result=original(self,*args,**kwargs)
            except Exception as exc:
                log.write(canonical(dict(index=count,error=type(exc).__name__,message=str(exc))));log.flush()
                raise
            last.update(residual=result.residual,hessian=result.hessian,hessian_low=result.hessian_low)
            log.write(canonical(dict(index=count,internal_residual=float(np.linalg.norm(result.residual[18:])))))
            log.flush()
            if count==1 or count%20==0:print(dict(stage='operator',evaluation=count),flush=True)
            return result
        monkeypatch.setattr(RetainedGeneralizedOperator,'evaluate',observed)
        try:
            result,events=solve_distributed_model(model,load,steps=2,max_iterations=24)
        except Exception as exc:
            save(tmp_path/'failure.json',dict(error=type(exc).__name__,message=str(exc),
                evaluations=count,model_unchanged=model_identity(model)==binding,
                supplied_virgin_unchanged=canonical(virgin)==initial,production_qualified=False))
            if last:save(tmp_path/'last-evaluation.json',last)
            raise
        finally:
            # Recovery replays the operator after the observation stream closes.
            # Always remove the observer before leaving its owned stream scope.
            monkeypatch.setattr(RetainedGeneralizedOperator,'evaluate',original)
    if last:save(tmp_path/'last-evaluation.json',last)
    save(tmp_path/'result.json',dict(status=result.status,displacements=result.displacements,
        states=result.element_states,events=events,evaluations=count,production_qualified=False))
    assert result.status=='completed',result.info
    assert model_identity(model)==binding and canonical(virgin)==initial
    state=result.element_states[1];before=canonical(state)
    recovery=recover_native_fields(e,model.mesh,state,expected_committed_total_u=result.displacements)
    free_error=float(np.linalg.norm(state['response'].residual[6:]))
    internal_error=state['response'].internal_error
    save(tmp_path/'accepted.json',dict(recovery=recovery,free_residual=free_error,
        internal_residual=internal_error,states_unchanged=canonical(state)==before,
        model_unchanged=model_identity(model)==binding,production_qualified=False))
    assert free_error<=1e-11 and internal_error<=1e-11 and canonical(state)==before
    assert np.linalg.norm(result.displacements)>1e-8
    print(dict(stage='completed',rho=rho,curved=curved,frame=name),flush=True)
