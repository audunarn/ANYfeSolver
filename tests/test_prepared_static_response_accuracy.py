"""Historical recovery arithmetic counterexamples on the source-response route."""
import numpy as np
from anyfesolver import PatternCase, RecoveryConfig, prepare_static_response_evaluator
from anyfesolver.recovery import recover_stress_result
from anyfesolver.prepared_static_measures import shell_surface_tensor
from test_prepared_static_evaluator import _shell_model, _shell_case


def test_response_principal_original_same_u_qualification():
    model = _shell_model()
    pattern = _shell_case(0,pressure=2.).load_case
    config = RecoveryConfig(components=['principal_stresses_top','principal_stresses_bottom'],include_stresses=True)
    records = []
    with prepare_static_response_evaluator(model,{'p':pattern},recovery_config=config) as owner:
        block = next(owner.iter_batches([PatternCase(str(c),{'p':c}) for c in (1.,.1,.3,3.7,-1.25)]))
        for j in range(5):
            raw = recover_stress_result(model,block.displacements[:,j],RecoveryConfig(include_stresses=True))
            for eid,fields in raw.element_stresses.items():
                for surface in ('top','bottom'):
                    expected = np.linalg.eigvalsh(shell_surface_tensor(fields,surface))[...,::-1]
                    actual = block.case_results[j]['element_stresses'][eid]['principal_stresses_'+surface]
                    errors = np.abs(actual-expected)
                    records.append((block.case_ids[j],eid,surface,float(errors.max()),int(np.count_nonzero(errors>1e-12+1e-12*np.abs(expected)))))
        print('principal same-u original1e-12/1e-12:',records)
        assert sum(row[-1] for row in records) == 0


def test_response_mixed_historical_roundoff_diagnostic():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from benchmark_static_batches import mixed_free_model,free_cases,numeric_leaves
    from anyfesolver.prepared_static import PreparedStaticEvaluator
    model,groups,balancing = mixed_free_model()
    patterns = {name:load for name,time,load in free_cases(model,groups,balancing,[0,48,576])}
    config = RecoveryConfig(element_ids=[1,4,5,6],include_stresses=True,include_reactions=False)
    coefficients = [{'case-048':.3},{'case-000':.25,'case-048':.75},{'case-048':1.,'case-576':-.7}]
    records = []
    with prepare_static_response_evaluator(model,patterns,recovery_config=config) as owner:
        cases = [PatternCase(str(i),c) for i,c in enumerate(coefficients)]
        block = next(owner.iter_batches(cases))
        for j,case in enumerate(cases):
            raw = recover_stress_result(model,block.displacements[:,j],config).element_stresses
            actual = block.case_results[j]['element_stresses']
            a,b = dict(numeric_leaves(actual)),dict(numeric_leaves(raw))
            fails = sum(abs(a[k]-b[k])>1e-12+1e-12*abs(b[k]) for k in a)
            _,_,packet = owner._ingest_case(case)
            F,_ = owner._load_vector(owner._snapshot,packet)
            snapshot = owner._snapshot
            rhs = snapshot.transformation.T@(F-snapshot.stiffness@snapshot.prescribed_offset)
            rhs = np.concatenate((rhs,np.zeros(snapshot.rigid_modes.shape[1])))
            direct = PreparedStaticEvaluator._solve_block(owner,rhs[:,None],())
            direct_u = snapshot.transformation@direct[:snapshot.reduced_stiffness.shape[0],0]+snapshot.prescribed_offset
            direct_raw = recover_stress_result(model,direct_u,config).element_stresses
            d = dict(numeric_leaves(direct_raw))
            direct_fails = sum(abs(a[k]-d[k])>1e-12+1e-12*abs(d[k]) for k in a)
            u_fails = int(np.count_nonzero(np.abs(block.displacements[:,j]-direct_u)>1e-13+1e-11*np.abs(direct_u)))
            records.append(dict(case=j,same_u_conservative_stress_failures=fails,direct_conservative_stress_failures=direct_fails,
                same_u_beam5_shear_error=abs(actual[5]['shear_stress_z']-raw[5]['shear_stress_z']),
                direct_u_max_error=float(np.max(np.abs(block.displacements[:,j]-direct_u))),direct_u_original_tolerance_failures=u_fails,
                backward_error=block.case_results[j]['solver_info']['relative_backward_error']))
        print('historical mixed diagnostic:',records)
        assert all(r['same_u_conservative_stress_failures']==0 and r['direct_u_original_tolerance_failures']==0 for r in records)

def test_stress_canonical_rhs_layout_independent_of_ingress_block(monkeypatch):
    config = RecoveryConfig(include_stresses=True,include_reactions=True)
    model = _shell_model()
    pattern = _shell_case(0,pressure=2.).load_case
    cases = [PatternCase(str(c),{'p':c}) for c in (.1,.3,3.7)]
    with prepare_static_response_evaluator(model,{'p':pattern},recovery_config=config) as owner:
        original = owner._operator_handle.solve_many
        layouts = []
        def observe(rhs):
            layouts.append((rhs.shape[1],rhs.flags.c_contiguous))
            return original(rhs)
        monkeypatch.setattr(owner._operator_handle,'solve_many',observe)
        block = next(owner.iter_batches(cases,batch_size=3))
        assert layouts == [(1,True)]*3
        singles = list(owner.iter_batches(cases,batch_size=1))
        np.testing.assert_array_equal(block.displacements,np.column_stack([b.displacements[:,0] for b in singles]))
        for j,single in enumerate(singles):
            for eid,fields in block.case_results[j]['element_stresses'].items():
                for name,value in fields.items():
                    if isinstance(value,(np.ndarray,int,float)):
                        np.testing.assert_array_equal(value,single.case_results[0]['element_stresses'][eid][name])
