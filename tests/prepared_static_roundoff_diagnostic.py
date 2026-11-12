"""Bounded one-case counterexample separating load and recovery arithmetic."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_static_batches import mixed_free_model, free_cases, numeric_leaves
from anyfesolver import AnalysisSession, RecoveryConfig, StaticCase, prepare_static_evaluator, iter_linear_batches
from anyfesolver.matrix_assembly import assemble_load_matrix
from anyfesolver.recovery import recover_stress_result
from anyfesolver.prepared_static_loads import ingest_static_case, evaluate_load_vector
from anyfesolver.prepared_static_recovery import apply_compiled_recovery

model, groups, balancing = mixed_free_model()
name, instant, load = next(free_cases(model, groups, balancing, [48]))
case = StaticCase(name, load, instant)
recovery = RecoveryConfig(element_ids=[1, 4, 5, 6], include_stresses=True, include_reactions=False)
def difference(a, b):
    leaves_a = dict(numeric_leaves(a))
    leaves_b = dict(numeric_leaves(b))
    result = [(abs(leaves_a[k]-leaves_b[k]), k, leaves_a[k], leaves_b[k]) for k in leaves_a]
    worst = max(result, key=lambda x:x[0])
    return dict(max_absolute=worst[0], key=repr(worst[1]), a=worst[2], b=worst[3],
                tolerance_failures=sum(err > 1e-12 + 1e-12*abs(ref) for err, _key, ref, _ in result))
with AnalysisSession(model) as session, prepare_static_evaluator(model, recovery_config=recovery) as owner:
    stream = next(iter_linear_batches(model, [case], recovery_config=recovery, session=session))
    prepared = next(owner.iter_batches([case]))
    us, up = stream.displacements[:, 0], prepared.displacements[:, 0]
    fs = assemble_load_matrix(model, [load])[0][:, 0]
    _, _, owned = ingest_static_case(case)
    fp = evaluate_load_vector(owner._snapshot.load_maps, owned)[0]
    cs = recover_stress_result(model, us, recovery).element_stresses
    cp = recover_stress_result(model, up, recovery).element_stresses
    hs = apply_compiled_recovery(owner._snapshot.recovery_compilation, us)[0]
    hp = apply_compiled_recovery(owner._snapshot.recovery_compilation, up)[0]
    operator = session.static_operator_plan(model=model)
    dense_difference = (operator.matrix - owner._operator_matrix).toarray()
    element = model.mesh.elements[5]
    L, transform = element._beam_frame_and_transform(element.get_node_coordinates(model.mesh))
    primitive = {}
    compiled = owner._snapshot.recovery_compilation.elements[5]
    H = next(matrix for key, baseline, matrix, shape in compiled.array_components if key == 'shear_stress_z')
    for label, u in [('stream', us), ('prepared', up)]:
        ue = u[compiled.dof_mapping]
        ul = transform @ ue
        first = (ul[8] - ul[2])/L
        second = -.5*(ul[4]+ul[10])
        primitive[label] = dict(translation_term=float(first), rotation_term=float(second),
            canonical_gamma=float(first-second), canonical_tau=float((cs if label=='stream' else cp)[5]['shear_stress_z']),
            collapsed_tau=float((hs if label=='stream' else hp)[5]['shear_stress_z']),
            collapsed_sum_abs=float(np.sum(np.abs(H*ue[None,:]))))
    print(json.dumps(dict(case=name, full_F_max_abs=float(np.max(abs(fs-fp))),
        u_max_abs=float(np.max(abs(us-up))), operator_max_abs=float(np.max(abs(dense_difference))),
        canonical_vs_collapsed_on_stream_u=difference(cs, hs),
        canonical_vs_collapsed_on_prepared_u=difference(cp, hp),
        canonical_on_stream_vs_prepared_u=difference(cs, cp), beam5=primitive), indent=2))
