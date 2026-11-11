"""Bounded, identical-output comparison of exact stream and prepared execution.

Run through the task's process-wall budget runner. No timing claim is made by
this driver alone. Meshing is excluded; evaluator/session creation, numerical
factorization, load evaluation, recovery and consumed envelopes are included.
"""
from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from benchmark_static_batches import (mixed_free_model, free_cases, positions,
                                      numeric_leaves, case_digest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=['stream', 'prepared'], required=True)
    parser.add_argument('--family', choices=['barge', 'free', 'multiple', 'independent'], required=True)
    parser.add_argument('--history', choices=['pilot', 'primary', 'holdout'], default='pilot')
    parser.add_argument('--cases', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=8)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--preparation-log', type=Path, help='optional phase progress evidence; profiling runs are not promotion evidence')
    args = parser.parse_args()
    if not 1 <= args.cases <= len(positions(args.history)):
        parser.error('cases must fit the selected frozen history')
    driver_start = time.perf_counter()
    import anyfesolver
    import psutil
    from threadpoolctl import threadpool_info
    from anyfesolver import AnalysisSession, RecoveryConfig, ResourceConfig, StaticCase
    from anyfesolver.linalg import backend_operation_counts
    from anyfesolver.static_envelopes import EnvelopeReducer
    indices = [positions(args.history)[i] for i in np.linspace(
        0, len(positions(args.history))-1, args.cases, dtype=int)]
    built = None
    application = None
    fixture_digest = None
    if args.family == 'barge':
        if args.fixture is None:
            parser.error('barge requires a frozen --fixture; this driver never meshes')
        import anyfem
        from anyfem.io.project_file import project_from_dict
        from anyfem.solve.build import build_fe_model
        from anymesher.serialize import mesh_from_dict
        data = args.fixture.read_bytes()
        fixture_digest = hashlib.sha256(data).hexdigest()
        fixture = json.loads(data)
        project = project_from_dict(fixture['project'])
        mesh = mesh_from_dict(fixture['mesh'])
        built = build_fe_model(project, mesh, load_case=None, require_loads=False)
        model = built.fe_model
        application = anyfem.__file__
        steps = [case for case in project.load_cases.values() if case.parent is not None]
        def records(selected):
            from anyfem.solve.static_batch import StaticLoadRequest
            for position in selected:
                case = copy(project.resolve_case(steps[position]))
                t = float(case.time)
                case.set_acceleration(.15*math.sin(t), .11*math.cos(1.7*t), -.2*math.sin(.7*t)**2)
                yield StaticLoadRequest(case_id=case.name, load_case=case, time=t)
    else:
        model, groups, balancing = mixed_free_model(2 if args.family == 'multiple' else 1)
        if args.family == 'independent':
            from anyfesolver import FixedSupport, LoadCase
            model.add_boundary_condition(FixedSupport('prepared benchmark support', groups[0][0][:1]))
            def records(selected):
                for position in selected:
                    rng = np.random.default_rng(93017+position)
                    case = LoadCase(f'independent-{position}')
                    for _, shells in groups:
                        for element in shells:
                            case.add_pressure_load(element, float(rng.uniform(-500, 500)))
                    case.set_acceleration(*rng.uniform(-.2, .2, 3))
                    yield StaticCase(case.name, case, float(position)*.1)
        else:
            def records(selected):
                for name, instant, case in free_cases(model, groups, balancing, selected):
                    yield StaticCase(name, case, instant)
    nodes = tuple(model.mesh.nodes)
    elements = tuple(model.mesh.elements)
    selected_nodes = tuple(dict.fromkeys((nodes[0], nodes[len(nodes)//2], nodes[-1])))
    selected_elements = tuple(dict.fromkeys((elements[0], elements[len(elements)//2], elements[-1])))
    dofs = np.array([d for n in selected_nodes for d in model.mesh.nodes[n].dofs], dtype=int)
    recovery = RecoveryConfig(node_ids=selected_nodes, element_ids=selected_elements,
                              history_mode='selected', store_full_histories=False)
    resources = ResourceConfig(solver_threads=1, assembly_threads=1, recovery_threads=1)
    fixture_seconds = time.perf_counter()-driver_start

    def preparation_progress(event):
        with args.preparation_log.open('a') as handle:
            handle.write(json.dumps(dict(stage=event.stage, monotonic=time.perf_counter()))+'\n')

    preparation_options = {} if args.preparation_log is None else dict(progress_callback=preparation_progress)

    def evaluate(selected):
        reducer = EnvelopeReducer()
        cases = []
        phase = {}
        before = backend_operation_counts()
        started = time.perf_counter()
        if args.route == 'prepared':
            if built is not None:
                from anyfem.solve import prepare_linear_static_evaluator
                owner = prepare_linear_static_evaluator(built=built, output_dofs=dofs,
                    recovery_config=recovery, resource_config=resources, **preparation_options)
            else:
                from anyfesolver import prepare_static_evaluator
                owner = prepare_static_evaluator(model, output_dofs=dofs,
                    recovery_config=recovery, resource_config=resources, **preparation_options)
        else:
            owner = AnalysisSession(model)
        preparation_seconds = time.perf_counter()-started
        def consume():
            if args.route == 'prepared':
                stream = owner.iter_batches(records(selected), batch_size=args.block_size)
            elif built is not None:
                from anyfem.solve import iter_linear_static_batches
                stream = iter_linear_static_batches(built=built, load_cases=records(selected),
                    batch_size=args.block_size, output_dofs=dofs, recovery_config=recovery,
                    resource_config=resources, session=owner)
            else:
                from anyfesolver import iter_linear_batches
                stream = iter_linear_batches(model, records(selected), batch_size=args.block_size,
                    output_dofs=dofs, recovery_config=recovery, resource_config=resources, session=owner)
            try:
                for block in stream:
                    for key, value in block.diagnostics.get('phase_timings_seconds', {}).items():
                        phase[key] = phase.get(key, 0.) + float(value)
                    for col, result in enumerate(block.case_results):
                        if result['status'] != 'converged':
                            raise ValueError('uncertified case in benchmark output')
                        values = {('u', int(d)):float(v) for d,v in zip(dofs, block.displacements[:, col])}
                        associated = {}
                        for element, stress in result['element_stresses'].items():
                            components = dict(numeric_leaves(stress))
                            for key,value in components.items():
                                quantity = ('stress', int(element), *key)
                                values[quantity] = value
                                associated[quantity] = components
                        for bucket in ('support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces'):
                            for node in selected_nodes:
                                vector = result['constraint_forces'].get(bucket, {}).get(node, np.zeros(6))
                                for component,value in enumerate(vector):
                                    values[(bucket,node,component)] = float(value)
                        reducer.update(values, case_id=block.case_ids[col], input_index=block.input_indices[col],
                            time=block.times[col], associated=associated)
                        cases.append(dict(case_id=block.case_ids[col], time=block.times[col],
                            digest=case_digest(values), values={repr(k):v for k,v in values.items()}))
            finally:
                stream.close()
            reducer.finish()
        try:
            consume()
            first_seconds = time.perf_counter()-started
            owner_info = owner.diagnostics()
            first_phases = dict(phase)
            reused_seconds = None
            if args.reuse:
                phase.clear()
                first_cases = cases.copy()
                reducer = EnvelopeReducer()
                cases.clear()
                reuse_started = time.perf_counter()
                consume()
                reused_seconds = time.perf_counter()-reuse_started
                if [x['digest'] for x in cases] != [x['digest'] for x in first_cases]:
                    raise AssertionError('same-owner reused evaluation changed output bits')
        finally:
            owner.close()
        after = backend_operation_counts()
        envelope = {repr(k):{name:dict(value=getattr(v,name).value,
            case_id=getattr(v,name).case_id, input_index=getattr(v,name).input_index,
            time=getattr(v,name).time) for name in ('minimum','maximum','absolute_maximum')}
            for k,v in reducer.extrema.items()}
        return dict(complete_seconds=first_seconds, eager_preparation_seconds=preparation_seconds,
            reused_seconds=reused_seconds, cases=cases, envelope=envelope, complete=reducer.complete,
            phases=first_phases, reused_phases=phase if args.reuse else None, owner=owner_info, operations={k:after[k]-before[k] for k in before})

    warmup_seconds = None
    if args.warmup:
        warmed = time.perf_counter()
        evaluate(indices[:1])
        warmup_seconds = time.perf_counter()-warmed
    result = evaluate(indices)
    memory = psutil.Process().memory_info()
    from anyfesolver.jit_compiler import JIT_ENABLED, JIT_DISABLED_REASON
    import numba
    result.update(route=args.route, family=args.family, history=args.history, positions=indices,
        selected_nodes=selected_nodes, selected_elements=selected_elements, output_dofs=dofs.tolist(),
        model_dofs=model.mesh.dof_manager.total_dofs, fixture_sha256=fixture_digest,
        fixture_seconds=fixture_seconds, warmup_seconds=warmup_seconds,
        peak_working_set_bytes=getattr(memory, 'peak_wset', memory.rss),
        memory_scope='process peak includes fixture and optional warmup; comparison sink retains selected values',
        runtime=dict(python=sys.executable, solver=anyfesolver.__file__, application=application,
                     jit_enabled=JIT_ENABLED, jit_disabled_reason=JIT_DISABLED_REASON,
                     numba_disable_jit=numba.config.DISABLE_JIT, numba_threads=numba.get_num_threads(),
                     threadpools=threadpool_info()), driver_seconds=time.perf_counter()-driver_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k:result[k] for k in ('route','family','positions','complete_seconds',
                     'eager_preparation_seconds','reused_seconds','peak_working_set_bytes','operations')}))


if __name__ == '__main__':
    main()
