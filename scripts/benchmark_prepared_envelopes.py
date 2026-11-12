"""Whole-model envelope screen. Run through the cumulative budget harness.

Histories are lazy prefixes of 10,000 distinct nonlinear samples. Meshing is
excluded; preparation, factors, loads, recovery and consumed output are timed.
Selected histories stream to JSONL; the comparison sink retains no case list.
"""
from __future__ import annotations
import argparse
from copy import copy
from types import SimpleNamespace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from benchmark_static_batches import mixed_free_model, numeric_leaves


def sample_time(index, history):
    # Low-discrepancy prefixes span the same interval even for short screens.
    offset = .137 if history == 'primary' else .419
    return 60. * ((offset + index * ((math.sqrt(5.) - 1.) / 2.)) % 1.)


def normalized_envelopes(result):
    if hasattr(result.extrema, 'items'):
        for key, entry in result.extrema.items():
            values = {}
            for name in ('minimum', 'maximum', 'absolute_maximum'):
                row = getattr(entry, name)
                values[name] = dict(value=row.value, case_id=row.case_id,
                    input_index=row.input_index, time=row.time,
                    associated={repr(k): v for k, v in row.associated.items()})
            yield repr(key), values
    else:
        for entry in result.extrema:
            for index in np.ndindex(entry.original_shape):
                component_path = entry.component if isinstance(entry.component, tuple) else (entry.component,)
                key = ((entry.quantity, entry.entity_id, *component_path, *index)
                       if entry.quantity == 'stress' else
                       (entry.quantity, entry.entity_id, *index))
                values = {}
                for name in ('minimum', 'maximum', 'absolute_maximum'):
                    row = getattr(entry, name)
                    associated = {}
                    if entry.quantity == 'stress':
                        for field, array in row.associated_fields.items():
                            associated.update({repr(k): v for k, v in
                                numeric_leaves(np.asarray(array)[index], field if isinstance(field, tuple) else (field,))})
                    instant = row.times[index]
                    values[name] = dict(value=float(row.values[index]),
                        case_id=row.case_ids[index], input_index=int(row.input_indices[index]),
                        time=None if instant is None else float(instant), associated=associated)
                yield repr(key), values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('stream', 'prepared'), required=True)
    parser.add_argument('--family', choices=('free', 'multiple', 'supported', 'barge'), required=True)
    parser.add_argument('--history', choices=('primary', 'holdout'), default='primary')
    parser.add_argument('--cases', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=32)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.cases <= 10000:
        parser.error('cases must be a prefix length between 1 and 10000')
    import anyfesolver
    import psutil
    import numba
    from threadpoolctl import threadpool_info
    from anyfesolver import (LoadCase, StaticCase, RecoveryConfig, ResourceConfig,
                           AnalysisSession, EnvelopeReducer)
    from anyfesolver.linalg import backend_operation_counts
    built = None
    fixture_digest = None
    if args.family == 'barge':
        if args.fixture is None:
            parser.error('barge requires the frozen built-model fixture')
        from anyfem.io.project_file import project_from_dict
        from anyfem.solve.build import build_fe_model
        from anymesher.serialize import mesh_from_dict
        from anyfem.solve.static_batch import StaticLoadRequest
        data = args.fixture.read_bytes()
        fixture_digest = hashlib.sha256(data).hexdigest()
        fixture = json.loads(data)
        project = project_from_dict(fixture['project'])
        mesh = mesh_from_dict(fixture['mesh'])
        built = build_fe_model(project, mesh, load_case=None, require_loads=False)
        model = built.fe_model
        template = next(case for case in project.load_cases.values() if case.parent is not None)
        def records():
            for i in range(args.cases):
                t = sample_time(i, args.history)
                case = copy(project.resolve_case(template))
                case.time = t
                case.set_acceleration(.15*math.sin(t), .11*math.cos(1.7*t), -.2*math.sin(.7*t)**2)
                yield StaticLoadRequest(f'{args.history}-{i:05}', load_case=case, time=t)
    else:
        model, groups, balancing = mixed_free_model(2 if args.family == 'multiple' else 1)
        if args.family == 'supported':
            from anyfesolver import FixedSupport
            model.add_boundary_condition(FixedSupport('screen support', [groups[0][0][0]]))
        def records():
            for i in range(args.cases):
                t = sample_time(i, args.history)
                load = LoadCase(f'{args.history}-{i:05}')
                acceleration = np.array([.17*math.sin(t), .11*math.cos(1.7*t), -.23*math.sin(.7*t)**2])
                load.gravity = acceleration
                for nodes, shells in groups:
                    pressure = 400*(math.sin(.9*t)+.2*math.cos(2.3*t))
                    for eid, weight in zip(shells, (1., -2., 1.)):
                        load.add_pressure_load(eid, pressure*weight)
                    amplitude = 120*(math.cos(.4*t)**2-.3) + (17. if t >= 31. else -11.)
                    for nid in nodes:
                        force = sum((acceleration[a]*balancing[a][nid] for a in range(3)), np.zeros(3))
                        if nid == nodes[0]: force[0] -= amplitude
                        if nid == nodes[3]: force[0] += amplitude
                        load.add_nodal_load(nid, forces=force)
                yield StaticCase(load.name, load, t)
    nodes, elements = tuple(model.mesh.nodes), tuple(model.mesh.elements)
    selected_nodes = tuple(dict.fromkeys((nodes[0], nodes[len(nodes)//2], nodes[-1])))
    selected_elements = tuple(dict.fromkeys((elements[0], elements[len(elements)//2], elements[-1])))
    dofs = np.array([d for node in selected_nodes for d in model.mesh.nodes[node].dofs], dtype=int)
    history_config = RecoveryConfig(node_ids=selected_nodes, element_ids=selected_elements,
        include_displacements=False, include_stresses=True, include_reactions=True, store_full_histories=False)
    envelope_config = RecoveryConfig(include_displacements=False, include_stresses=True,
        include_reactions=True, store_full_histories=False)
    resources = ResourceConfig(solver_threads=1, assembly_threads=1, recovery_threads=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    before = backend_operation_counts()
    started = time.perf_counter()
    if args.route == 'prepared':
        if built is None:
            owner = anyfesolver.prepare_static_evaluator(model, output_dofs=dofs,
                recovery_config=history_config, envelope_recovery_config=envelope_config,
                resource_config=resources)
        else:
            from anyfem.solve import prepare_linear_static_evaluator
            owner = prepare_linear_static_evaluator(built=built, output_dofs=dofs,
                recovery_config=history_config, envelope_recovery_config=envelope_config,
                resource_config=resources)
    else:
        owner = AnalysisSession(model)
    preparation_seconds = time.perf_counter() - started

    def consume(suffix):
        reducer = EnvelopeReducer()
        phase = {}
        seen = 0
        if args.route == 'prepared':
            stream = owner.iter_batches(records(), batch_size=args.block_size)
        elif built is None:
            stream = anyfesolver.iter_linear_batches(model, records(), session=owner,
                output_dofs=dofs, recovery_config=envelope_config,
                resource_config=resources, batch_size=args.block_size)
        else:
            from anyfem.solve import iter_linear_static_batches
            stream = iter_linear_static_batches(built=built, load_cases=records(), session=owner,
                output_dofs=dofs, recovery_config=envelope_config,
                resource_config=resources, batch_size=args.block_size)
        trace_path = args.output.with_suffix('.'+suffix+'.jsonl')
        try:
            with trace_path.open('w') as sink:
                for block in stream:
                    for name, value in block.diagnostics.get('phase_timings_seconds', {}).items():
                        phase[name] = phase.get(name, 0.) + float(value)
                    if args.route == 'stream':
                        reducer.update_batch(SimpleNamespace(output_dofs=np.empty(0, dtype=int),
                            displacements=np.empty((0, len(block.case_ids))),
                            case_results=block.case_results, case_ids=block.case_ids,
                            input_indices=block.input_indices, times=block.times))
                    for j, result in enumerate(block.case_results):
                        if result['status'] != 'converged': raise ValueError('uncertified case')
                        values = {repr(('displacement', int(d))): float(v)
                                  for d, v in zip(dofs, block.displacements[:, j])}
                        for eid in selected_elements:
                            values.update({repr(('stress', eid, *key)): value for key, value in
                                numeric_leaves(result['element_stresses'][eid])})
                        for bucket in ('support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces'):
                            for node in selected_nodes:
                                force = result['constraint_forces'][bucket].get(node, np.zeros(6))
                                values.update({repr((bucket, node, k)): float(v) for k, v in enumerate(force)})
                        sink.write(json.dumps(dict(case_id=block.case_ids[j], time=block.times[j], values=values))+'\n')
                        seen += 1
        finally:
            stream.close()
        if seen != args.cases: raise ValueError('history not completely consumed')
        if args.route == 'prepared':
            summary = owner.envelope_result()
            if not summary.complete: raise ValueError('envelope incomplete')
        else:
            reducer.finish()
            summary = reducer
        envelope_path = args.output.with_suffix('.'+suffix+'.envelopes.jsonl')
        with envelope_path.open('w') as sink:
            for key, values in normalized_envelopes(summary):
                sink.write(json.dumps(dict(key=key, extrema=values))+'\n')
        return dict(phases=phase, history_path=str(trace_path), envelope_path=str(envelope_path),
                    retained_history_cases=0, history_output_bytes=trace_path.stat().st_size,
                    envelope_output_bytes=envelope_path.stat().st_size,
                    owner_diagnostics=owner.diagnostics() if hasattr(owner, 'diagnostics') else {},
                    envelope_diagnostics=getattr(summary, 'diagnostics', {}))
    try:
        cold = consume('cold')
        complete_seconds = time.perf_counter() - started
        warm = None
        warm_seconds = None
        if args.reuse:
            warm_started = time.perf_counter()
            warm = consume('warm')
            warm_seconds = time.perf_counter() - warm_started
    finally:
        owner.close()
    after = backend_operation_counts()
    memory = psutil.Process().memory_info()
    result = dict(route=args.route, family=args.family, history=args.history, cases=args.cases,
        complete_seconds=complete_seconds, preparation_seconds=preparation_seconds,
        warm_seconds=warm_seconds, cold=cold, warm=warm,
        peak_working_set_bytes=getattr(memory, 'peak_wset', memory.rss),
        memory_scope='isolated process peak including model build; histories streamed to disk',
        fixture_sha256=fixture_digest, model_elements=len(elements), model_nodes=len(nodes),
        selected_elements=selected_elements, selected_nodes=selected_nodes,
        operations={k: after[k]-before[k] for k in before},
        runtime=dict(python=sys.executable, solver=anyfesolver.__file__,
            numba_disable_jit=numba.config.DISABLE_JIT, numba_threads=numba.get_num_threads(),
            threadpools=threadpool_info()))
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ('route','family','cases','complete_seconds',
        'preparation_seconds','warm_seconds','peak_working_set_bytes','operations')}))


if __name__ == '__main__':
    main()
