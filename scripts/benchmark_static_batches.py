"""Reproduce exact selected-output static batch baseline/candidate workloads.

Select the source checkout with PYTHONPATH. Baseline imports only pre-stream
public APIs. Run each measurement in a fresh process for isolated peak memory.
Numerical backend/thread settings are reported, never inferred from flags.

Honest measurement contract of this driver:
- Phase buckets record actually timed operations. Baseline recovery and
  constraint diagnostics are timed separately instead of being folded into a
  "solve" bucket; candidate load evaluation, solve and recovery happen inside
  the batch stream and are reported as one documented bucket. Cross-route
  phase comparison is valid only for the documented buckets and the total.
- Backend factorization counts are actual operations. The candidate exposes
  ``anyfesolver.linalg.backend_operation_counts``; on the baseline the driver
  installs counting wrappers around the equivalent linalg call sites
  (PARDISO phase-12 full factorizations, phase-22 symbolic refreshes, stale
  handle rebuilds and scipy/superlu factorizations). Wrappers only count;
  pypardiso is never imported by this driver, and the solver class is only
  touched after the solver's own MKL bootstrap constructed it.
- Every case contributes a sha256 digest over its exact selected response
  values plus compact magnitude norms, so all cases can be compared directly
  between routes; the envelope identities are retained unchanged.
"""
from __future__ import annotations

import argparse
from copy import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np


def positions(history: str):
    # Primary/holdout span the full 60-second wave/connection envelope and are
    # disjoint. Their definitions are frozen before candidate timing/tuning.
    return {'pilot': list(range(0, 120, 6)),
            'primary': list(range(0, 600, 6)),
            'holdout': list(range(2, 600, 3))}[history]


def mixed_free_model(bodies=1):
    from anyfesolver import FEModel, LoadCase, assemble_load_vector
    from anyfesolver.elements import BeamElement, create_shell_element
    model = FEModel('balanced mixed static stream')
    model.add_material('steel', 210e9, .3, density=7850.)
    groups = []
    for body in range(bodies):
        offset = body * 8
        nodes = []
        for j in range(2):
            for i in range(4):
                nid = offset + j*4+i+1
                model.add_node(nid, float(i+body*5), float(j), 0.)
                nodes.append(nid)
        shells=[]
        for i in range(3):
            eid=body*6+i+1
            connectivity=[offset+i+1,offset+i+2,offset+i+6,offset+i+5]
            model.add_element(eid, create_shell_element(eid,connectivity,'steel',
                              thickness=.01,formulation='e4-pl'))
            shells.append(eid)
            beam_id=body*6+i+4
            model.add_element(beam_id,BeamElement(beam_id,connectivity[:2],'steel',
                {'area':.003,'Iy':2e-5,'Iz':2e-5,'J':1e-5}))
        groups.append((nodes,shells))
    # Exact fixed-geometry force/moment balancing map; applying nodal forces
    # only preserves a nontrivial local acceleration stress field.
    balancing=[]
    for axis in range(3):
        unit=LoadCase('unit acceleration')
        unit.gravity=np.eye(3)[axis]
        force,_=assemble_load_vector(model,unit)
        nodal={}
        for nodes,_ in groups:
            B=np.zeros((6,3*len(nodes)))
            resultant=np.zeros(6)
            for j,nid in enumerate(nodes):
                node=model.mesh.nodes[nid]
                x,y,z=node.x,node.y,node.z
                cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
                B[:3,j*3:j*3+3]=np.eye(3)
                B[3:,j*3:j*3+3]=cross
                f=force[np.asarray(node.dofs,dtype=int)]
                resultant[:3]+=f[:3]
                resultant[3:]+=cross@f[:3]+f[3:]
            counter=-B.T@np.linalg.solve(B@B.T,resultant)
            for j,nid in enumerate(nodes): nodal[nid]=counter[j*3:j*3+3]
        balancing.append(nodal)
    return model,groups,balancing


def free_cases(model,groups,balancing,indices):
    from anyfesolver import LoadCase
    for position in indices:
        t=.1*position
        load=LoadCase(f'case-{position:03}')
        acceleration=np.array([.17*math.sin(t), .11*math.cos(1.7*t), -.23*math.sin(.7*t)**2])
        load.gravity=acceleration
        for nodes,shells in groups:
            pressure=400*(math.sin(.9*t)+.2*math.cos(2.3*t))
            for eid,weight in zip(shells,[1.,-2.,1.]):
                load.add_pressure_load(eid,pressure*weight)
            amplitude=120*(math.cos(.4*t)**2-.3)
            for nid in nodes:
                force=sum((acceleration[a]*balancing[a][nid] for a in range(3)),np.zeros(3))
                if nid==nodes[0]: force[0]-=amplitude
                if nid==nodes[3]: force[0]+=amplitude
                load.add_nodal_load(nid,forces=force)
        yield load.name,t,load


def numeric_leaves(value,prefix=()):
    if isinstance(value,dict):
        for key in sorted(value,key=str):
            yield from numeric_leaves(value[key],prefix+(str(key),))
    elif isinstance(value,(int,float,np.number)) and not isinstance(value,(bool,np.bool_)):
        yield prefix,float(value)
    elif isinstance(value,(np.ndarray,list,tuple)):
        array=np.asarray(value)
        if array.dtype.kind in 'fiu':
            for index in np.ndindex(array.shape): yield prefix+index,float(array[index])


def canonical_case_bytes(values):
    """Exact-order canonical bytes of one case's selected response values."""
    parts=[]
    for key in sorted(values,key=repr):
        parts.append(repr(key))
        parts.append(float(values[key]).hex())
    return '\n'.join(parts).encode()


def case_digest(values):
    return hashlib.sha256(canonical_case_bytes(values)).hexdigest()


def case_summary(values):
    """Compact magnitudes for diagnosing a digest mismatch between routes."""
    def extreme(predicate):
        selected=[abs(v) for k,v in values.items() if predicate(k)]
        return max(selected) if selected else 0.
    return {'u_abs_max':extreme(lambda key:type(key) is tuple and key[0]=='u'),
            'stress_abs_max':extreme(lambda key:type(key) is tuple and key[0]=='stress'),
            'reaction_abs_max':extreme(lambda key:type(key) is tuple and key[0]!='u' and key[0]!='stress')}


def install_backend_counters(linalg):
    """Equivalent actual-backend-operation counting for baseline and candidate.

    The candidate exposes ``backend_operation_counts`` natively. The baseline
    checkout predates it, so this driver wraps the same linalg call sites the
    candidate counts at: PARDISO phase-12 full factorizations
    (``_pardiso_full_factorize``), phase-22 symbolic refreshes (the retained
    slot solver's ``set_phase(22)``), stale-handle rebuilds
    (``_PardisoFactorization._active_solver`` via its own
    ``stale_rebuild_count``) and scipy/superlu factorizations (``splu``).
    Wrappers only count calls; no mechanics, locking order or pypardiso import
    timing changes. The pypardiso solver class is patched only after the
    solver's own bootstrap constructed it, never through a direct import.
    """
    native=getattr(linalg,'backend_operation_counts',None)
    if callable(native):
        return native,{'mode':'native','wrapped_symbols':[]}
    required=('_pardiso_full_factorize','_new_pypardiso_solver','_PardisoFactorization','splu')
    missing=[name for name in required if not hasattr(linalg,name)]
    if missing:
        raise RuntimeError('baseline anyfesolver.linalg lacks symbols required '
                           f'for equivalent backend instrumentation: {missing}')
    lock=threading.Lock()
    counters={'scipy_factorization_attempts':0,
              'pardiso_factorization_attempts':0,
              'pardiso_stale_rebuilds':0}
    def count(name):
        with lock: counters[name]+=1
    original_full_factorize=linalg._pardiso_full_factorize
    def wrapped_full_factorize(solver,prepared):
        count('pardiso_factorization_attempts')
        return original_full_factorize(solver,prepared)
    linalg._pardiso_full_factorize=wrapped_full_factorize
    original_new_solver=linalg._new_pypardiso_solver
    patched_solver_classes=set()
    def wrapped_new_solver(*,mtype):
        solver=original_new_solver(mtype=mtype)
        solver_class=type(solver)
        if solver_class not in patched_solver_classes:
            patched_solver_classes.add(solver_class)
            original_set_phase=solver_class.set_phase
            def counted_set_phase(self,phase):
                if int(phase)==22: count('pardiso_factorization_attempts')
                return original_set_phase(self,phase)
            solver_class.set_phase=counted_set_phase
        return solver
    linalg._new_pypardiso_solver=wrapped_new_solver
    factorization_class=linalg._PardisoFactorization
    original_active_solver=factorization_class._active_solver
    def wrapped_active_solver(self):
        before=self.stale_rebuild_count
        solver=original_active_solver(self)
        if self.stale_rebuild_count>before: count('pardiso_stale_rebuilds')
        return solver
    factorization_class._active_solver=wrapped_active_solver
    original_splu=linalg.splu
    def wrapped_splu(*args,**kwargs):
        count('scipy_factorization_attempts')
        return original_splu(*args,**kwargs)
    linalg.splu=wrapped_splu
    def counts():
        with lock: return dict(counters)
    return counts,{'mode':'driver-wrapper',
                   'wrapped_symbols':['anyfesolver.linalg._pardiso_full_factorize',
                                      'anyfesolver.linalg._new_pypardiso_solver'
                                      ' (+ solver-class set_phase(22) counting)',
                                      'anyfesolver.linalg._PardisoFactorization._active_solver',
                                      'anyfesolver.linalg.splu'],
                   'note':'counting wrappers only; no mechanical change'}


PHASE_SEMANTICS={
    'baseline':{
        'load_evaluation':'constructing each load record (case resolution / LoadCase definition)',
        'batch_solve':'one anyfem.solve_linear_static_many call; includes its internal load '
                      'assembly and all solves',
        'solve':'one anyfesolver.solve_linear call per case; includes its internal assembly, '
                'factorization and solve work',
        'stress_recovery':'one anyfesolver.recover_stress_result call per case',
        'constraint_diagnostics':'one compute_constraint_force_diagnostics call per case',
        'envelope_output':'selected-output extraction and envelope reduction'},
    'candidate':{
        'stream_load_solve_recover':'time pulling blocks from iter_linear_batches / '
            'iter_linear_static_batches; load evaluation, assembly, solve and selected '
            'recovery happen inside the stream and cannot be separated without changing '
            'product code',
        'envelope_output':'selected-output extraction and envelope reduction'},
    'note':'Cross-route phase comparison is valid only for envelope_output and the '
           'total; baseline recovery is timed separately while candidate recovery is '
           'inside the stream bucket. Totals include every timed operation.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--route',choices=['baseline','candidate'],required=True)
    parser.add_argument('--family',choices=['supported','free-free','multiple-body'],required=True)
    parser.add_argument('--history',choices=['pilot','primary','holdout'],default='pilot')
    parser.add_argument('--target-size',type=float,default=.7)
    parser.add_argument('--acceleration',action='store_true')
    parser.add_argument('--warmup',action='store_true')
    parser.add_argument('--warmup-cases',type=int,default=1,
                        help='Cases used to warm the same operator and selected recovery kernels; measured history stays unchanged')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--block-size',type=int,default=32)
    parser.add_argument('--fixture',type=Path,help='Trusted local frozen project/neutral mesh for identical A/B inputs')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--limit-cases',type=int,help='Diagnostic subset only; never a primary/holdout result')
    parser.add_argument('--sample-cases',type=int,
                        help='Evenly spaced subset of the frozen history for bounded checks; not a full primary/holdout campaign')
    args=parser.parse_args()
    driver_start=time.perf_counter()
    import anyfesolver
    import anyfesolver.linalg as solver_linalg
    from anyfesolver import AnalysisSession,RecoveryConfig,ResourceConfig
    from anyfesolver.assembly import compute_constraint_force_diagnostics
    from threadpoolctl import threadpool_info
    import psutil
    counts,instrumentation=install_backend_counters(solver_linalg)
    indices=positions(args.history)
    if args.sample_cases is not None:
        if args.limit_cases is not None or not 1 <= args.sample_cases <= len(indices):
            raise ValueError('sample-cases must fit the frozen history and cannot combine with limit-cases')
        indices=[indices[index] for index in np.linspace(0,len(indices)-1,args.sample_cases,dtype=int)]
    if args.limit_cases is not None: indices=indices[:args.limit_cases]
    prepare_start=time.perf_counter()
    print(json.dumps({'phase':'prepare','family':args.family,'route':args.route}),flush=True)
    if args.family=='supported':
        import anyfem
        from anyfem.application.examples import floating_barge_seaway
        from anyfem.solve.build import build_fe_model
        from anyfem.io.project_file import project_to_dict,project_from_dict
        from anymesher.serialize import mesh_to_dict,mesh_from_dict
        if args.fixture is not None and args.fixture.exists():
            fixture=json.loads(args.fixture.read_text())
            if fixture['target_size']!=args.target_size: raise ValueError('fixture target_size differs')
            project=project_from_dict(fixture['project'])
            mesh=mesh_from_dict(fixture['mesh'])
        else:
            project=floating_barge_seaway()
            mesh=project.generate_mesh(args.target_size)
            if args.fixture is not None:
                args.fixture.parent.mkdir(parents=True,exist_ok=True)
                payload={'project':project_to_dict(project),'mesh':mesh_to_dict(mesh),'target_size':args.target_size}
                args.fixture.write_text(json.dumps(payload))
        print(json.dumps({'phase':'mesh_complete','elapsed':time.perf_counter()-prepare_start}),flush=True)
        built=build_fe_model(project,mesh,load_case=None,require_loads=False)
        model=built.fe_model
        steps=[case for case in project.load_cases.values() if case.parent is not None]
        def records():
            for position in indices:
                case=copy(project.resolve_case(steps[position]))
                if args.acceleration:
                    t=float(case.time)
                    case.set_acceleration(.15*math.sin(t),.11*math.cos(1.7*t),-.2*math.sin(.7*t)**2)
                yield case.name,case.time,case
    else:
        anyfem=None
        model,groups,balancing=mixed_free_model(2 if args.family=='multiple-body' else 1)
        def records(): return free_cases(model,groups,balancing,indices)
    nodes=tuple(model.mesh.nodes)
    selected_nodes=tuple(dict.fromkeys([nodes[0],nodes[len(nodes)//2],nodes[-1]]))
    elements=tuple(model.mesh.elements)
    selected_elements=tuple(dict.fromkeys([elements[0],elements[len(elements)//2],elements[-1]]))
    dofs=np.asarray([d for nid in selected_nodes for d in model.mesh.nodes[nid].dofs],dtype=int)
    config=RecoveryConfig(node_ids=selected_nodes,element_ids=selected_elements,
                          history_mode='selected',store_full_histories=False)
    resources=ResourceConfig(solver_threads=1,assembly_threads=1,recovery_threads=1)
    preparation_seconds=time.perf_counter()-prepare_start
    print(json.dumps({'phase':'model_prepared','elapsed':preparation_seconds,'dofs':model.mesh.dof_manager.total_dofs}),flush=True)
    if args.prepare_only:
        args.output.write_text(json.dumps({'preparation_seconds':preparation_seconds,'dofs':model.mesh.dof_manager.total_dofs}))
        return
    # The identical reducer implementation is used in both measurements. It is
    # standalone and imports no solver package, so baseline mechanics stay old.
    spec=importlib.util.spec_from_file_location('benchmark_envelope',Path(__file__).parents[1]/'src/anyfesolver/static_envelopes.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    def run(session):
        reducer=module.EnvelopeReducer()
        if args.route=='candidate':
            phases={'stream_load_solve_recover':0.,'envelope_output':0.}
        elif args.family=='supported':
            phases={'load_evaluation':0.,'batch_solve':0.,'stress_recovery':0.,
                    'constraint_diagnostics':0.,'envelope_output':0.}
        else:
            phases={'load_evaluation':0.,'solve':0.,'stress_recovery':0.,
                    'constraint_diagnostics':0.,'envelope_output':0.}
        case_records=[]
        stream_phases={}
        first_case_seconds=None
        def consume(index,case_id,instant,u,stresses,forces):
            start=time.perf_counter()
            values={('u',int(d)):float(v) for d,v in zip(dofs,u)}
            associated={}
            for eid,stress in stresses.items():
                leaves=dict(numeric_leaves(stress))
                for key,value in leaves.items():
                    quantity=('stress',int(eid),*key)
                    values[quantity]=value
                    associated[quantity]=leaves
            for bucket in ['support_reactions','mpc_slave_forces','mpc_master_equivalent_forces']:
                for nid in selected_nodes:
                    vector=forces.get(bucket,{}).get(nid,np.zeros(6))
                    for component,value in enumerate(vector): values[(bucket,nid,component)]=float(value)
            reducer.update(values,case_id=case_id,input_index=index,time=instant,associated=associated)
            phases['envelope_output']+=time.perf_counter()-start
            case_records.append({'input_index':int(index),'case_id':case_id,
                                 'time':instant,'digest':case_digest(values),
                                 'selected_values':{repr(key):value for key,value in values.items()},
                                 **case_summary(values)})
        start=time.perf_counter()
        if args.route=='candidate':
            if args.family=='supported':
                from anyfem import iter_linear_static_batches,StaticLoadRequest
                stream=iter_linear_static_batches(built=built,
                    load_cases=(StaticLoadRequest(case_id=name,load_case=case,time=t) for name,t,case in records()),
                    batch_size=args.block_size,output_dofs=dofs,recovery_config=config,
                    resource_config=resources,session=session)
            else:
                from anyfesolver import iter_linear_batches,StaticCase
                stream=iter_linear_batches(model,(StaticCase(name,load,t) for name,t,load in records()),
                    batch_size=args.block_size,output_dofs=dofs,recovery_config=config,
                    resource_config=resources,session=session)
            blocks=iter(stream)
            pull_start=time.perf_counter()
            block=next(blocks,None)
            while block is not None:
                phases['stream_load_solve_recover']+=time.perf_counter()-pull_start
                for key,value in block.diagnostics.get('phase_timings_seconds',{}).items():
                    stream_phases[key]=stream_phases.get(key,0.)+value
                for j,result in enumerate(block.case_results):
                    consume(block.input_indices[j],block.case_ids[j],block.times[j],
                            block.displacements[:,j],result['element_stresses'],result['constraint_forces'])
                if first_case_seconds is None:
                    first_case_seconds=time.perf_counter()-start
                pull_start=time.perf_counter()
                block=next(blocks,None)
            phases['stream_load_solve_recover']+=time.perf_counter()-pull_start
        else:
            if args.family=='supported':
                from anyfem import solve_linear_static_many
                load_start=time.perf_counter()
                cases=list(records())
                phases['load_evaluation']+=time.perf_counter()-load_start
                solve_start=time.perf_counter()
                solved=solve_linear_static_many(built=built,load_cases=[case for _,_,case in cases],
                    load_tolerance=None,session=session,resource_config=resources)
                phases['batch_solve']+=time.perf_counter()-solve_start
                print(json.dumps({'phase':'baseline_batch_solved','elapsed':time.perf_counter()-start}),flush=True)
                if solved.status!='converged': raise RuntimeError(solved.info)
                for i,(name,t,_) in enumerate(cases):
                    shape=solved.shapes[i]
                    recovery_start=time.perf_counter()
                    stresses=anyfesolver.recover_stress_result(model,shape.displacements,
                        recovery_config=config,resource_config=resources).element_stresses
                    phases['stress_recovery']+=time.perf_counter()-recovery_start
                    diagnostics_start=time.perf_counter()
                    forces=compute_constraint_force_diagnostics(model,shape.displacements,shape.built.load_case)
                    phases['constraint_diagnostics']+=time.perf_counter()-diagnostics_start
                    consume(i,name,t,shape.displacements[dofs],stresses,forces)
                    if first_case_seconds is None:
                        first_case_seconds=time.perf_counter()-start
                        print(json.dumps({'phase':'first_case_recovered','elapsed':first_case_seconds}),flush=True)
            else:
                case_iterator=records()
                index=0
                while True:
                    load_start=time.perf_counter()
                    record=next(case_iterator,None)
                    phases['load_evaluation']+=time.perf_counter()-load_start
                    if record is None: break
                    name,t,load=record
                    solve_start=time.perf_counter()
                    u,info=anyfesolver.solve_linear(model,load,session=session,resource_config=resources)
                    phases['solve']+=time.perf_counter()-solve_start
                    if info['convergence_info']['status']!='converged': raise RuntimeError(info)
                    recovery_start=time.perf_counter()
                    stresses=anyfesolver.recover_stress_result(model,u,recovery_config=config,
                        resource_config=resources).element_stresses
                    phases['stress_recovery']+=time.perf_counter()-recovery_start
                    diagnostics_start=time.perf_counter()
                    forces=compute_constraint_force_diagnostics(model,u,load)
                    phases['constraint_diagnostics']+=time.perf_counter()-diagnostics_start
                    consume(index,name,t,u[dofs],stresses,forces)
                    if first_case_seconds is None:
                        first_case_seconds=time.perf_counter()-start
                        print(json.dumps({'phase':'first_case_recovered','elapsed':first_case_seconds}),flush=True)
                    index+=1
        elapsed=time.perf_counter()-start
        reducer.finish()
        return elapsed,reducer,phases,case_records,first_case_seconds,stream_phases
    with AnalysisSession(model) as session:
        rss_after_prepare=psutil.Process().memory_info().rss
        warmup_elapsed=None
        warmup_operations=None
        if args.warmup:
            warmup_counts_before=counts()
            if args.warmup_cases < 1:
                raise ValueError('warmup-cases must be positive')
            measured_indices=indices
            indices=indices[:args.warmup_cases]
            try:
                warmup_elapsed,warmup_reducer,_,_,_,_=run(session)
            finally:
                indices=measured_indices
            warmup_operations={k:v-warmup_counts_before[k] for k,v in counts().items()}
            print(json.dumps({'phase':'warmup_complete','elapsed':warmup_elapsed,
                              'cases':warmup_reducer.case_count}),flush=True)
        rss_after_warmup=psutil.Process().memory_info().rss
        before=counts()
        elapsed,reducer,phases,case_records,first_case_seconds,stream_phases=run(session)
        after=counts()
        session_info=session.diagnostics()
        rss_end=psutil.Process().memory_info().rss
    envelope={repr(key):{name:{'value':getattr(extrema,name).value,
                                  'input_index':getattr(extrema,name).input_index,
                                  'case_id':getattr(extrema,name).case_id,
                                  'time':getattr(extrema,name).time}
                         for name in ['minimum','maximum','absolute_maximum']}
              for key,extrema in reducer.extrema.items()}
    memory=psutil.Process().memory_info()
    selected_digest=hashlib.sha256('\n'.join(
        f"{record['case_id']}|{float(record['time']).hex()}|{record['digest']}"
        for record in case_records).encode()).hexdigest()
    payload={'route':args.route,'family':args.family,'history':args.history,'positions':indices,
             'acceleration':args.acceleration,'warmup':args.warmup,'elapsed_seconds':elapsed,
             'preparation_seconds':preparation_seconds,'phase_seconds':phases,
             'phase_semantics':PHASE_SEMANTICS,'total_measured_seconds':sum(phases.values()),
             'stream_internal_phase_seconds':stream_phases,
             'first_case_seconds':first_case_seconds,
             'cases':reducer.case_count,'complete':reducer.complete,'envelope':envelope,
             'case_digests':case_records,'selected_response_digest':selected_digest,
             'backend_operations':{k:after[k]-before[k] for k in before},
             'backend_operations_cumulative':after,
             'backend_count_instrumentation':instrumentation,
             'warmup_elapsed_seconds':warmup_elapsed,'warmup_backend_operations':warmup_operations,
             'warmup_cases':min(args.warmup_cases,len(indices)) if args.warmup else 0,
             'session':session_info,
             'peak_working_set_bytes':getattr(memory,'peak_wset',memory.rss),
             'memory':{'rss_after_prepare_bytes':rss_after_prepare,
                       'rss_after_warmup_bytes':rss_after_warmup,
                       'rss_end_bytes':rss_end,
                       'peak_working_set_bytes':getattr(memory,'peak_wset',memory.rss),
                       'retained_instrumentation_note':'case_digests retains selected numerical values for cross-route tolerance checks; this small O(cases*selected quantities) measurement sink is separate from solver-owned block storage',
                       'peak_note':'process-lifetime peak; includes the warmup run when --warmup'},
             'selected_nodes':selected_nodes,'selected_elements':selected_elements,
             'block_size':args.block_size,'limit_cases':args.limit_cases,
             'sample_cases':args.sample_cases,
             'fixture':str(args.fixture) if args.fixture is not None else None,
             'resource_config':{'solver_threads':1,'assembly_threads':1,'recovery_threads':1},
             'runtime':{'python':sys.executable,'solver':anyfesolver.__file__,
                        'application':getattr(anyfem,'__file__',None) if anyfem is not None else None,
                        'threadpools':threadpool_info()},
             'model_dofs':model.mesh.dof_manager.total_dofs,
             'total_driver_seconds':time.perf_counter()-driver_start}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,default=str))
    print(json.dumps({k:payload[k] for k in ['route','family','cases','elapsed_seconds',
                                             'preparation_seconds','peak_working_set_bytes',
                                             'model_dofs','backend_operations']}))


if __name__=='__main__': main()
