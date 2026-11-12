"""Private objective native pseudo-arclength predictor/corrector.

Full bordered solves, native load derivatives and accepted-origin transactions.
No automatic cutback/retry, branch switching, public routing or qualification.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
import time

import numpy as np
from scipy import sparse

from anyfesolver.control import cancellation_safe_point, SolveCancelled
from anyfesolver.linalg import factorize, MatrixClass
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from anyfesolver.nonlinear_state import create_model_native_rotation_store, native_trial_full_coordinates
from anyfesolver._native_rotation_state import rotation_exponential
from anyfesolver._ge_beam3_load_program import ForceProgram, ForceProgramResult, _capture as force_capture, _physical_force, TOLERANCE
from anyfesolver._ge_beam3_displacement_program import _parameter_column
from anyfesolver._ge_beam3_arc_geometry import constraint, POLICY
from anyfesolver._ge_beam3_p5.arrays import _readonly
from anyfesolver._ge_beam3_p5.typed_schema import _decode
from anyfesolver._ge_beam3_p5_loads import LoadStateStore
from anyfesolver._ge_beam3_p5_loads.core import canonical, sha
from anyfesolver._ge_beam3_p5_loads.codec import _load
from anyfesolver._ge_beam3_p5_loads.state import parameter_value, LOAD_POLICY


PROGRAM = 'GE_BEAM3_NATIVE_FRAME_CHORD_ARC_PROGRAM_V1'
SCHEMA = 'GE_BEAM3_NATIVE_FRAME_CHORD_ARC_CHECKPOINT_V1'


def _inverse_square(value):
    value = parameter_value(value)
    if value <= 0. or not math.isfinite(value*value) or value*value == 0.:
        raise ValueError('metric scale is outside the finite positive weight range')
    inverse = 1./value
    square = inverse*inverse
    if value <= 0. or not math.isfinite(square) or square <= 0.:
        raise ValueError('metric scale is outside the finite positive weight range')
    return square


@dataclass(frozen=True)
class ArcProgram:
    steps: tuple
    length_scale: float
    parameter_scale: float = 1.
    initial_load_sign: float = 1.
    nodal_forces: tuple = ()
    max_iterations: int = 16
    max_backtracks: int = 8

    def __post_init__(self):
        ForceProgram(self.steps,self.nodal_forces,self.max_iterations,self.max_backtracks)
        if any(not 0. < step <= .25 for step in self.steps): raise ValueError('positive bounded explicit arc steps required')
        for value in (self.length_scale,self.parameter_scale):
            if parameter_value(value) <= 0.: raise ValueError('positive explicit metric scale required')
            _inverse_square(value)
        if parameter_value(self.initial_load_sign) not in (-1.,1.): raise ValueError('explicit initial load sign required')

    def descriptor(self):
        return dict(schema=PROGRAM,steps=self.steps,length_scale=self.length_scale,parameter_scale=self.parameter_scale,
            initial_load_sign=self.initial_load_sign,nodal_forces=self.nodal_forces,max_iterations=self.max_iterations,
            max_backtracks=self.max_backtracks,tolerance=TOLERANCE,geometry_policy=POLICY,load_policy=LOAD_POLICY)


def _capture(model,program):
    if type(program) is not ArcProgram: raise ValueError('exact arc program required')
    pattern = ForceProgram((0.,),program.nodal_forces,program.max_iterations,program.max_backtracks)
    elements,n,free,nodal,model_identity,model_guard = force_capture(model,pattern)
    if not np.any(nodal) and not any(np.any(e.core.line_force) for _,e in elements):
        raise ValueError('arc continuation requires a nonzero declared load pattern')
    maps = np.array([model.mesh.dof_manager.get_node_dofs(i) for i in sorted(model.mesh.nodes)],dtype=int)
    if maps.shape != (len(model.mesh.nodes),6) or sorted(maps.ravel()) != list(range(n)):
        raise ValueError('complete disjoint native node mapping required')
    metric = np.zeros(n+1)
    metric[maps[:,:3]] = _inverse_square(program.length_scale)/len(maps)
    metric[maps[:,3:]] = 1./len(maps)
    metric[-1] = _inverse_square(program.parameter_scale)
    if not np.isfinite(metric).all() or np.any(metric <= 0.): raise ValueError('finite positive metric required')
    initial = np.zeros(n+1); initial[-1] = program.initial_load_sign*program.parameter_scale
    descriptor = sha(program.descriptor())
    identity = sha(dict(model=model_identity,program=descriptor,metric=metric,node_maps=maps))
    def guard():
        model_guard()
        if sha(program.descriptor()) != descriptor: raise ValueError('arc program changed')
    return elements,n,free,nodal,identity,guard,maps,_readonly(metric),_readonly(initial)


def _bordered(matrix,column,free,row):
    return sparse.bmat([[matrix[free][:,free],sparse.csr_matrix(column[free,None])],
        [sparse.csr_matrix(row[free][None,:]),sparse.csr_matrix([[row[-1]]])]],format='csr')


def _direction(matrix,column,free,previous,metric):
    row = metric*previous; rhs = np.zeros(len(free)+1); rhs[-1] = 1.
    handle = factorize(_bordered(matrix,column,free,row),MatrixClass.GENERAL)
    reduced = np.asarray(handle.solve(rhs),dtype=float).reshape(-1)
    made = np.zeros(len(metric)); made[free] = reduced[:-1]; made[-1] = reduced[-1]
    norm = math.sqrt(float(np.sum(metric*made*made)))
    if not math.isfinite(norm) or norm <= 0.: raise ValueError('unresolved arc tangent; no automatic branch selection')
    made /= norm
    if not np.isfinite(made).all() or float(row@made) <= 0.: raise ValueError('unresolved arc orientation')
    return _readonly(made)


def _arc(total,origin,direction,metric,p,p0,step,maps):
    value,local = constraint((total-origin)[maps],direction[:-1][maps],metric[:-1][maps],
        float(p-p0),float(direction[-1]),float(metric[-1]),step)
    row = np.empty(len(metric)); row[maps.ravel()] = local[:-1]; row[-1] = local[-1]
    return value,row


def _states(model,elements,states):
    return [dict(element_id=i,state=e.serialize_native_material_state(model.mesh,states[i])) for i,e in elements]


def _decode_states(rows,model,elements,total,p,epoch):
    if type(rows) is not list or len(rows) != len(elements): raise ValueError('complete ordered arc states required')
    states = {}
    for row,(i,e) in zip(rows,elements):
        if type(row) is not dict or set(row) != {'element_id','state'} or type(row['element_id']) is not int or row['element_id'] != i:
            raise ValueError('arc element identity/order mismatch')
        states[i] = e.validate_model_bound_nonlinear_state(model.mesh,e.core.section,row['state'],1,
            expected_committed_total_u=total[e.get_dof_mapping(model.mesh)])
        inner = states[i]['material_state']
        if inner['epoch'] != epoch or inner['load_parameter'] != p: raise ValueError('arc history epoch/parameter mismatch')
    return states


def _store(model,states,total):
    store = LoadStateStore.from_shell_layouts((),states)
    store.attach_native_rotation_store(create_model_native_rotation_store(model,states,total))
    return store


def _assemble(model,store,total,p,nodal):
    with store.load_parameter_scope(p): force,matrix,trial = _assemble_nonlinear_system(model,total,store,1)
    return force-p*nodal,matrix,trial


def _capsule(model,program,identity,elements,states,total,reaction,cursor,p,direction,metric,records,origin):
    body = dict(schema=SCHEMA,model_sha256=identity,program=program.descriptor(),program_sha256=sha(program.descriptor()),
        completed_steps=cursor,accepted_parameter=p,total_displacement=total.tolist(),physical_imbalance=reaction.tolist(),
        direction=direction.tolist(),metric=metric.tolist(),records=records,last_origin=origin,
        element_states=_states(model,elements,states))
    raw = canonical({**body,'checkpoint_sha256':sha(body)}); _load(raw.decode('ascii'))
    return raw


def _restore(raw,model,program,identity,elements,n,free,nodal,maps,metric,initial):
    if type(raw) is not bytes: raise ValueError('canonical arc checkpoint bytes required')
    value = _load(raw.decode('ascii'))
    keys = {'schema','model_sha256','program','program_sha256','completed_steps','accepted_parameter','total_displacement',
        'physical_imbalance','direction','metric','records','last_origin','element_states','checkpoint_sha256'}
    if type(value) is not dict or set(value) != keys: raise ValueError('exact arc checkpoint schema required')
    body = {k:v for k,v in value.items() if k != 'checkpoint_sha256'}
    if (value['schema'] != SCHEMA or value['model_sha256'] != identity or value['program_sha256'] != sha(program.descriptor())
            or canonical(value['program']) != canonical(program.descriptor()) or value['checkpoint_sha256'] != sha(body)):
        raise ValueError('arc checkpoint model/program/hash mismatch')
    cursor = value['completed_steps']; p = parameter_value(value['accepted_parameter'])
    if type(cursor) is not int or not 0 <= cursor <= len(program.steps): raise ValueError('bounded arc cursor required')
    array = lambda item,size: _decode(item,('array',(size,)))
    total = array(value['total_displacement'],n); reaction = array(value['physical_imbalance'],n)
    direction = array(value['direction'],n+1)
    if not np.array_equal(array(value['metric'],n+1),metric): raise ValueError('arc metric changed')
    fixed = list(set(range(n))-set(free))
    def direction_check(d):
        if np.any(d[fixed]) or abs(float(np.sum(metric*d*d))-1.) > TOLERANCE: raise ValueError('arc orientation normalization/support mismatch')
    direction_check(direction)
    if np.any(total[fixed]): raise ValueError('arc checkpoint violates supports')
    states = _decode_states(value['element_states'],model,elements,total,p,cursor)
    physical = _physical_force(model,elements,states,p,nodal)
    if not np.array_equal(physical,reaction) or np.linalg.norm(physical[free]) > TOLERANCE*max(1.,abs(p)*np.linalg.norm(nodal)):
        raise ValueError('arc accepted equilibrium/reaction mismatch')
    records = value['records']
    if type(records) is not list or len(records) != cursor: raise ValueError('complete arc path records required')
    for index,row in enumerate(records):
        if type(row) is not dict or set(row) != {'step','step_size','parameter','iterations','arc_residual','direction'}:
            raise ValueError('exact arc record required')
        if type(row['step']) is not int or row['step'] != index+1 or parameter_value(row['step_size']) != program.steps[index]:
            raise ValueError('arc step ordering/size mismatch')
        if type(row['iterations']) is not int or not 0 <= row['iterations'] <= program.max_iterations: raise ValueError('arc iteration record mismatch')
        if not 0. <= parameter_value(row['arc_residual']) <= TOLERANCE: raise ValueError('arc residual record mismatch')
        parameter_value(row['parameter']); direction_check(array(row['direction'],n+1))
    origin = value['last_origin']
    if cursor == 0:
        if p != 0. or np.any(total) or origin is not None or not np.array_equal(direction,initial):
            raise ValueError('invalid virgin arc origin')
    else:
        if records[-1]['parameter'] != p or not np.array_equal(array(records[-1]['direction'],n+1),direction):
            raise ValueError('accepted arc direction/parameter record mismatch')
        if type(origin) is not dict or set(origin) != {'total','parameter','direction','element_states'}:
            raise ValueError('complete preceding arc origin required')
        u0 = array(origin['total'],n); p0 = parameter_value(origin['parameter']); previous = array(origin['direction'],n+1)
        expected_previous = initial if cursor == 1 else array(records[-2]['direction'],n+1)
        if np.any(u0[fixed]) or p0 != (0. if cursor == 1 else records[-2]['parameter']) or not np.array_equal(previous,expected_previous):
            raise ValueError('arc predecessor direction/parameter mismatch')
        old = _decode_states(origin['element_states'],model,elements,u0,p0,cursor-1)
        old_physical = _physical_force(model,elements,old,p0,nodal)
        if np.linalg.norm(old_physical[free]) > TOLERANCE*max(1.,abs(p0)*np.linalg.norm(nodal)):
            raise ValueError('arc predecessor is not an equilibrium')
        for i,e in elements:
            before = old[i]['material_state']; after = states[i]['material_state']
            if after['origins'] != before['histories']: raise ValueError('arc material origin linkage mismatch')
            delta = (total-u0)[e.get_dof_mapping(model.mesh)].reshape(3,6)[:,3:]
            expected = np.array([rotation_exponential(v)@q for v,q in zip(delta,before['committed_nodal_rotation_matrices'])])
            if not np.array_equal(expected,after['committed_nodal_rotation_matrices']): raise ValueError('arc multiplicative origin linkage mismatch')
        gap,_ = _arc(total,u0,direction,metric,p,p0,program.steps[cursor-1],maps)
        if abs(gap) > TOLERANCE or abs(gap) != records[-1]['arc_residual']: raise ValueError('arc hyperplane does not replay')
        probe = _store(model,old,u0)
        try:
            _,matrix,trial = _assemble(model,probe,u0,p0,nodal)
            column = _parameter_column(model,elements,probe,trial,n,nodal)
            expected = _direction(matrix,column,free,previous,metric)
            if not np.array_equal(expected,direction): raise ValueError('arc predictor does not replay from saved origin')
        finally:
            if probe.has_active_trial: probe.discard_trial(probe.active_trial_token())
    return cursor,p,total.copy(),reaction.copy(),states,direction.copy(),deepcopy(records),deepcopy(origin)


def solve_arc_program(model,program,*,checkpoint=None,stop_after=None,cancellation_token=None,progress=None):
    started = time.monotonic(); cancellation_safe_point(cancellation_token,'native_arc.start')
    elements,n,free,nodal,identity,guard,maps,metric,initial = _capture(model,program)
    if progress is not None and not callable(progress): raise ValueError('callable arc observer required')
    end = len(program.steps) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(program.steps): raise ValueError('bounded arc stop count required')
    if checkpoint is None:
        states = {i:e.init_model_bound_nonlinear_state(model.mesh,e.core.section,1) for i,e in elements}
        cursor = 0; p = 0.; total = np.zeros(n); records = []; direction = initial.copy(); origin = None
        reaction = _physical_force(model,elements,states,p,nodal)
        capsule = _capsule(model,program,identity,elements,states,total,reaction,cursor,p,direction,metric,records,origin)
    else:
        cursor,p,total,reaction,states,direction,records,origin = _restore(checkpoint,model,program,identity,elements,n,free,nodal,maps,metric,initial)
        capsule = checkpoint
    if cursor > end: raise ValueError('cannot rewind an accepted arc cursor')
    guard(); store = _store(model,states,total)
    status = 'completed' if end == len(program.steps) else 'paused'; failure = None
    def observe(stage,index,iteration=0,**data):
        cancellation_safe_point(cancellation_token,stage); guard()
        if time.monotonic()-started > 600.: raise RuntimeError('arc cooperative deadline exceeded')
        if progress is not None: progress(dict(stage=stage,step=index,iteration=iteration,**data))
        cancellation_safe_point(cancellation_token,stage); guard()
    def evaluate(u,parameter,index,iteration):
        observe('native_arc.before_assembly',index,iteration)
        force,matrix,trial = _assemble(model,store,u,parameter,nodal)
        observe('native_arc.after_assembly',index,iteration)
        physical = _physical_force(model,elements,trial,parameter,nodal)
        return force,matrix,trial,physical
    def force_norm(force,physical,parameter):
        return max(np.linalg.norm(force[free]),np.linalg.norm(physical[free]))/max(1.,abs(parameter)*np.linalg.norm(nodal))
    try:
        observe('native_arc.initialized',cursor)
        for index in range(cursor,end):
            step = program.steps[index]
            origin = dict(total=total.tolist(),parameter=p,direction=direction.tolist(),element_states=_states(model,elements,store.materialize()))
            force,matrix,trial,physical = evaluate(total,p,index+1,0)
            if force_norm(force,physical,p) > TOLERANCE: raise ValueError('arc origin is not an equilibrium')
            column = _parameter_column(model,elements,store,trial,n,nodal)
            observe('native_arc.before_predictor',index+1)
            tangent = _direction(matrix,column,free,direction,metric)
            candidate = total+step*tangent[:-1]; parameter = float(p+step*tangent[-1]); accepted = False
            for iteration in range(program.max_iterations+1):
                force,matrix,trial,physical = evaluate(candidate,parameter,index+1,iteration)
                gap,row = _arc(candidate,total,tangent,metric,parameter,p,step,maps)
                norm = max(force_norm(force,physical,parameter),abs(gap))
                observe('native_arc.iteration',index+1,iteration,parameter=parameter,merit=float(norm),arc_error=float(gap))
                if norm <= TOLERANCE:
                    next_records = [*records,dict(step=index+1,step_size=step,parameter=parameter,iterations=iteration,
                        arc_residual=float(abs(gap)),direction=tangent.tolist())]
                    staged = _capsule(model,program,identity,elements,trial,candidate,physical,index+1,parameter,tangent,metric,next_records,origin)
                    next_total = candidate.copy(); next_reaction = physical.copy()
                    observe('native_arc.before_commit',index+1,iteration)
                    coordinates = native_trial_full_coordinates(store,model,candidate)
                    store.commit(store.active_trial_token(),accepted_full_displacement=candidate,accepted_full_coordinates=coordinates)
                    capsule,total,reaction,records = staged,next_total,next_reaction,next_records
                    cursor,p,direction = index+1,parameter,tangent.copy(); accepted = True
                    observe('native_arc.committed',cursor,iteration)
                    break
                if iteration == program.max_iterations: break
                column = _parameter_column(model,elements,store,trial,n,nodal)
                observe('native_arc.before_corrector',index+1,iteration)
                handle = factorize(_bordered(matrix,column,free,row),MatrixClass.GENERAL)
                delta = np.asarray(handle.solve(-np.r_[force[free],gap]),dtype=float).reshape(-1)
                if not np.isfinite(delta).all(): raise ValueError('nonfinite arc corrector')
                for cut in range(program.max_backtracks+1):
                    proposed = candidate.copy(); proposed[free] += (.5**cut)*delta[:-1]
                    next_p = float(parameter+(.5**cut)*delta[-1])
                    changed,_,_,changed_physical = evaluate(proposed,next_p,index+1,iteration)
                    changed_gap,_ = _arc(proposed,total,tangent,metric,next_p,p,step,maps)
                    if max(force_norm(changed,changed_physical,next_p),abs(changed_gap)) < norm:
                        candidate,parameter = proposed,next_p; break
                else: raise RuntimeError('arc line search exhausted')
            if not accepted: raise RuntimeError('arc iteration bound exhausted')
    except SolveCancelled as error:
        status,failure = 'cancelled',str(error)
    except Exception as error:
        status,failure = 'failed',type(error).__name__+': '+str(error)
    finally:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())
    return ForceProgramResult(status,cursor,p,_readonly(total),_readonly(reaction),capsule,failure)
