"""Private native displacement control with an unknown common load parameter.

Solves the full bordered Jacobian, not K-inverse load sensitivities. This keeps
the control equation available at a simple load limit point. No new mechanics,
arc-length policy, public routing or qualification authority is introduced.
"""

from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse

from anyfesolver.control import cancellation_safe_point, SolveCancelled
from anyfesolver.linalg import factorize, MatrixClass
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from anyfesolver.nonlinear_state import create_model_native_rotation_store, native_trial_full_coordinates
from anyfesolver._ge_beam3_load_program import ForceProgram, ForceProgramResult, _capture as force_capture, _physical_force, TOLERANCE
from anyfesolver._ge_beam3_p5.arrays import _readonly
from anyfesolver._ge_beam3_p5.typed_schema import _decode
from anyfesolver._ge_beam3_p5_loads import LoadStateStore
from anyfesolver._ge_beam3_p5_loads.core import canonical, sha
from anyfesolver._ge_beam3_p5_loads.codec import _load
from anyfesolver._ge_beam3_p5_loads.state import parameter_value, LOAD_POLICY


PROGRAM = 'GE_BEAM3_TRANSLATION_CONTROL_PROGRAM_V1'
SCHEMA = 'GE_BEAM3_TRANSLATION_CONTROL_CHECKPOINT_V1'


@dataclass(frozen=True)
class DisplacementProgram:
    targets: tuple
    control_node: int
    control_component: str
    nodal_forces: tuple = ()
    max_iterations: int = 12
    max_backtracks: int = 8

    def __post_init__(self):
        # Validate the numeric schedule/pattern/bounds, not its interpretation.
        ForceProgram(self.targets,self.nodal_forces,self.max_iterations,self.max_backtracks)
        if type(self.control_node) is not int or self.control_component not in ('ux','uy','uz'):
            raise ValueError('explicit node and spatial translation control required')
        if type(self.control_component) is not str:
            raise ValueError('exact translation component required')

    def descriptor(self):
        return dict(schema=PROGRAM,targets=self.targets,control_node=self.control_node,
            control_component=self.control_component,nodal_forces=self.nodal_forces,
            max_iterations=self.max_iterations,max_backtracks=self.max_backtracks,
            tolerance=TOLERANCE,load_policy=LOAD_POLICY)


def _capture(model,program):
    if type(program) is not DisplacementProgram: raise ValueError('exact displacement program required')
    # Reuse only the supported-model capture/pattern guard. The dummy force
    # target is never executed, serialized as a path or supplied to mechanics.
    pattern = ForceProgram((0.,),program.nodal_forces,program.max_iterations,program.max_backtracks)
    elements,n,free,nodal,model_identity,model_guard = force_capture(model,pattern)
    if program.control_node not in model.mesh.nodes: raise ValueError('control node is absent')
    control = model.mesh.dof_manager.get_node_dofs(program.control_node)[('ux','uy','uz').index(program.control_component)]
    where = np.flatnonzero(free == control)
    if len(where) != 1: raise ValueError('control coordinate must be a free translation')
    descriptor = sha(program.descriptor())
    identity = sha(dict(model_identity=model_identity,program_sha256=descriptor,control_dof=control))
    def guard():
        model_guard()
        if sha(program.descriptor()) != descriptor: raise ValueError('displacement program changed')
    return elements,n,free,nodal,identity,guard,control,int(where[0])


def _parameter_column(model,elements,store,states,n,nodal):
    """Exact native chart derivative of net residual at this owned trial."""
    result = -nodal.copy(); token = store.active_trial_token()
    for i,element in elements:
        context = store.native_material_context(token,i)
        view = store.native_element_rotation_view(token,i,element.node_ids,element.native_reference_directors(model.mesh))
        value = element.native_load_parameter_derivative(model.mesh,states[i],
            native_rotation_trial=view,native_material_context=context)
        if value['load_policy'] != LOAD_POLICY or value['load_parameter'] != store.native_load_parameter(token,i):
            raise ValueError('native parameter column policy/parameter mismatch')
        np.add.at(result,element.get_dof_mapping(model.mesh),value['residual_parameter_derivative'])
    if result.shape != (n,) or not np.isfinite(result).all(): raise ValueError('invalid assembled parameter column')
    return result


def _bordered(matrix,column,free,control_index):
    row = np.zeros(len(free)); row[control_index] = 1.
    return sparse.bmat([[matrix[free][:,free],sparse.csr_matrix(column[free,None])],
        [sparse.csr_matrix(row[None,:]),sparse.csr_matrix((1,1))]],format='csr')


def _capsule(model,program,identity,elements,states,total,reaction,cursor,parameter,records):
    body = dict(schema=SCHEMA,model_sha256=identity,program=program.descriptor(),
        program_sha256=sha(program.descriptor()),completed_targets=cursor,accepted_parameter=parameter,
        total_displacement=total.tolist(),physical_imbalance=reaction.tolist(),records=records,
        element_states=[dict(element_id=i,state=e.serialize_native_material_state(model.mesh,states[i])) for i,e in elements])
    raw = canonical({**body,'checkpoint_sha256':sha(body)})
    _load(raw.decode('ascii'))
    return raw


def _restore(raw,model,program,identity,elements,n,free,nodal,control):
    if type(raw) is not bytes: raise ValueError('canonical displacement checkpoint bytes required')
    value = _load(raw.decode('ascii'))
    keys = {'schema','model_sha256','program','program_sha256','completed_targets','accepted_parameter',
        'total_displacement','physical_imbalance','records','element_states','checkpoint_sha256'}
    if type(value) is not dict or set(value) != keys: raise ValueError('exact displacement checkpoint schema required')
    body = {k:v for k,v in value.items() if k != 'checkpoint_sha256'}
    if (value['schema'] != SCHEMA or value['model_sha256'] != identity
            or value['program_sha256'] != sha(program.descriptor())
            or canonical(value['program']) != canonical(program.descriptor())
            or value['checkpoint_sha256'] != sha(body)):
        raise ValueError('displacement checkpoint model/program/hash mismatch')
    cursor = value['completed_targets']
    if type(cursor) is not int or not 0 <= cursor <= len(program.targets): raise ValueError('bounded cursor required')
    parameter = parameter_value(value['accepted_parameter'])
    total = _decode(value['total_displacement'],('array',(n,)))
    reaction = _decode(value['physical_imbalance'],('array',(n,)))
    if np.any(total[list(set(range(n))-set(free))]): raise ValueError('checkpoint violates supports')
    target = 0. if cursor == 0 else program.targets[cursor-1]
    if abs(total[control]-target) > TOLERANCE*max(1.,abs(target)):
        raise ValueError('checkpoint violates displacement control')
    if cursor == 0 and (parameter != 0. or np.any(total)):
        raise ValueError('virgin program requires zero parameter and displacement')
    rows = value['element_states']
    if type(rows) is not list or len(rows) != len(elements): raise ValueError('complete element states required')
    states = {}
    for row,(i,element) in zip(rows,elements):
        if type(row) is not dict or set(row) != {'element_id','state'} or type(row['element_id']) is not int or row['element_id'] != i:
            raise ValueError('element checkpoint identity/order mismatch')
        states[i] = element.validate_model_bound_nonlinear_state(model.mesh,element.core.section,row['state'],1,
            expected_committed_total_u=total[element.get_dof_mapping(model.mesh)])
        inner = states[i]['material_state']
        if inner['load_parameter'] != parameter or inner['epoch'] != cursor:
            raise ValueError('native history epoch/parameter mismatch')
    physical = _physical_force(model,elements,states,parameter,nodal)
    if not np.array_equal(physical,reaction): raise ValueError('physical reaction replay mismatch')
    if np.linalg.norm(physical[free]) > TOLERANCE*max(1.,abs(parameter)*np.linalg.norm(nodal)):
        raise ValueError('checkpoint is not an equilibrium')
    records = value['records']
    if type(records) is not list or len(records) != cursor: raise ValueError('complete path records required')
    for index,row in enumerate(records):
        if type(row) is not dict or set(row) != {'target','displacement_target','parameter','iterations'}:
            raise ValueError('exact control step record required')
        if type(row['target']) is not int or row['target'] != index+1:
            raise ValueError('ordered step identity required')
        if type(row['iterations']) is not int or not 0 <= row['iterations'] <= program.max_iterations:
            raise ValueError('bounded iteration record required')
        if parameter_value(row['displacement_target']) != program.targets[index]:
            raise ValueError('recorded target mismatch')
        parameter_value(row['parameter'])
    if cursor and records[-1]['parameter'] != parameter: raise ValueError('last accepted parameter mismatch')
    return cursor,parameter,total.copy(),reaction.copy(),states,deepcopy(records)


def solve_displacement_program(model,program,*,checkpoint=None,stop_after=None,cancellation_token=None,progress=None):
    """Solve equilibrium and one translation control with parameter unknown.

    Uses [K, dR/dp; e_control.T, 0], even when K alone is singular.
    This does not prove a particular beam's post-buckling branch or stability.
    """
    started = time.monotonic(); cancellation_safe_point(cancellation_token,'native_displacement.start')
    elements,n,free,nodal,identity,guard,control,local_control = _capture(model,program)
    if progress is not None and not callable(progress): raise ValueError('callable progress observer required')
    end = len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(program.targets): raise ValueError('bounded stop target required')
    if checkpoint is None:
        states = {i:e.init_model_bound_nonlinear_state(model.mesh,e.core.section,1) for i,e in elements}
        cursor = 0; parameter = 0.; total = np.zeros(n); records = []
        reaction = _physical_force(model,elements,states,parameter,nodal)
        capsule = _capsule(model,program,identity,elements,states,total,reaction,cursor,parameter,records)
    else:
        cursor,parameter,total,reaction,states,records = _restore(checkpoint,model,program,identity,elements,n,free,nodal,control)
        capsule = checkpoint
    if cursor > end: raise ValueError('cannot rewind accepted cursor')
    guard(); store = LoadStateStore.from_shell_layouts((),states)
    store.attach_native_rotation_store(create_model_native_rotation_store(model,states,total))
    status = 'completed' if end == len(program.targets) else 'paused'; failure = None
    def safe(stage):
        cancellation_safe_point(cancellation_token,stage)
        if time.monotonic()-started > 600.: raise RuntimeError('displacement cooperative deadline exceeded')
        guard()
    def observe(stage,index,iteration=0,**diagnostics):
        safe(stage)
        if progress is not None: progress(dict(stage=stage,target=index,iteration=iteration,**diagnostics))
        safe(stage)
    def assemble(candidate,p,index,iteration):
        observe('native_displacement.before_assembly',index,iteration)
        with store.load_parameter_scope(parameter_value(p)):
            force,matrix,trial = _assemble_nonlinear_system(model,candidate,store,1)
        safe('native_displacement.after_assembly')
        return force-p*nodal,matrix,trial
    def merit(force,physical,p,candidate,target):
        force_scale = max(1.,abs(p)*np.linalg.norm(nodal))
        return max(np.linalg.norm(force[free])/force_scale,np.linalg.norm(physical[free])/force_scale,
            abs(candidate[control]-target)/max(1.,abs(target)))
    try:
        observe('native_displacement.initialized',cursor)
        for index in range(cursor,end):
            target = program.targets[index]; candidate = total.copy(); p = parameter; accepted = False
            # Translation control is an exact affine equation, not a soft
            # residual to approach gradually from the preceding target. Seed
            # and keep every trial on that plane; all other coordinates and
            # the unknown parameter remain in the bordered Newton solve.
            # This changes no accepted state or material origin.
            candidate[control] = target
            for iteration in range(program.max_iterations+1):
                force,matrix,trial = assemble(candidate,p,index+1,iteration)
                physical = _physical_force(model,elements,trial,p,nodal)
                norm = merit(force,physical,p,candidate,target)
                observe('native_displacement.iteration',index+1,iteration,parameter=p,merit=float(norm),
                    control_error=float(candidate[control]-target),force_norm=float(np.linalg.norm(force[free])),
                    physical_norm=float(np.linalg.norm(physical[free])))
                if norm <= TOLERANCE:
                    next_records = [*records,dict(target=index+1,displacement_target=target,parameter=p,iterations=iteration)]
                    staged = _capsule(model,program,identity,elements,trial,candidate,physical,index+1,p,next_records)
                    next_total = candidate.copy(); next_reaction = physical.copy()
                    observe('native_displacement.before_commit',index+1,iteration)
                    coordinates = native_trial_full_coordinates(store,model,candidate)
                    store.commit(store.active_trial_token(),accepted_full_displacement=candidate,accepted_full_coordinates=coordinates)
                    capsule,total,reaction,records = staged,next_total,next_reaction,next_records
                    cursor,parameter = index+1,p; accepted = True
                    observe('native_displacement.committed',cursor,iteration)
                    break
                if iteration == program.max_iterations: break
                observe('native_displacement.before_parameter_column',index+1,iteration)
                column = _parameter_column(model,elements,store,trial,n,nodal)
                safe('native_displacement.after_parameter_column')
                augmented = _bordered(matrix,column,free,local_control)
                rhs = -np.r_[force[free],candidate[control]-target]
                safe('native_displacement.before_factorization')
                handle = factorize(augmented,MatrixClass.GENERAL,signature=f'native_displacement:{identity}:{index}:{iteration}')
                increment = np.asarray(handle.solve(rhs),dtype=float).reshape(-1)
                if not np.isfinite(increment).all(): raise ValueError('nonfinite control increment')
                safe('native_displacement.after_factorization')
                for cut in range(program.max_backtracks+1):
                    proposed = candidate.copy(); proposed[free] += (.5**cut)*increment[:-1]
                    proposed[control] = target
                    next_p = float(p+(.5**cut)*increment[-1])
                    changed,_,changed_states = assemble(proposed,next_p,index+1,iteration)
                    changed_physical = _physical_force(model,elements,changed_states,next_p,nodal)
                    if merit(changed,changed_physical,next_p,proposed,target) < norm:
                        candidate,p = proposed,next_p; break
                else: raise RuntimeError('displacement line search exhausted')
            if not accepted: raise RuntimeError('displacement iteration bound exhausted')
    except SolveCancelled as error:
        status,failure = 'cancelled',str(error)
    except Exception as error:
        status,failure = 'failed',type(error).__name__+': '+str(error)
    finally:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())
    return ForceProgramResult(status,cursor,parameter,_readonly(total),_readonly(reaction),capsule,failure)
