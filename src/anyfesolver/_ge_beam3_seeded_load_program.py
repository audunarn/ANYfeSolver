# V5 private successor: authoritative seed and conservative globalization.
"""Private bounded native force-program driver and global restart capsule.

Uses the existing scalar assembly, linear algebra and native state machinery.
No public selector, legacy routing, displacement/arc control or qualification.
Line work is already included in the element residual; only the separate nodal
force pattern is subtracted here. All loads use the same explicit parameter.
"""

from copy import deepcopy
from dataclasses import dataclass
import time
from math import fsum

import numpy as np
from scipy import sparse

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.control import cancellation_safe_point, SolveCancelled
from anyfesolver.linalg import MatrixClass, factorize
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from anyfesolver.nonlinear_state import create_model_native_rotation_store, native_trial_full_coordinates
from anyfesolver._ge_beam3_p5.arrays import _readonly
from anyfesolver._ge_beam3_p5_seeded import NativeP5BeamElement, LoadStateStore
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_p5_seeded.codec import _load
from anyfesolver._ge_beam3_p5_seeded.state import parameter_value, LOAD_POLICY
from anyfesolver._ge_beam3_conservative_backtrack import acceptance


SCHEMA = 'GE_BEAM3_SEEDED_FORCE_PROGRAM_CHECKPOINT_V1'
PROGRAM = 'GE_BEAM3_SEEDED_CONSERVATIVE_FORCE_PROGRAM_V2'
TOLERANCE = 1e-11


@dataclass(frozen=True)
class ForceProgram:
    targets: tuple
    nodal_forces: tuple = ()  # Sorted (node ID, Fx, Fy, Fz); no nodal moments.
    max_iterations: int = 12
    max_backtracks: int = 8

    def __post_init__(self):
        if type(self.targets) is not tuple or not 1 <= len(self.targets) <= 64:
            raise ValueError('one to 64 explicit parameter targets required')
        for value in self.targets: parameter_value(value)
        if type(self.nodal_forces) is not tuple:
            raise ValueError('exact nodal force tuple required')
        ids = []
        for row in self.nodal_forces:
            if type(row) is not tuple or len(row) != 4 or type(row[0]) is not int:
                raise ValueError('explicit node and three spatial dead force components required')
            ids.append(row[0])
            for value in row[1:]: parameter_value(value)
        if ids != sorted(set(ids)):
            raise ValueError('unique ascending nodal force IDs required')
        for value,limit in ((self.max_iterations,32),(self.max_backtracks,12)):
            if type(value) is not int or not 0 <= value <= limit:
                raise ValueError('bounded exact Newton/backtracking limits required')

    def descriptor(self):
        return dict(schema=PROGRAM,targets=self.targets,nodal_forces=self.nodal_forces,
            max_iterations=self.max_iterations,max_backtracks=self.max_backtracks,
            tolerance=TOLERANCE,load_policy=LOAD_POLICY,
            line_search='CONSERVATIVE_ARMIJO_1E_MINUS4_ROUNDOFF_RESIDUAL_V1')


@dataclass(frozen=True)
class ForceProgramResult:
    status: str
    completed_targets: int
    parameter: float
    displacements: np.ndarray
    physical_imbalance: np.ndarray
    checkpoint: bytes
    failure: str | None
    production_qualified: bool = False


def _capture(model, program):
    if type(program) is not ForceProgram: raise ValueError('exact native force program required')
    elements = tuple(sorted(model.mesh.elements.items()))
    n = model.mesh.dof_manager.total_dofs
    if not elements or not 1 <= n+6*len(elements) <= 256:
        raise ValueError('bounded standalone native model required before mechanics')
    if any(type(e) is not NativeP5BeamElement for _,e in elements):
        raise ValueError('exact load-aware native elements required; no mixed formulations or joints')
    _,_,transform,offset,free,info = build_constraint_transformation(sparse.eye(n,format='csr'),np.zeros(n),model)
    if info['slave_dofs'] or np.any(offset) or len(free) == n:
        raise ValueError('supported homogeneous non-MPC constraints required')
    constrained = set(range(n))-set(free)
    for node in model.mesh.nodes:
        rotations = set(model.mesh.dof_manager.get_node_dofs(node)[3:6])
        if len(rotations & constrained) not in (0,3):
            raise ValueError('partial rotational supports need a separately qualified constraint contract')
    if transform.nnz != len(free) or np.any(transform.data != 1.):
        raise ValueError('native force program requires a pure support selector')
    nodal = np.zeros(n)
    for node,*force in program.nodal_forces:
        if node not in model.mesh.nodes: raise ValueError('nodal force targets an absent node')
        nodal[list(model.mesh.dof_manager.get_node_dofs(node)[:3])] = force

    def snapshot():
        if (tuple(sorted(model.mesh.elements.items())) != elements or model.mesh.element_activity is not None
                or model.constraint_equations or model.mesh.point_masses):
            raise ValueError('native force model ownership/activity/constraint mismatch')
        for _,element in elements:
            element._check(model.mesh)
            if model.materials.get(element.material_name) is not element.core.section:
                raise ValueError('native force section ownership changed')
        return sha(dict(elements=[(i,e.to_dict(),list(e.get_dof_mapping(model.mesh))) for i,e in elements],
            nodes=[(i,node.coords()) for i,node in sorted(model.mesh.nodes.items())],
            boundaries=[vars(bc) for bc in model.boundary_conditions],
            dofs=model.mesh.dof_manager.total_dofs,
            constrained_dofs=sorted(model.mesh.dof_manager._constrained_dofs),
            program=program.descriptor()))
    identity = snapshot()
    def guard():
        if snapshot() != identity: raise ValueError('native force model/program inputs changed')
    return elements,n,np.array(free,copy=True),_readonly(nodal),identity,guard


def _physical_force(model, elements, states, parameter, nodal):
    # At an accepted state use the physical spatial force, not the last
    # increment's rotation-coordinate pullback. Supports above admit complete
    # rotational blocks only, so both free-equilibrium conditions are enforced.
    result = -parameter*nodal.copy()
    for i,element in elements:
        np.add.at(result,element.get_dof_mapping(model.mesh),states[i]['material_state']['response'].residual)
    if not np.isfinite(result).all(): raise ValueError('nonfinite accepted physical imbalance')
    return result


def _potential(elements, states, total, parameter, nodal):
    # Element potential already contains the distributed spatial-dead work.
    # Add only nodal translation work. History origins stay fixed throughout
    # the global trial; this is the native incremental potential.
    terms = [float(states[i]['material_state']['response'].potential) for i,_ in elements]
    work = fsum(float(parameter)*float(f)*float(u) for f,u in zip(nodal,total))
    value = fsum([*terms,-work]); scale = fsum([*(abs(v) for v in terms),abs(work)])
    if not np.isfinite(value) or not np.isfinite(scale): raise ValueError('finite incremental energy required')
    return value,scale


def _checkpoint(model, program, identity, elements, states, total, reaction, cursor, records):
    body = dict(schema=SCHEMA,model_sha256=identity,program=program.descriptor(),program_sha256=sha(program.descriptor()),
        completed_targets=cursor,accepted_parameter=0. if cursor == 0 else program.targets[cursor-1],
        total_displacement=total.tolist(),physical_imbalance=reaction.tolist(),records=records,
        element_states=[dict(element_id=i,state=e.serialize_native_material_state(model.mesh,states[i])) for i,e in elements])
    raw = canonical({**body,'checkpoint_sha256':sha(body)})
    # Stage a complete bounded canonical capsule BEFORE committing the trial.
    _load(raw.decode('ascii'))
    return raw


def _restore(raw, model, program, identity, elements, n, free, nodal):
    if type(raw) is not bytes: raise ValueError('canonical checkpoint bytes required')
    value = _load(raw.decode('ascii'))
    keys = {'schema','model_sha256','program','program_sha256','completed_targets','accepted_parameter',
        'total_displacement','physical_imbalance','records','element_states','checkpoint_sha256'}
    if type(value) is not dict or set(value) != keys: raise ValueError('exact global force checkpoint schema required')
    body = {k:v for k,v in value.items() if k != 'checkpoint_sha256'}
    if (value['schema'] != SCHEMA or value['model_sha256'] != identity
            or value['program_sha256'] != sha(program.descriptor())
            or canonical(value['program']) != canonical(program.descriptor())
            or value['checkpoint_sha256'] != sha(body)):
        raise ValueError('global force checkpoint model/program/hash mismatch')
    cursor = value['completed_targets']
    if type(cursor) is not int or not 0 <= cursor <= len(program.targets): raise ValueError('bounded checkpoint cursor required')
    parameter = parameter_value(value['accepted_parameter'])
    if parameter != (0. if cursor == 0 else program.targets[cursor-1]): raise ValueError('checkpoint schedule/parameter mismatch')
    from anyfesolver._ge_beam3_p5.typed_schema import _decode
    total = _decode(value['total_displacement'],('array',(n,)))
    reaction = _decode(value['physical_imbalance'],('array',(n,)))
    if np.any(total[list(set(range(n))-set(free))]): raise ValueError('checkpoint violates prescribed supports')
    rows = value['element_states']
    if type(rows) is not list or len(rows) != len(elements): raise ValueError('complete ordered element checkpoint required')
    states = {}
    for row,(i,element) in zip(rows,elements):
        if type(row) is not dict or set(row) != {'element_id','state'} or type(row['element_id']) is not int or row['element_id'] != i:
            raise ValueError('element checkpoint identity/order mismatch')
        states[i] = element.validate_model_bound_nonlinear_state(model.mesh,element.core.section,row['state'],1,
            expected_committed_total_u=total[element.get_dof_mapping(model.mesh)])
        inner = states[i]['material_state']
        if inner['load_parameter'] != parameter or inner['epoch'] != cursor:
            raise ValueError('element history epoch/load parameter disagrees with global cursor')
    physical = _physical_force(model,elements,states,parameter,nodal)
    if not np.array_equal(physical,reaction): raise ValueError('checkpoint physical reaction does not replay')
    if np.linalg.norm(physical[free]) > TOLERANCE*max(1.,abs(parameter)*np.linalg.norm(nodal)):
        raise ValueError('checkpoint is not a supported equilibrium')
    records = value['records']
    if type(records) is not list or len(records) != cursor: raise ValueError('complete checkpoint path records required')
    for index,row in enumerate(records):
        if type(row) is not dict or set(row) != {'target','parameter','iterations'}:
            raise ValueError('exact checkpoint step record required')
        if type(row['target']) is not int or row['target'] != index+1 or type(row['iterations']) is not int or not 0 <= row['iterations'] <= program.max_iterations:
            raise ValueError('checkpoint step ordering/bounds mismatch')
        if parameter_value(row['parameter']) != program.targets[index]: raise ValueError('checkpoint recorded schedule mismatch')
    return cursor,parameter,total.copy(),reaction.copy(),states,deepcopy(records)


def solve_force_program(model, program, *, checkpoint=None, stop_after=None, cancellation_token=None, progress=None):
    """Bounded standalone force schedule; not a public analysis dispatcher.

    Targets can unload/reverse. stop_after is a total target count, so a paused
    capsule binds the complete unchanged program. Failure/cancellation returns
    the last accepted capsule; an invalid input checkpoint raises before trial.
    The 600-second check is cooperative; formal runs still need process guards.
    """
    started = time.monotonic()
    cancellation_safe_point(cancellation_token,'native_force.start')
    elements,n,free,nodal,identity,guard = _capture(model,program)
    if progress is not None and not callable(progress): raise ValueError('callable progress observer required')
    end = len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(program.targets): raise ValueError('bounded stop target required')
    if checkpoint is None:
        states = {i:e.init_model_bound_nonlinear_state(model.mesh,e.core.section,1) for i,e in elements}
        cursor = 0; parameter = 0.; total = np.zeros(n); records = []
        reaction = _physical_force(model,elements,states,parameter,nodal)
        capsule = _checkpoint(model,program,identity,elements,states,total,reaction,cursor,records)
    else:
        cursor,parameter,total,reaction,states,records = _restore(checkpoint,model,program,identity,elements,n,free,nodal)
        capsule = checkpoint
    if cursor > end: raise ValueError('cannot rewind an accepted program cursor')
    guard()
    store = LoadStateStore.from_shell_layouts((),states)
    store.attach_native_rotation_store(create_model_native_rotation_store(model,states,total))
    status = 'completed' if end == len(program.targets) else 'paused'; failure = None

    def safe(stage):
        cancellation_safe_point(cancellation_token,stage)
        if time.monotonic()-started > 600.: raise RuntimeError('native force cooperative deadline exceeded')
        guard()

    def observe(stage, target, iteration=0):
        safe(stage)
        if progress is not None: progress(dict(stage=stage,target=target,iteration=iteration))
        safe(stage)

    def assemble(candidate, target, index, iteration):
        observe('native_force.before_assembly',index,iteration)
        with store.load_parameter_scope(target):
            f,k,trial = _assemble_nonlinear_system(model,candidate,store,1)
        safe('native_force.after_assembly')
        return f-target*nodal,k,trial

    try:
        observe('native_force.initialized',cursor)
        for index in range(cursor,end):
            target = program.targets[index]; candidate = total.copy()
            threshold = TOLERANCE*max(1.,abs(target)*np.linalg.norm(nodal))
            accepted = False
            for iteration in range(program.max_iterations+1):
                f,k,trial = assemble(candidate,target,index+1,iteration)
                physical = _physical_force(model,elements,trial,target,nodal)
                norm = max(np.linalg.norm(f[free]),np.linalg.norm(physical[free]))
                energy, energy_scale = _potential(elements,trial,candidate,target,nodal)
                if norm <= threshold:
                    next_records = [*records,dict(target=index+1,parameter=target,iterations=iteration)]
                    staged = _checkpoint(model,program,identity,elements,trial,candidate,physical,index+1,next_records)
                    next_total = candidate.copy(); next_reaction = physical.copy()
                    observe('native_force.before_commit',index+1,iteration)
                    coordinates = native_trial_full_coordinates(store,model,candidate)
                    store.commit(store.active_trial_token(),accepted_full_displacement=candidate,accepted_full_coordinates=coordinates)
                    capsule,total,reaction,records = staged,next_total,next_reaction,next_records
                    cursor,parameter = index+1,target; accepted = True
                    observe('native_force.committed',cursor,iteration)
                    break
                if iteration == program.max_iterations: break
                safe('native_force.before_factorization')
                reduced = k[free][:,free].tocsr()
                handle = factorize(reduced,MatrixClass.SYMMETRIC_INDEFINITE,
                    signature=f'native_force:{identity}:{index}:{iteration}')
                increment = np.asarray(handle.solve(-f[free]),dtype=float).reshape(-1)
                if not np.isfinite(increment).all(): raise ValueError('nonfinite native force increment')
                slope = float(f[free]@increment)
                if not np.isfinite(slope) or slope >= 0.:
                    raise RuntimeError('no conservative descent direction; explicit continuation required')
                safe('native_force.after_factorization')
                for cut in range(program.max_backtracks+1):
                    proposed = candidate.copy(); proposed[free] += (.5**cut)*increment
                    changed,_,changed_states = assemble(proposed,target,index+1,iteration)
                    changed_physical = _physical_force(model,elements,changed_states,target,nodal)
                    new_norm = max(np.linalg.norm(changed[free]),np.linalg.norm(changed_physical[free]))
                    new_energy, new_scale = _potential(elements,changed_states,proposed,target,nodal)
                    decision = acceptance(energy,new_energy,slope,.5**cut,max(energy_scale,new_scale),norm,new_norm)
                    # An already equilibrated trial meets the actual stopping
                    # test. Do not reject it on a rounded energy difference;
                    # the unchanged chart AND physical residual tests still
                    # run again before staging/commit on the next iteration.
                    if new_norm <= threshold or decision.startswith('ACCEPT_'):
                        candidate = proposed; break
                else: raise RuntimeError('native force line search exhausted')
            if not accepted: raise RuntimeError('native force iteration bound exhausted')
    except SolveCancelled as error:
        status,failure = 'cancelled',str(error)
    except Exception as error:
        status,failure = 'failed',type(error).__name__+': '+str(error)
    finally:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())
    return ForceProgramResult(status,cursor,parameter,_readonly(total),_readonly(reaction),capsule,failure)
