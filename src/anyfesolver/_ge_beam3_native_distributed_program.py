"""Private distributed-couple force control through the actual native solver.

Restart accepts only the externally authenticated, complete distributed-load
chain. Raw initial states and cross-formulation checkpoints are not admitted.
"""
from contextvars import ContextVar
from dataclasses import dataclass,field
import numpy as np
from ._ge_beam3_native_line_loading import LinePattern,nodal_force_vector
from ._ge_beam3_native_distributed_loading import DistributedPattern,_Scope,_ACTIVE
from ._ge_beam3_p5_seeded.core import sha

_PROGRAM=ContextVar('ge_beam3_native_distributed_program',default=None)


def model_identity(model):
    from ._ge_beam3_native_distributed_element import NativeDistributedFibreStaticElement
    elements=tuple(sorted(model.mesh.elements.items()));n=model.mesh.dof_manager.total_dofs
    if not 1<=len(elements)<=16 or n!=6*len(model.mesh.nodes) or n>512:raise ValueError('bounded standalone distributed model required')
    if model.constraint_equations or model.mesh.point_masses or model.mesh.element_activity is not None:
        raise ValueError('distributed MPC/activity/dynamics not admitted')
    for eid,e in elements:
        if type(e) is not NativeDistributedFibreStaticElement or eid!=e.element_id or model.materials.get(e.material_name) is not e.section:
            raise ValueError('exact distributed element and section authority required')
        e._check(model.mesh)
    if set(model.mesh.nodes)!={i for _,e in elements for i in e.node_ids}:raise ValueError('unconnected distributed nodes')
    fixed=set()
    for bc in model.boundary_conditions:
        for dof,value in bc.get_constrained_dofs(model.mesh.dof_manager):
            if value!=0.:raise ValueError('homogeneous distributed supports required')
            fixed.add(int(dof))
    if not fixed:raise ValueError('supported distributed model required')
    for node in model.mesh.nodes:
        if len(set(model.mesh.dof_manager.get_node_dofs(node)[3:])&fixed) not in (0,3):
            raise ValueError('partial distributed rotation support not admitted')
    return sha(dict(elements=[(i,e.to_dict()) for i,e in elements],
        nodes=[(i,node.coords(),list(model.mesh.dof_manager.get_node_dofs(i))) for i,node in sorted(model.mesh.nodes.items())],
        boundaries=[vars(b) for b in model.boundary_conditions],fixed=sorted(fixed)))


@dataclass(frozen=True)
class _Program:
    model: object
    identity: str
    proportional: DistributedPattern
    constant: DistributedPattern
    events: list
    initial: object=None
    restart_sha256: object=None
    input_sha256: str=field(init=False)

    def __post_init__(self):
        if (self.initial is None)!=(self.restart_sha256 is None):raise ValueError('distributed programme restart binding')
        object.__setattr__(self,'input_sha256',self._input_identity())

    def _input_identity(self):
        values=(self.identity,self.proportional.signature,self.constant.signature)
        if self.initial is not None:values+= (self.restart_sha256,sha(self.initial))
        return sha(values)

    def require(self,model):
        if model is not self.model or model_identity(model)!=self.identity:raise ValueError('distributed programme model changed')
        self.proportional.require(model.mesh);self.constant.require(model.mesh)
        if self._input_identity()!=self.input_sha256:
            raise ValueError('distributed programme load authority changed')

    def effective(self,parameter):
        if not np.isfinite(parameter) or not 0.<=parameter<=1.:raise ValueError('bounded distributed load parameter')
        lines=[];couples=[]
        for eid in sorted(self.model.mesh.elements):
            with np.errstate(over='ignore',invalid='ignore'):
                force=self.constant.force(eid)+float(parameter)*self.proportional.force(eid)
                density=self.constant.density(eid)+float(parameter)*self.proportional.density(eid)
            if np.any(force):lines.append((eid,*map(float,force)))
            if np.any(density):couples.append((eid,*map(float,density)))
        return DistributedPattern(LinePattern(tuple(lines)),tuple(couples))


def require_active(model):
    programme=_PROGRAM.get()
    if type(programme) is not _Program:raise ValueError('live native distributed force programme required')
    programme.require(model)
    return programme


def require_solver_initial(model,states,displacements):
    from ._ge_beam3_p5_seeded.core import canonical
    programme=require_active(model)
    if programme.initial is None:
        if states is not None or displacements is not None:raise ValueError('distributed initial state requires an authenticated chain')
    elif (canonical(states)!=canonical(programme.initial['states'])
          or not np.array_equal(displacements,programme.initial['displacements'])):
        raise ValueError('distributed solver initial data differ from authenticated chain')


def assemble_at(parameter,model,displacements,store,num_layers,**kwargs):
    from .nonlinear_static import _assemble_nonlinear_system
    programme=require_active(model)
    if _ACTIVE.get() is not None or num_layers!=1:raise ValueError('unnested distributed trial programme required')
    pattern=programme.effective(parameter);signature=pattern.signature
    scope=_Scope(model.mesh,store,pattern,tuple((i,e.identity) for i,e in sorted(model.mesh.elements.items())))
    token=_ACTIVE.set(scope)
    try:
        result=_assemble_nonlinear_system(model,displacements,store,num_layers,**kwargs)
        programme.require(model);pattern.require(model.mesh)
        if pattern.signature!=signature:raise ValueError('distributed effective load changed')
        programme.events.append(dict(parameter=float(parameter),pattern_sha256=signature,
            tangent=bool(kwargs.get('tangent',True)),reaction=bool(kwargs.get('require_full_coordinates',False))))
        return result
    except BaseException:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
        raise
    finally:_ACTIVE.reset(token)


def solve_distributed_model(model,proportional,*,constant=None,steps=2,max_iterations=24,line_search=True,
                            initial_checkpoint=None,expected_sha256=None):
    from .boundary import LoadCase
    from .nonlinear_static import solve_static_nonlinear,NonlinearConvergenceSettings
    if _PROGRAM.get() is not None or _ACTIVE.get() is not None:raise ValueError('nested native distributed programme forbidden')
    if type(proportional) is not DistributedPattern or (constant is not None and type(constant) is not DistributedPattern):
        raise ValueError('exact distributed patterns required')
    if type(steps) is not int or not 1<=steps<=16 or type(max_iterations) is not int or not 1<=max_iterations<=24 or type(line_search) is not bool:
        raise ValueError('bounded native distributed controls required')
    initial=None
    if initial_checkpoint is not None:
        from ._ge_beam3_native_combined_couples import active_for,decode_active_restart
        if active_for(model):
            chain=decode_active_restart(model,initial_checkpoint,expected_sha256=expected_sha256)
            point=chain[-1]['load_point'].distributed
        else:
            from ._ge_beam3_native_distributed_restart import decode_checkpoint
            chain=decode_checkpoint(model,initial_checkpoint,expected_sha256=expected_sha256)
            point=chain[-1]['load_point']
        initial=chain[-1];accepted=point.effective(model)
        if constant is None:constant=accepted
        elif constant!=accepted:raise ValueError('distributed restart constant must match accepted load')
    elif expected_sha256 is not None:raise ValueError('distributed restart hash without checkpoint')
    constant=DistributedPattern(LinePattern(()),()) if constant is None else constant
    proportional.require(model.mesh);constant.require(model.mesh)
    programme=_Program(model,model_identity(model),DistributedPattern(LinePattern(proportional.line.rows),proportional.couples),
        DistributedPattern(LinePattern(constant.line.rows),constant.couples),[],initial,expected_sha256)
    def load(pattern):
        vector=nodal_force_vector(model,pattern.line);value=LoadCase('private-native-distributed-line')
        for node in sorted(model.mesh.nodes):
            dofs=list(model.mesh.dof_manager.get_node_dofs(node))
            if np.any(vector[dofs[3:]]):raise ValueError('distributed nodal line moment not admitted')
            value.add_nodal_load(node,forces=vector[dofs[:3]])
        return value
    prop,const=load(programme.proportional),load(programme.constant)
    token=_PROGRAM.set(programme)
    try:
        result=solve_static_nonlinear(model,prop,constant_load_case=const,num_steps=steps,
            max_iterations=max_iterations,tolerance=1e-12,num_layers=1,min_step_fraction=1.,
            record_increment_snapshots=True,equilibrate_initial_state=False,
            initial_displacements=None if initial is None else initial['displacements'],
            initial_element_states=None if initial is None else initial['states'],
            convergence_settings=NonlinearConvergenceSettings(profile='legacy',line_search='always' if line_search else 'never',
                max_step_factor=1.,max_line_search_cuts=8))
        programme.require(model);proportional.require(model.mesh);constant.require(model.mesh)
        evidence=dict(programme_sha256=programme.input_sha256,model_sha256=programme.identity,
            events=tuple(programme.events),general_matrix_required=True,production_qualified=False,restart_authorized=initial is not None)
        if initial is not None:evidence['restart_sha256']=expected_sha256
        return result,evidence
    finally:_PROGRAM.reset(token)
