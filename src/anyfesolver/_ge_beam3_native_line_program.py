"""Private proportional reference-line force control through real global Newton.

Only authenticated typed line-chain restart is admitted. No staged loading,
public routing or dynamic authority is granted.
The frozen trial scope owns the effective load at every assembly call.
"""
from contextvars import ContextVar
from dataclasses import dataclass, field
import numpy as np
from ._ge_beam3_native_line_loading import LinePattern, _Scope, _ACTIVE, nodal_force_vector
from ._ge_beam3_p5_seeded.core import sha

_PROGRAM = ContextVar('ge_beam3_native_line_program', default=None)


def model_identity(model):
    from ._ge_beam3_native_line_static_element import NativeLineFibreStaticElement
    elements = tuple(sorted(model.mesh.elements.items()))
    n = model.mesh.dof_manager.total_dofs
    if not 1 <= len(elements) <= 16 or n != 6 * len(model.mesh.nodes) or n > 512:
        raise ValueError('bounded standalone native line model required')
    if model.constraint_equations or model.mesh.point_masses or model.mesh.element_activity is not None:
        raise ValueError('native line MPC/activity/dynamics not admitted')
    for eid, element in elements:
        if type(element) is not NativeLineFibreStaticElement or eid != element.element_id or model.materials.get(element.material_name) is not element.section:
            raise ValueError('exact native line elements and sections required')
        element._check(model.mesh)
    if set(model.mesh.nodes) != {i for _, e in elements for i in e.node_ids}:
        raise ValueError('unconnected native line nodes')
    fixed = set()
    for bc in model.boundary_conditions:
        for dof, value in bc.get_constrained_dofs(model.mesh.dof_manager):
            if value != 0.: raise ValueError('homogeneous native line supports required')
            fixed.add(int(dof))
    if not fixed: raise ValueError('supported native line model required')
    for node in model.mesh.nodes:
        if len(set(model.mesh.dof_manager.get_node_dofs(node)[3:]) & fixed) not in (0, 3):
            raise ValueError('partial rotation support not admitted')
    return sha(dict(elements=[(i,e.to_dict()) for i,e in elements],
        nodes=[(i,node.coords(),list(model.mesh.dof_manager.get_node_dofs(i))) for i,node in sorted(model.mesh.nodes.items())],
        boundaries=[vars(b) for b in model.boundary_conditions], fixed=sorted(fixed)))


@dataclass(frozen=True)
class _Program:
    model: object
    identity: str
    proportional: LinePattern
    constant: LinePattern
    events: list
    input_sha256: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'input_sha256', sha((self.identity,self.proportional.signature,self.constant.signature)))

    def require(self, model):
        if model is not self.model or model_identity(model) != self.identity:
            raise ValueError('native line program model authority changed')
        self.proportional.require(model.mesh); self.constant.require(model.mesh)
        if sha((self.identity,self.proportional.signature,self.constant.signature)) != self.input_sha256:
            raise ValueError('native line program frozen load authority changed')

    def effective(self, parameter):
        if not np.isfinite(parameter) or not 0. <= parameter <= 1.:
            raise ValueError('native line force parameter outside registered path')
        rows = []
        for eid in sorted(self.model.mesh.elements):
            force = self.constant.force(eid) + float(parameter) * self.proportional.force(eid)
            if np.any(force): rows.append((eid, *map(float, force)))
        return LinePattern(tuple(rows))


def assemble_at(parameter, model, displacements, store, num_layers, **kwargs):
    from .nonlinear_static import _assemble_nonlinear_system
    program = _PROGRAM.get()
    if type(program) is not _Program or _ACTIVE.get() is not None or num_layers != 1:
        raise ValueError('live unnested native line force program required')
    program.require(model)
    pattern = program.effective(parameter)
    expected_pattern_sha256 = pattern.signature
    scope = _Scope(model.mesh, store, pattern, tuple((i,e.identity) for i,e in sorted(model.mesh.elements.items())))
    token = _ACTIVE.set(scope)
    try:
        result = _assemble_nonlinear_system(model, displacements, store, num_layers, **kwargs)
        program.require(model); pattern.require(model.mesh)
        if pattern.signature != expected_pattern_sha256:
            raise ValueError('native line effective load authority changed')
        program.events.append(dict(parameter=float(parameter), pattern_sha256=pattern.signature,
            tangent=bool(kwargs.get('tangent', True)), reaction=bool(kwargs.get('require_full_coordinates', False))))
        return result
    except BaseException:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())
        raise
    finally:
        _ACTIVE.reset(token)


def solve_line_static(model, proportional, *, constant=None, steps=2, max_iterations=12, line_search=False,
                      initial_checkpoint=None, expected_sha256=None):
    from .boundary import LoadCase
    from .nonlinear_static import solve_static_nonlinear, NonlinearConvergenceSettings
    if _PROGRAM.get() is not None or _ACTIVE.get() is not None:
        raise ValueError('nested native line program forbidden')
    if type(proportional) is not LinePattern or (constant is not None and type(constant) is not LinePattern):
        raise ValueError('exact native line patterns required')
    if type(steps) is not int or not 1 <= steps <= 16 or type(max_iterations) is not int or not 1 <= max_iterations <= 24 or type(line_search) is not bool:
        raise ValueError('bounded native line controls required')
    proportional.require(model.mesh)
    initial=None
    if initial_checkpoint is not None:
        from ._ge_beam3_native_spatial_couples import active_for,decode_active_restart
        if active_for(model):
            chain=decode_active_restart(model,initial_checkpoint,expected_sha256=expected_sha256)
            point=chain[-1]['load_point'].line
        else:
            from ._ge_beam3_native_line_restart import decode_checkpoint
            chain=decode_checkpoint(model,initial_checkpoint,expected_sha256=expected_sha256)
            point=chain[-1]['load_point']
        initial=chain[-1]
        accepted=point.effective(model)
        if constant is None: constant=accepted
        elif constant!=accepted: raise ValueError('native line restart constant must match accepted load')
    elif expected_sha256 is not None:
        raise ValueError('native line restart hash without checkpoint')
    constant = LinePattern(()) if constant is None else constant
    constant.require(model.mesh)
    program = _Program(model, model_identity(model), LinePattern(proportional.rows), LinePattern(constant.rows), [])

    def load(pattern):
        vector = nodal_force_vector(model, pattern); value = LoadCase('private-native-reference-line')
        for node in sorted(model.mesh.nodes):
            mapping = list(model.mesh.dof_manager.get_node_dofs(node))
            if np.any(vector[mapping[3:]]): raise ValueError('reference line nodal moments not admitted')
            value.add_nodal_load(node, forces=vector[mapping[:3]])
        return value

    prop, const = load(program.proportional), load(program.constant)
    token = _PROGRAM.set(program)
    try:
        result = solve_static_nonlinear(model, prop, constant_load_case=const, num_steps=steps,
            max_iterations=max_iterations, tolerance=1e-12, num_layers=1, min_step_fraction=1.,
            record_increment_snapshots=True, equilibrate_initial_state=False,
            initial_displacements=None if initial is None else initial['displacements'],
            initial_element_states=None if initial is None else initial['states'],
            convergence_settings=NonlinearConvergenceSettings(profile='legacy', line_search='always' if line_search else 'never',
                max_step_factor=1., max_line_search_cuts=8))
        program.require(model); proportional.require(model.mesh); constant.require(model.mesh)
        return result, tuple(dict(event) for event in program.events)
    finally:
        _PROGRAM.reset(token)
