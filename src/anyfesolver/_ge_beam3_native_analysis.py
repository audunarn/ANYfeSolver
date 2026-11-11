"""Model-owned native beam dispatch, with distinct solver and state owners.

Private integration candidate. This is not a replacement for the earlier
straight ge-beam3 facade, a static-mass shortcut, or production qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from threading import Lock

import numpy as np

from .boundary import BoundaryCondition, LoadCase
from .control import cancellation_safe_point
from .fe_core import FEModel
from ._ge_beam3_native_definition import NativeBeamDefinition, _describe
from ._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from ._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
from ._ge_beam3_native_generalized_loading import DistributedPattern
from ._ge_beam3_native_line_loading import LinePattern
from ._ge_beam3_p5_seeded.core import canonical

SCHEMA = 'GE_BEAM3_MODEL_OWNED_ANALYSIS_CHECKPOINT_V1'
FORCE_WORKFLOW_SCHEMA = 'GE_BEAM3_MODEL_OWNED_FORCE_WORKFLOW_CHECKPOINT_V1'
COMBINED_WORKFLOW = 'GENERALIZED_COMBINED_SPATIAL_COUPLES_V1'
MAX_CHECKPOINT_BYTES = 64*1024**2


class NativeBeamAnalysisError(ValueError):
    pass


class NativeBeamWorkflowError(NativeBeamAnalysisError):
    pass


def _require(ok, message):
    if not ok:
        raise NativeBeamAnalysisError(message)


def _controls(steps, iterations):
    _require(type(steps) is int and 1 <= steps <= 16
             and type(iterations) is int and 1 <= iterations <= 24,
             'explicit bounded native force controls required')


@dataclass(frozen=True)
class NativeBeamRun:
    status: str
    checkpoint: bytes
    backend_result: object
    production_qualified: bool = field(default=False, init=False)

    @property
    def checkpoint_sha256(self):
        return sha256(self.checkpoint).hexdigest()


class NativeBeamAnalysis:
    """An owned FEModel constructed from exact native beam definitions.

Current drivers admit one section family per model. Different section data
within that family are permitted; cross-family or beam-shell assembly awaits
its own integration. Model mutation or concurrent use fails closed.
"""

    def __init__(self, definitions, boundaries, *, retained_refinement=False):
        _require(type(retained_refinement) is bool, 'explicit retained refinement flag required')
        self._retained_refinement = retained_refinement
        limit = 32 if retained_refinement else 16
        _require(type(definitions) is tuple and 1 <= len(definitions) <= limit
                 and all(type(d) is NativeBeamDefinition for d in definitions),
                 'bounded explicit native definitions required')
        _require(type(boundaries) is tuple and all(type(b) is BoundaryCondition for b in boundaries),
                 'explicit boundary-condition tuple required')
        elements = []; inertias = {}; nodes = {}; ids = []
        for definition in definitions:
            element, inertia = definition.instantiate()
            ids.append(element.element_id); elements.append(element); inertias[element.element_id] = inertia
            for node, coordinates in zip(element.node_ids, element.operator.reference.coordinates):
                if node in nodes:
                    _require(np.array_equal(nodes[node], coordinates), 'shared node reference coordinates disagree')
                nodes[node] = coordinates.copy()
        _require(ids == sorted(set(ids)), 'definitions must have unique ordered element IDs')
        _require(len({type(e) for e in elements}) == 1,
                 'mixed native section families require a separately integrated driver')
        self.model = FEModel('native-beam-analysis')
        for node, coordinates in sorted(nodes.items()):
            self.model.add_node(node, *coordinates)
        for element in elements:
            self.model.add_element(element.element_id, element)
            self.model.materials[element.material_name] = element.section
        for boundary in boundaries:
            self.model.add_boundary_condition(deepcopy(boundary))
        self._admit_boundaries()
        self._elements = tuple(elements)
        self._definitions = tuple(d.raw for d in definitions)
        self._inertias = inertias
        self._family = 'GENERALIZED_DISTRIBUTED' if type(elements[0]) is NativeGeneralizedStaticElement else 'PHYSICAL_FIBRE_NODAL'
        self._lock = Lock()
        self._structure = self._snapshot()
        self.identity = self._graph_identity()
        self._identity = self.identity
        self._guard()

    def _admit_boundaries(self):
        names = ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')
        fixed = set()
        for boundary in self.model.boundary_conditions:
            _require(type(boundary) is BoundaryCondition and type(boundary.node_ids) is list
                     and boundary.node_ids and all(type(i) is int and i in self.model.mesh.nodes for i in boundary.node_ids)
                     and len(set(boundary.node_ids)) == len(boundary.node_ids), 'explicit existing support nodes required')
            values = boundary.dof_constraints
            _require(type(values) is dict and values and set(values) <= set(names)
                     and all(type(v) in (float, int) and v == 0 for v in values.values()),
                     'homogeneous native support components required')
            rebuilt = BoundaryCondition(boundary.name, list(boundary.node_ids), dict(values))
            _require(canonical(vars(rebuilt)) == canonical(vars(boundary)), 'support declaration/cache mismatch')
            fixed.update(int(dof) for dof, _ in boundary.get_constrained_dofs(self.model.mesh.dof_manager))
        for node in self.model.mesh.nodes:
            rotations = set(self.model.mesh.dof_manager.get_node_dofs(node)[3:])
            _require(len(rotations & fixed) in (0, 3), 'partial rotational supports need a separate chart contract')
        return tuple(sorted(fixed))

    def _snapshot(self):
        m = self.model
        _require(not m.constraint_equations and m.mesh.element_activity is None and not m.mesh.point_masses,
                 'MPC/activity/point-mass integration is not authorized by this driver')
        return canonical(dict(nodes=[(i, n.coords(), list(m.mesh.dof_manager.get_node_dofs(i)))
                                     for i, n in sorted(m.mesh.nodes.items())],
                              boundaries=[vars(b) for b in m.boundary_conditions],
                              dofs=m.mesh.dof_manager.total_dofs,
                              materials=sorted(m.materials), elements=sorted(m.mesh.elements)))

    def _graph_identity(self):
        _require(type(self._retained_refinement) is bool, 'retained refinement flag changed')
        data = dict(definitions=[sha256(b).hexdigest() for b in self._definitions],
                    structure=self._structure.decode('ascii'), owner=self._family)
        if self._retained_refinement:
            data['retained_refinement'] = 'GE_BEAM3_N32_RETAINED_ANALYSIS_V1'
        return sha256(canonical(data)).hexdigest()

    def _guard(self):
        graph = self._graph_identity()
        _require(self.identity == self._identity == graph and self._snapshot() == self._structure,
                 'native beam analysis model changed')
        _require(tuple(self.model.mesh.elements[i] for i in sorted(self.model.mesh.elements)) == self._elements,
                 'native beam element ownership changed')
        for element, raw in zip(self._elements, self._definitions):
            _require(self.model.materials.get(element.material_name) is element.section,
                     'native beam section ownership changed')
            _require(canonical(_describe(element, self._inertias[element.element_id])) == raw,
                     'native beam definition/inertia changed')
            element._check(self.model.mesh)

    @contextmanager
    def _operation(self):
        _require(self._lock.acquire(blocking=False), 'native beam analysis is already in use')
        try:
            self._guard()
            yield
            self._guard()
        finally:
            self._lock.release()

    def _family_required(self, family):
        if self._family != family:
            raise NativeBeamWorkflowError('requested workflow has a different native section/state owner')

    def _initial(self, cancellation_token=None):
        states = {}
        for e in self._elements:
            cancellation_safe_point(cancellation_token, 'native-force.initialize-element')
            states[e.element_id] = e.init_model_bound_nonlinear_state(self.model.mesh, e.section, 1)
        cancellation_safe_point(cancellation_token, 'native-force.initialized')
        return states

    def _force_workflow_schema(self, workflow):
        if workflow is None:
            return SCHEMA
        _require(type(workflow) is str and workflow == COMBINED_WORKFLOW, 'unknown native force workflow')
        self._family_required('GENERALIZED_DISTRIBUTED')
        return FORCE_WORKFLOW_SCHEMA

    def _envelope(self, backend, *, workflow=None):
        self._guard()
        schema = self._force_workflow_schema(workflow)
        data = dict(schema=schema, definition_graph_sha256=self.identity,
                             owner=self._family, backend=backend.decode('ascii'),
                             backend_sha256=sha256(backend).hexdigest(), production_qualified=False)
        if workflow is not None:
            data['workflow'] = workflow
        raw = canonical(data)
        _require(len(raw) <= MAX_CHECKPOINT_BYTES, 'native analysis checkpoint byte bound')
        return raw

    def _backend(self, raw, expected, *, workflow=None):
        schema = self._force_workflow_schema(workflow)
        _require(type(raw) is bytes and 0 < len(raw) <= MAX_CHECKPOINT_BYTES
                 and type(expected) is str and sha256(raw).hexdigest() == expected,
                 'external analysis checkpoint authority mismatch')
        def pairs(rows):
            result = {}
            for key, value in rows:
                _require(key not in result, 'duplicate analysis checkpoint key')
                result[key] = value
            return result
        def forbidden(value):
            raise NativeBeamAnalysisError('nonfinite analysis checkpoint')
        try:
            data = json.loads(raw.decode('ascii'), object_pairs_hook=pairs, parse_constant=forbidden)
        except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
            raise NativeBeamAnalysisError('invalid analysis checkpoint JSON') from error
        keys = {'schema', 'definition_graph_sha256', 'owner', 'backend', 'backend_sha256', 'production_qualified'}
        if workflow is not None:
            keys.add('workflow')
        _require(type(data) is dict and set(data) == keys and canonical(data) == raw,
                 'canonical complete analysis checkpoint required')
        _require(data['schema'] == schema and data.get('workflow') == workflow
                 and data['definition_graph_sha256'] == self.identity
                 and data['owner'] == self._family and data['production_qualified'] is False
                 and type(data['backend']) is str, 'analysis definition/state owner mismatch')
        backend = data['backend'].encode('ascii')
        _require(sha256(backend).hexdigest() == data['backend_sha256'], 'backend checkpoint hash mismatch')
        return backend, data['backend_sha256']

    def solve_distributed(self, proportional, *, steps=2, max_iterations=24,
                          checkpoint=None, expected_sha256=None,
                          cancellation_token=None, progress_callback=None):
        """Generalized native Newton; resume changes loads, never state owners."""
        from ._ge_beam3_native_generalized_program import solve_distributed_model
        from ._ge_beam3_native_generalized_restart import LoadPoint, encode_checkpoint, decode_checkpoint
        cancellation_safe_point(cancellation_token, 'native-force.capture')
        _require(progress_callback is None or callable(progress_callback), 'callable force progress observer required')
        _controls(steps, max_iterations)
        self._family_required('GENERALIZED_DISTRIBUTED')
        _require(type(proportional) is DistributedPattern, 'exact distributed pattern required')
        _require((checkpoint is None) == (expected_sha256 is None), 'checkpoint/hash pair required')
        with self._operation():
            _require(bool(self._admit_boundaries()), 'supported force model required before state initialization')
            proportional.require(self.model.mesh)
            constant = DistributedPattern(LinePattern(()), ())
            backend = digest = None
            if checkpoint is None:
                chain = (dict(load_point=LoadPoint(0., constant, proportional),
                              displacements=np.zeros(self.model.mesh.dof_manager.total_dofs), states=self._initial(cancellation_token)),)
            else:
                backend, digest = self._backend(checkpoint, expected_sha256)
                chain = decode_checkpoint(self.model, backend, expected_sha256=digest)
                constant = chain[-1]['load_point'].effective(self.model)
            cancellation_safe_point(cancellation_token, 'native-force.captured')
            result, _ = solve_distributed_model(self.model, proportional, steps=steps,
                max_iterations=max_iterations, initial_checkpoint=backend, expected_sha256=digest,
                cancellation_token=cancellation_token, progress_callback=progress_callback)
            chain += tuple(dict(load_point=LoadPoint(float(s.load_factor), constant, proportional),
                                displacements=s.displacements, states=s.element_states) for s in result.snapshots)
            self._guard()
            cancellation_safe_point(cancellation_token, 'native-force.checkpoint')
            raw = self._envelope(encode_checkpoint(self.model, chain))
            cancellation_safe_point(cancellation_token, 'native-force.complete')
            return NativeBeamRun(result.status, raw, result)

    def solve_nodal(self, forces, *, target_factor=1., steps=2, max_iterations=24,
                    checkpoint=None, expected_sha256=None,
                    cancellation_token=None, progress_callback=None):
        """Physical-fibre native Newton with a fixed spatial dead-force pattern."""
        from .nonlinear_static import solve_static_nonlinear
        from ._ge_beam3_native_fibre_restart import _forces, encode_checkpoint, decode_checkpoint
        cancellation_safe_point(cancellation_token, 'native-force.capture')
        _require(progress_callback is None or callable(progress_callback), 'callable force progress observer required')
        _controls(steps, max_iterations)
        self._family_required('PHYSICAL_FIBRE_NODAL')
        _require(type(target_factor) is float and isfinite(target_factor), 'explicit finite load factor required')
        _require((checkpoint is None) == (expected_sha256 is None), 'checkpoint/hash pair required')
        with self._operation():
            _require(bool(self._admit_boundaries()), 'supported force model required before state initialization')
            _forces(forces, tuple(sorted(self.model.mesh.nodes)), self.model.mesh.dof_manager.total_dofs)
            if checkpoint is None:
                chain = (dict(load_factor=0., displacements=np.zeros(self.model.mesh.dof_manager.total_dofs),
                              states=self._initial(cancellation_token)),)
                initial = None
            else:
                backend, digest = self._backend(checkpoint, expected_sha256)
                original, chain = decode_checkpoint(self.model, backend, expected_sha256=digest)
                _require(canonical(original) == canonical(forces), 'restart force pattern changed')
                initial = chain[-1]
            cancellation_safe_point(cancellation_token, 'native-force.captured')
            accepted = chain[-1]['load_factor']
            def load(factor):
                value = LoadCase('native-beam-owned-dead-force')
                for node, *vector in forces:
                    value.add_nodal_load(node, forces=factor*np.array(vector))
                return value
            result = solve_static_nonlinear(self.model, load(target_factor-accepted),
                constant_load_case=load(accepted), num_steps=steps, max_iterations=max_iterations,
                tolerance=1e-12, num_layers=1, min_step_fraction=1., record_increment_snapshots=True,
                equilibrate_initial_state=False,
                initial_element_states=None if initial is None else initial['states'],
                initial_displacements=None if initial is None else initial['displacements'],
                cancellation_token=cancellation_token, progress_callback=progress_callback)
            chain += tuple(dict(load_factor=float(accepted+s.load_factor*(target_factor-accepted)),
                                displacements=s.displacements, states=s.element_states) for s in result.snapshots)
            self._guard()
            cancellation_safe_point(cancellation_token, 'native-force.checkpoint')
            raw = self._envelope(encode_checkpoint(self.model, forces, chain))
            cancellation_safe_point(cancellation_token, 'native-force.complete')
            return NativeBeamRun(result.status, raw, result)

    def _decode(self, checkpoint, expected):
        backend, digest = self._backend(checkpoint, expected)
        if self._family == 'GENERALIZED_DISTRIBUTED':
            from ._ge_beam3_native_generalized_restart import decode_checkpoint
            return decode_checkpoint(self.model, backend, expected_sha256=digest)
        from ._ge_beam3_native_fibre_restart import decode_checkpoint
        return decode_checkpoint(self.model, backend, expected_sha256=digest)[1]

    def recover(self, checkpoint, *, expected_sha256):
        """Recover through the matching native owner from an authenticated chain."""
        with self._operation():
            state = self._decode(checkpoint, expected_sha256)[-1]
            if self._family == 'GENERALIZED_DISTRIBUTED':
                from ._ge_beam3_native_generalized_recovery import recover_native_fields
            else:
                from ._ge_beam3_native_fibre_recovery import recover_native_fields
            before = canonical(state)
            rows = {}
            for element in self._elements:
                mapping = list(element.get_dof_mapping(self.model.mesh))
                rows[element.element_id] = recover_native_fields(element, self.model.mesh,
                    state['states'][element.element_id], expected_committed_total_u=state['displacements'][mapping])
            _require(canonical(state) == before, 'native analysis recovery changed history')
            return rows

    def checkpoint_prefix(self, checkpoint, accepted_steps, *, expected_sha256):
        """Issue a validated prefix without rerunning or inventing a state."""
        _require(type(accepted_steps) is int and accepted_steps >= 0, 'explicit prefix step count required')
        with self._operation():
            backend, digest = self._backend(checkpoint, expected_sha256)
            if self._family == 'GENERALIZED_DISTRIBUTED':
                from ._ge_beam3_native_generalized_restart import encode_checkpoint, decode_checkpoint
                chain = decode_checkpoint(self.model, backend, expected_sha256=digest)
                _require(accepted_steps < len(chain), 'prefix exceeds accepted chain')
                raw = encode_checkpoint(self.model, chain[:accepted_steps+1])
            else:
                from ._ge_beam3_native_fibre_restart import encode_checkpoint, decode_checkpoint
                forces, chain = decode_checkpoint(self.model, backend, expected_sha256=digest)
                _require(accepted_steps < len(chain), 'prefix exceeds accepted chain')
                raw = encode_checkpoint(self.model, forces, chain[:accepted_steps+1])
            return self._envelope(raw)

    def reference_modes(self, *, num_modes=6, cancellation_token=None, bounds=None):
        """Distinct retained physical-inertia pencil; never a static mass matrix."""
        cancellation_safe_point(cancellation_token, 'native-reference-modal.capture')
        if self._family == 'PHYSICAL_FIBRE_NODAL':
            if bounds is None:
                raise NativeBeamWorkflowError('physical-fibre reference modes require explicit spectral bounds')
            from ._ge_beam3_fibre_reference_modal import solve
            return solve(self,bounds=bounds,num_modes=num_modes,cancellation_token=cancellation_token)
        from ._ge_beam3_native_generalized_modal import solve_modes
        self._family_required('GENERALIZED_DISTRIBUTED')
        _require(type(num_modes) is int and num_modes > 0, 'positive modal count required')
        options = {}
        if bounds is not None:
            from ._ge_beam3_analysis_translation_modal import _controls
            from ._ge_beam3_native_generalized_paired_modal import solve_modes
            _controls(self, bounds, num_modes, 1e-10, 1e-12)
            options['bounds'] = bounds
        with self._operation():
            zeros = np.zeros(self.model.mesh.dof_manager.total_dofs)
            return solve_modes(self.model, self._initial(cancellation_token), zeros, self._inertias, zeros,
                               num_modes=num_modes, cancellation_token=cancellation_token, **options)

    def solve_nodal_program(self, program, **kwargs):
        """Generalized dead forces: resume the same frozen absolute schedule."""
        from ._ge_beam3_analysis_nodal_program import solve
        return solve(self, program, **kwargs)

    def import_nodal_program_checkpoint(self, program, backend, *, expected_sha256):
        from ._ge_beam3_analysis_nodal_program import adopt
        return adopt(self, program, backend, expected_sha256=expected_sha256)

    def recover_nodal_program(self, program, checkpoint, *, expected_sha256):
        from ._ge_beam3_analysis_nodal_program import recover
        return recover(self, program, checkpoint, expected_sha256=expected_sha256)

    def nodal_program_checkpoint_prefix(self, program, checkpoint, accepted_steps, *, expected_sha256):
        from ._ge_beam3_analysis_nodal_program import prefix
        return prefix(self, program, checkpoint, accepted_steps, expected_sha256=expected_sha256)

    def nodal_program_modes(self, program, checkpoint, *, expected_sha256, **kwargs):
        from ._ge_beam3_analysis_nodal_program import modes
        return modes(self, program, checkpoint, expected_sha256=expected_sha256, **kwargs)

    def _translation_route(self):
        if self._family == 'PHYSICAL_FIBRE_NODAL':
            from . import _ge_beam3_analysis_fibre_translation as route
        else:
            from . import _ge_beam3_analysis_translation as route
        return route

    def solve_translation(self, program, **kwargs):
        """Actual retained displacement control; distinct from force checkpoints."""
        return self._translation_route().solve(self, program, **kwargs)

    def import_translation_checkpoint(self, program, backend, *, expected_sha256, **kwargs):
        """Adopt only after native mechanical replay of the exact external bytes."""
        return self._translation_route().adopt(self, program, backend, expected_sha256=expected_sha256, **kwargs)

    def recover_translation(self, program, checkpoint, *, expected_sha256, **kwargs):
        return self._translation_route().recover(self, program, checkpoint, expected_sha256=expected_sha256, **kwargs)

    def translation_checkpoint_prefix(self, program, checkpoint, accepted_steps, *, expected_sha256, **kwargs):
        return self._translation_route().prefix(self, program, checkpoint, accepted_steps, expected_sha256=expected_sha256, **kwargs)

    def current_modes(self, checkpoint, *, expected_sha256, num_modes=6, cancellation_token=None, bounds=None):
        """Only the backend's conservative elastic-interior current-rest scope."""
        from .control import cancellation_safe_point
        cancellation_safe_point(cancellation_token,'native-force-current-modal.capture')
        if self._family == 'PHYSICAL_FIBRE_NODAL':
            if bounds is None:
                raise NativeBeamWorkflowError('physical-fibre force-state modes require explicit spectral bounds')
            from ._ge_beam3_native_fibre_current_modal import solve
            return solve(self,checkpoint,expected_sha256=expected_sha256,bounds=bounds,
                num_modes=num_modes,cancellation_token=cancellation_token)
        from ._ge_beam3_native_generalized_modal import solve_modes
        self._family_required('GENERALIZED_DISTRIBUTED')
        _require(type(num_modes) is int and num_modes > 0, 'positive modal count required')
        options={}
        if bounds is not None:
            from ._ge_beam3_analysis_translation_modal import _controls
            from ._ge_beam3_native_generalized_paired_modal import solve_modes
            _controls(self,bounds,num_modes,1e-10,1e-12)
            options['bounds']=bounds
        with self._operation():
            state = self._decode(checkpoint, expected_sha256)[-1]
            # Captured native operators already subtract the complete line
            # load gradient, including nodal work. This owner has no additional
            # independent nodal force pattern; subtracting the line again
            # incorrectly rejects an actual conservative equilibrium.
            external = np.zeros(self.model.mesh.dof_manager.total_dofs)
            return solve_modes(self.model, state['states'], state['displacements'], self._inertias, external,
                               num_modes=num_modes, cancellation_token=cancellation_token,**options)

    def translation_modes(self, program, checkpoint, *, expected_sha256, **kwargs):
        """Conservative current-rest modes; preserve the native continuation owner."""
        if self._family == 'PHYSICAL_FIBRE_NODAL':
            from ._ge_beam3_analysis_fibre_translation import modes
            return modes(self, program, checkpoint, expected_sha256=expected_sha256, **kwargs)
        from ._ge_beam3_analysis_translation_modal import solve
        return solve(self, program, checkpoint, expected_sha256=expected_sha256, **kwargs)

    def buckling_modes(self, checkpoint, *, expected_sha256, bounds, num_modes=6, cancellation_token=None):
        """Frozen-current conservative multipliers, not nonlinear limit loads.

        Retains all free nodal and cell coordinates and the real force-state
        owner. Active yielding, follower couples and unsupported states fail
        closed; accepted history and existing shell buckling are unchanged.
        """
        from ._ge_beam3_native_buckling import solve
        return solve(self, checkpoint, expected_sha256=expected_sha256, bounds=bounds,
                     num_modes=num_modes, cancellation_token=cancellation_token)

    def solve_spatial_couples(self, nodal_moments, **kwargs):
        """Generalized distributed forces/couples plus fixed-spatial nodal couples.

        Uses its own complete load-history envelope. It does not relabel
        nonconservative spatial work as a conservative force checkpoint.
        """
        from ._ge_beam3_analysis_combined import solve
        return solve(self, nodal_moments, **kwargs)

    def recover_spatial_couples(self, checkpoint, *, expected_sha256):
        """Physical section recovery from the authenticated combined history."""
        from ._ge_beam3_analysis_combined import recover
        return recover(self, checkpoint, expected_sha256=expected_sha256)

    def spatial_couple_checkpoint_prefix(self, checkpoint, accepted_steps, *, expected_sha256):
        """Select only already accepted combined-load snapshots, without a solve."""
        from ._ge_beam3_analysis_combined import prefix
        return prefix(self, checkpoint, accepted_steps, expected_sha256=expected_sha256)
