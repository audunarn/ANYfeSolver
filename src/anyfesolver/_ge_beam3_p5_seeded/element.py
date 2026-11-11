# V5 successor of the preserved V4 load-aware candidate. Mechanical expressions
# remain unchanged; authoritative operator seeding is bound by the V5 core.
"""Load-parameter native successor to the frozen centered P5 package.

Private development candidate: no public selector or qualification authority.
Unchanged physical operators are imported from the preserved P5 package.
"""

from copy import deepcopy

import numpy as np

from anyfesolver.elements import Element
from anyfesolver._native_material_protocol import PROTOCOL, NativeMaterialContext, NativeMaterialValidator
from anyfesolver._ge_beam3_p5_seeded.core import (
    StationaryCore, NativeMaterialError, sha, seal,
)
from anyfesolver._ge_beam3_p5.chart import pullback
from anyfesolver._ge_beam3_p5.arrays import _array, _readonly
from anyfesolver._ge_beam3_p5_seeded import codec as codec
from anyfesolver._ge_beam3_p5_centered.positions import POLICY, from_total, station_position
from .state import LOAD_POLICY, LoadStateStore


SCHEMA='GE_BEAM3_P5_LOAD_DRIVER_STATE_V5'
FORMULATION='CANDIDATE_GE_BEAM3_P5_NATIVE_LOAD_V5'


class NativeP5BeamElement(Element):
    """Scalar native-driver test candidate; public registration remains absent."""

    formulation_native_total_lagrangian=True
    native_material_state_protocol=PROTOCOL
    formulation_id=FORMULATION
    num_nodes=3
    dofs_per_node=6

    def __init__(self, element_id, nodes, reference, section, *, line_force, order=8):
        self._core=StationaryCore(element_id, nodes, reference, section, line_force=line_force, order=order)
        super().__init__(element_id, nodes, self.core.material_name)
        self.core.section.checkpoint_descriptor=self._section_descriptor()
        self._layout=None
        self.driver_sha256=sha(dict(schema=SCHEMA, formulation=FORMULATION,
                                   element=element_id, nodes=nodes, core=self.core.model_sha256))

    @property
    def core(self): return self._core

    def _section_descriptor(self):
        law=self.core.section
        return dict(schema='P5_DIRECTED_HARDENING_SECTION_DESCRIPTOR_V1', elastic=law._elastic.tolist(),
                    direction=law._direction.tolist(), yield_force=law._yield, hardening=law._hardening)

    def _check(self, mesh):
        if type(self) is not NativeP5BeamElement:
            raise NativeMaterialError('exact private driver candidate required')
        if mesh.elements.get(self.element_id) is not self or any(type(e) is not NativeP5BeamElement for e in mesh.elements.values()):
            raise NativeMaterialError('private driver candidate is standalone-only; no unqualified joints or shells')
        self.core._check_model(mesh)
        if sha(self.core.section.checkpoint_descriptor)!=sha(self._section_descriptor()):
            raise NativeMaterialError('section checkpoint descriptor differs from physical law')
        expected=sha(dict(schema=SCHEMA, formulation=FORMULATION, element=self.element_id,
                          nodes=self.node_ids, core=self.core.model_sha256))
        if expected!=self.driver_sha256 or tuple(self.node_ids)!=self.core.node_ids:
            raise NativeMaterialError('driver identity changed')
        layout=tuple(self.get_dof_mapping(mesh))
        if len(layout)!=18 or len(set(layout))!=18:
            raise NativeMaterialError('driver requires eighteen distinct nodal DOFs')
        if self._layout is None: self._layout=layout
        elif layout!=self._layout: raise NativeMaterialError('driver DOF ownership changed')

    def get_node_coordinates(self, mesh): return self.core.get_node_coordinates(mesh)

    def native_reference_directors(self, mesh):
        self._check(mesh)
        return self.core.native_reference_directors(mesh)

    def _wrap(self, state):
        return seal(dict(driver_schema=SCHEMA, driver_sha256=self.driver_sha256,
            committed_total_u=state['committed_total_u'],
            committed_nodal_rotation_matrices=state['committed_nodal_rotation_matrices'], material_state=state))

    def _validate(self, mesh, state, expected=None):
        self._check(mesh)
        keys={'driver_schema', 'driver_sha256', 'committed_total_u',
              'committed_nodal_rotation_matrices', 'material_state', 'state_sha256'}
        if type(state) is not dict or set(state)!=keys:
            raise NativeMaterialError('exact private driver state required')
        if (state['driver_schema']!=SCHEMA or state['driver_sha256']!=self.driver_sha256
            or seal(state)['state_sha256']!=state['state_sha256']):
            raise NativeMaterialError('driver state identity/hash mismatch')
        inner=self.core.validate_model_bound_nonlinear_state(mesh, None, state['material_state'], 1,
                                                            expected_committed_total_u=expected)
        for key in ('committed_total_u', 'committed_nodal_rotation_matrices'):
            if not np.array_equal(state[key], inner[key]):
                raise NativeMaterialError('driver/native state redundancy mismatch')
        return inner

    def init_model_bound_nonlinear_state(self, mesh, material, num_layers):
        self._check(mesh)
        return self._wrap(self.core.init_model_bound_nonlinear_state(mesh, material, num_layers))

    def validate_model_bound_nonlinear_state(self, mesh, material, state, num_layers,
                                            *, expected_committed_total_u=None):
        self.core._arguments(mesh, material, num_layers)
        if type(state) is dict and state.get('schema')==codec.SCHEMA:
            state=codec.decode(state, order=self.core.order)
        self._validate(mesh, state, expected_committed_total_u)
        return deepcopy(state)

    def serialize_native_material_state(self, mesh, state):
        self._validate(mesh, state)
        return codec.encode(state, order=self.core.order)

    def to_dict(self):
        # Deterministic model descriptor only. No public factory deserialization.
        return dict(formulation_id=FORMULATION, schema=SCHEMA, codec=codec.SCHEMA,
            element_id=self.element_id, node_ids=list(self.node_ids), driver_sha256=self.driver_sha256,
            reference_coordinates=self.core.reference.coordinates.tolist(),
            reference_triads=self.core.reference.nodal_triads.tolist(), section=self._section_descriptor(),
            quadrature_order=self.core.order, update='SPATIAL_MULTIPLICATIVE',
            coordinate_policy=POLICY, load_policy=LOAD_POLICY, line_force=self.core.line_force.tolist(),
            reference_evaluation=self.core.reference.evaluation_id,
            reference_fingerprint=self.core.reference.fingerprint(),
            reduction='STATIONARY_CELL_ROTATIONS_AND_ENDPOINT_MOMENTS', production_qualified=False)

    def recover_native_fields(self, mesh, state, *, expected_committed_total_u=None):
        """Read accepted station state; no new material trial or history advance."""
        validated=self.validate_model_bound_nonlinear_state(mesh, self.core.section, state, 1,
            expected_committed_total_u=expected_committed_total_u)
        inner=validated['material_state'];response=inner['response'];ref=self.core.reference
        reference_frames=[];current_frames=[];positions=[];position_lows=[];global_resultants=[]
        points, _ = np.polynomial.legendre.leggauss(self.core.order)
        for station in response.stations:
            xi=station.reference_coordinate;cell=station.cell;t=float((points[station.index]+1)/2)
            if xi != cell-1+t: raise NativeMaterialError('station geometry schedule mismatch')
            r0=ref.frame(xi);q=response.local_rotations[cell]@r0
            lift=ref.half_cell_lift(cell,t)
            point, point_low = station_position(inner['committed_positions'], inner['committed_position_low'],
                cell, t, response.local_rotations[cell], lift)
            resultant=station.response.resultants
            reference_frames.append(r0);current_frames.append(q);positions.append(point);position_lows.append(point_low)
            global_resultants.append(np.r_[q@resultant[:3], q@resultant[3:]])
        return dict(formulation_id=FORMULATION, state_schema=SCHEMA, production_qualified=False,
            driver_sha256=self.driver_sha256, state_sha256=validated['state_sha256'],
            station_ids=tuple((s.cell, s.index) for s in response.stations),
            reference_coordinates=_readonly(np.array([s.reference_coordinate for s in response.stations])),
            reference_frames=_readonly(np.array(reference_frames)), current_frames=_readonly(np.array(current_frames)),
            current_positions=_readonly(np.array(positions)),
            current_position_low=_readonly(np.array(position_lows)), coordinate_policy=POLICY,
            reference_evaluation=ref.evaluation_id, reference_fingerprint=ref.fingerprint(),
            load_policy=LOAD_POLICY, load_parameter=inner['load_parameter'],
            position_display_policy='CURRENT_POSITIONS_IS_ROUNDED_HIGH_COMPONENT',
            strains=_readonly(np.array([s.response.strain for s in response.stations])),
            resultants=_readonly(np.array([s.response.resultants for s in response.stations])),
            global_resultants=_readonly(np.array(global_resultants)),
            strain_order=('eps_x', 'gamma_xy', 'gamma_xz', 'kappa_x', 'kappa_y', 'kappa_z'),
            resultant_order=('N', 'V_y', 'V_z', 'T', 'M_y', 'M_z'),
            fibre_stress_status='SECTION_DOES_NOT_SUPPLY_FIBRE_STRESSES')

    @staticmethod
    def _pose(inner, view, *, committed):
        positions=view.committed_coordinates if committed else view.trial_coordinates
        matrices=view.committed_rotation_matrices if committed else view.trial_rotation_matrices
        rotations=view.committed_rotation_coordinates if committed else view.trial_rotation_coordinates
        if (not np.array_equal(inner['committed_positions'], positions)
            or not np.array_equal(inner['committed_nodal_rotation_matrices'], matrices)
            or not np.array_equal(inner['committed_total_u'].reshape(3, 6)[:, 3:], rotations)):
            raise NativeMaterialError('driver material/native pose mismatch')

    def create_native_material_validator(self, mesh):
        self._check(mesh)
        def validate(state, *, previous_state, native_view, displacement, phase):
            inner=self._validate(mesh, state, displacement)
            if phase=='committed':
                if previous_state is not None: raise NativeMaterialError('invalid initial validator phase')
                self._pose(inner, native_view, committed=True)
            elif phase=='trial':
                previous=self._validate(mesh, previous_state)
                self._pose(previous, native_view, committed=True)
                self._pose(inner, native_view, committed=False)
                if inner['epoch']!=previous['epoch']+1 or inner['origins']!=previous['histories']:
                    raise NativeMaterialError('driver accepted-origin linkage mismatch')
            else: raise NativeMaterialError('unknown native validation phase')
        return NativeMaterialValidator(validate)

    def compute_nonlinear_response(self, mesh, material, displacement, state=None, num_layers=1,
                                   tangent=True, *, native_rotation_trial=None, native_material_context=None):
        self.core._arguments(mesh, material, num_layers)
        context=native_material_context;view=native_rotation_trial
        if type(context) is not NativeMaterialContext or context.element_id!=self.element_id:
            raise NativeMaterialError('live driver material context required')
        context.require_view(view)
        if type(context.store) is not LoadStateStore:
            raise NativeMaterialError('explicit load-aware native state store required')
        parameter=context.store.native_load_parameter(context.token,self.element_id)
        previous=self._validate(mesh, state)
        self._pose(previous, view, committed=True)
        total=_array(displacement, (18,), 'driver element displacement')
        if not np.array_equal(total.reshape(3, 6)[:, 3:], view.trial_rotation_coordinates):
            raise NativeMaterialError('driver total displacement/view mismatch')
        from_total(self.core.reference.coordinates, total, view.trial_coordinates)
        response=self.core._solve(view.trial_coordinates, view.trial_rotation_matrices, previous['histories'], total=total, parameter=parameter)
        force, matrix, _=pullback(response.residual, response.tangent, view.rotation_coordinate_increment)
        context.require_view(view)
        inner=self.core._state(previous['epoch']+1, total, view.trial_coordinates,
                               view.trial_rotation_matrices, previous['histories'], response, parameter=parameter)
        return force.copy(), matrix.copy() if tangent else None, self._wrap(inner)

    def _unsupported(self, *args, **kwargs):
        raise NativeMaterialError('private driver candidate: workflow not yet authorized')

    def native_load_parameter_derivative(self, mesh, state, *, native_rotation_trial, native_material_context):
        context=native_material_context; view=native_rotation_trial
        if type(context) is not NativeMaterialContext or context.element_id!=self.element_id or type(context.store) is not LoadStateStore:
            raise NativeMaterialError('owned native load context required')
        context.require_view(view)
        parameter=context.store.native_load_parameter(context.token,self.element_id)
        inner=self._validate(mesh,state)
        previous=self._validate(mesh,context.store[self.element_id])
        self._pose(inner,view,committed=False); self._pose(previous,view,committed=True)
        if (inner['load_parameter']!=parameter or inner['epoch']!=previous['epoch']+1
                or inner['origins']!=previous['histories']):
            raise NativeMaterialError('load derivative state is not this trial candidate')
        spatial,work=self.core._load_parameter_derivative(inner)
        from anyfesolver._ge_beam3_p5.chart import exp_chart_terms
        result=spatial.copy()
        for node,increment in enumerate(view.rotation_coordinate_increment):
            slot=slice(6*node+3,6*node+6)
            result[slot]=exp_chart_terms(increment)[0].T@spatial[slot]
        context.require_view(view)
        return dict(load_policy=LOAD_POLICY,load_parameter=parameter,
            residual_parameter_derivative=_readonly(result),potential_parameter_derivative=float(work))

    def compute_stiffness_matrix(self, mesh, material):
        # The actual Newton driver requests a virgin reference K0 to build its
        # constraint reduction. This is the native stationary Hessian, not a
        # legacy/reference-beam surrogate or a current plastic tangent.
        initial=self.init_model_bound_nonlinear_state(mesh, material, 1)
        return initial['material_state']['response'].tangent.copy()

    compute_mass_matrix=_unsupported
    compute_geometric_stiffness_matrix=_unsupported
    compute_internal_forces=_unsupported
    compute_stresses=_unsupported
