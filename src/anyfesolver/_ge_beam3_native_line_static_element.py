"""Private load-aware successor using actual native state tokens.

Mechanical kernels and zero-load adapter are preserved; this successor binds
reference-line work into local equilibrium and replay. Global Newton load-case
routing and typed load-chain restart are not yet exposed.

No public selector or dynamic route is registered. Existing beam mechanics
are not inherited. The 24 internal static coordinates remain explicit state.
"""
from copy import deepcopy
from decimal import localcontext
from fractions import Fraction
import numpy as np
from scipy import linalg
from .elements import Element
from ._native_material_protocol import PROTOCOL,NativeMaterialContext,NativeMaterialValidator
from ._ge_beam3_retained_fibre import RetainedFibreOperator
from ._ge_beam3_fibre_static_boundary import solve_static,StaticTrial,POLICY as STATIC_POLICY
from ._ge_beam3_p5.chart import pullback
from ._ge_beam3_p5.arrays import _array,_frames
from ._ge_beam3_p5_seeded.core import canonical,sha
from ._native_reference_modal import _owned
from ._ge_beam3_native_line_loading import LinePattern,current_pattern
from ._ge_beam3_fibre_line_work import evaluate as line_work

FORMULATION='CANDIDATE_GE_BEAM3_NATIVE_PHYSICAL_FIBRE_LINE_STATIC_V1'
SCHEMA='GE_BEAM3_NATIVE_PHYSICAL_FIBRE_LINE_STATIC_STATE_V1'


def seal(value):
    body={k:v for k,v in value.items() if k!='state_sha256'}
    return {**body,'state_sha256':sha(body)}


class NativeLineFibreStaticElement(Element):
    num_nodes=3
    dofs_per_node=6
    formulation_id=FORMULATION
    formulation_native_total_lagrangian=True
    native_material_state_protocol=PROTOCOL
    production_qualified=False

    def __init__(self,element_id,node_ids,reference,section,*,order=4):
        if type(element_id) is not int or element_id<=0 or type(node_ids) is not tuple or len(node_ids)!=3 or len(set(node_ids))!=3 or any(type(i) is not int or i<=0 for i in node_ids):
            raise ValueError('explicit native fibre element/node identities')
        super().__init__(element_id,node_ids,'private-native-fibre-'+str(element_id))
        self.operator=RetainedFibreOperator(reference,section,order=order); self.section=section
        self.identity=sha(dict(formulation=FORMULATION,element_id=element_id,nodes=node_ids,operator=self.operator.identity))
        self._mapping=None; self._mesh=None; self._validator=None; self._issued=None

    def _check(self,mesh):
        if type(self) is not NativeLineFibreStaticElement or mesh.elements.get(self.element_id) is not self:
            raise ValueError('exact native fibre element ownership')
        if self._mesh is None: self._mesh=mesh
        elif self._mesh is not mesh: raise ValueError('native fibre element belongs to another mesh')
        if any(type(e) is not NativeLineFibreStaticElement for e in mesh.elements.values()): raise ValueError('private standalone native fibre adapter only')
        self.operator.guard()
        if self.section is not self.operator.section or self.formulation_id!=FORMULATION or self.production_qualified is not False or self.num_nodes!=3 or self.dofs_per_node!=6:
            raise ValueError('native fibre formulation/section changed')
        if sha(dict(formulation=FORMULATION,element_id=self.element_id,nodes=self.node_ids,operator=self.operator.identity))!=self.identity:
            raise ValueError('native fibre identity changed')
        if not np.array_equal(self.get_node_coordinates(mesh),self.operator.reference.coordinates): raise ValueError('native reference coordinates changed')
        mapping=tuple(self.get_dof_mapping(mesh))
        if len(mapping)!=18 or len(set(mapping))!=18: raise ValueError('eighteen native external DOFs required')
        if self._mapping is None: self._mapping=mapping
        elif mapping!=self._mapping: raise ValueError('native fibre DOF ownership changed')

    def get_node_coordinates(self,mesh):
        return np.array([mesh.nodes[i].coords() for i in self.node_ids])

    def native_reference_directors(self,mesh):
        self._check(mesh); return self.operator.reference.nodal_triads[:,:,2].copy()

    def to_dict(self):
        return dict(formulation_id=FORMULATION,state_schema=SCHEMA,element_id=self.element_id,node_ids=self.node_ids,
            operator=self.operator.identity,reference=self.operator.reference.fingerprint(),section=self.section.identity,
            quadrature=self.operator.order,reduction='STATIC_ONLY_24_INTERNAL_COORDINATES',production_qualified=False)

    def _coordinates(self,total):
        total=_array(total,(18,),'native total displacement'); reference=self.operator.reference.coordinates
        high=reference+total.reshape(3,6)[:,:3]; low=np.empty((3,3))
        for i in range(3):
            for j in range(3): low[i,j]=float(Fraction(float(reference[i,j]))+Fraction(float(total[6*i+j]))-Fraction(float(high[i,j])))
        return _owned(high),_owned(low)

    def _state(self,epoch,total,operators,origin,seed_u,seed_p,response,previous,pattern):
        positions,low=self._coordinates(total)
        return seal(dict(schema=SCHEMA,element_identity=self.identity,epoch=epoch,previous_state_sha256=previous,
            committed_total_u=_owned(total),committed_nodal_rotation_matrices=_owned(operators),positions=positions,position_low=low,
            origins=origin,seed_rotations=_owned(seed_u),seed_resultants=_owned(seed_p),response=response,load_pattern=pattern))

    def _validate(self,mesh,state,expected=None):
        self._check(mesh)
        keys={'schema','element_identity','epoch','previous_state_sha256','committed_total_u','committed_nodal_rotation_matrices',
            'positions','position_low','origins','seed_rotations','seed_resultants','response','state_sha256','load_pattern'}
        if type(state) is not dict or set(state)!=keys or state['schema']!=SCHEMA or state['element_identity']!=self.identity or seal(state)['state_sha256']!=state['state_sha256']:
            raise ValueError('native fibre state schema/identity/hash')
        pattern=state['load_pattern']
        if type(pattern) is not LinePattern:raise ValueError('exact native line state pattern')
        pattern.require(mesh);load=pattern.force(self.element_id)
        epoch=state['epoch']
        if type(epoch) is not int or epoch<0: raise ValueError('native fibre state epoch')
        previous_hash=state['previous_state_sha256']
        if epoch and (type(previous_hash) is not str or len(previous_hash)!=64 or any(c not in '0123456789abcdef' for c in previous_hash)):
            raise ValueError('native fibre predecessor hash')
        total=_array(state['committed_total_u'],(18,),'native fibre state displacement')
        if expected is not None and not np.array_equal(total,expected): raise ValueError('native fibre displacement mismatch')
        positions,low=self._coordinates(total)
        if not np.array_equal(positions,state['positions']) or not np.array_equal(low,state['position_low']): raise ValueError('native compensated position mismatch')
        operators=_frames(state['committed_nodal_rotation_matrices'],3,'native fibre operators')
        response=state['response']
        if type(response) is not StaticTrial: raise ValueError('exact native fibre static response')
        with localcontext() as context:
            context.prec=80; self.operator.cell._origins(state['origins'])
        if epoch==0 and (pattern.rows or state['previous_state_sha256'] is not None or np.any(total) or not np.array_equal(operators,np.tile(np.eye(3),(3,1,1))) or canonical(state['origins'])!=canonical(self.operator.cell.virgin())):
            raise ValueError('native fibre virgin state mismatch')
        seed_u=_frames(state['seed_rotations'],2,'native seed rotations'); seed_p=_array(state['seed_resultants'],(18,),'native seed resultants')
        q=operators@self.operator.reference.nodal_triads
        input_id=sha(dict(policy=STATIC_POLICY,operator=self.operator.identity,positions=positions,position_low=low,
            nodal_frames=q,origin=state['origins'],line_force=load,initial_rotations=seed_u,initial_resultants=seed_p))
        if response.input_sha256!=input_id or type(response.iterations) is not int or not 0<=response.iterations<=24 or response.production_qualified or response.dynamic_reduction_authorized:
            raise ValueError('native fibre response provenance')
        replay=self.operator.evaluate(positions,low,q,response.rotations,response.resultants,origin=state['origins'])
        h=replay.hessian+replay.hessian_low;residual=replay.residual.copy();potential=replay.potential
        if np.any(load):
            work=line_work(self.operator.reference,positions,low,response.rotations,load,order=self.operator.order)
            residual-=work.gradient;h=h-work.hessian;potential-=work.value
        condensed=h[:18,:18]-h[:18,18:]@linalg.solve(h[18:,18:],h[18:,:18],assume_a='gen',check_finite=True)
        length=float(np.linalg.norm(self.operator.reference.coordinates[-1]-self.operator.reference.coordinates[0]))
        error=float(np.linalg.norm(residual[18:]/np.r_[np.full(12,length),np.ones(12)]))
        expected_response=StaticTrial(response.rotations,response.resultants,residual[:18],condensed,
            residual,h,replay.history,float(potential),error,response.iterations,input_id)
        if error>1e-11 or canonical(expected_response)!=canonical(response): raise ValueError('native fibre static/history replay mismatch')
        return state

    def _pose(self,state,view,*,committed):
        coordinates=view.committed_coordinates if committed else view.trial_coordinates
        operators=view.committed_rotation_matrices if committed else view.trial_rotation_matrices
        rotations=view.committed_rotation_coordinates if committed else view.trial_rotation_coordinates
        if not np.array_equal(state['positions'],coordinates) or not np.array_equal(state['committed_nodal_rotation_matrices'],operators) or not np.array_equal(state['committed_total_u'].reshape(3,6)[:,3:],rotations):
            raise ValueError('native fibre material/pose mismatch')

    def init_model_bound_nonlinear_state(self,mesh,material,num_layers):
        self._arguments(mesh,material,num_layers)
        total=np.zeros(18); operators=np.tile(np.eye(3),(3,1,1)); seed_u=np.tile(np.eye(3),(2,1,1)); seed_p=np.zeros(18)
        positions,low=self._coordinates(total); origin=self.operator.cell.virgin()
        response=solve_static(self.operator,positions,low,operators@self.operator.reference.nodal_triads,origin=origin,
            initial_rotations=seed_u,initial_resultants=seed_p,spatial_line_force=np.zeros(3))
        return self._state(0,total,operators,origin,seed_u,seed_p,response,None,LinePattern(()))

    def _arguments(self,mesh,material,num_layers):
        self._check(mesh)
        if material is not self.section or type(num_layers) is not int or num_layers!=1: raise ValueError('native fibre section/layer authority')

    def validate_model_bound_nonlinear_state(self,mesh,material,state,num_layers,*,expected_committed_total_u=None):
        self._arguments(mesh,material,num_layers); self._validate(mesh,state,expected_committed_total_u); return deepcopy(state)

    def create_native_material_validator(self,mesh):
        self._check(mesh)
        self._issued=None  # Each store registration receives a distinct validator.
        def validate(state,*,previous_state,native_view,displacement,phase):
            if phase=="trial":
                issued=self._issued
                if issued is None or state.get("state_sha256")!=issued[1]:raise ValueError("native line candidate was not issued by its load scope")
                binding=issued[0].store._native_element_bindings.get(self.element_id)
                if binding is None or binding.material_validator is not validator:raise ValueError("foreign native line validator")
                issued[0].require_view(native_view)
            self._validate(mesh,state,displacement)
            if phase=='committed':
                if previous_state is not None: raise ValueError('native fibre committed validation phase')
                self._pose(state,native_view,committed=True)
            elif phase=='trial':
                previous=self._validate(mesh,previous_state); self._pose(previous,native_view,committed=True); self._pose(state,native_view,committed=False)
                if state['epoch']!=previous['epoch']+1 or state['previous_state_sha256']!=previous['state_sha256'] or canonical(state['origins'])!=canonical(previous['response'].history): raise ValueError('native fibre accepted-origin linkage')
                if not np.array_equal(state['seed_rotations'],previous['response'].rotations) or not np.array_equal(state['seed_resultants'],previous['response'].resultants): raise ValueError('native fibre internal seed linkage')
            else: raise ValueError('native fibre validation phase')
        validator=NativeMaterialValidator(validate);self._validator=validator
        return validator

    def _live(self,context,view,state,*,check_state=True):
        if type(context) is not NativeMaterialContext or context.element_id!=self.element_id: raise ValueError('live native fibre material context required')
        context.require_view(view)
        # This private adapter explicitly binds the actual store's validator,
        # not only a numerically identical pose from a different model.
        binding=context.store._native_element_bindings.get(self.element_id)
        if self._validator is None or binding is None or binding.material_validator is not self._validator:
            raise ValueError('foreign native fibre material-store binding')
        if check_state and canonical(context.store[self.element_id])!=canonical(state):
            raise ValueError('native fibre input is not the current store state')

    def compute_nonlinear_response(self,mesh,material,displacement,state=None,num_layers=1,tangent=True,*,native_rotation_trial=None,native_material_context=None):
        self._arguments(mesh,material,num_layers); context=native_material_context; view=native_rotation_trial
        self._live(context,view,state); pattern=current_pattern(self,context,view);load=pattern.force(self.element_id)
        previous=self._validate(mesh,state); self._pose(previous,view,committed=True)
        total=_array(displacement,(18,),'native fibre trial displacement'); positions,low=self._coordinates(total)
        if not np.array_equal(positions,view.trial_coordinates) or not np.array_equal(total.reshape(3,6)[:,3:],view.trial_rotation_coordinates): raise ValueError('native fibre trial pose mismatch')
        seed=previous['response']; origin=seed.history
        response=solve_static(self.operator,positions,low,view.trial_rotation_matrices@self.operator.reference.nodal_triads,
            origin=origin,initial_rotations=seed.rotations,initial_resultants=seed.resultants,spatial_line_force=load,check=lambda: (self._live(context,view,state,check_state=False),current_pattern(self,context,view)))
        physical_residual=response.residual.copy()
        if np.any(load):
            work=line_work(self.operator.reference,positions,low,response.rotations,load,order=self.operator.order)
            physical_residual+=work.gradient[:18]  # Nodal external work is assembled once, outside this internal response.
        force,matrix,_=pullback(physical_residual,response.hessian,view.rotation_coordinate_increment)
        self._live(context,view,state)
        candidate=self._state(previous['epoch']+1,total,view.trial_rotation_matrices,origin,seed.rotations,seed.resultants,response,previous['state_sha256'],pattern)
        self._issued=(context,candidate['state_sha256'])
        return force.copy(),matrix.copy() if tangent else None,candidate

    def compute_stiffness_matrix(self,mesh,material):
        return self.init_model_bound_nonlinear_state(mesh,material,1)['response'].hessian.copy()

    def _unsupported(self,*args,**kwargs): raise ValueError('private native fibre static adapter: workflow not qualified')
    compute_mass_matrix=_unsupported
    compute_geometric_stiffness_matrix=_unsupported
    compute_internal_forces=_unsupported
    compute_stresses=_unsupported
    compute_native_dead_pressure_load=_unsupported
    compute_native_current_pressure_load=_unsupported
    compute_native_current_pressure_tangent=_unsupported
