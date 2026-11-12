"""Private spatial nodal couples in the actual native force Newton driver.

Work is m dot spatial virtual rotation. A constant spatial couple is not a
conservative SO(3) potential; its additive Exp-chart force/tangent are A.T m
and (dA/dtheta).T m. Internal element forces/Hessians remain untouched.
Distributed couples remain unsupported. Moment-bearing restart is admitted
only through the complete externally authenticated spatial-load chain.
"""
from contextvars import ContextVar
from dataclasses import dataclass,field
from hashlib import sha256
import numpy as np
from scipy import sparse
from ._ge_beam3_spatial_nodal_moments import SpatialNodalMoments,POLICY
from ._ge_beam3_native_line_loading import LinePattern
from ._ge_beam3_native_line_program import model_identity,solve_line_static
from ._ge_beam3_p5.chart import exp_chart_terms
from ._ge_beam3_p5_seeded.core import sha

_RUN=ContextVar('ge_beam3_native_spatial_couples',default=None)


def _capture(value,model):
    if value is None:return None
    if type(value) is not SpatialNodalMoments:raise ValueError('exact spatial nodal couple pattern required')
    made=SpatialNodalMoments(value.rows)
    if any(row[0] not in model.mesh.nodes for row in made.rows):raise ValueError('spatial couple node absent')
    return made


def _rows(value):return () if value is None else value.rows


@dataclass
class _Run:
    model: object
    proportional: object
    constant: object=None
    restart_sha256: object=None
    events: list=field(default_factory=list,init=False)
    store: object=field(default=None,init=False,repr=False)
    identity: str=field(init=False)
    model_sha256: str=field(init=False)

    def __post_init__(self):
        self.proportional=_capture(self.proportional,self.model);self.constant=_capture(self.constant,self.model)
        if self.proportional is None and self.constant is None:raise ValueError('explicit nonzero spatial couple programme required')
        self.model_sha256=model_identity(self.model);self.identity=sha(self.descriptor())

    def descriptor(self):
        value=dict(policy=POLICY,model_sha256=self.model_sha256,proportional=_rows(self.proportional),constant=_rows(self.constant),
            axes='FIXED_SPATIAL',virtual_work='M_DOT_SPATIAL_MULTIPLICATIVE_VIRTUAL_ROTATION',
            chart_force='A_TRANSPOSE_M',chart_tangent='DERIVATIVE_A_TRANSPOSE_M',
            general_matrix_required=True,conservative_potential=False,conservative_spectral_authority=False,
            moment_restart_authorized=False,production_qualified=False)
        if self.restart_sha256 is not None:
            value.update(restart_sha256=self.restart_sha256,moment_restart_authorized=True,
                restart_policy='GE_BEAM3_SUPPORTED_NATIVE_SPATIAL_COUPLE_RESTART_V1')
        return value

    def require(self,model):
        if model is not self.model or model_identity(model)!=self.model_sha256:raise ValueError('spatial couple model authority changed')
        _capture(self.proportional,model);_capture(self.constant,model)
        if sha(self.descriptor())!=self.identity:raise ValueError('spatial couple frozen authority changed')

    def effective(self,parameter):
        if not np.isfinite(parameter) or not 0.<=parameter<=1.:raise ValueError('bounded spatial couple parameter required')
        result={}
        for factor,pattern in ((1.,self.constant),(float(parameter),self.proportional)):
            for node,*moment in _rows(pattern):
                with np.errstate(over='ignore',invalid='ignore'):
                    result[node]=result.get(node,np.zeros(3))+factor*np.array(moment)
                if not np.isfinite(result[node]).all():raise ValueError('effective spatial couple exceeds finite range')
        return result


def active_for(model):
    run=_RUN.get()
    if run is None:return False
    if type(run) is not _Run:raise ValueError('exact spatial couple programme scope required')
    run.require(model);return True


def decode_active_restart(model,raw,*,expected_sha256):
    from ._ge_beam3_native_spatial_restart import decode_checkpoint
    run=_RUN.get()
    if type(run) is not _Run:raise ValueError('live spatial restart programme required')
    run.require(model)
    if type(raw) is not bytes or run.restart_sha256 is None or expected_sha256!=run.restart_sha256 or sha256(raw).hexdigest()!=run.restart_sha256:
        raise ValueError('spatial restart programme checkpoint mismatch')
    chain=decode_checkpoint(model,raw,expected_sha256=expected_sha256)
    if run.constant!=chain[-1]['load_point'].effective_moments(model):
        raise ValueError('spatial restart constant moment mismatch')
    run.require(model)
    return chain


def external_at(model,store,displacements,parameter,*,tangent):
    from .nonlinear_state import NonlinearStateStore
    run=_RUN.get()
    if type(run) is not _Run or type(store) is not NonlinearStateStore or store.native_rotation_store is None:
        raise ValueError('live spatial couple programme and native state store required')
    run.require(model)
    if run.store is None:run.store=store
    elif run.store is not store:raise ValueError('foreign spatial couple state store')
    u=np.asarray(displacements)
    n=model.mesh.dof_manager.total_dofs
    if u.dtype!=np.float64 or u.shape!=(n,) or not np.isfinite(u).all() or type(tangent) is not bool:
        raise ValueError('exact spatial couple displacement/tangent request')
    generation=(store.generation,store.native_rotation_store.generation)
    force=np.zeros(n);matrix=np.zeros((n,n)) if tangent else None;increments={}
    try:
        for eid,e in sorted(model.mesh.elements.items()):
            binding=store._native_element_bindings.get(eid)
            if binding is None or binding.material_validator is not e._validator:
                raise ValueError('foreign spatial couple element registration')
            committed=store._native_view_for_binding(eid,trial=False)
            mapping=list(e.get_dof_mapping(model.mesh));total=u[mapping].reshape(3,6)
            delta=total[:,3:]-committed.committed_rotation_coordinates
            if store.has_active_trial:
                token=store.active_trial_token()
                view=store.native_element_rotation_view(token,eid,e.node_ids,e.native_reference_directors(model.mesh))
                store.native_material_context(token,eid).require_view(view)
                if not np.array_equal(view.rotation_coordinate_increment,delta) or not np.array_equal(view.trial_coordinates,e._coordinates(u[mapping])[0]):
                    raise ValueError('spatial couple load does not match live trial pose')
            elif np.any(delta) or not np.array_equal(committed.committed_coordinates,e._coordinates(u[mapping])[0]):
                raise ValueError('unissued spatial couple trial pose')
            for local,node in enumerate(e.node_ids):
                if node in increments and not np.array_equal(increments[node],delta[local]):raise ValueError('spatial couple shared chart mismatch')
                increments[node]=delta[local].copy()
        for node,moment in sorted(run.effective(parameter).items()):
            a,da=exp_chart_terms(increments[node]);dofs=list(model.mesh.dof_manager.get_node_dofs(node)[3:])
            force[dofs]=a.T@moment
            if tangent:matrix[np.ix_(dofs,dofs)]=np.column_stack([da[:,:,k].T@moment for k in range(3)])
        with np.errstate(over='ignore',invalid='ignore'):
            norm=float(np.linalg.norm(force))
        if not np.isfinite(force).all() or not np.isfinite(norm) or (tangent and not np.isfinite(matrix).all()):
            raise ValueError('spatial couple chart force/tangent exceeds finite norm range')
        run.require(model)
        if generation!=(store.generation,store.native_rotation_store.generation):raise ValueError('spatial couple committed state changed')
        run.events.append(dict(parameter=float(parameter),tangent=tangent,generation=int(store.generation),
            active_trial=bool(store.has_active_trial),pattern_sha256=sha(tuple((i,tuple(map(float,m))) for i,m in sorted(run.effective(parameter).items())))))
        return force,sparse.csr_matrix(matrix) if tangent else None
    except BaseException:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
        raise


def solve_spatial_static(model,nodal_moments,*,line_pattern=None,constant_line=None,constant_moments=None,
                         steps=2,max_iterations=24,line_search=True,initial_checkpoint=None,expected_sha256=None):
    if _RUN.get() is not None:raise ValueError('nested spatial couple programme forbidden')
    if initial_checkpoint is not None:
        from ._ge_beam3_native_spatial_restart import decode_checkpoint
        chain=decode_checkpoint(model,initial_checkpoint,expected_sha256=expected_sha256)
        accepted=chain[-1]['load_point'];moments=accepted.effective_moments(model);line=accepted.line.effective(model)
        if constant_moments is None:constant_moments=moments
        elif _capture(constant_moments,model)!=moments:raise ValueError('spatial restart constant must match accepted moments')
        if constant_line is None:constant_line=line
        elif constant_line!=line:raise ValueError('spatial restart constant must match accepted line')
    elif expected_sha256 is not None:raise ValueError('spatial restart hash without checkpoint')
    run=_Run(model,nodal_moments,constant_moments,restart_sha256=expected_sha256)
    original=sha(dict(proportional=_rows(nodal_moments),constant=_rows(constant_moments)))
    token=_RUN.set(run)
    try:
        result,line_events=solve_line_static(model,LinePattern(()) if line_pattern is None else line_pattern,
            constant=constant_line,steps=steps,max_iterations=max_iterations,line_search=line_search,
            initial_checkpoint=initial_checkpoint,expected_sha256=expected_sha256)
        run.require(model)
        if sha(dict(proportional=_rows(nodal_moments),constant=_rows(constant_moments)))!=original:raise ValueError('caller spatial couple inputs changed')
        evidence=dict(program=run.descriptor(),program_sha256=run.identity,line_assembly_events=line_events,external_events=tuple(run.events),
            production_qualified=False,moment_restart_authorized=initial_checkpoint is not None)
        result.info['native_spatial_couples']=run.descriptor()
        return result,evidence
    finally:
        if run.store is not None and run.store.has_active_trial:run.store.discard_trial(run.store.active_trial_token())
        _RUN.reset(token)
