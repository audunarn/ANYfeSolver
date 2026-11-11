"""Analytic load column for an issued generalized native stationary trial.

With g_i(y_i,y_e,lambda)=0, y_i,lambda=-J_ii^-1 b_i and
d g_e/d lambda=b_e+J_ei y_i,lambda. The accepted material origin and
external nodal pose stay fixed. This is a continuation building block, not
a new element law or an authorization for conservative spectral analysis.
"""
import numpy as np
from scipy import linalg
from .nonlinear_state import NonlinearStateStore
from ._ge_beam3_native_generalized_loading import DistributedPattern
from ._ge_beam3_native_generalized_program import model_identity
from ._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from ._ge_beam3_generalized_static_boundary import cell_couple_load
from ._ge_beam3_fibre_line_work import evaluate as line_work
from ._ge_beam3_p5.chart import exp_chart_terms
from ._ge_beam3_p5_seeded.core import canonical,sha
from ._native_reference_modal import _owned

POLICY='GE_BEAM3_NATIVE_GENERALIZED_ANALYTIC_LOAD_COLUMN_V1'


def parameter_column(model,store,states,proportional,*,nodal_moments=None):
    """Return net-residual d/dlambda in the current accepted-origin chart.

    The supplied patterns are derivatives, not the effective trial loads.
    Nodal moments are accumulated once globally, including shared nodes.
    Requires an active issued trial; it never commits or advances history.
    """
    if type(store) is not NonlinearStateStore or store.native_rotation_store is None or not store.has_active_trial:
        raise ValueError('issued active native parameter trial required')
    token=store.active_trial_token()
    try:
        if type(proportional) is not DistributedPattern:
            raise ValueError('exact proportional distributed pattern required')
        if nodal_moments is not None and type(nodal_moments) is not SpatialNodalMoments:
            raise ValueError('exact proportional nodal moments required')
        identity=model_identity(model);proportional.require(model.mesh)
        rows=() if nodal_moments is None else nodal_moments.rows
        if nodal_moments is not None:SpatialNodalMoments(rows)
        if any(row[0] not in model.mesh.nodes for row in rows):raise ValueError('parameter moment node absent')
        signature=sha((proportional,rows));generation=(store.generation,store.native_rotation_store.generation)
        state_identity=sha(dict(states))
        if set(states)!=set(model.mesh.elements):raise ValueError('complete parameter trial states required')
        n=model.mesh.dof_manager.total_dofs;column=np.zeros(n);lifts={};charts={}
        for eid,e in sorted(model.mesh.elements.items()):
            binding=store._native_element_bindings.get(eid)
            if binding is None or binding.material_validator is not e._validator:
                raise ValueError('foreign native parameter store binding')
            context=store.native_material_context(token,eid)
            view=store.native_element_rotation_view(token,eid,e.node_ids,e.native_reference_directors(model.mesh))
            context.require_view(view)
            state=states[eid]
            # Context wrappers are re-created; bind their store/token/view.
            if e._issued is None or e._issued[1]!=state['state_sha256'] or e._issued[0].store is not store:
                raise ValueError('parameter state was not issued by this trial')
            e._issued[0].require_view(view)
            e._validate(model.mesh,state);e._pose(state,view,committed=False)
            if canonical(store.trial_view(token)[eid])!=canonical(state):raise ValueError('parameter state differs from active trial')
            response=state['response']
            work=line_work(e.operator.reference,state['positions'],state['position_low'],response.rotations,
                proportional.force(eid),order=e.operator.order)
            partial=-work.gradient-cell_couple_load(e.operator,proportional.density(eid))
            j=response.full_spatial_jacobian
            lift=-linalg.solve(j[18:,18:],partial[18:],assume_a='gen',check_finite=True)
            reduced=partial[:18]+j[:18,18:]@lift
            for local,node in enumerate(e.node_ids):
                a,_=exp_chart_terms(view.rotation_coordinate_increment[local])
                if node in charts and not np.array_equal(charts[node],a):raise ValueError('parameter shared chart mismatch')
                charts[node]=a
                reduced[6*local+3:6*local+6]=a.T@reduced[6*local+3:6*local+6]
            np.add.at(column,e.get_dof_mapping(model.mesh),reduced)
            lifts[eid]=_owned(lift)
        for node,*moment in rows:
            dofs=list(model.mesh.dof_manager.get_node_dofs(node)[3:])
            column[dofs]-=charts[node].T@np.array(moment)
        if not np.isfinite(column).all() or not np.isfinite(np.linalg.norm(column)) or any(not np.isfinite(v).all() for v in lifts.values()):
            raise ValueError('parameter derivative exceeds finite range')
        proportional.require(model.mesh)
        if model_identity(model)!=identity or sha((proportional,() if nodal_moments is None else nodal_moments.rows))!=signature or sha(dict(states))!=state_identity:
            raise ValueError('parameter input authority changed')
        if generation!=(store.generation,store.native_rotation_store.generation) or store.active_trial_token()!=token:
            raise ValueError('parameter accepted state or trial changed')
        return dict(policy=POLICY,column=_owned(column),internal_parameter_lifts=lifts,
            model_sha256=identity,state_sha256=state_identity,pattern_sha256=signature,
            fixed_accepted_origin=True,production_qualified=False,conservative_spectral_authority=False)
    except BaseException:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
        raise
