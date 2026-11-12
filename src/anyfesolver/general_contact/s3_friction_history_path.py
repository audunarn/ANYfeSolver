"""Declared equilibrated subincrement history, with invocation publication rollback.

This finite-resolution convention does not certify absence of interior events.
It reuses the existing coupled owner; no interpolated pressure history is used.
"""
import copy
import numpy as np
from contextvars import ContextVar
from dataclasses import replace
from .s3_friction_swept_gap import S3MeshWholeChordGapFrictionManager

PROFILE='S3_MESH_EQUILIBRATED_MATERIAL_SUBINCREMENTS_STATIC_DEV_V0'
POLICY='uniform_equilibrated_physical_subincrements_v1'
STATE_MODE='validated_owned_mapping_invocation_publication_v1'
_ENTRY=ContextVar('s3_history_invocation_entry',default=None)

def expand_program(program,ratio):
    expanded=replace(program,steps_per_segment=program.steps_per_segment*ratio)
    return expanded,dict(policy=POLICY,physical_subincrements_per_user_increment=ratio,
        original_user_program=program.contract(),state_mode=STATE_MODE,
        event_scope='finite_resolution_equilibria_not_event_free_certificate')

def register_entry(owner):
    entries=_ENTRY.get()
    if entries is None:raise RuntimeError('history profile requires invocation publication guard')
    entries.append((owner,copy.deepcopy((owner.committed_displacement,owner.history,
        owner.last_committed_summary,owner.normal_multipliers))))

def guarded_invocation(operation):
    entries=[];token=_ENTRY.set(entries)
    try:return operation()
    except BaseException:
        for owner,state in reversed(entries):
            pose,history,summary,normal=state
            owner.committed_displacement=pose;owner.history=history
            owner.last_committed_summary=summary
            owner.normal.set_normal_multipliers(normal)
        raise
    finally:_ENTRY.reset(token)

class S3MeshHistoryPathFrictionManager(S3MeshWholeChordGapFrictionManager):
    def validate_successor(self,u,z):
        pose,history,summary=super().validate_successor(u,z)
        force,_history,records=self._force(u,z)
        corners,n,_frame,_origin=self._patch(u)
        dn=self._normal_variation(corners,n,local=True)
        du=np.asarray(u)-self.committed_displacement;local_du=du[self.work_columns]
        projector=np.eye(3)-np.outer(n,n);material=numerical=0.
        return_drop=plastic_dissipation=projection_excess=0.
        kt=self.friction_settings.tangential_stiffness_Pa_per_m
        mu=self.friction_settings.friction_coefficient
        for j,(point,area,record) in enumerate(zip(self.points,self.weights,records)):
            element=record['target_element']
            if element is None:continue
            bary=np.asarray(record['target_barycentric']);traction=np.asarray(record['traction_pa'])
            js=self.point_maps[j];jt=np.einsum('i,ijk->jk',bary,self.target_maps[element])
            x=self.point_reference[j]+js@np.asarray(u)[self.work_columns]
            y=bary@corners[element];gap=n@(x-y)
            material+=area*traction@(projector@((js-jt)@local_du))
            numerical-=area*gap*traction@(dn@local_du)
            old=self.history[j]
            if old.target_element is not None and record['branch'] in ('stick','slip'):
                old_anchor=np.asarray(old.barycentric)@corners[old.target_element]
                trial=projector@(x-old_anchor);returned=projector@(x-y);dp=trial-returned
                return_drop+=.5*area*kt*(trial@trial-returned@returned)
                plastic_dissipation+=area*mu*record['pressure_pa']*np.linalg.norm(dp)
                projection_excess+=.5*area*kt*(dp@dp)
        summary.update(profile=PROFILE,material_history_policy=POLICY,
            material_history_event_scope='accepted_physical_subincrements_only',
            material_relative_endpoint_contraction_j=float(material),
            numerical_normal_variation_endpoint_contraction_j=float(numerical),
            friction_residual_endpoint_contraction_j=float(force@du),
            endpoint_work_identity_error_j=float(-force@du-material-numerical),
            work_reporting_scope='endpoint_contractions_not_path_integrals',
            return_work_reporting_schema='material_anchor_return_algebra_v1',
            return_storage_drop_anchor_j=float(return_drop),
            endpoint_pressure_plastic_dissipation_anchor_j=float(plastic_dissipation),
            hypothetical_projection_excess_anchor_j=float(projection_excess),
            return_algebra_error_j=float(return_drop-plastic_dissipation-projection_excess),
            return_projection_loss_meaning='hypothetical_elastic_trial_excess_not_physical_dissipation',
            return_dissipation_scope='endpoint_pressure_discrete_material_anchor_slip_not_variable_pressure_path_integral',
            power_components_scope='material_and_normal_terms_not_separately_objective_sum_is_full_work',
            physical_subincrements_per_user_increment=self.friction_settings.physical_subincrements_per_user_increment)
        return pose,history,summary

    def restart_contract(self):
        contract=super().restart_contract()
        contract.update(friction_profile=PROFILE,material_history_policy=POLICY,
            physical_subincrements_per_user_increment=self.friction_settings.physical_subincrements_per_user_increment,
            structural_state_mode=STATE_MODE)
        return contract


def summarize_work_history(history):
    """Aggregate only declared-schema rows; legacy rows are never treated as zero."""
    schema='material_anchor_return_algebra_v1'
    fields=('material_relative_endpoint_contraction_j','numerical_normal_variation_endpoint_contraction_j',
            'return_storage_drop_anchor_j','endpoint_pressure_plastic_dissipation_anchor_j',
            'hypothetical_projection_excess_anchor_j','return_algebra_error_j')
    totals={key:0. for key in fields};covered=[];missing=[]
    for row in history:
        friction=row.get('friction',{})
        if friction.get('return_work_reporting_schema')!=schema:
            missing.append(row['step_index']);continue
        values={key:float(friction[key]) for key in fields}
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('nonfinite declared material work row')
        covered.append(row['step_index'])
        for key,v in values.items():totals[key]+=v
    return dict(schema=schema,status='no_accepted_rows' if not history else 'complete' if not missing else 'partial_schema_coverage',
        covered_physical_steps=covered,excluded_physical_steps=missing,totals_on_covered_rows_only=totals,
        scope='endpoint_contractions_and_return_algebra_not_path_integrals')
