"""Explicit weak-pressure finite-gap material friction, fully covered S3 mesh.

Numerical pressure support does not assert pointwise normal complementarity.
Old hard fully closed successor remains in its separate owner unchanged.
"""
import numpy as np
from .s3_friction_mesh_monolithic import S3MeshMovingNormalFrictionManager
from .s3_friction_moving_coverage import verified_norm

PROFILE='S3_MESH_WEAK_P1_PRESSURE_MATERIAL_FRICTION_STATIC_DEV_V0'

class S3MeshWeakPressureFrictionManager(S3MeshMovingNormalFrictionManager):
    def validate_successor(self,u,z):
        self.require_friction_path(self.committed_displacement,u)
        z=np.asarray(z,float)
        if z.shape!=(len(self.normal.law.pressure_mode_ids),) or not np.isfinite(z).all() or np.any(z<0):
            raise ValueError('weak pressure friction requires finite nonnegative supported forces')
        corners,n,frame,origin=self._patch(u)
        source,_target=self.normal._positions(u)
        dense=self.point_reference+self.point_maps@np.asarray(u)[self.work_columns]
        points=np.concatenate((source,dense));gaps=(points-origin)@n
        facet_gaps={}
        active=np.zeros(len(points),dtype=bool)
        for adapter,indices in zip(self.source_adapters,self.normal.law.source_face_indices):
            modes=[self.normal.law.source_node_ids.index(node) for node in adapter.node_ids]
            if np.any(z[modes]>0):
                active[indices]=True
                for j,point in enumerate(self.points):
                    if point.adapter.element_id==adapter.element_id:active[len(source)+j]=True
        for element,c in corners.items():
            raw=np.cross(c[1]-c[0],c[2]-c[0]);fn=raw/verified_norm(raw)
            fg=(points-c[0])@fn;facet_gaps[str(element)]=fg.tolist()
            if np.any(np.abs(fg[active])>self.friction_settings.finite_material_gap_error_m):
                raise ValueError('weak pressure finite material-gap error budget exceeded on active source face')
        force,history,records=self._force(u,z)
        previous_corners,old_n,_,_=self._patch(self.committed_displacement)
        normal_delta=float(np.linalg.norm(n-old_n))
        # End-pose discrepancy envelope, NOT integrated friction work.
        error_bound=sum(area*abs(gaps[len(source)+j])*np.linalg.norm(rec['traction_pa'])*normal_delta
                        for j,(area,rec) in enumerate(zip(self.weights,records)))
        summary=dict(profile=PROFILE,traction_measure='force_per_source_reference_area',
            pressure_measure='weak_mortar_lambda_over_source_target_cosine_per_source_reference_area',
            activation='numerical_pressure_positive_current_projection_birth',
            inactive_meaning='zero_numerical_pressure_not_geometric_separation',
            finite_material_gap_error_m=self.friction_settings.finite_material_gap_error_m,
            finite_material_gap_error_scope="accepted_endpoint_only",
            active_face_max_abs_gap_m=max((max(abs(np.asarray(v)[active]),default=0.) for v in facet_gaps.values()),default=0.),
            pressure_inactive_points=[j for j,r in enumerate(records) if r['pressure_pa']==0.],
            positive_pressure_locally_open_points=[j for j,r in enumerate(records) if r['pressure_pa']>0. and gaps[len(source)+j]>0.],
            physical_all_points_strictly_open=all(min(v)>0. for v in facet_gaps.values()),
            numerical_normal_variation_endpoint_work_bound_j=float(error_bound),
            numerical_work_bound_scope='accepted_endpoint_normal_change_not_path_integral',
            target_normal_delta=normal_delta,branch=[r['branch'] for r in records],
            point_gap_m=gaps.tolist(),facet_point_gap_m=facet_gaps,supported_force_n=z.tolist(),
            pressure_pa=[r['pressure_pa'] for r in records],target_elements=[r['target_element'] for r in records],
            traction_pa=[r['traction_pa'] for r in records],
            **{key:sum(r[key] for r in records) for key in ('storage_j','dissipation_increment_j','released_storage_j','return_projection_loss_j')})
        return np.asarray(u).copy(),history,summary

    def restart_contract(self):
        contract=super().restart_contract()
        contract['friction_profile']=PROFILE
        contract['pressure_activation']='weak_P1_numerical_support_no_geometric_friction_clip_v1'
        contract['finite_material_gap_error_m']=self.friction_settings.finite_material_gap_error_m
        contract['hard_closed_point_control']='compatibility_default_only_not_weak_acceptance'
        return contract
