"""Full-face hard-normal S3 material friction, fixed committed history."""
from dataclasses import replace
import numpy as np
from scipy import sparse
from .s3_sliding_friction import S3SlidingFrictionManager
from .s3_friction_moving_coverage import full_moving_coverage, verified_norm

PROFILE = "S3_FULL_FACE_HARD_MATERIAL_FRICTION_STATIC_DEV_V0"


class S3MovingNormalFrictionManager(S3SlidingFrictionManager):
    def __init__(self,model,normal,settings):
        super().__init__(model,normal,settings)
        self.translation_dofs=np.asarray([model.mesh.dof_manager.get_node_dofs(n)[:3]
            for n in normal.law.source_node_ids+normal.law.target_node_ids],dtype=int).reshape(-1)

    def require_friction_path(self, first, last):
        if not full_moving_coverage(self, first, last):
            raise ValueError("moving-normal full-face friction path is unsafe or uncertified")

    def admissible_step_fraction(self, first, last):
        # The legacy slider's shortening guard deliberately requires a fixed
        # target normal. This named owner uses the moving-chart certificate.
        fraction=self.normal.admissible_step_fraction(first,last)
        for _ in range(self.normal.settings.maximum_swept_subdivisions+1):
            candidate=np.asarray(first)+fraction*(np.asarray(last)-first)
            if (fraction>0 and full_moving_coverage(self,first,candidate)
                    and full_moving_coverage(self,self.committed_displacement,candidate)):
                return fraction
            fraction*=.5
        return 0.

    def _patch(self, u, **kwargs):
        corners,n,frame,origin=super()._patch(u,**kwargs)
        c=next(iter(corners.values()))
        raw=np.cross(c[1]-c[0],c[2]-c[0])
        n=raw/verified_norm(raw)
        first=frame[:,0]
        return corners,n,np.column_stack((first,np.cross(n,first),n)),origin

    def _force(self,u,z,*,tangent_topology=None):
        source,target=self.normal._positions(u)
        state=self.normal.law.nodal_force_linearization(source,target,z,tangent=False).state
        options={} if tangent_topology is None else dict(
            branches=tangent_topology[0],linearized_elements=tangent_topology[1],
            linearized_barycentric=tangent_topology[2],linearized_touch=tangent_topology[3],
            validate=False)
        return self._friction(u,normal_state=state,**options)

    def linearize_contact(self,u,z,*,tangent=True):
        from .semismooth import linearize
        self.require_friction_path(self.committed_displacement,u)
        normal=linearize(self.normal,u,z,tangent=tangent)
        force,history,records=self._force(u,z)
        if not tangent:
            return replace(normal,force=normal.force+force,
                           local_force=normal.local_force+force[self.translation_dofs])
        topology=([r['branch'] for r in records],[r['target_element'] for r in records],
                  [r['target_barycentric'] for r in records],[r['first_touch'] for r in records])
        if any(r['sliding_rim'] for r in records):
            raise ValueError("moving-normal friction rim tangent is unsupported")
        evaluations=1+2*(len(self.columns)+len(z))
        if evaluations>self.friction_settings.maximum_candidate_evaluations:
            raise MemoryError("moving-normal coupled friction derivative exceeds evaluation cap")
        kq=np.zeros((len(self.work_columns),len(self.columns)))
        kz_local=np.zeros((len(self.work_columns),len(z)))
        for local,column in enumerate(self.columns):
            delta=np.zeros(self.n_dof);delta[column]=self.friction_settings.steps_by_dof[column]
            kq[:,local]=(self._force(np.asarray(u)+delta,z,tangent_topology=topology)[0][self.work_columns]
                         -self._force(np.asarray(u)-delta,z,tangent_topology=topology)[0][self.work_columns])/(2*delta[column])
        for j in range(len(z)):
            delta=np.zeros(len(z));delta[j]=self.friction_settings.pressure_step_n
            kz_local[:,j]=(self._force(u,np.asarray(z)+delta,tangent_topology=topology)[0][self.work_columns]
                     -self._force(u,np.asarray(z)-delta,tangent_topology=topology)[0][self.work_columns])/(2*delta[j])
        rows,cols=np.nonzero(kq)
        kq=sparse.coo_matrix((kq[rows,cols],(self.work_columns[rows],np.asarray(self.columns)[cols])),
                            shape=(self.n_dof,self.n_dof)).tocsr()
        kz=np.zeros((self.n_dof,len(z)));kz[self.work_columns]=kz_local
        return replace(normal,force=normal.force+force,
            tangent=normal.tangent+kq,
            local_force=normal.local_force+force[self.translation_dofs],
            force_pressure_gradient=-normal.gap_gradient.T+kz)

    def evaluate(self,u,*,tangent=True):
        source,target=self.normal._positions(u)
        state=self.normal.law.geometry.evaluate(source,target)
        z=np.asarray(self.normal.normal_multipliers)*state.reference_support
        contact=self.linearize_contact(u,z,tangent=tangent)
        _force,_tangent,certificate=self.normal.evaluate(u,tangent=False)
        return contact.force,contact.tangent,certificate

    def validate_successor(self,u,z):
        self.require_friction_path(self.committed_displacement,u)
        z=np.asarray(z,float)
        if np.any(z<0.): raise ValueError("moving-normal friction requires nonnegative supported forces")
        source,target=self.normal._positions(u)
        c=next(iter(self.target_adapters.values())).current_positions(u)
        normal=np.cross(c[1]-c[0],c[2]-c[0]);normal/=verified_norm(normal)
        dense_points=self.point_reference+self.point_maps@np.asarray(u)[self.work_columns]
        gaps=(np.concatenate((source,dense_points))-c[0])@normal
        if np.all(z==0.):
            if np.min(gaps)<=0.: raise ValueError("released moving-normal state is not strictly open")
        elif (not np.all(z>0.) or
              np.max(np.abs(gaps))>self.friction_settings.closed_point_gap_tolerance_m):
            raise ValueError("partial pressure/nonzero point-gap moving friction is unsupported")
        _force,history,records=self._force(u,z)
        return np.asarray(u).copy(),history,dict(profile=PROFILE,
            traction_measure='force_per_source_reference_area',
            pressure_measure='mortar_lambda_over_source_target_normal_cosine_per_source_reference_area',
            branch=[r['branch'] for r in records],
            point_gap_m=gaps.tolist(),supported_force_n=z.tolist(),
            traction_pa=[r['traction_pa'] for r in records],
            **{key:sum(r[key] for r in records) for key in
               ('storage_j','dissipation_increment_j','released_storage_j','return_projection_loss_j')})

    def restart_contract(self):
        contract=super().restart_contract()
        contract['friction_profile']=PROFILE
        contract['material_path']='single_target_affine_map_quartic_interval_roundoff_v1'
        contract['pressure_measure']='source_reference_area_normal_density_lambda_over_cosine'
        return contract
