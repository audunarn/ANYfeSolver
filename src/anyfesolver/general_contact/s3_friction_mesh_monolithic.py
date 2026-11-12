"""Connected conforming S3 full-coverage hard-normal material friction."""
import numpy as np
from .s3_friction_monolithic import S3MovingNormalFrictionManager
from .s3_sliding_friction import S3SlidingFrictionManager
from .s3_friction_moving_coverage import verified_norm
from .s3_friction_mesh_coverage import mesh_moving_coverage

PROFILE="S3_FULL_MESH_HARD_MATERIAL_FRICTION_STATIC_DEV_V0"


def _geometry_identity(value,depth=0):
    """Exact internal prepared-geometry identity, including nested pair laws."""
    if depth>12:raise ValueError("mesh geometry identity exceeds bounded nesting")
    if isinstance(value,np.generic):return (value.dtype.str,value.tobytes())
    if isinstance(value,np.ndarray):return (value.shape,value.dtype.str,value.tobytes())
    if hasattr(value,'tocsr'):
        a=value.tocsr();return (a.shape,_geometry_identity(a.data),_geometry_identity(a.indices),_geometry_identity(a.indptr))
    if value is None or type(value) in {str,int,float,bool}:return (type(value).__name__,repr(value))
    if isinstance(value,(tuple,list)):return tuple(_geometry_identity(v,depth+1) for v in value)
    if isinstance(value,dict):return tuple((repr(k),_geometry_identity(v,depth+1)) for k,v in sorted(value.items(),key=lambda x:repr(x[0])))
    if hasattr(value,'__dict__'):return (type(value).__module__,type(value).__qualname__,_geometry_identity(vars(value),depth+1))
    raise ValueError("unbound mesh prepared-geometry operand")


class S3MeshMovingNormalFrictionManager(S3MovingNormalFrictionManager):
    def __init__(self,model,normal,settings):
        super().__init__(model,normal,settings)
        self._mesh_path_success={}
        self._prepared_normal_geometry=self._normal_geometry()
        self._prepared_friction_geometry=self._friction_geometry()
        self._physical_ray_expected=normal.settings.path_predictor == "control_projection_dev"
        # Source is also one conforming connected material disk. Shared work
        # rows were checked by the normal/physical-adapter constructor.
        owners={}
        for a in self.source_adapters:
            for j,n in enumerate(a.node_ids):
                e=(n,a.node_ids[(j+1)%3]);owners.setdefault(tuple(sorted(e)),[]).append(e)
        if any(len(v)>2 or (len(v)==2 and v[0]!=v[1][::-1]) for v in owners.values()):
            raise ValueError("source mesh friction requires a conforming oriented disk")
        boundary=[v[0] for v in owners.values() if len(v)==1];following=dict(boundary)
        nodes=set(n for a in self.source_adapters for n in a.node_ids)
        seen=set();n=boundary[0][0] if boundary else None
        while n is not None and n not in seen and n in following:seen.add(n);n=following[n]
        if not boundary or len(following)!=len(boundary) or n!=boundary[0][0] or len(seen)!=len(boundary) or len(nodes)-len(owners)+len(self.source_adapters)!=1:
            raise ValueError("source mesh friction requires one connected disk")
        if normal.settings.path_predictor == "control_projection_dev":
            normal._mesh_physical_ray_owner=self
            normal._mesh_physical_ray_bound_owner=self

    def _patch(self,u,**kwargs):
        corners,n,frame,origin=S3SlidingFrictionManager._patch(self,u,**kwargs)
        raw=sum((np.cross(c[1]-c[0],c[2]-c[0]) for c in corners.values()),np.zeros(3))
        n=raw/verified_norm(raw);first=frame[:,0]
        return corners,n,np.column_stack((first,np.cross(n,first),n)),origin

    def _normal_geometry(self):
        g=self.normal.law.geometry
        return _geometry_identity((self.normal.settings,self.normal.profile,self.normal.reference,self.normal.map,g.reference_source,g.reference_target,
            g.source_faces,g.target_faces,g.source_face_indices,g.target_face_indices,
            g.source_node_ids,g.target_node_ids,g.pair_laws))

    def _friction_geometry(self):
        return _geometry_identity((self.source_adapters,self.target_adapters,self.target_maps,
            self.points,self.point_reference,self.point_maps,self.work_columns,
            self.pressure_indices,self.weights,self.translation_dofs,
            tuple(sorted(self.boundary_edges)),self.friction_settings,self.n_dof))

    def _require_prepared_geometry(self):
        if self._physical_ray_expected and (getattr(self.normal,"_mesh_physical_ray_bound_owner",None) is not self or getattr(self.normal,"_mesh_physical_ray_owner",None) is not self):
            raise ValueError("prepared mesh physical-ray owner binding changed")
        if self._normal_geometry()!=self._prepared_normal_geometry:
            raise ValueError("prepared mesh normal geometry changed after material-map binding")
        if self._friction_geometry()!=self._prepared_friction_geometry:
            raise ValueError("prepared mesh friction geometry changed after material-map binding")
    def require_friction_path(self,first,last):
        self._require_prepared_geometry()
        # Pure, bounded certificate reuse. Bind ALL mutable geometric operands
        # as well as the actual binary chord; no physical history is cached.
        arrays=[np.asarray(first,float),np.asarray(last,float),self.point_reference,self.point_maps,
                np.asarray(self.work_columns),np.asarray([self.friction_settings.geometry_relative_tolerance])]
        for a in (*self.source_adapters,*self.target_adapters.values()):
            arrays.extend((np.asarray([a.element_id,*a.node_ids]),a.reference_positions,
                           a.work_map.data,a.work_map.indices,a.work_map.indptr))
        arrays=[np.asarray(x) for x in arrays]
        prepared=_geometry_identity((self.normal.reference,self.normal.map,
            self.normal.law,self.normal.settings,self.normal.source_coordinates))
        key=(_geometry_identity(arrays),prepared)
        if key in self._mesh_path_success:return
        if self.normal.settings.path_predictor == 'control_projection_dev':
            from .s3_friction_dyadic_interval import outward_dyadic_160
            with outward_dyadic_160():
                covered=mesh_moving_coverage(self,first,last)
        else:
            covered=mesh_moving_coverage(self,first,last)
        if not covered:
            raise ValueError("moving mesh union/coplanarity/rounded-chart path is unsafe or uncertified")
        # Endpoint branch differentiability is distinct from whole-union coverage.
        # Evaluate the authoritative normal geometry before ANY material trial.
        self.normal.law.nodal_force_linearization(*self.normal._positions(last),
            np.zeros(len(self.normal.law.pressure_mode_ids)),tangent=False)
        if len(self._mesh_path_success)>=16:self._mesh_path_success.pop(next(iter(self._mesh_path_success)))
        self._mesh_path_success[key]=True

    def admissible_step_fraction(self,first,last):
        fraction=self.normal.admissible_step_fraction(first,last)
        for _ in range(self.normal.settings.maximum_swept_subdivisions+1):
            candidate=np.asarray(first)+fraction*(np.asarray(last)-first)
            try:
                self.require_friction_path(first,candidate)
                self.require_friction_path(self.committed_displacement,candidate)
                return fraction
            except ValueError: fraction*=.5
        return 0.

    def validate_successor(self,u,z):
        self.require_friction_path(self.committed_displacement,u)
        z=np.asarray(z,float)
        if np.any(z<0):raise ValueError("mesh friction requires nonnegative supported forces")
        corners,n,frame,origin=self._patch(u)
        source,_target=self.normal._positions(u)
        dense=self.point_reference+self.point_maps@np.asarray(u)[self.work_columns]
        points=np.concatenate((source,dense));gaps=(points-origin)@n
        facet_gaps=[]
        for element,c in corners.items():
            raw=np.cross(c[1]-c[0],c[2]-c[0]);facet_n=raw/verified_norm(raw)
            facet_gaps.append((element,(points-c[0])@facet_n))
        if np.all(z==0):
            if any(np.min(g)<=0 for _e,g in facet_gaps):raise ValueError("released mesh friction must be strictly open on every facet")
        elif not np.all(z>0) or any(np.max(np.abs(g))>self.friction_settings.closed_point_gap_tolerance_m for _e,g in facet_gaps):
            raise ValueError("partial pressure/nonzero material-point mesh friction unsupported")
        _force,history,records=self._force(u,z)
        return np.asarray(u).copy(),history,dict(profile=PROFILE,
            traction_measure='force_per_source_reference_area',
            pressure_measure='mortar_lambda_over_source_target_normal_cosine_per_source_reference_area',
            branch=[r['branch'] for r in records],point_gap_m=gaps.tolist(),supported_force_n=z.tolist(),
            facet_point_gap_m={str(e):g.tolist() for e,g in facet_gaps},
            target_elements=[r['target_element'] for r in records],
            traction_pa=[r['traction_pa'] for r in records],
            **{key:sum(r[key] for r in records) for key in ('storage_j','dissipation_increment_j','released_storage_j','return_projection_loss_j')})

    def restart_contract(self):
        contract=super().restart_contract()
        contract['friction_profile']=PROFILE
        contract['material_path']='conforming_convex_union_affine_map_interval_roundoff_v1'
        if self.normal.settings.path_predictor == 'control_projection_dev':
            from .s3_friction_dyadic_interval import METHOD
            contract['material_path_interval']=METHOD
            from .s3_friction_mesh_gap import METHOD as GAP_METHOD
            contract['physical_gap_certificate']=GAP_METHOD
        return contract
