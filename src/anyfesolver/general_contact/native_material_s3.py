"""Straight interior retained material patch / small-rotation physical S3 skin.

Fixed reference-area normal penalty; no line-law substitution, caps, friction or
native dynamics. All 42 native and authoritative shell work columns remain.
"""
from dataclasses import dataclass
from math import isfinite
import numpy as np
from scipy import sparse
from anyfesolver._ge_beam3_p5_seeded.core import sha
from .native_retained_station import _readonly
from .native_surface_extrema import straight_patch_plane_extrema,straight_patch_s3_gap
from .native_surface_normal import StraightNormalIntegrationSettings,integrate_straight_normal
from .native_surface_quadrature import NativeRoundSurfaceRule,certify_station_orientation_path
from .native_surface_s3_friction import NativeS3MaterialConfiguration
from .s3_friction_point import S3SkinMaterialPoint
from .s3_skin_work import S3SkinWorkAdapter

PROFILE='NATIVE_B3_GE_RETAINED_MATERIAL_PATCH_S3_STATIC_DEV_V1'


def _skin_translation_dofs(shell):
    """Recover vertex order from the exact six-coordinate S3 work contract.

    Translation columns, rather than material node labels or corner coordinates,
    determine authoritative force order. Verify the complete offset rotation map.
    """
    mapping=shell.work_map
    if (not sparse.isspmatrix_csr(mapping) or mapping.shape[0]!=9
            or mapping.shape[1]%6 or mapping.shape[1]<18):
        raise ValueError('material S3 requires the authoritative six-coordinate shell work map')
    columns=[]
    for row in range(9):
        begin,end=mapping.indptr[row:row+2]
        indices=mapping.indices[begin:end];values=mapping.data[begin:end]
        positions=np.flatnonzero(indices%6<3)
        if (len(positions)!=1 or values[positions[0]]!=1.
                or indices[positions[0]]%6!=row%3):
            raise ValueError('material S3 translation columns are not authoritative')
        columns.append(int(indices[positions[0]]))
    dofs=np.array(columns,dtype=int).reshape(3,3)
    bases=dofs[:,0]
    if len(set(bases))!=3 or not np.array_equal(dofs,bases[:,None]+np.arange(3)):
        raise ValueError('material S3 vertex translation blocks are ambiguous')
    rows=[];cols=[];data=[]
    for vertex,base in enumerate(bases):
        for component in range(3):
            rows.append(3*vertex+component);cols.append(int(base+component));data.append(1.)
        for rotation in range(3):
            variation=shell.physical_offset*np.cross(np.eye(3)[rotation],shell.director)
            for component,value in enumerate(variation):
                if value:
                    rows.append(3*vertex+component);cols.append(int(base+3+rotation));data.append(float(value))
    expected=sparse.coo_matrix((data,(rows,cols)),shape=mapping.shape).tocsr()
    if (mapping-expected).nnz:
        raise ValueError('material S3 offset rotation map is not authoritative')
    dofs.setflags(write=False)
    return dofs


class NativeMaterialS3Patch:
    """Model-bound source patch and explicit area-law/geometry controls."""
    def __init__(self,rule,beam_node_ids,*,normal_stiffness_Pa_per_m,
                 integration_settings,geometry_tolerance_m,rim_clearance_m):
        if type(rule) is not NativeRoundSurfaceRule or type(integration_settings) is not StraightNormalIntegrationSettings:
            raise TypeError('exact material rule and normal integration settings required')
        lo,hi=rule.definition['xi_range']
        if not(-1.<lo<hi<0. or 0.<lo<hi<1.):
            raise ValueError('material S3 patch must be strictly interior to one half-cell')
        if (type(beam_node_ids) is not tuple or len(beam_node_ids)!=3 or len(set(beam_node_ids))!=3
                or any(type(n) is not int or n<1 for n in beam_node_ids)):
            raise ValueError('three distinct positive native node identities required')
        values=(normal_stiffness_Pa_per_m,geometry_tolerance_m,rim_clearance_m)
        if any(isinstance(v,(bool,np.bool_)) or not isfinite(float(v)) or float(v)<=0. for v in values):
            raise ValueError('positive finite explicit area stiffness / geometry controls required')
        self.rule=rule;self.beam_node_ids=beam_node_ids
        self.beam_reference=rule.samples[0].station.reference
        self.beam_element_id=rule.samples[0].station.element_id
        self.radius=float(rule.definition['radius']);self.cell=0 if hi<0. else 1
        self.stiffness=float(normal_stiffness_Pa_per_m)
        self.geometry_tolerance=float(geometry_tolerance_m);self.rim_clearance=float(rim_clearance_m)
        self.settings=integration_settings;self.identity=self._fingerprint()

    def definition(self):
        from dataclasses import asdict
        return dict(profile=PROFILE,rule=self.rule.definition,beam_nodes=self.beam_node_ids,
            beam_element=self.beam_element_id,reference=self.beam_reference.canonical_data(),
            radius_m=self.radius,cell=self.cell,normal_stiffness_Pa_per_m=self.stiffness,
            integration=asdict(self.settings),geometry_tolerance_m=self.geometry_tolerance,
            rim_clearance_m=self.rim_clearance,measure='fixed_reference_material_area',
            numerical_revision='precise_angular_projection_radius_reference_full_shell_work_v3',
            wrench_origin_m=(0.,0.,0.))

    def _fingerprint(self):return sha(self.definition())


@dataclass(frozen=True)
class NativeMaterialS3PhysicalResult:
    potential_j: float
    beam_force_n: np.ndarray
    shell_force_n: np.ndarray
    beam_moment_nm: np.ndarray
    shell_moment_nm: np.ndarray
    shell_vertex_force_n: np.ndarray
    minimum_gap_m: float
    minimum_rim_clearance_m: float
    active_intervals: None  # Line intervals are not a material-area result.
    integration_intervals: int
    integration_evaluations: int
    energy_error_j: float
    force_error_n: float
    tangent_error_n_per_m: float
    minimum_gap_estimate_m: float


@dataclass(frozen=True)
class NativeMaterialS3WorkResult:
    contact: NativeMaterialS3PhysicalResult
    residual: np.ndarray
    tangent: sparse.csr_matrix
    profile: str = PROFILE


class NativeRetainedMaterialS3Work:
    """Authoritative area work, including moving target normal and offsets."""
    def __init__(self,contact,shell):
        if type(contact) is not NativeMaterialS3Patch or type(shell) is not S3SkinWorkAdapter:
            raise TypeError('exact material patch and physical S3 skin required')
        if shell.side!='top':
            raise ValueError('material patch requires a top S3 skin')
        _skin_translation_dofs(shell)
        reference=shell.reference_positions
        normal=np.cross(reference[1]-reference[0],reference[2]-reference[0])
        area=float(np.linalg.norm(normal))
        if (not np.isfinite(area) or area<=contact.geometry_tolerance**2
                or float(normal@shell.director)<=256.*np.finfo(float).eps*area):
            raise ValueError('material S3 top skin requires positive director/reference orientation')
        self.contact=contact;self.shell=shell
        self.target=S3SkinMaterialPoint(shell,(1/3,1/3,1/3))
        self.identity=self._fingerprint()

    @property
    def work_map(self):
        """Geometry provenance for the shared owner; area work is already mapped."""
        return sparse.block_diag((sparse.eye(42,format='csr'),self.shell.work_map),format='csr')

    @property
    def shell_translation_dofs(self):
        return _skin_translation_dofs(self.shell)

    def _fingerprint(self):
        s=self.shell;m=s.work_map
        return sha(dict(profile=PROFILE,contact=self.contact._fingerprint(),
            body=s.body_id,element=s.element_id,side=s.side,nodes=s.node_ids,
            director=s.director,offset=s.physical_offset,reference=s.reference_positions,
            map_shape=m.shape,map_data=m.data,map_indices=m.indices,map_indptr=m.indptr))

    def _configuration(self,positions,position_low,rotations,u):
        if (self._fingerprint()!=self.identity or self.contact._fingerprint()!=self.contact.identity
                or sha(self.contact.rule.samples[0].station.reference.canonical_data())
                   !=sha(self.contact.beam_reference.canonical_data())
                or self.target.adapter is not self.shell):
            raise ValueError('native material S3 definition/settings changed')
        return NativeS3MaterialConfiguration(positions,position_low,rotations,u)

    def _geometry(self,pose):
        k=self.contact
        corners=self.shell.current_positions(pose.shell_displacement)
        edges=[corners[(i+1)%3]-corners[i] for i in range(3)]
        if (min(np.linalg.norm(e) for e in edges)<=k.geometry_tolerance
                or np.linalg.norm(np.cross(edges[0],-edges[2]))<=k.geometry_tolerance**2):
            raise ValueError('native material S3 has unresolved target geometry')
        extent=straight_patch_s3_gap(k.rule,self.target,pose)
        normal=np.asarray(self.target.physical_pose(pose.shell_displacement).material_frame)[:,2]
        clearance=min(straight_patch_plane_extrema(k.rule,pose,corners[i],
            np.cross(normal,(corners[(i+1)%3]-corners[i])/np.linalg.norm(corners[(i+1)%3]-corners[i]))).lower_bound
            for i in range(3))
        if clearance<=k.rim_clearance:
            raise ValueError('native material S3 patch approaches unadmitted physical rim')
        return extent,clearance

    def geometry_gap(self,positions,position_low,rotations,u):
        extent,clearance=self._geometry(self._configuration(positions,position_low,rotations,u))
        return extent.lower_bound,clearance

    def certify_orientation_path(self,start,translations,angular,shell_increment,maximum_intervals):
        return certify_station_orientation_path(self.contact.rule.samples[0].station,start,
            translations,angular,shell_increment,maximum_intervals=maximum_intervals)

    def evaluate(self,positions,position_low,rotations,u):
        pose=self._configuration(positions,position_low,rotations,u)
        extent,clearance=self._geometry(pose);k=self.contact
        integral=integrate_straight_normal(k.rule,self.target,pose,k.stiffness,k.settings,
            compensated_projection=True)
        residual=integral.residual
        beam_forces=np.array([-residual[6*i:6*i+3] for i in range(3)])
        shell_forces=-residual[42:][self.shell_translation_dofs]
        beam_moment=sum((np.cross(pose.positions[i],beam_forces[i])
            +np.cross(pose.position_low[i],beam_forces[i]) for i in range(3)),start=np.zeros(3))
        beam_moment-=residual[18:24].reshape(2,3).sum(axis=0)
        corners=self.shell.current_positions(u)
        shell_moment=np.cross(corners,shell_forces).sum(axis=0)
        d=integral.diagnostics
        physical=NativeMaterialS3PhysicalResult(integral.normal_storage_j,
            _readonly(beam_forces.sum(axis=0)),_readonly(shell_forces.sum(axis=0)),
            _readonly(beam_moment),_readonly(shell_moment),_readonly(shell_forces),
            extent.lower_bound,clearance,None,d.accepted_intervals,d.integrand_evaluations,
            d.estimated_energy_error,d.estimated_force_error,d.estimated_tangent_error,extent.minimum)
        return NativeMaterialS3WorkResult(physical,_readonly(residual),sparse.csr_matrix(integral.tangent))
