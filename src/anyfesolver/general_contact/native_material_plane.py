"""Frictionless retained material-section patch against a fixed analytic plane.

One strictly interior, straight half-cell patch; fixed reference area. Uses the
existing section map, axial active-branch integration and angular error controls.
This is distinct from the centreline-tube line penalty and admits no caps,
friction, native dynamics or generic FEModel insertion.
"""
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import Jet2
from anyfesolver._ge_beam3_p5_seeded.core import sha
from .native_retained_line_plane import NativeRetainedLinePlaneResult
from .native_retained_station import _readonly
from .native_surface_extrema import straight_patch_plane_extrema
from .native_surface_normal import (StraightNormalIntegrationSettings,
    _material_cell_normal_data, _integrate_material_normal_data)
from .native_surface_quadrature import NativeRoundSurfaceRule
from .native_surface_s3_friction import NativeS3MaterialConfiguration

PROFILE='NATIVE_B3_GE_RETAINED_MATERIAL_PATCH_PLANE_STATIC_DEV_V1'


@dataclass(frozen=True)
class NativeMaterialPlaneResult(NativeRetainedLinePlaneResult):
    minimum_gap_estimate_m: float

    def spatial_tangent(self):
        """The shared integral has already converted its section chart once."""
        return self.tangent


class NativeRoundMaterialPlane:
    def __init__(self, rule, *, plane_origin_m, plane_normal,
                 normal_stiffness_Pa_per_m, integration_settings):
        if type(rule) is not NativeRoundSurfaceRule or type(integration_settings) is not StraightNormalIntegrationSettings:
            raise TypeError('exact native surface rule and normal integration settings required')
        lo,hi=rule.definition['xi_range']
        if not (-1.<lo<hi<0. or 0.<lo<hi<1.):
            raise ValueError('material plane patch must be strictly interior to one half-cell')
        if isinstance(normal_stiffness_Pa_per_m,(bool,np.bool_)):
            raise ValueError('dimensioned normal stiffness must be numerical, not boolean')
        stiffness=float(normal_stiffness_Pa_per_m)
        if not isfinite(stiffness) or stiffness<=0.:
            raise ValueError('positive normal_stiffness_Pa_per_m (N/m^3) required')
        origin=np.asarray(plane_origin_m,dtype=float); normal=np.asarray(plane_normal,dtype=float)
        if (origin.shape!=(3,) or normal.shape!=(3,) or not np.all(np.isfinite(origin))
                or not np.all(np.isfinite(normal)) or abs(np.linalg.norm(normal)-1.)>1e-12):
            raise ValueError('finite plane origin and unit normal required')
        self.rule=rule; self.reference=rule.samples[0].station.reference
        self.element_id=rule.samples[0].station.element_id
        self.radius=float(rule.definition['radius'])
        self.origin=_readonly(origin); self.normal=_readonly(normal)
        self.stiffness=stiffness; self.integration_settings=integration_settings
        self.cell=0 if hi<0. else 1
        self.identity=sha(self.definition())
        # Shared exact-reference, full-patch orientation and extrema admission.
        self.minimum_gap(self.reference.coordinates,np.zeros((3,3)),np.tile(np.eye(3),(2,1,1)))

    def definition(self):
        return dict(profile=PROFILE,rule=self.rule.definition,element_id=self.element_id,
            reference=self.reference.canonical_data(),radius_m=self.radius,cell=self.cell,
            plane_origin_m=self.origin,plane_normal=self.normal,
            normal_stiffness_Pa_per_m=self.stiffness,
            integration_settings=asdict(self.integration_settings),
            numerical_gap_revision='precise_angular_projection_partition_grazing_guard_v2',
            measure='fixed_reference_material_area',wrench_origin_m=(0.,0.,0.))

    def _configuration(self, positions, position_low, cell_rotations):
        if sha(self.definition())!=self.identity:
            raise ValueError('native material plane definition/settings changed')
        return NativeS3MaterialConfiguration(positions,position_low,cell_rotations,np.zeros(0))

    def minimum_gap(self,positions,position_low,cell_rotations):
        pose=self._configuration(positions,position_low,cell_rotations)
        return straight_patch_plane_extrema(self.rule,pose,self.origin,self.normal).lower_bound

    def evaluate(self,positions,position_low,cell_rotations):
        pose=self._configuration(positions,position_low,cell_rotations)
        extent=straight_patch_plane_extrema(self.rule,pose,self.origin,self.normal)
        q=[Jet2.variable(0.,i,9) for i in range(9)]
        normal=[Jet2.constant(v,9) for v in self.normal]
        stationary_origin=[Jet2.constant(0.,9) for _ in range(3)]
        mapping=np.zeros((9,42)); cell=self.cell
        mapping[:3,6*cell:6*cell+3]=np.eye(3)
        mapping[3:6,6*(cell+1):6*(cell+1)+3]=np.eye(3)
        mapping[6:9,18+3*cell:21+3*cell]=np.eye(3)
        lo,hi=self.rule.definition['xi_range']
        data=_material_cell_normal_data(self.rule,pose,cell,lo,hi,self.stiffness,
            q,normal,self.origin,stationary_origin,mapping,compensated_projection=True)
        try:integrated=_integrate_material_normal_data([data],42,self.integration_settings)
        except MemoryError as exc:
            if str(exc)!='straight normal integration interval budget exhausted':raise
            exc.contact_diagnostics=dict(category='normal_integration_resource_admission',
                definition=self.definition(),integration=getattr(exc,'integration_diagnostics',None),
                configuration=dict(positions=pose.positions,position_low=pose.position_low,
                                   cell_rotations=pose.cell_rotations))
            raise
        residual=integrated.residual
        forces=np.array([-residual[6*i:6*i+3] for i in range(3)])
        force=forces.sum(axis=0)
        # Include section couples; nodal trace rotations do not carry position.
        moment=sum((np.cross(pose.positions[i],forces[i])+np.cross(pose.position_low[i],forces[i])
                    for i in range(3)),start=np.zeros(3))-residual[18:24].reshape(2,3).sum(axis=0)
        d=integrated.diagnostics
        return NativeMaterialPlaneResult(integrated.normal_storage_j,residual,integrated.tangent,
            _readonly(force),_readonly(-force),_readonly(moment),_readonly(-moment),
            extent.lower_bound,d.accepted_intervals,d.integrand_evaluations,
            d.estimated_energy_error,d.estimated_force_error,d.estimated_tangent_error,extent.minimum)
