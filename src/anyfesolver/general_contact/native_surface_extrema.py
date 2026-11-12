"""Continuous plane-distance extrema for an exactly straight reference patch.

Uses native coordinates directly, independently of contact quadrature. This
bounded profile excludes reference curvature/twist; both cell-interface traces
are checked. It supplies an endpoint geometry check, not a swept certificate.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import atan2, cos, sin, pi, fsum

import numpy as np


@dataclass(frozen=True)
class NativeSurfacePlaneExtrema:
    minimum: float
    maximum: float
    minimum_location: tuple
    maximum_location: tuple
    roundoff_bound: float

    @property
    def lower_bound(self):
        return self.minimum-self.roundoff_bound


def straight_patch_plane_extrema(rule,configuration,origin,normal):
    rule.evaluate(configuration.positions,configuration.position_low,configuration.cell_rotations)
    reference=rule.samples[0].station.reference
    coordinates=reference.coordinates; frames=reference.nodal_triads
    if (any(Fraction(float(coordinates[0,i]))+Fraction(float(coordinates[2,i]))!=2*Fraction(float(coordinates[1,i])) for i in range(3))
            or not np.array_equal(frames,np.tile(frames[0],(3,1,1)))):
        raise ValueError('continuous patch extrema require exact straight untwisted reference')
    origin=np.asarray(origin,dtype=float); normal=np.asarray(normal,dtype=float)
    if (origin.shape!=(3,) or normal.shape!=(3,) or not np.all(np.isfinite(origin))
            or not np.all(np.isfinite(normal)) or abs(np.linalg.norm(normal)-1.)>1e-12):
        raise ValueError('finite plane origin and unit normal required')
    lo,hi=rule.definition['xi_range']; alo,ahi=rule.definition['angle_range']
    radius=rule.definition['radius']; values=[]
    for cell,left,right in ((0,max(lo,-1.),min(hi,0.)),(1,max(lo,0.),min(hi,1.))):
        if right<=left:continue
        frame=configuration.cell_rotations[cell]@frames[0]
        a=radius*float(normal@frame[:,1]); b=radius*float(normal@frame[:,2])
        phase=atan2(b,a)
        angles=[alo,ahi]+[phase+k*pi for k in range(-2,5) if alo<=phase+k*pi<=ahi]
        for xi in (left,right):
            t=xi+1 if cell==0 else xi
            base=np.array([fsum(((1-t)*float(configuration.positions[cell,i]),
                (1-t)*float(configuration.position_low[cell,i]),t*float(configuration.positions[cell+1,i]),
                t*float(configuration.position_low[cell+1,i]),-float(origin[i]))) for i in range(3)])
            offset=float(normal@base)
            for angle in angles:
                values.append((fsum((offset,a*cos(angle),b*sin(angle))),(cell,float(xi),float(angle))))
    low=min(values,key=lambda item:item[0]); high=max(values,key=lambda item:item[0])
    scale=max(1.,np.max(np.abs(configuration.positions)),np.max(np.abs(origin)),radius)
    padding=512*np.finfo(float).eps*scale
    return NativeSurfacePlaneExtrema(low[0],high[0],low[1],high[1],padding)


def straight_patch_s3_gap(rule,target,configuration):
    """Full patch lies strictly within the target; return signed gap enclosure."""
    corners=target.adapter.current_positions(configuration.shell_displacement)
    frame=np.asarray(target.physical_pose(configuration.shell_displacement).material_frame)
    normal=frame[:,2]
    altitude=max(np.linalg.norm(corners[(i+1)%3]-corners[i]) for i in range(3))
    for i in range(3):
        edge=corners[(i+1)%3]-corners[i]
        inward=np.cross(normal,edge/np.linalg.norm(edge))
        minimum=straight_patch_plane_extrema(rule,configuration,corners[i],inward)
        if minimum.lower_bound<=1e-12*altitude:
            raise ValueError('native/S3 material history path requires subdivision: continuous patch target ownership')
    return straight_patch_plane_extrema(rule,configuration,corners[0],normal)
