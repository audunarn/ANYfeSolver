"""Ideal physical source-ray gap over enclosed S3 skins, not computed Jet2 gap.

Only the canonical mesh friction/control-lift owner installs this certificate.
Coverage/chart checks remain prerequisites; no legacy full-pair fallback.
"""
from fractions import Fraction as F
import math
import numpy as np
from .s3_friction_coverage import mapped_path, vsub, magnitude, dot_axis, positive
from .s3_friction_moving_coverage import _raw_normal, _dot
from .s3_friction_mesh_coverage import mesh_moving_coverage, restrict
from .s3_friction_dyadic_interval import outward_dyadic_160
from . import s3_friction_chart_interval as I

METHOD='covered_source_ray_interval_v1'

def pair_lower(source,target):
    ns=_raw_normal(source);nt=_raw_normal(target)
    denominator=I.box(_dot(ns,nt))[0]
    if denominator<=0:raise ValueError('uncertified source-ray normal alignment')
    norm_upper=sum(map(magnitude,ns))
    lower=F(0)
    for corner in source:
        numerator=I.box(_dot(vsub(corner,target[0]),nt))[0]
        if numerator<0:lower=min(lower,numerator*norm_upper/denominator)
    value=float(lower)
    return math.nextafter(value,-math.inf) if F(value)>lower else value


def _constant(points):
    return [tuple(tuple(((F(float(x)),F(float(x))),(F(0),F(0))) for x in p) for p in face) for face in points]


def endpoint_lower(owner,positions):
    owner._require_prepared_geometry()
    source,target=positions
    sources=_constant([source[si] for si in owner.normal.law.source_face_indices])
    targets=dict(zip(owner.target_adapters,_constant([target[ti] for ti in owner.normal.law.target_face_indices])))
    zero=np.zeros(owner.n_dof)
    with outward_dyadic_160():
        if not mesh_moving_coverage(owner,zero,zero,physical_paths=(sources,targets,[])):
            raise ValueError('physical source-ray endpoint union/chart is uncertified')
        for face,adapter in zip(sources,owner.source_adapters):
            ref=np.cross(adapter.reference_positions[1]-adapter.reference_positions[0],adapter.reference_positions[2]-adapter.reference_positions[0])
            if not positive(dot_axis(_raw_normal(face),tuple(F(float(x)) for x in ref))):
                raise ValueError("source reference polarity unresolved")
        return min(pair_lower(s,t) for s in sources for t in targets.values())


def path_lower(owner,first,last):
    owner._require_prepared_geometry()
    owner.require_friction_path(first,last)
    with outward_dyadic_160():
        sources=[mapped_path(a,first,last)[0] for a in owner.source_adapters]
        targets=[mapped_path(a,first,last)[0] for a in owner.target_adapters.values()]
        # Restrict the original enclosed map polynomial, not rounded child chords.
        for level in range(min(8, owner.normal.settings.maximum_swept_subdivisions)+1):
            bounds=[]
            try:
                for j in range(2**level):
                    a,b=F(j,2**level),F(j+1,2**level)
                    rv=lambda v:tuple(restrict(p,a,b) for p in v)
                    ss=[tuple(rv(v) for v in f) for f in sources]
                    tt=[tuple(rv(v) for v in f) for f in targets]
                    for s,adapter in zip(ss,owner.source_adapters):
                        ref=np.cross(adapter.reference_positions[1]-adapter.reference_positions[0],adapter.reference_positions[2]-adapter.reference_positions[0])
                        if not positive(dot_axis(_raw_normal(s),tuple(F(float(x)) for x in ref))):
                            raise ValueError('source reference polarity unresolved')
                    bounds.extend(pair_lower(s,t) for s in ss for t in tt)
            except ValueError:
                continue
            lower=min(bounds)
            if lower>=-owner.normal.gap_tolerance:return lower
        return -math.inf
