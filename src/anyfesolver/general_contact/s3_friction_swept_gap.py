"""Opt-in whole-chord absolute perpendicular gap geometry for S3 mesh friction.

Original rounded-map interval polynomials are restricted, never rebuilt from
sampled/rounded child poses. This certificate does not resolve pressure or
history events and does not qualify floating Jet2 projection values.
"""
from fractions import Fraction as F
import numpy as np
from .s3_friction_weak_pressure import S3MeshWeakPressureFrictionManager
from .s3_friction_coverage import mapped_path, material_point_paths, product, vsub
from .s3_friction_moving_coverage import _raw_normal, _dot
from .s3_friction_mesh_coverage import restrict
from .s3_friction_dyadic_interval import outward_dyadic_160
from . import s3_friction_chart_interval as I

PROFILE='S3_MESH_WEAK_P1_WHOLE_CHORD_GAP_FRICTION_STATIC_DEV_V0'
METHOD='original_map_perpendicular_squared_bernstein_v1'
MAX_SEGMENTS=256
class _EvaluationLimit(ValueError):
    pass

def _difference(a,b):
    zero=(F(0),F(0));size=max(len(a),len(b))
    return tuple(((a[i] if i<len(a) else zero)[0]-(b[i] if i<len(b) else zero)[1],
                  (a[i] if i<len(a) else zero)[1]-(b[i] if i<len(b) else zero)[0]) for i in range(size))

def _poly(p):return tuple(I.compress(v) for v in p)

def _facet_basis(facet):
    normal=tuple(_poly(p) for p in _raw_normal(facet))
    norm2=_poly(_dot(normal,normal))
    if I.box(norm2)[0]<=0:raise ValueError('whole-chord target normal unresolved')
    return normal,norm2

def point_facet_certified(point,facet,budget,*,basis=None):
    """Sufficient squared-distance inequality for an entire interval chord."""
    normal,norm2=_facet_basis(facet) if basis is None else basis
    numerator=_poly(_dot(vsub(point,facet[0]),normal))
    scaled=tuple((lo*budget**2,hi*budget**2) for lo,hi in norm2)
    return I.box(_poly(_difference(scaled,_poly(product(numerator,numerator)))))[0]>=0

def certify_material_gap(owner,first,last):
    """Every original-map corner/dense point against every facet; fail closed."""
    owner._require_prepared_geometry()
    first=np.asarray(first,float);last=np.asarray(last,float)
    if first.shape!=(owner.n_dof,) or last.shape!=first.shape or not np.isfinite([first,last]).all():
        raise ValueError('whole-chord gap requires finite full displacement endpoints')
    budget=F(float(owner.friction_settings.finite_material_gap_error_m))
    limit=owner.friction_settings.maximum_material_gap_evaluations
    if budget<=0 or type(limit) is not int or limit<=0:raise ValueError('invalid whole-chord gap budget/evaluation cap')
    with outward_dyadic_160():
        sources=[mapped_path(a,first,last)[0] for a in owner.source_adapters]
        targets=[mapped_path(a,first,last)[0] for a in owner.target_adapters.values()]
        points=[p for face in sources for p in face]+list(material_point_paths(owner,first,last))
        levels=min(8,owner.normal.settings.maximum_swept_subdivisions)
        evaluations=0
        for level in range(levels+1):
            complete=True
            for j in range(2**level):
                a,b=F(j,2**level),F(j+1,2**level)
                rv=lambda p:tuple(restrict(v,a,b) for v in p)
                pp=[rv(p) for p in points];tt=[tuple(rv(p) for p in t) for t in targets]
                try:
                    bases=[_facet_basis(t) for t in tt]
                    for point in pp:
                        for target,basis in zip(tt,bases):
                            if evaluations>=limit:
                                raise _EvaluationLimit('whole-chord material-gap evaluation cap exhausted')
                            evaluations+=1
                            if not point_facet_certified(point,target,budget,basis=basis):
                                complete=False;break
                        if not complete:break
                except _EvaluationLimit:raise
                except (ValueError,OverflowError):complete=False
                if not complete:break
            if complete:
                return dict(method=METHOD,scope='whole_original_dof_chord_all_corners_dense_all_facets',
                    finite_material_gap_error_m=float(budget),subdivision_level=level,
                    certified_segments=2**level,point_facet_evaluations=evaluations,
                    maximum_segments=min(MAX_SEGMENTS,2**levels),maximum_subdivision_level=levels,
                    maximum_point_facet_evaluations=limit,
                    history_event_scope='not_certified')
    raise ValueError('whole-chord absolute material-gap budget unsafe or unresolved')

class S3MeshWholeChordGapFrictionManager(S3MeshWeakPressureFrictionManager):
    def __init__(self,*args,**kwargs):
        self._whole_gap_success={}
        super().__init__(*args,**kwargs)

    def require_friction_path(self,first,last):
        # Mandatory union/chart/coplanarity/integrity guards remain unchanged.
        super().require_friction_path(first,last)
        key=(np.asarray(first,float).tobytes(),np.asarray(last,float).tobytes())
        if key not in self._whole_gap_success:
            proof=certify_material_gap(self,first,last)
            if len(self._whole_gap_success)>=16:self._whole_gap_success.pop(next(iter(self._whole_gap_success)))
            self._whole_gap_success[key]=proof
        self._last_whole_gap_certificate=self._whole_gap_success[key]

    def validate_successor(self,u,z):
        pose,history,summary=super().validate_successor(u,z)
        summary.update(profile=PROFILE,finite_material_gap_error_scope='whole_original_dof_chord',
            whole_chord_geometry_certificate=dict(self._last_whole_gap_certificate),
            pressure_history_event_scope='not_certified')
        return pose,history,summary

    def restart_contract(self):
        contract=super().restart_contract()
        contract.update(friction_profile=PROFILE,whole_chord_material_gap_method=METHOD,
            whole_chord_maximum_evaluations=self.friction_settings.maximum_material_gap_evaluations,
            whole_chord_maximum_segments=MAX_SEGMENTS,whole_chord_point_scope='all_source_corners_dense_all_target_facets')
        return contract
