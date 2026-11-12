"""Explicit prescribed axis translation with actual cylinder-cap work.

Uses the fixed-cylinder geometry and the existing complete reaction/work map.
The fixed/legacy routes remain separate. Exact affine centres and radius work
are versioned in the opt-in restart contract.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
import numpy as np
from .core import ContactValidationReport, validate_general_contact
from .beam_hemisphere_cylinder import (HemisphereCylinderManager,
    HemisphereCylinderResults, validate_hemisphere_cylinder_contact)

PROFILE='CLASSICAL_B2_B3_HEMISPHERE_TRANSLATING_CYLINDER_STATIC_DEV_V1'


def validate_translating_hemisphere_cylinder(model,definition,settings,*,analysis_control='force'):
    issues=[]
    if len(definition.rigid_bodies)!=1:
        return ContactValidationReport('invalid',('one prescribed cylinder required',),PROFILE)
    body=definition.rigid_bodies[0]
    if body.kind!='cylinder' or body.path_translation is None:
        issues.append('translating cylinder requires explicit prescribed axis translation')
    if analysis_control!='force':
        issues.append('translating actual cylinder-cap profile currently requires force continuation')
    fixed=replace(definition,rigid_bodies=(replace(body,path_translation=None),))
    issues.extend(validate_hemisphere_cylinder_contact(model,fixed,settings).issues)
    issues.extend(validate_general_contact(model,definition,
        replace(settings,beam_end_cap_work='owned_point'),analysis_control=analysis_control).issues)
    return ContactValidationReport('invalid' if issues else 'ok',tuple(issues),PROFILE)


@dataclass(frozen=True)
class TranslatingHemisphereCylinderResults(HemisphereCylinderResults):
    def to_dict(self):
        result=super().to_dict()
        result['rigid_motion_profile']='exact_literal_axis_translation_affine_static_v2'
        return result


class TranslatingHemisphereCylinderManager(HemisphereCylinderManager):
    profile=PROFILE
    validator=staticmethod(validate_translating_hemisphere_cylinder)
    results_type=TranslatingHemisphereCylinderResults

    def point_at(self,factor):
        value=float(factor)
        if not np.isfinite(value) or value<0:
            raise ValueError('radial tool path factor must be finite and nonnegative')
        value=Fraction.from_float(value)
        # Preserve the literal affine path before subtracting global positions.
        # Conversion to float happens only after local geometry is formed.
        point=np.array([Fraction.from_float(float(origin))+value*Fraction.from_float(float(velocity))
            for origin,velocity in zip(self.reference_point,self.path_translation)],dtype=object)
        try:finite=all(np.isfinite(float(coordinate)) for coordinate in point)
        except OverflowError:finite=False
        if not finite:
            raise ValueError('radial tool path position is nonfinite')
        return point

    def tool_position_report(self):
        # Display floats are explicitly approximate. Checkpoint definition and
        # continuation factor reconstruct the exact path; rational output keeps
        # sub-ULP coordinates inspectable and JSON-safe across applications.
        point=self.point_at(self.path_factor)
        return dict(rigid_tool_position=[float(v) for v in point],
            rigid_tool_position_representation='float_display_with_exact_rational_m',
            rigid_tool_position_rational_m=[dict(numerator=str(v.numerator),denominator=str(v.denominator))
                for v in point])

    def tool_work_increment(self,start,end,start_factor,end_factor):
        if not np.any(self.path_translation):return 0.
        previous=self.path_factor
        try:
            self.set_path_factor(start_factor)
            first=np.asarray(self.evaluate(start,tangent=False)[2].rigid_force)
            self.set_path_factor(end_factor)
            last=np.asarray(self.evaluate(end,tangent=False)[2].rigid_force)
        finally:self.set_path_factor(previous)
        delta=Fraction.from_float(float(end_factor))-Fraction.from_float(float(start_factor))
        motion=np.array([float(delta*Fraction.from_float(float(v))) for v in self.path_translation])
        work=float(.5*(first+last)@motion)
        if not np.isfinite(work):raise ValueError('radial beam tool work is nonfinite')
        return work

    def restart_contract(self):
        result=super().restart_contract()
        result['prescribed_rigid_kinematics']='exact_literal_axis_translation_affine_static_v2'
        return result
