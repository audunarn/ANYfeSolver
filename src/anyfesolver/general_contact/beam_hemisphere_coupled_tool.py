"""Opt-in certified coupled load seed; unchanged actual cylinder-cap work."""
from dataclasses import dataclass, replace
from .core import ContactValidationReport
from .beam_hemisphere_cylinder_motion import (TranslatingHemisphereCylinderManager,
    TranslatingHemisphereCylinderResults, validate_translating_hemisphere_cylinder)

PROFILE='CLASSICAL_B2_B3_HEMISPHERE_COUPLED_TRANSLATING_CYLINDER_STATIC_DEV_V1'


def validate_coupled_hemisphere_tool(model,definition,settings,*,analysis_control='force'):
    if settings.beam_end_cap_work!='hemisphere_coupled_tool_dev':
        return ContactValidationReport('invalid',('explicit coupled tool selection required',),PROFILE)
    report=validate_translating_hemisphere_cylinder(model,definition,
        replace(settings,beam_end_cap_work='hemisphere_support_dev'),analysis_control=analysis_control)
    return ContactValidationReport(report.status,report.issues,PROFILE)


@dataclass(frozen=True)
class CoupledHemisphereToolResults(TranslatingHemisphereCylinderResults):
    def to_dict(self):
        result=super().to_dict()
        result.update(beam_end_cap_work='hemisphere_coupled_tool_dev',
            load_staging='active_origin_one_newton_seed_complete_path_v1')
        return result


class CoupledHemisphereToolManager(TranslatingHemisphereCylinderManager):
    profile=PROFILE
    validator=staticmethod(validate_coupled_hemisphere_tool)
    results_type=CoupledHemisphereToolResults

    def coupled_load_seed_active_origin(self,displacement,factor):
        # First approach remains subject to the existing event/cutback policy.
        # This is an algorithmic query, never a material/contact commit.
        previous=self.path_factor
        try:
            self.set_path_factor(factor)
            result=self.evaluate(displacement,tangent=False)[2]
            return result.active_endpoints>0 or result.active_line_length>0.
        finally:
            self.set_path_factor(previous)

    def coupled_load_seed_pose_admissible(self, displacement, factor):
        # The new-tool/old-structure pose may exceed the penetration budget,
        # but force linearization still needs a certified exterior centreline.
        # A lower enclosure that cannot establish this causes ordinary cutback.
        previous = self.path_factor
        try:
            self.set_path_factor(factor)
            return self._minimum_gap(displacement) > -min(
                self.beam_radius, self.definition.rigid_bodies[0].radius
            )
        finally:
            self.set_path_factor(previous)

    def restart_contract(self):
        result=super().restart_contract()
        result.update(load_staging='active_origin_one_newton_seed_complete_path_v1',
            beam_end_cap_work='hemisphere_coupled_tool_dev')
        result['settings']['beam_end_cap_work']='hemisphere_coupled_tool_dev'
        return result
