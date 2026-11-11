"""Sum explicitly owned round-beam/fixed-rigid work on shared coordinates.

Reuses the single-pair kernels; no automatic pairs, moving tools, friction,
pressure multipliers or new structural solver are introduced.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport

PROFILE = "MULTIPLE_EXPLICIT_ROUND_BEAM_FIXED_RIGID_STATIC_DEV_V0"
PROFILE_ENDS = "MULTIPLE_EXPLICIT_ROUND_BEAM_FIXED_RIGID_PHYSICAL_ENDS_STATIC_DEV_V1"
PROFILE_SUPPORT = "MULTIPLE_EXPLICIT_B2_B3_HEMISPHERE_SUPPORT_PLANES_STATIC_DEV_V1"
PROFILE_SUPPORT_CYLINDER = "MULTIPLE_EXPLICIT_B2_B3_HEMISPHERE_SUPPORT_ANALYTIC_CYLINDERS_STATIC_DEV_V1"
PROFILE_SUPPORT_ANALYTIC = "MULTIPLE_EXPLICIT_B2_B3_HEMISPHERE_SUPPORT_PLANES_SPHERES_STATIC_DEV_V1"
_SUPPORT_PROFILES = {PROFILE_SUPPORT, PROFILE_SUPPORT_ANALYTIC, PROFILE_SUPPORT_CYLINDER}
_END_PROFILES = {PROFILE_ENDS, *_SUPPORT_PROFILES}


def _profile(definition, settings=None):
    if settings is not None and settings.beam_end_cap_work == 'hemisphere_support_dev':
        if any(body.kind=='cylinder' for body in definition.rigid_bodies):return PROFILE_SUPPORT_CYLINDER
        return (PROFILE_SUPPORT if all(body.kind=='plane' for body in definition.rigid_bodies)
                else PROFILE_SUPPORT_ANALYTIC)
    return PROFILE_ENDS if (any(surface.beam_end_caps=='hemisphere' for surface in definition.surfaces)
        or any(body.kind=='capped_cylinder' for body in definition.rigid_bodies)) else PROFILE


def _parts(model, definition, settings, analysis_control):
    from .core import validate_general_contact

    if analysis_control not in {"force", "displacement"}:
        raise ValueError("multiple beam interactions require force or displacement control")
    if settings.enforcement_solver != "nested_augmented":
        raise ValueError("multiple beam line penalties do not admit monolithic pressure enforcement")
    if len(definition.interactions) < 2 or not definition.rigid_bodies:
        raise ValueError("multiple beam-rigid profile requires at least two explicit interactions")
    surfaces = {surface.name: surface for surface in definition.surfaces}
    bodies = {body.name: body for body in definition.rigid_bodies}
    used_surfaces, used_bodies, owned, parts = set(), set(), set(), []
    candidate_count = 0
    for interaction in sorted(definition.interactions, key=lambda item: item.name):
        if interaction.surface not in surfaces or interaction.target not in bodies:
            raise ValueError("each beam interaction must name a declared beam and rigid body")
        source, body = surfaces[interaction.surface], bodies[interaction.target]
        if settings.beam_end_cap_work == 'hemisphere_support_dev' and body.kind not in {'plane','sphere','cylinder'}:
            raise ValueError('multiple hemisphere-support profile requires fixed planes, spheres or infinite cylinders')
        if source.beam_radius is None:
            raise ValueError("multiple beam-rigid profile requires explicit round radius")
        if body.kind not in {"plane", "sphere", "cylinder", "capped_cylinder"} or body.path_translation is not None:
            raise ValueError("multiple beam-rigid profile admits fixed analytic obstacles only")
        if interaction.normal_law != "penalty":
            raise ValueError("multiple beam-rigid profile requires explicitly dimensioned line penalties")
        used_surfaces.add(source.name)
        used_bodies.add(body.name)
        candidate_count += len(source.element_ids)
        if candidate_count > settings.maximum_candidate_pairs:
            raise MemoryError("multiple beam-rigid aggregate candidate limit exceeded; no contacts dropped")
        for element in source.element_ids:
            feature = (element, body.name)
            if feature in owned:
                raise ValueError("duplicate beam element/rigid-body feature pair in explicit interactions")
            owned.add(feature)
        part = ContactDefinition(surfaces=(source,), rigid_bodies=(body,), interactions=(interaction,))
        report = validate_general_contact(model, part, settings, analysis_control=analysis_control)
        if not report.valid:
            raise ValueError(f"{interaction.name}: " + "; ".join(report.issues))
        parts.append((interaction.name, part))
    if used_surfaces != set(surfaces) or used_bodies != set(bodies):
        raise ValueError("multiple beam-rigid definition contains an unreferenced surface or body")
    return tuple(parts)


def validate_multiple_beam_rigid(model, definition, settings=None, *, analysis_control="force"):
    controls = ContactSolverSettings() if settings is None else settings
    try:
        _parts(model, definition, controls, analysis_control)
    except (ValueError, MemoryError, KeyError) as exc:
        return ContactValidationReport("invalid", (str(exc),), _profile(definition, controls))
    return ContactValidationReport("ok", (), _profile(definition, controls))


@dataclass(frozen=True)
class MultipleBeamRigidResults:
    profile: str
    pairs: tuple
    body_names: tuple = ()

    @property
    def minimum_gap(self):
        return min(result.minimum_gap for _, result in self.pairs)

    @property
    def gap_tolerance_met(self):
        return all(result.gap_tolerance_met for _, result in self.pairs)

    def to_dict(self):
        fields = [result.to_dict() for _, result in self.pairs]
        result={"profile": self.profile, "minimum_gap": self.minimum_gap,
                "gap_tolerance_met": self.gap_tolerance_met,
                "work_convention": "sum_of_explicit_pair_potentials_same_authoritative_coordinates",
                "normal_law": "penalty",
                **{name: np.sum([field[name] for field in fields], axis=0).tolist()
                   for name in ("beam_force", "rigid_force", "beam_moment", "rigid_moment")},
                "algorithmic_contact_potential": float(sum(field["algorithmic_contact_potential"]
                                                            for field in fields)),
                "candidate_cells": sum(field["candidate_cells"] for field in fields),
                "pairs": [{"interaction": name, "contact": field}
                          for (name, _), field in zip(self.pairs, fields)]}
        if self.profile in _END_PROFILES:
            result.update(line_contact_energy=sum(field['line_contact_energy'] for field in fields),
                endpoint_contact_energy=sum(field.get('endpoint_contact_energy',0.) for field in fields),
                active_endpoints=sum(field.get('active_endpoints',0) for field in fields),
                candidate_endpoints=sum(field.get('candidate_endpoints',field.get('endpoint_candidates',0))
                    for field in fields),
                moment_origin=[0.,0.,0.],moment_units='N*m')
            for pair,body in zip(result['pairs'],self.body_names):pair['rigid_body']=body
            bodies={}
            for body,field in zip(self.body_names,fields):
                force,moment=bodies.setdefault(body,(np.zeros(3),np.zeros(3)))
                force+=np.asarray(field['rigid_force']);moment+=np.asarray(field['rigid_moment'])
            result['rigid_bodies']=[{'name':body,'rigid_force':bodies[body][0].tolist(),
                'rigid_moment':bodies[body][1].tolist()} for body in sorted(bodies)]
            if self.profile in _SUPPORT_PROFILES:
                result['beam_end_cap_work'] = 'hemisphere_support_dev'
        return result


class MultipleBeamRigidManager:
    profile = PROFILE
    augmented = False

    def __init__(self, model, definition, settings=None, initial_displacement=None,
                 *, analysis_control="force"):
        from .beam_plane import _BeamPlaneManager
        self.settings = ContactSolverSettings() if settings is None else settings
        parts = _parts(model, definition, self.settings, analysis_control)
        self.definition = definition
        self.profile=_profile(definition, self.settings)
        self.n_dof = model.mesh.dof_manager.total_dofs
        def contributor(part):
            kind=part.rigid_bodies[0].kind
            if kind=='plane':
                if self.settings.beam_end_cap_work == 'hemisphere_support_dev':
                    from .beam_hemisphere_plane import HemispherePlaneManager
                    owner = HemispherePlaneManager
                else:owner=_BeamPlaneManager
            elif kind=='capped_cylinder':
                from .capped_cylinder import CappedCylinderBeamManager
                owner=CappedCylinderBeamManager
            elif kind=='cylinder' and self.settings.beam_end_cap_work=='hemisphere_support_dev':
                from .beam_hemisphere_cylinder import HemisphereCylinderManager
                owner=HemisphereCylinderManager
            elif kind=='sphere' and self.settings.beam_end_cap_work=='hemisphere_support_dev':
                from .beam_hemisphere_sphere import HemisphereSphereManager
                owner=HemisphereSphereManager
            elif part.surfaces[0].beam_end_caps=='hemisphere':
                from .beam_radial_ends import BeamRadialEndManager
                owner=BeamRadialEndManager
            else:
                from .beam_radial import _BeamRadialManager
                owner=_BeamRadialManager
            return owner(model,part,self.settings,initial_displacement)
        self.pairs=tuple((name,contributor(part)) for name,part in parts)
        self.body_names=tuple(part.rigid_bodies[0].name for _,part in parts)
        self.physical_cells={cell.element_id:cell for _,manager in self.pairs for cell in manager.cells}
        if self.profile in _END_PROFILES:
            initial=np.zeros(self.n_dof) if initial_displacement is None else np.asarray(initial_displacement)
            if not self._regular_sweep(initial,initial):
                raise ValueError('multiple physical-end initial curve is singular or unresolved')
        self.swept_trial_cutbacks = 0

    def _regular_sweep(self,start,end):
        from .beam_end_radial_geometry import regular_curve_path
        first,last=np.asarray(start,dtype=float).reshape(-1),np.asarray(end,dtype=float).reshape(-1)
        if (first.size!=self.n_dof or last.size!=self.n_dof or
            not np.all(np.isfinite(first)) or not np.all(np.isfinite(last))):
            raise ValueError('multiple physical-end displacement is invalid')
        return all(regular_curve_path(cell.reference+first[cell.dofs],cell.reference+last[cell.dofs],
            self.settings.maximum_swept_subdivisions) for cell in self.physical_cells.values())

    def evaluate(self, displacement, *, tangent=True):
        if self.profile in _END_PROFILES and not self._regular_sweep(displacement,displacement):
            raise ValueError('multiple physical-end candidate curve is singular or unresolved')
        residual = np.zeros(self.n_dof)
        stiffness = sparse.csr_matrix((self.n_dof, self.n_dof)) if tangent else None
        results = []
        for name, manager in self.pairs:
            force, block, result = manager.evaluate(displacement, tangent=tangent)
            residual += force
            if tangent:
                stiffness = stiffness + block
            results.append((name, result))
        return residual, stiffness, MultipleBeamRigidResults(self.profile, tuple(results),
            self.body_names if self.profile in _END_PROFILES else ())

    def swept_increment_admissible(self, start, end, start_factor, end_factor):
        if self.profile in _END_PROFILES and not self._regular_sweep(start,end):
            self.swept_trial_cutbacks+=1
            return False
        admitted = [manager.swept_increment_admissible(start, end, start_factor, end_factor)
                    for _, manager in self.pairs]
        if not all(admitted):
            self.swept_trial_cutbacks += 1
        return all(admitted)

    def admissible_step_fraction(self, start, end):
        first, last = np.asarray(start), np.asarray(end)
        if self.profile in _END_PROFILES:
            scale=1.
            for _ in range(self.settings.maximum_swept_subdivisions+1):
                if self._regular_sweep(first,first+scale*(last-first)):break
                if scale==1.:self.swept_trial_cutbacks+=1
                scale*=.5
            else:return 0.
            candidate=first+scale*(last-first)
            fraction=scale*min(manager.admissible_step_fraction(first,candidate) for _,manager in self.pairs)
        else:
            fraction = min(manager.admissible_step_fraction(first, last) for _, manager in self.pairs)
        # Recheck the actual common endpoint: roundoff at one child's chosen
        # fraction must not hide another pair's independent dimensional gate.
        for _ in range(self.settings.maximum_swept_subdivisions + 1):
            if self.swept_increment_admissible(first, first+fraction*(last-first), 0., 1.):
                return fraction
            fraction *= .5
        return 0.

    def restart_contract(self):
        return {"profile": self.profile, "composition_revision": (
                    'explicit_hemisphere_support_planes_spheres_cylinders_regular_path_v1' if self.profile==PROFILE_SUPPORT_CYLINDER
                    else 'explicit_hemisphere_support_planes_spheres_regular_path_v1' if self.profile==PROFILE_SUPPORT_ANALYTIC
                    else 'explicit_hemisphere_support_planes_regular_path_v1' if self.profile==PROFILE_SUPPORT
                    else 'explicit_physical_end_work_regular_path_v1' if self.profile==PROFILE_ENDS
                    else "explicit_beam_rigid_work_sum_v1"),
                "pairs": [{"interaction": name, "contract": manager.restart_contract()}
                          for name, manager in self.pairs],
                "aggregate_candidate_limit": self.settings.maximum_candidate_pairs}

    def summary_settings(self):
        return {"normal_law": "penalty", "pair_enforcement": [
            {"interaction": name, "line_stiffness_N_per_m2": manager.stiffness,
             "gap_tolerance_m": manager.gap_tolerance,
             **({'endpoint_stiffness_N_per_m':manager.endpoint_stiffness}
                if (self.profile in _SUPPORT_PROFILES or
                    self.profile==PROFILE_ENDS and getattr(manager,'end_caps','none')=='hemisphere') else {})}
            for name, manager in self.pairs]}
