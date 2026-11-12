"""Development S3/S3 static contact owner for arbitrary nodal skin motion.

The pointwise penalty law is intentionally distinct from the selected mortar
formulation. Its signed gap certificate covers all vertex pairs and is therefore
conservative even when the projected overlap is only part of either triangle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport
from .s3_skin_work import S3SkinWorkAdapter
from .triangle_pair_energy import MovingS3PairPenalty


POINTWISE_PROFILE = "S3_S3_GENERAL_NODAL_POINTWISE_PENALTY_STATIC_DEV_V0"
MORTAR_PROFILE = "S3_S3_GENERAL_NODAL_PRIMAL_MORTAR_PENALTY_STATIC_DEV_V0"


def validate_general_s3_pair(model, definition: ContactDefinition,
                             settings: ContactSolverSettings) -> ContactValidationReport:
    profile = (MORTAR_PROFILE if settings.normal_discretization == "primal_mortar"
               else POINTWISE_PROFILE)
    issues: list[str] = []
    if tuple(map(len, (definition.surfaces, definition.rigid_bodies,
                       definition.interactions))) != (2, 0, 1):
        return ContactValidationReport("invalid", (
            "general S3 pair needs two shell surfaces and one interaction",
        ), profile)
    if settings.pair_motion != "general":
        issues.append("general S3 pair needs pair_motion='general'")
    if settings.general_s3_difference_step is None:
        issues.append("general S3 pair needs explicit general_s3_difference_step")
    if settings.sliding_difference_step is not None:
        issues.append("general S3 pair does not use sliding_difference_step")
    interaction = definition.interactions[0]
    names = {surface.name: surface for surface in definition.surfaces}
    if (interaction.surface not in names or interaction.target not in names
            or interaction.surface == interaction.target):
        issues.append("general S3 interaction must name two different surfaces")
    if (interaction.normal_law != "penalty" or interaction.normal_stiffness is None
            or interaction.line_stiffness is not None
            or interaction.endpoint_stiffness is not None):
        issues.append("general S3 pair needs penalty surface-pressure stiffness only")
    from anymaterial import IsotropicMaterial
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    nodes: list[set[int]] = []
    for surface in definition.surfaces:
        if surface.beam_radius is not None or len(surface.element_ids) != 1:
            issues.append(f"surface {surface.name} needs exactly one S3 skin")
            nodes.append(set())
            continue
        element_id = surface.element_ids[0]
        element = model.mesh.elements.get(element_id)
        if type(element) is not NativeParityE4PLS3V2DShellElement:
            issues.append(f"element {element_id} is not admitted S3 V2D")
            nodes.append(set())
            continue
        nodes.append(set(map(int, element.node_ids)))
        material = model.materials.get(element.material_name)
        if (type(material) is not IsotropicMaterial
                or material.yield_stress != 0.0
                or material.hardening_curve is not None):
            issues.append(f"element {element_id} needs isotropic elastic material")
        try:
            S3SkinWorkAdapter.from_model(model, 1, element_id, surface.side)
        except ValueError as exc:
            issues.append(f"surface {surface.name}: {exc}")
    if nodes[0] & nodes[1]:
        issues.append("general S3 pair does not admit shared material nodes")
    if not issues:
        source = names[interaction.surface]
        target = names[interaction.target]
        try:
            a = S3SkinWorkAdapter.from_model(model, 1, source.element_ids[0], source.side)
            b = S3SkinWorkAdapter.from_model(model, 2, target.element_ids[0], target.side)
            law_type = (MovingS3PairPenalty
                        if settings.normal_discretization == "pointwise_penalty"
                        else _mortar_law_type())
            law_type(
                a.reference_positions, b.reference_positions,
                interaction.normal_stiffness,
                difference_step=settings.general_s3_difference_step,
                maximum_energy_evaluations=settings.maximum_general_s3_energy_evaluations,
            )
            scale = max(1.0, float(np.max(np.abs(a.reference_positions))),
                        float(np.max(np.abs(b.reference_positions))))
            if (_vertex_gap_lower(a.reference_positions, b.reference_positions)
                    < -256.0*np.finfo(float).eps*scale):
                issues.append("source S3 reference normal must point from target to source")
            normal = _normal(a.reference_positions)
            if source.side != "midsurface" and (
                (-1.0 if source.side == "bottom" else 1.0)
                * float(a.director @ normal) >= 0.0
            ):
                issues.append(f"surface {source.name}: physical side faces away from target")
            if target.side != "midsurface" and (
                (-1.0 if target.side == "bottom" else 1.0)
                * float(b.director @ normal) <= 0.0
            ):
                issues.append(f"surface {target.name}: physical side faces away from source")
        except ValueError as exc:
            issues.append(str(exc))
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), profile)


def _mortar_law_type():
    from .triangle_mortar_energy import MovingS3PrimalMortarPenalty

    return MovingS3PrimalMortarPenalty


def _normal(positions: np.ndarray) -> np.ndarray:
    cross = np.cross(positions[1] - positions[0], positions[2] - positions[0])
    magnitude = float(np.linalg.norm(cross))
    if magnitude <= 0.0 or not np.isfinite(magnitude):
        raise ValueError("S3 pair has degenerate physical face")
    return cross / magnitude


def _vertex_gap_lower(a: np.ndarray, b: np.ndarray) -> float:
    normal = _normal(a)
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    roundoff = 128.0 * np.finfo(float).eps * scale
    return float(np.min((a[:, None, :] - b[None, :, :]) @ normal) - roundoff)


def _normal_controls(first: np.ndarray, last: np.ndarray) -> np.ndarray:
    middle = 0.5 * (first + last)
    def cross(face):
        return np.cross(face[1] - face[0], face[2] - face[0])
    n0, nm, n1 = cross(first), cross(middle), cross(last)
    return np.stack((n0, 2.0*nm - 0.5*(n0+n1), n1))


def _swept_gap_lower(a0: np.ndarray, a1: np.ndarray,
                     b0: np.ndarray, b1: np.ndarray) -> float:
    """Lower bound for every physical point pair over a linear trial path."""
    ac, bc = _normal_controls(a0, a1), _normal_controls(b0, b1)
    an = _normal(0.5*(a0+a1))
    # The middle Bézier control is not the value at t=1/2.
    amid = 0.25*ac[0] + 0.5*ac[1] + 0.25*ac[2]
    bmid = 0.25*bc[0] + 0.5*bc[1] + 0.25*bc[2]
    da = max(float(np.linalg.norm(item-amid)) for item in ac)
    db = max(float(np.linalg.norm(item-bmid)) for item in bc)
    norm_lower = float(np.linalg.norm(amid)) - da
    if norm_lower <= 0.0:
        return -np.inf
    if (float(amid @ bmid) - float(np.linalg.norm(amid))*db
            - float(np.linalg.norm(bmid))*da - da*db <= 0.0):
        return -np.inf
    if float(amid @ ac[0]) <= da * float(np.linalg.norm(ac[0])):
        return -np.inf
    if float(bmid @ bc[0]) <= db * float(np.linalg.norm(bc[0])):
        return -np.inf
    d0 = a0[:, None, :] - b0[None, :, :]
    d1 = a1[:, None, :] - b1[None, :, :]
    projected = min(float(np.min(d0 @ an)), float(np.min(d1 @ an)))
    size = max(float(np.max(np.linalg.norm(d0, axis=2))),
               float(np.max(np.linalg.norm(d1, axis=2))))
    error = size * 2.0 * da / norm_lower
    scale = max(1.0, size, float(np.max(np.abs(a0))), float(np.max(np.abs(b0))),
                float(np.max(np.abs(a1))), float(np.max(np.abs(b1))))
    return projected - error - 128.0*np.finfo(float).eps*scale


@dataclass(frozen=True)
class _Results:
    profile: str
    minimum_gap: float
    sampled_overlap_gap: float | None
    penalty_energy: float
    overlap_area: float
    active_area: float | None
    integration_force: tuple[float, float, float]
    target_force: tuple[float, float, float]
    integration_moment: tuple[float, float, float]
    target_moment: tuple[float, float, float]
    gap_tolerance_met: bool
    energy_evaluations: int
    active_pressure_dofs: int | None
    normalized_weighted_gap_m: tuple[float, ...] | None
    numerical_pressure_pa: tuple[float, ...] | None
    reference_support_m2: tuple[float, ...] | None

    def to_dict(self):
        return dict(vars(self), integration_force=list(self.integration_force),
                    target_force=list(self.target_force),
                    integration_moment=list(self.integration_moment),
                    target_moment=list(self.target_moment),
                    normalized_weighted_gap_m=(
                        None if self.normalized_weighted_gap_m is None
                        else list(self.normalized_weighted_gap_m)),
                    numerical_pressure_pa=(
                        None if self.numerical_pressure_pa is None
                        else list(self.numerical_pressure_pa)),
                    reference_support_m2=(
                        None if self.reference_support_m2 is None
                        else list(self.reference_support_m2)))


class GeneralS3PairManager:
    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        self.settings = ContactSolverSettings() if settings is None else settings
        report = validate_general_s3_pair(model, definition, self.settings)
        if not report.valid:
            raise ValueError("unsupported general S3 pair: " + "; ".join(report.issues))
        self.definition = definition
        self.profile = report.profile
        interaction = definition.interactions[0]
        names = {surface.name: surface for surface in definition.surfaces}
        selected = (names[interaction.surface], names[interaction.target])
        self.adapters = tuple(S3SkinWorkAdapter.from_model(
            model, index+1, surface.element_ids[0], surface.side,
        ) for index, surface in enumerate(selected))
        self.map = sparse.vstack([item.work_map for item in self.adapters]).tocsr()
        self.n_dof = model.mesh.dof_manager.total_dofs
        self.stiffness = interaction.normal_stiffness
        self.gap_tolerance = interaction.gap_tolerance
        self.swept_trial_cutbacks = 0
        law_type = (MovingS3PairPenalty
                    if self.settings.normal_discretization == "pointwise_penalty"
                    else _mortar_law_type())
        self.law = law_type(
            self.adapters[0].reference_positions,
            self.adapters[1].reference_positions,
            self.stiffness,
            difference_step=self.settings.general_s3_difference_step,
            maximum_energy_evaluations=self.settings.maximum_general_s3_energy_evaluations,
        )
        initial = np.zeros(self.n_dof) if initial_displacement is None else initial_displacement
        a, b = self._positions(initial)
        lower = _vertex_gap_lower(a, b)
        scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        if lower < -256.0*np.finfo(float).eps*scale:
            raise ValueError(f"general S3 initial penetration is unresolved: lower bound {lower:.6g} m")

    def _positions(self, displacement):
        return tuple(item.current_positions(displacement) for item in self.adapters)

    def evaluate(self, displacement, *, tangent=True):
        a, b = self._positions(displacement)
        lower = _vertex_gap_lower(a, b)
        energy = self.law.evaluate(a, b)
        if energy.energy == 0.0:
            local_force = np.zeros(18)
            local_tangent = sparse.csr_matrix((18, 18)) if tangent else None
            evaluations = 1
        else:
            work = self.law.work(a, b, tangent=tangent)
            local_force = work.residual
            local_tangent = (sparse.csr_matrix(work.tangent) if tangent else None)
            evaluations = work.energy_evaluations
        force = np.asarray(self.map.T @ local_force).reshape(-1)
        stiffness = ((self.map.T @ local_tangent @ self.map).tocsr()
                     if tangent else None)
        physical = -local_force.reshape(6, 3)
        fa, fb = np.sum(physical[:3], axis=0), np.sum(physical[3:], axis=0)
        ma = np.sum(np.cross(a, physical[:3]), axis=0)
        mb = np.sum(np.cross(b, physical[3:]), axis=0)
        mortar = self.settings.normal_discretization == "primal_mortar"
        results = _Results(
            self.profile, lower,
            energy.minimum_physical_gap if mortar else energy.minimum_gap,
            energy.energy, energy.overlap_area,
            None if mortar else energy.active_area,
            tuple(map(float, fa)), tuple(map(float, fb)),
            tuple(map(float, ma)), tuple(map(float, mb)),
            lower >= -self.gap_tolerance, evaluations,
            energy.active_pressure_dofs if mortar else None,
            tuple(map(float, energy.normalized_gap)) if mortar else None,
            tuple(map(float, energy.pressure)) if mortar else None,
            tuple(map(float, energy.reference_support)) if mortar else None,
        )
        return force, stiffness, results

    def _swept_admissible(self, start, end):
        first = self._positions(start)
        last = self._positions(end)
        if _vertex_gap_lower(*first) < -self.gap_tolerance:
            raise ValueError("committed general S3 gap exceeds tolerance")
        if _vertex_gap_lower(*last) < -self.gap_tolerance:
            return False

        def certify(left, right, depth):
            if _swept_gap_lower(left[0], right[0], left[1], right[1]) >= -self.gap_tolerance:
                return True
            if depth >= self.settings.maximum_swept_subdivisions:
                return False
            middle = tuple(0.5*(a+b) for a, b in zip(left, right))
            if _vertex_gap_lower(*middle) < -self.gap_tolerance:
                return False
            return certify(left, middle, depth+1) and certify(middle, right, depth+1)

        return certify(first, last, 0)

    def swept_increment_admissible(self, start, end, _start_factor, _end_factor):
        allowed = self._swept_admissible(start, end)
        if not allowed:
            self.swept_trial_cutbacks += 1
        return allowed

    def admissible_step_fraction(self, start, end):
        first = np.asarray(start, dtype=float).reshape(-1)
        last = np.asarray(end, dtype=float).reshape(-1)
        if self._swept_admissible(first, last):
            return 1.0
        self.swept_trial_cutbacks += 1
        fraction = 0.5
        for _ in range(self.settings.maximum_swept_subdivisions):
            if self._swept_admissible(first, first + fraction*(last-first)):
                return fraction
            fraction *= 0.5
        return 0.0

    def restart_contract(self):
        interaction = self.definition.interactions[0]
        return {
            "profile": self.profile,
            "surfaces": [{"name": surface.name,
                          "element_ids": list(surface.element_ids),
                          "side": surface.side}
                         for surface in self.definition.surfaces],
            "interaction": {"name": interaction.name,
                            "surface": interaction.surface,
                            "target": interaction.target,
                            "normal_stiffness": self.stiffness,
                            "gap_tolerance": self.gap_tolerance,
                            "normal_law": interaction.normal_law},
            "settings": {"pair_motion": self.settings.pair_motion,
                         "normal_discretization": self.settings.normal_discretization,
                         "general_s3_difference_step": self.settings.general_s3_difference_step,
                         "maximum_general_s3_energy_evaluations": self.settings.maximum_general_s3_energy_evaluations,
                         "maximum_swept_subdivisions": self.settings.maximum_swept_subdivisions,
                         "maximum_candidate_pairs": self.settings.maximum_candidate_pairs},
        }
