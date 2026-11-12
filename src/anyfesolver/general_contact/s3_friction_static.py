"""Transactional fixed-material S3 friction for the development static path.

Normal pressure comes from the shared mortar pressure space. This bounded
quadrature contributor does not implement moving material partners or facet
transfer. Analytic traction and bounded numerical pressure/work-map derivatives
form a development tangent candidate.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import sparse

from .friction_point import (
    FrictionPointState, initial_friction_point, trial_coulomb_point_directional,
)
from .s3_friction_pair import S3FrictionPointPair
from .s3_skin_work import S3SkinWorkAdapter


@dataclass(frozen=True)
class S3MaterialFrictionSettings:
    pairs: tuple[S3FrictionPointPair, ...]
    steps_by_dof: tuple[float, ...]
    maximum_candidate_evaluations: int = 145

    def __post_init__(self):
        pairs = tuple(self.pairs)
        steps = tuple(float(x) for x in self.steps_by_dof)
        if (not pairs or any(type(p) is not S3FrictionPointPair for p in pairs)
                or len({p.partner_key for p in pairs}) != len(pairs)):
            raise ValueError("friction needs unique fixed-material S3 pairs")
        if not steps or not np.all(np.isfinite(steps)) or min(steps) <= 0:
            raise ValueError("friction needs explicit positive finite FE derivative steps")
        if (type(self.maximum_candidate_evaluations) is not int
                or self.maximum_candidate_evaluations < 1):
            raise ValueError("friction evaluation cap must be positive")
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "steps_by_dof", steps)


class S3MaterialFrictionManager:
    def __init__(self, model, normal, settings):
        if type(settings) is not S3MaterialFrictionSettings:
            raise TypeError("friction needs S3MaterialFrictionSettings")
        from .s3_patch_static import GeneralS3PatchManager
        if (type(normal) is not GeneralS3PatchManager or normal.augmented
                or normal.settings.normal_discretization != "primal_mortar"):
            raise ValueError("fixed-material friction needs the S3 patch penalty owner")
        self.normal = normal
        self.friction_settings = settings
        self.n_dof = model.mesh.dof_manager.total_dofs
        if len(settings.steps_by_dof) != self.n_dof:
            raise ValueError("friction derivative steps must match authoritative FE coordinates")
        interaction = normal.definition.interactions[0]
        surfaces = {s.name: s for s in normal.definition.surfaces}
        source, target = surfaces[interaction.surface], surfaces[interaction.target]
        columns = set(normal.map.indices)
        self.pressure_indices = []
        quadrature_support = {}
        for pair in settings.pairs:
            for point, surface in ((pair.source, source), (pair.target, target)):
                adapter = point.adapter
                if adapter.element_id not in surface.element_ids or adapter.side != surface.side:
                    raise ValueError("friction point is outside the declared normal surface")
                expected = S3SkinWorkAdapter.from_model(
                    model, adapter.body_id, adapter.element_id, adapter.side)
                if ((expected.work_map != adapter.work_map).nnz
                        or not np.array_equal(expected.reference_positions, adapter.reference_positions)):
                    raise ValueError("friction point belongs to a different model/skin")
                columns.update(adapter.work_map.indices)
            self.pressure_indices.append(tuple(
                normal.law.source_node_ids.index(n) for n in pair.source.adapter.node_ids))
            key = pair.source.adapter.element_id
            region, weight = quadrature_support.get(key, (pair.target.adapter.element_id, 0.0))
            if region != pair.target.adapter.element_id:
                raise ValueError("fixed friction source facet needs one declared target facet")
            quadrature_support[key] = (region, weight + pair.reference_weight_m2)
        if set(quadrature_support) != set(source.element_ids):
            raise ValueError("fixed friction quadrature must cover all source facets")
        for element, (_region, weight) in quadrature_support.items():
            reference = next(p.source.adapter.reference_positions for p in settings.pairs
                             if p.source.adapter.element_id == element)
            area = .5*np.linalg.norm(np.cross(reference[1]-reference[0],reference[2]-reference[0]))
            if abs(weight-area) > 1e-12*area:
                raise ValueError("friction quadrature weights must sum to source reference area")
        self.columns = tuple(sorted(columns))
        if 1 + 2*len(self.columns) > settings.maximum_candidate_evaluations:
            raise MemoryError("coupled friction derivative evaluation cap exceeded")
        self.committed_displacement = np.zeros(self.n_dof)
        self.history = tuple(initial_friction_point(
            p.target.physical_pose(self.committed_displacement).material_frame
        ) for p in settings.pairs)
        self.last_committed_summary = None

    def __getattr__(self, name):
        return getattr(self.normal, name)

    def restrict_tangent_to_reduced_variations(self, transform):
        self.normal.restrict_tangent_to_reduced_variations(transform)
        reachable = np.asarray(sparse.csr_matrix(transform).getnnz(axis=1)).reshape(-1) > 0
        self.columns = tuple(c for c in self.columns if reachable[c])

    def restart_contract(self):
        pairs = []
        for p in self.friction_settings.pairs:
            pairs.append(dict(partner=p.partner_key, weight_m2=p.reference_weight_m2,
                              mu=p.friction_coefficient, stiffness_Pa_per_m=p.tangential_stiffness_Pa_per_m,
                              maximum_active_columns=p.maximum_active_columns))
        return dict(normal=self.normal.restart_contract(),
                    friction_profile="S3_FIXED_MATERIAL_PRESSURE_COUPLED_STATIC_DEV_V1",
                    pairs=pairs, steps_by_dof=list(self.friction_settings.steps_by_dof),
                    maximum_candidate_evaluations=self.friction_settings.maximum_candidate_evaluations)

    def _friction(self, displacement):
        source, target = self.normal._positions(displacement)
        state = (self.normal.law.prepare_work_state(
            source, target, self.normal.normal_multipliers).state
            if self.normal.projected_work else self.normal.law.evaluate(source, target))
        force = np.zeros(self.n_dof)
        trials, pressures = [], []
        for pair, old, indices in zip(self.friction_settings.pairs, self.history, self.pressure_indices):
            a = pair.source.physical_pose(displacement)
            b = pair.target.physical_pose(displacement)
            corners = pair.target.adapter.current_positions(displacement)
            frame = np.asarray(b.material_frame)
            coordinates = (corners - corners[0]) @ frame[:, :2]
            source_corners = pair.source.adapter.current_positions(displacement)
            corner_uv = np.linalg.solve(coordinates[1:].T,
                ((source_corners-corners[0]) @ frame[:, :2]).T).T
            corner_weights = np.column_stack((1-corner_uv.sum(axis=1),corner_uv))
            if np.min(corner_weights) < -1e-12:
                raise ValueError("fixed friction requires the whole source facet within its target facet")
            projected = (np.asarray(a.position)-corners[0]) @ frame[:, :2]
            uv = np.linalg.solve(coordinates[1:].T, projected)
            weights = np.array((1-uv.sum(), *uv))
            if min(weights) < -1e-12:
                raise ValueError("fixed-material friction point left its target facet")
            gap = float((np.asarray(a.position)-corners[0]) @ frame[:, 2])
            pressure = (0.0 if gap >= 0 else float(
                np.asarray(pair.source.barycentric) @ state.pressure[list(indices)]))
            trial = pair.evaluate(self.committed_displacement, displacement, old,
                                  normal_pressure_Pa=pressure)
            # Pair force is physical; the structural equilibrium residual is opposite.
            force -= trial.structural_force
            trials.append(trial)
            pressures.append(pressure)
        return force, tuple(trials), tuple(pressures)

    def evaluate(self, displacement, *, tangent=True):
        force, stiffness, certificate = self.normal.evaluate(displacement, tangent=tangent)
        friction, trials, pressures = self._friction(displacement)
        if tangent:
            derivative = np.zeros((self.n_dof, len(self.columns)))
            for local, column in enumerate(self.columns):
                delta = np.zeros(self.n_dof)
                delta[column] = self.friction_settings.steps_by_dof[column]
                plus = self._friction(np.asarray(displacement)+delta)
                minus = self._friction(np.asarray(displacement)-delta)
                direction = np.zeros(self.n_dof)
                direction[column] = 1
                for j, (pair, base, old, pressure) in enumerate(zip(
                        self.friction_settings.pairs, trials, self.history, pressures)):
                    if base.trial.branch == "open":
                        continue  # Open member of the activation generalized Jacobian.
                    pose = pair.target.physical_pose(displacement)
                    dt = np.asarray(trial_coulomb_point_directional(
                        old, material_frame=pose.material_frame,
                        material_frame_variation=pair.target.frame_directional_variation(displacement,direction),
                        relative_increment=base.relative_increment,
                        relative_increment_variation=base.relative_increment_map[:,column],
                        normal_pressure=pressure,
                        pressure_variation=(plus[2][j]-minus[2][j])/(2*delta[column]),
                        friction_coefficient=pair.friction_coefficient,
                        tangential_stiffness=pair.tangential_stiffness_Pa_per_m,
                        partner_key=pair.partner_key, allow_stick_slip_equality=True))
                    # The slip work map is geometric. Keep its computation active
                    # even if a perturbed normal pose is open; never difference
                    # an active map against the open response's zero placeholder.
                    pos = plus[1][j]
                    neg = minus[1][j]
                    if pos.trial.branch == "open":
                        pos = pair.evaluate(self.committed_displacement,np.asarray(displacement)+delta,old,
                                            normal_pressure_Pa=pressure)
                    if neg.trial.branch == "open":
                        neg = pair.evaluate(self.committed_displacement,np.asarray(displacement)-delta,old,
                                            normal_pressure_Pa=pressure)
                    ds = (pos.slip_map-neg.slip_map)/(2*delta[column])
                    derivative[:,local] -= pair.reference_weight_m2 * (
                        ds.T @ np.asarray(base.trial.traction) + base.slip_map.T @ dt)
            if not np.all(np.isfinite(derivative)):
                raise ValueError("coupled friction tangent exceeded finite range")
            rows, local = np.nonzero(derivative)
            block = sparse.coo_matrix((derivative[rows, local],
                (rows, np.asarray(self.columns)[local])), shape=(self.n_dof, self.n_dof)).tocsr()
            stiffness = stiffness + block
        return force+friction, stiffness, certificate

    def prepare_commit(self, displacement):
        u = np.asarray(displacement, dtype=float).copy()
        if u.shape != (self.n_dof,) or not np.all(np.isfinite(u)):
            raise ValueError("invalid accepted friction displacement")
        _force, trials, pressures = self._friction(u)
        histories = tuple(t.trial.candidate_state for t in trials)
        summary = dict(profile="S3_FIXED_MATERIAL_PRESSURE_COUPLED_STATIC_DEV_V1",
                       branch=[t.trial.branch for t in trials], pressure_pa=list(pressures),
                       traction_pa=[list(t.trial.traction) for t in trials],
                       dissipation_increment_j=sum(p.reference_weight_m2*t.trial.dissipated_increment_density
                           for p, t in zip(self.friction_settings.pairs, trials)),
                       storage_j=sum(p.reference_weight_m2*t.trial.stored_energy_density
                           for p, t in zip(self.friction_settings.pairs, trials)),
                       released_storage_j=sum(p.reference_weight_m2*t.trial.released_storage_density
                           for p, t in zip(self.friction_settings.pairs, trials)))
        return u, histories, summary

    def commit(self, prepared):
        self.committed_displacement, self.history, self.last_committed_summary = prepared

    def checkpoint_state(self):
        return dict(displacement=self.committed_displacement.tolist(),
                    history=[asdict(h) for h in self.history])

    def restore(self, state, displacement):
        if type(state) is not dict or set(state) != {"displacement", "history"}:
            raise ValueError("invalid friction checkpoint")
        u = np.asarray(state["displacement"], dtype=float)
        if not np.array_equal(u, displacement) or not np.all(np.isfinite(u)):
            raise ValueError("friction checkpoint pose differs from structural state")
        if len(state["history"]) != len(self.history):
            raise ValueError("friction checkpoint point count differs")
        history = tuple(FrictionPointState(**h) for h in state["history"])
        for p, h in zip(self.friction_settings.pairs, history):
            if h.partner_key not in (None, p.partner_key):
                raise ValueError("friction checkpoint partner differs")
            if not np.allclose(h.material_frame, p.target.physical_pose(u).material_frame,
                               rtol=0, atol=1e-9):
                raise ValueError("friction checkpoint material frame differs")
        self.committed_displacement, self.history = u.copy(), history


def create_s3_friction_manager(model, normal, settings):
    """Explicit profile dispatch; never switch a failed candidate's friction law."""
    from .s3_sliding_friction import S3SlidingFrictionSettings, S3SlidingFrictionManager
    from .s3_sliding_friction import S3MovingNormalFrictionSettings, S3MeshMovingNormalFrictionSettings, S3MeshWeakPressureFrictionSettings, S3MeshWholeChordGapFrictionSettings, S3MeshHistoryPathFrictionSettings
    from .s3_friction_replay import S3ReplayFrictionSettings,S3ReplayFrictionManager,S3RimReplayFrictionSettings,S3RimReplayFrictionManager
    if type(settings) is S3RimReplayFrictionSettings:
        return S3RimReplayFrictionManager(model,normal,settings)
    if type(settings) is S3ReplayFrictionSettings:
        return S3ReplayFrictionManager(model,normal,settings)
    if type(settings) is S3MeshHistoryPathFrictionSettings:
        from .s3_friction_history_path import S3MeshHistoryPathFrictionManager
        return S3MeshHistoryPathFrictionManager(model,normal,settings)
    if type(settings) is S3MeshWholeChordGapFrictionSettings:
        from .s3_friction_swept_gap import S3MeshWholeChordGapFrictionManager
        return S3MeshWholeChordGapFrictionManager(model,normal,settings)
    if type(settings) is S3MeshWeakPressureFrictionSettings:
        from .s3_friction_weak_pressure import S3MeshWeakPressureFrictionManager
        return S3MeshWeakPressureFrictionManager(model,normal,settings)
    if type(settings) is S3MeshMovingNormalFrictionSettings:
        from .s3_friction_mesh_monolithic import S3MeshMovingNormalFrictionManager
        return S3MeshMovingNormalFrictionManager(model,normal,settings)
    if type(settings) is S3MovingNormalFrictionSettings:
        from .s3_friction_monolithic import S3MovingNormalFrictionManager
        return S3MovingNormalFrictionManager(model,normal,settings)
    if type(settings) is S3SlidingFrictionSettings:
        return S3SlidingFrictionManager(model, normal, settings)
    return S3MaterialFrictionManager(model, normal, settings)
