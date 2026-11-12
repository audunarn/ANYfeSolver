"""Internal coupled retained-coordinate native beam/plane static development owner.

One generalized native B3-GE beam with a fixed plane line kernel or an explicitly
selected interior material-area kernel. Restricted development checkpoints do
not establish public native engineering admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

import numpy as np
from anyfesolver.control import SolveCancelled, cancellation_safe_point

from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_retained_nodal_loading import (
    Context as NativeNodalContext,
    Program as NativeNodalProgram,
)
from anyfesolver._ge_beam3_p5_seeded.core import sha
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5_seeded.codec import MAX_BYTES, _load

from .native_retained_line_plane import (
    NativeRetainedLinePlaneResult,
    NativeRetainedRoundLinePlane,
)
from .native_material_plane import NativeRoundMaterialPlane, PROFILE as MATERIAL_PROFILE
from .native_curved_plane_work import NativeCurvedMaterialPlaneWork


PROFILE = "NATIVE_B3_GE_RETAINED_ROUND_LINE_PLANE_STATIC_DEV_V1"
CURVED_MATERIAL_PROFILE = "NATIVE_B3_GE_RETAINED_CURVED_MATERIAL_PLANE_STATIC_DEV_V1"
CHECKPOINT_SCHEMA = "NATIVE_RETAINED_ROUND_LINE_PLANE_STATIC_CHECKPOINT_DEV_V2"


@dataclass(frozen=True)
class NativeRetainedContactAccepted:
    mechanical: object
    histories: tuple
    completed_targets: int
    contact: NativeRetainedLinePlaneResult


@dataclass(frozen=True)
class NativeRetainedContactRun:
    status: str
    accepted: NativeRetainedContactAccepted
    failure: str | None
    checkpoint: bytes
    profile: str = PROFILE
    diagnostics: dict | None = None

    @property
    def checkpoint_sha256(self) -> str:
        return sha256(self.checkpoint).hexdigest()


@dataclass(frozen=True)
class _CurvedPlaneContribution(NativeRetainedLinePlaneResult):
    """Existing owner work contract; gap is a conservative whole-patch bound."""
    def spatial_tangent(self):
        return self.tangent


class NativeRetainedRoundPlaneAnalysis:
    """Separate owner; all native internal coordinates remain in the solve."""

    def __init__(self, definitions: tuple[NativeBeamDefinition, ...],
                 boundaries: tuple, program: NativeNodalProgram,
                 contact: NativeRetainedRoundLinePlane | NativeRoundMaterialPlane | NativeCurvedMaterialPlaneWork, *,
                 gap_tolerance_m: float, equilibrium_tolerance: float,
                 maximum_swept_intervals: int):
        if (type(definitions) is not tuple or len(definitions) != 1
                or type(definitions[0]) is not NativeBeamDefinition):
            raise ValueError("native contact development owner needs one exact beam definition")
        if type(program) is not NativeNodalProgram:
            raise TypeError("native contact owner needs exact retained nodal program")
        if type(contact) not in (NativeRetainedRoundLinePlane, NativeRoundMaterialPlane,NativeCurvedMaterialPlaneWork):
            raise TypeError("native contact owner needs exact retained line or material plane kernel")
        self.profile = (CURVED_MATERIAL_PROFILE if type(contact) is NativeCurvedMaterialPlaneWork else
                        MATERIAL_PROFILE if type(contact) is NativeRoundMaterialPlane else PROFILE)
        if (type(gap_tolerance_m) not in (int, float)
                or not isfinite(gap_tolerance_m) or gap_tolerance_m <= 0.0
                or type(equilibrium_tolerance) not in (int, float)
                or not isfinite(equilibrium_tolerance) or equilibrium_tolerance <= 0.0):
            raise ValueError("native contact owner needs positive gap/equilibrium tolerances")
        if type(maximum_swept_intervals) is not int or maximum_swept_intervals < 1:
            raise ValueError("native contact owner needs finite swept interval budget")
        self.native = NativeBeamAnalysis(definitions, boundaries)
        self.context = NativeNodalContext(self.native.model, program)
        if len(self.context.elements) != 1:
            raise ValueError("native contact owner needs exactly one element")
        element_id, element = self.context.elements[0]
        if (type(element) is not NativeGeneralizedStaticElement
                or element_id != contact.element_id
                or sha(element.operator.reference.canonical_data())
                != sha(contact.reference.canonical_data())):
            raise ValueError("native contact kernel does not bind the generalized beam")
        self.contact = contact
        self.gap_tolerance = float(gap_tolerance_m)
        self.equilibrium_tolerance = float(equilibrium_tolerance)
        self.maximum_swept_intervals = maximum_swept_intervals
        self.program = program
        self.curved_rejected_trials=[]
        self.identity = self._identity()
        initial = self.context.initial
        first = self._contact(initial.mechanical)
        roundoff = 256.0*np.finfo(float).eps*max(
            1.0, float(np.max(np.abs(element.operator.reference.coordinates))),
            contact.radius,
        )
        if first.minimum_gap_m < -roundoff:
            raise ValueError(
                f"unresolved initial native contact penetration {first.minimum_gap_m} m"
            )
        self.initial = NativeRetainedContactAccepted(
            initial.mechanical, initial.histories, 0, first,
        )

    def _identity(self) -> str:
        contact = self.contact
        if type(contact) in (NativeRoundMaterialPlane,NativeCurvedMaterialPlaneWork):
            definition=dict(profile=self.profile,native_model=self.context.identity,
                native_support_sha256=sha256(self.context.support_identity).hexdigest(),
                contact=contact.definition(),gap_tolerance_m=self.gap_tolerance,
                equilibrium_tolerance=self.equilibrium_tolerance,
                maximum_swept_intervals=self.maximum_swept_intervals)
            if type(contact) is NativeCurvedMaterialPlaneWork:
                definition['trial_policy']='whole_patch_gap_preflight_bounded_initial_fraction_v1'
            return sha(definition)
        return sha(dict(
            profile=self.profile,
            native_model=self.context.identity,
            native_support_sha256=sha256(self.context.support_identity).hexdigest(),
            contact=dict(element=contact.element_id,
                         reference=contact.reference.canonical_data(),
                         radius_m=contact.radius,
                         plane_origin_m=contact.origin,
                         plane_normal=contact.normal,
                         line_stiffness_n_per_m2=contact.stiffness,
                         energy_tolerance_j=contact.energy_tolerance,
                         force_tolerance_n=contact.force_tolerance,
                         tangent_tolerance_n_per_m=contact.tangent_tolerance,
                         maximum_intervals=contact.maximum_intervals),
            gap_tolerance_m=self.gap_tolerance,
            equilibrium_tolerance=self.equilibrium_tolerance,
            maximum_swept_intervals=self.maximum_swept_intervals,
        ))

    def _guard_contact(self) -> None:
        if self._identity() != self.identity:
            raise ValueError("native contact definition/settings changed")

    def _record(self, target: int, mechanical, origins, previous_sha256: str):
        if type(target) is not int or not 1 <= target <= len(self.program.targets):
            raise ValueError("native contact checkpoint target")
        mechanical = self.context.make(mechanical.descriptor())
        r, j, metrics, responses, contact = self._assemble(
            mechanical, self.program.targets[target-1], origins,
        )
        _, correction = self.context.step(mechanical, r, j)
        if max(*metrics, correction) > self.equilibrium_tolerance:
            raise ValueError("native contact checkpoint requires coupled equilibrium")
        if contact.minimum_gap_m < -self.gap_tolerance:
            raise ValueError("native contact checkpoint gap tolerance")
        histories = self.context.histories(
            tuple(response.history for response in responses)
        )
        contact_summary = dict(
            potential_j=contact.potential_j,
            minimum_gap_m=contact.minimum_gap_m,
            beam_force_n=contact.beam_force_n,
            plane_force_n=contact.plane_force_n,
            beam_moment_nm=contact.beam_moment_nm,
            plane_moment_nm=contact.plane_moment_nm,
            energy_error_j=contact.energy_error_j,
            force_error_n=contact.force_error_n,
            tangent_error_n_per_m=contact.tangent_error_n_per_m,
        )
        if type(self.contact) is NativeCurvedMaterialPlaneWork:
            contact_summary['minimum_gap_kind']='continuous_whole_material_patch_lower_bound'
        row = dict(target=target, previous_sha256=previous_sha256,
                   mechanical=mechanical.descriptor(), histories=histories,
                   contact=contact_summary)
        return NativeRetainedContactAccepted(
            mechanical, histories, target, contact,
        ), _load(canonical(row).decode("ascii"))

    def _checkpoint(self, records: tuple[dict, ...]) -> bytes:
        self._guard_contact()
        if type(records) is not tuple or len(records) > len(self.program.targets):
            raise ValueError("native contact checkpoint record extent")
        body = dict(schema=CHECKPOINT_SCHEMA, profile=self.profile,
                    identity=self.identity, completed_targets=len(records),
                    records=records)
        raw = canonical({**body, "checkpoint_sha256": sha(body)})
        if len(raw) > MAX_BYTES:
            raise ValueError("native contact checkpoint byte limit")
        self.context.guard()
        return raw

    def _restore(self, raw: bytes, expected_sha256: str):
        self._guard_contact()
        if (type(raw) is not bytes or type(expected_sha256) is not str
                or sha256(raw).hexdigest() != expected_sha256):
            raise ValueError("native contact external checkpoint hash mismatch")
        value = _load(raw.decode("ascii"))
        keys = {"schema", "profile", "identity", "completed_targets",
                "records", "checkpoint_sha256"}
        if type(value) is not dict or set(value) != keys:
            raise ValueError("native contact checkpoint schema")
        body = {key: item for key, item in value.items()
                if key != "checkpoint_sha256"}
        if (value["schema"] != CHECKPOINT_SCHEMA or value["profile"] != self.profile
                or value["identity"] != self.identity
                or value["checkpoint_sha256"] != sha(body)):
            raise ValueError("native contact checkpoint definition/settings binding")
        count = value["completed_targets"]
        if (type(count) is not int or not 0 <= count <= len(self.program.targets)
                or type(value["records"]) is not list or len(value["records"]) != count):
            raise ValueError("native contact checkpoint target extent")
        accepted = self.initial
        previous = sha(dict(schema=CHECKPOINT_SCHEMA, identity=self.identity))
        records = ()
        for index, row in enumerate(value["records"], 1):
            if (type(row) is not dict or set(row) !=
                    {"target", "previous_sha256", "mechanical", "histories", "contact"}
                    or row["target"] != index or row["previous_sha256"] != previous):
                raise ValueError("native contact checkpoint accepted chain")
            mechanical = self.context.make(row["mechanical"], decoded=True)
            proposed, replay = self._record(
                index, mechanical, accepted.histories, previous,
            )
            if canonical(row) != canonical(replay):
                raise ValueError("native contact checkpoint physical replay mismatch")
            accepted = proposed
            records = (*records, replay)
            previous = sha(replay)
        if self._checkpoint(records) != raw:
            raise ValueError("native contact noncanonical checkpoint replay")
        return accepted, records

    def _contact(self, mechanical) -> NativeRetainedLinePlaneResult:
        nodes = self.context.nodes[0]
        try:
            value=self.contact.evaluate(
                mechanical.positions[nodes], mechanical.position_low[nodes],
                mechanical.cell_rotations[0],
            )
        except (ValueError,MemoryError) as error:
            if type(self.contact) is NativeCurvedMaterialPlaneWork:
                error.contact_diagnostics=dict(category='curved_contact_evaluation_non_success',
                    definition_sha256=self.contact.identity,reason=str(error),
                    integration=getattr(error,'integration_diagnostics',None),
                    positions=mechanical.positions[nodes].tolist(),
                    position_low=mechanical.position_low[nodes].tolist(),
                    cell_rotations=mechanical.cell_rotations[0].tolist())
            raise
        if type(self.contact) is NativeCurvedMaterialPlaneWork:
            return _CurvedPlaneContribution(value.normal_storage_j,value.residual,value.tangent,
                value.source_force,value.target_force,value.source_moment,value.target_moment,
                value.geometry.gap_lower_m,value.intervals,value.integrand_evaluations,
                *value.estimated_errors)
        return value

    def _swept_admissible(self, start, increment, end,
                          start_gap: float, end_gap: float,*,cancellation_token=None) -> bool:
        """Bound the minimum plane gap along one native exponential update."""
        self._guard_contact()
        if min(start_gap, end_gap) < -self.gap_tolerance:
            return False
        nodes = self.context.nodes[0]
        if type(self.contact) is NativeCurvedMaterialPlaneWork:
            from .native_surface_s3_friction import NativeS3MaterialConfiguration
            pose=NativeS3MaterialConfiguration(start.positions[nodes],start.position_low[nodes],
                start.cell_rotations[0],np.zeros(0))
            local=increment[list(self.context.slots[0])]
            translations=np.array([local[6*i:6*i+3] for i in range(3)])
            control=self.contact.certify_path(pose,translations,local[18:24].reshape(2,3),
                gap_tolerance_m=self.gap_tolerance,maximum_intervals=self.maximum_swept_intervals,
                cancellation_token=cancellation_token)
            self.curved_last_path_control=control
            return control.certified
        before = start.positions[nodes]+start.position_low[nodes]
        after = end.positions[nodes]+end.position_low[nodes]
        normal = self.contact.normal
        translation_lipschitz = float(np.max(np.abs((after-before) @ normal)))
        cell_first = self.context.nodal_count
        rotation_lipschitz = max(
            float(np.linalg.norm(increment[cell_first+3*cell:
                                           cell_first+3*cell+3])
                  * (np.linalg.norm(self.contact.reference.half_cell_lift(cell, 0.5))
                     + (self.contact.radius if type(self.contact) is NativeRoundMaterialPlane else 0.)))
            for cell in (0, 1)
        )
        if type(self.contact) is NativeRoundMaterialPlane:
            from .native_surface_s3_friction import NativeS3MaterialConfiguration
            pose=NativeS3MaterialConfiguration(start.positions[nodes],start.position_low[nodes],
                                               start.cell_rotations[0],np.zeros(0))
            local=increment[np.asarray(self.context.slots[0],dtype=int)]
            translations=np.array([local[6*i:6*i+3] for i in range(3)])
            try:
                self.contact.rule.certify_orientation_path(pose,translations,local[18:24].reshape(2,3),
                    np.zeros(0),maximum_intervals=self.maximum_swept_intervals)
            except ValueError as exc:
                if 'requires subdivision' not in str(exc):raise
                return False
        lipschitz = translation_lipschitz+rotation_lipschitz
        roundoff = 256.0*np.finfo(float).eps*max(
            1.0, float(np.max(np.abs(before))), float(np.max(np.abs(after))),
            self.contact.radius,
        )
        stack = [(0.0, 1.0)]
        visited = 0
        while stack:
            left, right = stack.pop()
            visited += 1
            if visited > self.maximum_swept_intervals:
                return False
            middle = 0.5*(left+right)
            state = self.context.advance(start, middle*increment)
            gap = (self.contact.minimum_gap(state.positions[nodes],state.position_low[nodes],state.cell_rotations[0])
                   if type(self.contact) is NativeRoundMaterialPlane else self._contact(state).minimum_gap_m)
            if gap < -self.gap_tolerance:
                return False
            if gap-0.5*(right-left)*lipschitz-roundoff >= -self.gap_tolerance:
                continue
            stack.extend(((left, middle), (middle, right)))
        return True

    def _assemble(self, mechanical, parameter, origins):
        self._guard_contact()
        self.native._guard()
        r, j, _structural_metrics, responses, _works = self.context.assemble(
            mechanical, parameter, origins,
        )
        contact = self._contact(mechanical)
        slots = self.context.slots[0]
        r[slots] += contact.residual
        j[np.ix_(slots, slots)] += (contact.spatial_tangent()
            if type(self.contact) in (NativeRoundMaterialPlane,NativeCurvedMaterialPlaneWork) else
            spatial_jacobian(contact.residual, contact.tangent))
        scaled = r/self.context.scale
        metrics = (float(np.linalg.norm(scaled[self.context.equilibrium])),
                   float(np.linalg.norm(scaled[self.context.compatibility])))
        return r, j, metrics, responses, contact

    def solve(self, *, checkpoint: bytes | None = None,
              expected_sha256: str | None = None,
              stop_after: int | None = None,
              cancellation_token=None) -> NativeRetainedContactRun:
        end = len(self.program.targets) if stop_after is None else stop_after
        if type(end) is not int or not 0 <= end <= len(self.program.targets):
            raise ValueError("native contact bounded stop target")
        if checkpoint is None:
            if expected_sha256 is not None:
                raise ValueError("native contact hash requires checkpoint")
            accepted, records = self.initial, ()
            capsule = self._checkpoint(records)
        else:
            accepted, records = self._restore(checkpoint, expected_sha256)
            capsule = checkpoint
        if accepted.completed_targets > end:
            raise ValueError("native contact accepted target cannot rewind")
        try:
            cancellation_safe_point(cancellation_token, 'native contact continuation')
            for index in range(accepted.completed_targets, end):
                parameter = self.program.targets[index]
                trial = accepted.mechanical
                origins = accepted.histories
                for iteration in range(self.program.max_iterations+1):
                    cancellation_safe_point(cancellation_token, 'native contact equilibrium')
                    r, j, metrics, responses, contact = self._assemble(
                        trial, parameter, origins,
                    )
                    step, correction = self.context.step(trial, r, j)
                    if max(*metrics, correction) <= self.equilibrium_tolerance:
                        if contact.minimum_gap_m < -self.gap_tolerance:
                            raise RuntimeError("native contact endpoint gap tolerance not met")
                        previous = (sha(records[-1]) if records else sha(dict(
                            schema=CHECKPOINT_SCHEMA, identity=self.identity,
                        )))
                        proposed, record = self._record(
                            index+1, trial, origins, previous,
                        )
                        next_records = (*records, record)
                        staged = self._checkpoint(next_records)
                        cancellation_safe_point(cancellation_token, 'native contact atomic commit')
                        accepted, records, capsule = proposed, next_records, staged
                        break
                    if iteration == self.program.max_iterations:
                        raise RuntimeError("native contact Newton iteration limit")
                    angular = [step[6*i+3:6*i+6]
                               for i in range(len(self.context.node_ids))]
                    angular.extend((step[self.context.nodal_count+3*c:
                                         self.context.nodal_count+3*c+3]
                                    for c in (0, 1)))
                    fraction = min(
                        1.0, 0.45*np.pi/max(
                            max(float(np.linalg.norm(value)) for value in angular),
                            np.finfo(float).tiny,
                        ),
                    )
                    if type(self.contact) is NativeCurvedMaterialPlaneWork:
                        # A proposal only: the same endpoint, whole-path and
                        # merit gates still decide acceptance. Avoid spending
                        # every backtrack on the unconstrained first-contact step.
                        full=self.context.advance(trial,step*fraction)
                        nodes=self.context.nodes[0]
                        try:
                            full_gap=self.contact.minimum_gap_bound(full.positions[nodes],
                                full.position_low[nodes],full.cell_rotations[0])
                        except ValueError as error:
                            if ('orientation' not in str(error) and
                                'native round surface singular or reversed at sample' not in str(error)):raise
                        else:
                            if full_gap < -self.gap_tolerance:
                                room=contact.minimum_gap_m+self.gap_tolerance
                                decrease=contact.minimum_gap_m-full_gap
                                if room<=0 or decrease<=0:
                                    raise RuntimeError('native curved contact has no admissible gap step')
                                fraction*=min(1.,.8*room/decrease)
                    for cut in range(self.program.max_backtracks+1):
                        cancellation_safe_point(cancellation_token, 'native contact backtrack')
                        candidate = self.context.advance(
                            trial, step*(fraction*0.5**cut),
                        )
                        if type(self.contact) is NativeCurvedMaterialPlaneWork:
                            self._guard_contact()
                            nodes=self.context.nodes[0]
                            try:
                                bound=self.contact.minimum_gap_bound(candidate.positions[nodes],
                                    candidate.position_low[nodes],candidate.cell_rotations[0])
                            except ValueError as error:
                                if ('orientation' not in str(error) and
                                    'native round surface singular or reversed at sample' not in str(error)):raise
                                self.curved_rejected_trials.append(dict(target=index+1,iteration=iteration,
                                    backtrack=cut,reason=str(error)))
                                continue
                            if bound < -self.gap_tolerance:
                                self.curved_rejected_trials.append(dict(target=index+1,iteration=iteration,
                                    backtrack=cut,reason='whole_patch_gap_bound',gap_bound_m=bound))
                                continue
                        _, _, changed, _, candidate_contact = self._assemble(
                            candidate, parameter, origins,
                        )
                        path_ok=(candidate_contact.minimum_gap_m >= -self.gap_tolerance
                                and self._swept_admissible(
                                    trial, step*(fraction*0.5**cut), candidate,
                                    contact.minimum_gap_m,
                                    candidate_contact.minimum_gap_m,
                                    cancellation_token=cancellation_token,
                                ))
                        if path_ok and max(changed) < max(metrics):
                            trial = candidate
                            break
                        if type(self.contact) is NativeCurvedMaterialPlaneWork:
                            self.curved_rejected_trials.append(dict(target=index+1,iteration=iteration,
                                backtrack=cut,reason='coupled_merit' if path_ok else 'whole_patch_path',
                                metrics=tuple(changed),previous_metrics=tuple(metrics),
                                gap_bound_m=candidate_contact.minimum_gap_m,
                                path_reason=getattr(getattr(self,'curved_last_path_control',None),'reason',None)))
                    else:
                        raise RuntimeError("native contact line-search/gap limit")
            status = "completed" if end == len(self.program.targets) else "paused"
            return NativeRetainedContactRun(status, accepted, None, capsule, self.profile)
        except Exception as exc:
            return NativeRetainedContactRun(
                "cancelled" if isinstance(exc,SolveCancelled) else "failed",
                accepted, type(exc).__name__+": "+str(exc), capsule, self.profile,
                getattr(exc,'contact_diagnostics',None),
            )
