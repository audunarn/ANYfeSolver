"""Internal two-beam retained-coordinate native static contact development."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from fractions import Fraction
from math import fsum, isfinite, sqrt

import numpy as np

from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_retained_nodal_loading import (
    Context as NativeNodalContext,
    Program as NativeNodalProgram,
)
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_p5_seeded.codec import MAX_BYTES, _load
from anyfesolver._ge_beam3_p5.algebra import log_rotation

from .native_retained_cell_pair import (
    NativeRetainedCellPairResult,
    NativeRetainedSelectedCellPair,
)
from .native_retained_whole_pair import (
    NativeRetainedWholeCurvePair,
    NativeRetainedWholePairResult,
)
from .native_replay_diagnostics import replay_difference_message


def _directed_fraction(value, upward):
    result = float(value)
    if not isfinite(result):
        raise ValueError('native swept bound arithmetic is unresolved')
    if (Fraction(result) < value if upward else Fraction(result) > value):
        result = float(np.nextafter(result, np.inf if upward else -np.inf))
    return result


def _outward_norm(values):
    squared = sum((Fraction(float(value))**2 for value in values), Fraction(0))
    result = sqrt(_directed_fraction(squared, True))
    if Fraction(result)**2 < squared:
        result = float(np.nextafter(result, np.inf))
    return result


def _swept_distance_chord_lower(g0, g1, width, speed, acceleration, radius, roundoff):
    """Lower gap for the complete moving material-pair distance envelope.

    On a positive distance floor, every fixed pair has norm second derivative
    <= V**2/d0+A. Its endpoint distances exceed both global endpoint lowers.
    The chord remainder <= h**2*(V**2/d0+A)/8 therefore holds after taking
    the infimum over material pairs, including changes of nearest feature.
    Scalar arithmetic encloses the exact supplied binary64 values outward.
    """
    values = (g0, g1, width, speed, acceleration, radius, roundoff)
    if (not all(isfinite(value) for value in values)
            or min(width, speed, acceleration, roundoff) < 0 or radius <= 0):
        raise ValueError('native swept bound requires finite physical data')
    gap, h, v, a, rho, error = map(Fraction, (min(g0, g1), width, speed,
                                           acceleration, radius, roundoff))
    distance_floor = rho+gap-h*v/2-error
    if distance_floor <= 0:
        return -np.inf
    return _directed_fraction(gap-h*h*(v*v/distance_floor+a)/8-error, False)


PROFILE = "NATIVE_B3_GE_SELECTED_CELL_PAIR_STATIC_DEV_V1"
CHECKPOINT_SCHEMA = "NATIVE_RETAINED_SELECTED_CELL_PAIR_CHECKPOINT_DEV_V1"
WHOLE_PROFILE = "NATIVE_B3_GE_WHOLE_CURVE_PAIR_STATIC_DEV_V4"
WHOLE_CHECKPOINT_SCHEMA = "NATIVE_RETAINED_WHOLE_CURVE_PAIR_CHECKPOINT_DEV_V4"
FIBRE_WHOLE_PROFILE = "NATIVE_B3_GE_PHYSICAL_FIBRE_WHOLE_CURVE_PAIR_STATIC_DEV_V1"
FIBRE_WHOLE_CHECKPOINT_SCHEMA = "NATIVE_PHYSICAL_FIBRE_WHOLE_CURVE_PAIR_CHECKPOINT_DEV_V1"


def _gap_limited_fraction(fraction, origin_gap, candidate_gap, tolerance):
    """Suggest a shorter trial; gap and swept checks must still certify it.

    Interpolate towards the interior of the permitted gap budget. This is
    globalization, not a crossing certificate or a change to the normal law.
    Safeguards enforce strict progress even on a nonmonotone gap branch.
    """
    target=min(-.5*tolerance,.5*(origin_gap-tolerance))
    predicted=fraction*(origin_gap-target)/(origin_gap-candidate_gap)
    return min(.9*fraction,max(.1*fraction,predicted))


@dataclass(frozen=True)
class NativeRetainedPairAccepted:
    mechanical: object
    histories: tuple
    completed_targets: int
    contact: NativeRetainedCellPairResult | NativeRetainedWholePairResult


@dataclass(frozen=True)
class NativeRetainedPairRun:
    status: str
    accepted: NativeRetainedPairAccepted
    failure: str | None
    checkpoint: bytes
    profile: str = PROFILE

    @property
    def checkpoint_sha256(self) -> str:
        return sha256(self.checkpoint).hexdigest()


class NativeRetainedRoundPairAnalysis:
    """Two disjoint native beams with an exact admitted pair contact kernel."""

    def __init__(self, definitions: tuple[NativeBeamDefinition, ...],
                 boundaries: tuple, program: NativeNodalProgram,
                 contact: NativeRetainedSelectedCellPair | NativeRetainedWholeCurvePair, *,
                 gap_tolerance_m: float, equilibrium_tolerance: float,
                 maximum_swept_intervals: int, gap_step_control: bool = False,
                 initial_trial: dict | None = None, reuse_target_trials: bool = False):
        if type(gap_step_control) is not bool:
            raise TypeError('native pair gap step control must be bool')
        if type(reuse_target_trials) is not bool:
            raise TypeError('native pair repeated target trial control must be bool')
        if (type(definitions) is not tuple or len(definitions) != 2
                or any(type(item) is not NativeBeamDefinition for item in definitions)):
            raise ValueError("native pair owner needs two exact beam definitions")
        if type(program) is not NativeNodalProgram:
            raise TypeError("native pair owner needs exact retained nodal program")
        if type(contact) not in (NativeRetainedSelectedCellPair,
                                 NativeRetainedWholeCurvePair):
            raise TypeError("native pair owner needs exact distributed pair kernel")
        self.profile = (PROFILE if type(contact) is NativeRetainedSelectedCellPair
                        else WHOLE_PROFILE)
        self.checkpoint_schema = (
            CHECKPOINT_SCHEMA if self.profile == PROFILE
            else WHOLE_CHECKPOINT_SCHEMA
        )
        controls = (gap_tolerance_m, equilibrium_tolerance)
        if any(type(item) not in (int, float) or not isfinite(item) or item <= 0.0
               for item in controls):
            raise ValueError("native pair owner needs positive gap/equilibrium tolerances")
        if type(maximum_swept_intervals) is not int or maximum_swept_intervals < 1:
            raise ValueError("native pair owner needs finite swept interval budget")
        self.native = NativeBeamAnalysis(definitions, boundaries)
        if self.native._family=='PHYSICAL_FIBRE_NODAL':
            if type(contact) is not NativeRetainedWholeCurvePair:
                raise ValueError('physical-fibre pair currently requires whole-curve geometry')
            from .native_fibre_context import NativeFibreContactContext,NativeRetainedFibreElement
            self.context=NativeFibreContactContext(self.native,program)
            expected=NativeRetainedFibreElement
            self.profile=FIBRE_WHOLE_PROFILE
            self.checkpoint_schema=FIBRE_WHOLE_CHECKPOINT_SCHEMA
        else:
            if gap_step_control:
                raise ValueError('gap step control requires the physical-fibre whole-pair profile')
            if (type(contact) is NativeRetainedWholeCurvePair
                    and contact.certify_curved_ownership):
                raise ValueError('curved ownership requires the physical-fibre pair profile')
            if (type(contact) is NativeRetainedWholeCurvePair
                    and contact.certify_open_intervals):
                raise ValueError('chord open certificates require the physical-fibre pair profile')
            self.context = NativeNodalContext(self.native.model, program)
            expected=NativeGeneralizedStaticElement
        if (len(self.context.elements) != 2
                or any(type(element) is not expected
                       for _, element in self.context.elements)):
            raise ValueError("native pair owner needs two exact homogeneous retained elements")
        ids = (contact.cells_a[0].element_id, contact.cells_b[0].element_id)
        index = {element_id: i for i, (element_id, _) in enumerate(self.context.elements)}
        if set(index) != set(ids):
            raise ValueError("native pair contact beam identities do not bind model")
        self.indices = (index[ids[0]], index[ids[1]])
        a, b = (self.context.elements[i][1] for i in self.indices)
        if (sha(a.operator.reference.canonical_data())
                != sha(contact.cells_a[0].reference.canonical_data())
                or sha(b.operator.reference.canonical_data())
                != sha(contact.cells_b[0].reference.canonical_data())):
            raise ValueError("native pair contact references do not bind model")
        if set(a.node_ids) & set(b.node_ids):
            raise ValueError("selected native pair does not admit shared material nodes")
        if (type(contact) is NativeRetainedWholeCurvePair
                and (tuple(a.node_ids) != contact.node_ids_a
                     or tuple(b.node_ids) != contact.node_ids_b)):
            raise ValueError("native whole-pair material node identities do not bind model")
        self.contact = contact
        self.program = program
        self.gap_tolerance = float(gap_tolerance_m)
        self.equilibrium_tolerance = float(equilibrium_tolerance)
        self.maximum_swept_intervals = maximum_swept_intervals
        self.gap_step_control = gap_step_control
        self.reuse_target_trials = reuse_target_trials
        if reuse_target_trials and not self._uses_physical_gap():
            raise ValueError('repeated native pair trial requires explicit curved physical geometry')
        self.initial_trial = None
        self._trial_state = None
        if initial_trial is not None:
            if not self._uses_physical_gap():
                raise ValueError('initial native pair trial requires explicit curved physical geometry')
            if (type(initial_trial) is not dict or set(initial_trial) != {'native_model','mechanical'}
                    or initial_trial['native_model'] != self.context.identity):
                raise ValueError('initial native pair trial model binding')
            self._trial_state = self.context.make(initial_trial['mechanical'])
            self.initial_trial = canonical(dict(native_model=self.context.identity,
                                                mechanical=self._trial_state.descriptor()))
            if len(self.initial_trial) > MAX_BYTES:
                raise ValueError('initial native pair trial byte limit')
            # Validate supported motion/chart now; admission remains disposable
            # and is checked again inside the cancellation-aware solve operation.
            self._trial_increment(self.context.initial.mechanical, self._trial_state)
        self.identity = self._identity()
        initial = self.context.initial
        first = self._contact(initial.mechanical)
        roundoff = 256.0*np.finfo(float).eps*max(
            1.0, *(float(np.max(np.abs(element.operator.reference.coordinates)))
                   for _, element in self.context.elements),
            contact.radius,
        )
        if first.conservative_lower_gap_m < -roundoff:
            raise ValueError(
                f"initial native pair gap not certified: conservative lower "
                f"bound {first.conservative_lower_gap_m} m"
            )
        self.initial = NativeRetainedPairAccepted(
            initial.mechanical, initial.histories, 0, first,
        )

    def _identity(self) -> str:
        values=dict(
            profile=self.profile,
            native_model=self.context.identity,
            native_support_sha256=sha256(self.context.support_identity).hexdigest(),
            contact=self.contact.identity,
            gap_tolerance_m=self.gap_tolerance,
            equilibrium_tolerance=self.equilibrium_tolerance,
            maximum_swept_intervals=self.maximum_swept_intervals,
        )
        # Preserve the historical false/default checkpoint identity exactly.
        if self.gap_step_control:
            values['gap_step_control']='SAFEGUARDED_CERTIFIED_GAP_SECANT_GEOMETRY_PATH_DEV_V2'
            if (type(self.contact) is NativeRetainedWholeCurvePair
                    and self.contact.certify_curved_ownership):
                values['gap_step_control']='SAFEGUARDED_RESOLVED_PHYSICAL_GAP_SECANT_DEV_V3'
        if self._uses_physical_gap():
            # Bind every curved owner, including owners without secant control.
            values['swept_gap_policy']='RETAINED_EXP_PATH_DISTANCE_CHORD_DEV_V14'
        if self.initial_trial is not None:
            values['initial_trial']=dict(policy='VIRGIN_MODEL_BOUND_RETAINED_TRIAL_DEV_V1',
                                        sha256=sha256(self.initial_trial).hexdigest())
        if self.reuse_target_trials:
            values['repeated_target_trial']='LAST_ACCEPTED_SAME_PARAMETER_MECHANICAL_TRIAL_DEV_V1'
        return sha(values)

    def _guard(self) -> None:
        if type(self.reuse_target_trials) is not bool:
            raise ValueError('native pair repeated target trial setting changed')
        if self._uses_physical_gap() and self.program is not self.context.program:
            raise ValueError('native pair bound program changed')
        if self._identity() != self.identity:
            raise ValueError("native pair definition/settings changed")
        if self.contact._fingerprint() != self.contact.identity:
            raise ValueError("native pair contact definition changed")
        if self.initial_trial is not None and canonical(dict(
                native_model=self.context.identity,
                mechanical=self._trial_state.descriptor())) != self.initial_trial:
            raise ValueError('initial native pair trial definition changed')
        self.native._guard()
        if self.native._family=='PHYSICAL_FIBRE_NODAL':self.context.guard()

    def _record(self, target: int, mechanical, origins,
                previous_sha256: str, prepared=None):
        if type(target) is not int or not 1 <= target <= len(self.program.targets):
            raise ValueError("native pair checkpoint target")
        if prepared is not None and (prepared[0] is not mechanical
                or prepared[1] is not self.program.targets[target-1]
                or prepared[2] is not origins):
            raise ValueError("native pair prepared record requires the exact accepted pose, target and origins")
        mechanical = self.context.make(mechanical.descriptor())
        if prepared is None:
            residual, tangent, metrics, responses, contact = self._assemble(
                mechanical, self.program.targets[target-1], origins,
            )
        else:
            # The converged evaluation of this exact accepted pose: the same
            # deterministic assemble, parameter and origins that just
            # certified equilibrium. Reusing it removes the redundant second
            # full contact evaluation of an unchanged pose; the cached branch
            # rechecks the guard exactly as the assembly path does, every
            # checkpoint gate below still runs on those identical values, and
            # restore replay re-assembles each record from scratch.
            self._guard()
            residual, tangent, metrics, responses, contact = prepared[3]
        _, correction = self.context.step(mechanical, residual, tangent)
        if max(*metrics, correction) > self.equilibrium_tolerance:
            raise ValueError("native pair checkpoint requires coupled equilibrium")
        if contact.conservative_lower_gap_m < -self.gap_tolerance:
            raise ValueError("native pair checkpoint gap tolerance")
        histories = self.context.histories(
            tuple(response.history for response in responses)
        )
        summary = dict(
            potential_j=contact.potential_j,
            conservative_lower_gap_m=contact.conservative_lower_gap_m,
            beam_a_force_n=contact.beam_a_force_n,
            beam_b_force_n=contact.beam_b_force_n,
            beam_a_moment_nm=contact.beam_a_moment_nm,
            beam_b_moment_nm=contact.beam_b_moment_nm,
        )
        if type(contact) is NativeRetainedCellPairResult:
            summary.update(
                sampled_gap_m=contact.sampled_gap_m,
                integration_points=contact.integration_points,
                integrated_intervals=contact.integrated_intervals,
                projection_evaluations=contact.projection_evaluations,
                maximum_relative_integration_error=(
                    contact.maximum_relative_integration_error
                ),
                gap_cells_checked=contact.gap_cells_checked,
            )
        else:
            summary.update(
                minimum_sampled_gap_m=contact.minimum_sampled_gap_m,
                energy_error_j=contact.energy_error_j,
                force_error_n=contact.force_error_n,
                tangent_error_n_per_m=contact.tangent_error_n_per_m,
                search_cells=contact.search_cells,
                integration_intervals=contact.integration_intervals,
                integration_evaluations=contact.integration_evaluations,
                integration_cells_checked=contact.integration_cells_checked,
                bounded_omission_cells=contact.bounded_omission_cells,
                unresolved_search_intervals=contact.unresolved_search_intervals,
            )
        row = dict(target=target, previous_sha256=previous_sha256,
                   mechanical=mechanical.descriptor(), histories=histories,
                   contact=summary)
        return NativeRetainedPairAccepted(
            mechanical, histories, target, contact,
        ), _load(canonical(row).decode("ascii"))

    def _checkpoint(self, records: tuple[dict, ...]) -> bytes:
        self._guard()
        if type(records) is not tuple or len(records) > len(self.program.targets):
            raise ValueError("native pair checkpoint record extent")
        body = dict(schema=self.checkpoint_schema, profile=self.profile,
                    identity=self.identity, completed_targets=len(records),
                    records=records)
        raw = canonical({**body, "checkpoint_sha256": sha(body)})
        if len(raw) > MAX_BYTES:
            raise ValueError("native pair checkpoint byte limit")
        return raw

    def _restore(self, raw: bytes, expected_sha256: str):
        self._guard()
        if (type(raw) is not bytes or type(expected_sha256) is not str
                or sha256(raw).hexdigest() != expected_sha256):
            raise ValueError("native pair external checkpoint hash mismatch")
        value = _load(raw.decode("ascii"))
        keys = {"schema", "profile", "identity", "completed_targets",
                "records", "checkpoint_sha256"}
        if type(value) is not dict or set(value) != keys:
            raise ValueError("native pair checkpoint schema")
        body = {key: item for key, item in value.items()
                if key != "checkpoint_sha256"}
        if (value["schema"] != self.checkpoint_schema
                or value["profile"] != self.profile
                or value["identity"] != self.identity
                or value["checkpoint_sha256"] != sha(body)):
            raise ValueError("native pair checkpoint definition/settings binding")
        count = value["completed_targets"]
        if (type(count) is not int or not 0 <= count <= len(self.program.targets)
                or type(value["records"]) is not list
                or len(value["records"]) != count):
            raise ValueError("native pair checkpoint target extent")
        accepted = self.initial
        previous = sha(dict(schema=self.checkpoint_schema,
                            identity=self.identity))
        records = ()
        for index, row in enumerate(value["records"], 1):
            if (type(row) is not dict or set(row) !=
                    {"target", "previous_sha256", "mechanical", "histories", "contact"}
                    or row["target"] != index or row["previous_sha256"] != previous):
                raise ValueError("native pair checkpoint accepted chain")
            mechanical = self.context.make(row["mechanical"], decoded=True)
            proposed, replay = self._record(
                index, mechanical, accepted.histories, previous,
            )
            if canonical(row) != canonical(replay):
                raise ValueError("native pair checkpoint physical replay mismatch"
                                 + replay_difference_message(row, replay, "record %d" % index))
            accepted = proposed
            records = (*records, replay)
            previous = sha(replay)
        if self._checkpoint(records) != raw:
            raise ValueError("native pair noncanonical checkpoint replay")
        return accepted, records

    def _contact(self, mechanical) -> (
        NativeRetainedCellPairResult | NativeRetainedWholePairResult
    ):
        a, b = self.indices
        result = self.contact.evaluate(
            mechanical.positions[self.context.nodes[a]],
            mechanical.position_low[self.context.nodes[a]],
            mechanical.cell_rotations[a],
            mechanical.positions[self.context.nodes[b]],
            mechanical.position_low[self.context.nodes[b]],
            mechanical.cell_rotations[b],
        )
        if self._uses_physical_gap():
            result = replace(result, conservative_lower_gap_m=self._physical_gap(mechanical))
        return result

    def _endpoint_contact(self, mechanical):
        """Prefilter physical penetration before contact Jet/integration work."""
        if self._uses_physical_gap():
            lower = self._physical_gap(mechanical)
            if lower < -self.gap_tolerance:
                return None, lower
        contact = self._contact(mechanical)
        return contact, contact.conservative_lower_gap_m

    def _uses_physical_gap(self):
        return (type(self.contact) is NativeRetainedWholeCurvePair
                and self.contact.certify_curved_ownership)

    def _physical_gap(self, mechanical):
        self._guard()
        self.context._safe_point()
        a, b = self.indices
        value = self.contact.physical_gap_lower(
            mechanical.positions[self.context.nodes[a]],
            mechanical.position_low[self.context.nodes[a]],
            mechanical.cell_rotations[a],
            mechanical.positions[self.context.nodes[b]],
            mechanical.position_low[self.context.nodes[b]],
            mechanical.cell_rotations[b],
        )
        self._guard()
        self.context._safe_point()
        return value

    def _recoverable_trial_contact_error(self, exc: Exception) -> bool:
        """Only geometric/resource non-admission may trigger a path cutback."""
        message = str(exc)
        if type(self.contact) is NativeRetainedSelectedCellPair:
            return (type(exc) is ValueError
                    and "undeclared native half-cell pair" in message)
        return (
            type(exc) in (ValueError, MemoryError)
            and message.startswith((
                "native whole pair integration interval budget exhausted",
                "native whole pair integration parameter resolution exhausted",
                "native whole pair has competing target minima",
                "native whole pair target ownership is unresolved",
                "native union has competing",
                "native union search cell budget exhausted",
                "native union search depth budget exhausted",
                "B3 projection has competing equal-distance minima",
                "B3 closest-point branch lacks positive curvature",
                "coincident B3 centrelines cannot define a contact normal",
            ))
        )

    def _assemble(self, mechanical, parameter, origins, *, prepared_contact=None):
        self._guard()
        if prepared_contact is not None and prepared_contact[0] is not mechanical:
            raise ValueError('native pair prepared contact requires the exact candidate pose')
        residual, tangent, _, responses, _ = self.context.assemble(
            mechanical, parameter, origins,
        )
        contact = (self._contact(mechanical) if prepared_contact is None
                   else prepared_contact[1])
        slots = self.context.slots[self.indices[0]]+self.context.slots[self.indices[1]]
        np.add.at(residual, slots, contact.residual)
        block = contact.tangent.copy()
        for first in (0, 42):
            block[first:first+42, first:first+42] = spatial_jacobian(
                contact.residual[first:first+42],
                block[first:first+42, first:first+42],
            )
        rows, columns = np.meshgrid(slots, slots, indexing="ij")
        np.add.at(tangent, (rows, columns), block)
        scaled = residual/self.context.scale
        metrics = (float(np.linalg.norm(scaled[self.context.equilibrium])),
                   float(np.linalg.norm(scaled[self.context.compatibility])))
        return residual, tangent, metrics, responses, contact

    def _swept_gap(self, mechanical):
        if self._uses_physical_gap():
            return self._physical_gap(mechanical)
        if (not self.gap_step_control
                and type(self.contact) is not NativeRetainedWholeCurvePair):
            return self._contact(mechanical).conservative_lower_gap_m
        # A swept midpoint needs the complete physical gap certificate, not
        # residuals or tangents. The whole-pair geometry query validates every
        # retained cell and uses the same finite search controls. Its early
        # return proves the requested minimum over the complete pair; endpoint
        # equilibrium and checkpoint acceptance still use the full force law.
        # This does not enable the separate Newton gap-step controller.
        self._guard()
        a,b=self.indices
        value=self.contact.geometry_gap_lower(
            mechanical.positions[self.context.nodes[a]],
            mechanical.position_low[self.context.nodes[a]],
            mechanical.cell_rotations[a],
            mechanical.positions[self.context.nodes[b]],
            mechanical.position_low[self.context.nodes[b]],
            mechanical.cell_rotations[b],
            minimum_gap_m=-self.gap_tolerance,
        )
        self._guard()
        return value

    def _swept_admissible(self, start, increment, end,
                          start_gap: float, end_gap: float) -> bool:
        self._guard()
        if min(start_gap, end_gap) < -self.gap_tolerance:
            return False
        curved = self._uses_physical_gap()
        if curved:
            self.context._safe_point()
            speed, acceleration = self._swept_motion_bounds(start, increment)
        else:
            speed = 0.0
            acceleration = 0.0
        for index in self.indices:
            if curved:
                break
            nodes = self.context.nodes[index]
            before = start.positions[nodes]+start.position_low[nodes]
            after = end.positions[nodes]+end.position_low[nodes]
            translation = float(np.max(np.linalg.norm(after-before, axis=1)))
            reference = self.context.elements[index][1].operator.reference
            cell_first = self.context.nodal_count+24*index
            rotation = max(
                float(np.linalg.norm(increment[cell_first+3*c:
                                               cell_first+3*c+3])
                      * np.linalg.norm(reference.half_cell_lift(c, 0.5)))
                for c in (0, 1)
            )
            speed += translation+rotation
        roundoff = 256.0*np.finfo(float).eps*max(
            1.0, float(np.max(np.abs(start.positions))),
            float(np.max(np.abs(end.positions))), self.contact.radius,
            speed if curved else 0.0,
        )
        stack = [(0.0, 1.0, start_gap, end_gap)]
        visited = 0
        while stack:
            if curved:
                self._guard()
                self.context._safe_point()
            left, right, left_gap, right_gap = stack.pop()
            visited += 1
            if visited > self.maximum_swept_intervals:
                return False
            # Every path point is within half this interval of one endpoint.
            # Both stored gaps are conservative lower bounds, so this proves
            # safety without constructing another contact candidate.
            lower = min(left_gap, right_gap)-0.5*(right-left)*speed-roundoff
            if curved:
                first_order = Fraction(min(left_gap, right_gap))-Fraction(right-left)*Fraction(speed)/2-Fraction(roundoff)
                lower = max(_directed_fraction(first_order, False),
                            _swept_distance_chord_lower(left_gap, right_gap,
                                right-left, speed, acceleration, self.contact.radius, roundoff))
            if lower >= -self.gap_tolerance:
                continue
            middle = 0.5*(left+right)
            state = self.context.advance(start, middle*increment)
            try:
                gap = self._swept_gap(state)
            except (ValueError, MemoryError) as exc:
                if self._recoverable_trial_contact_error(exc):
                    return False
                raise
            if gap < -self.gap_tolerance:
                return False
            midpoint_lower = gap-0.5*(right-left)*speed-roundoff
            if curved:
                midpoint_lower = _directed_fraction(Fraction(gap)-Fraction(right-left)*Fraction(speed)/2-Fraction(roundoff), False)
            if midpoint_lower >= -self.gap_tolerance:
                continue
            stack.extend(((left, middle, left_gap, gap),
                          (middle, right, gap, right_gap)))
        return True

    def _swept_motion_bounds(self, start, increment):
        """Uniform relative Q2 point speed/acceleration on layout.advance.

        Nodal increments translate affinely; cell arms rotate by exp(t*w)R.
        The Q2 arm multiplier 4*s*(1-s) is in [0,1]. Nodal frame and retained
        resultant increments do not move the physical centerline.
        """
        increment = np.asarray(increment)
        if increment.shape != (self.context.count,) or not np.all(np.isfinite(increment)):
            raise ValueError('native swept increment requires finite retained coordinates')
        speeds, accelerations = [], []
        eps = Fraction(float(np.finfo(float).eps))
        for index in self.indices:
            nodes = self.context.nodes[index]
            translation = max(_outward_norm(increment[6*node:6*node+3]) for node in nodes)
            reference = self.context.elements[index][1].operator.reference
            first = self.context.nodal_count+24*index
            velocity, acceleration = [], []
            for cell in (0, 1):
                lift = reference.half_cell_lift(cell, .5)
                frame = start.cell_rotations[index, cell]
                arm = [sum((Fraction(float(frame[i, j]))*Fraction(float(lift[j]))
                            for j in range(3)), Fraction(0)) for i in range(3)]
                # Matrix/exp evaluation uncertainty is charged before any
                # geometric cancellation, including nearly aligned arms.
                arm_norm = _outward_norm([_directed_fraction(x, x >= 0) for x in arm])
                arm_norm = Fraction(arm_norm)*(1+128*eps)
                omega = Fraction(_outward_norm(increment[first+3*cell:first+3*cell+3]))
                velocity.append(omega*arm_norm)
                acceleration.append(omega*omega*arm_norm)
            speeds.append(Fraction(translation)+max(velocity))
            accelerations.append(max(acceleration))
        return (_directed_fraction(sum(speeds), True),
                _directed_fraction(sum(accelerations), True))

    def _trial_increment(self, start, target):
        """Supported exponential-chart motion to a disposable mechanical guess."""
        increment = np.zeros(self.context.count)
        for node in range(len(self.context.node_ids)):
            for axis in range(3):
                increment[6*node+axis] = fsum((float(target.positions[node,axis]),
                    float(target.position_low[node,axis]),-float(start.positions[node,axis]),
                    -float(start.position_low[node,axis])))
            increment[6*node+3:6*node+6] = log_rotation(
                target.nodal_frames[node]@start.nodal_frames[node].T)
        for index in range(len(self.context.elements)):
            first = self.context.nodal_count+24*index
            for cell in (0,1):
                increment[first+3*cell:first+3*cell+3] = log_rotation(
                    target.cell_rotations[index,cell]@start.cell_rotations[index,cell].T)
            increment[first+6:first+24] = target.resultants[index]-start.resultants[index]
        # advance owns the exact support/chart/finite validation. The returned
        # endpoint, rather than an approximately reconstructed guess, is used
        # in both path admission and Newton; no endpoint discrepancy is hidden.
        endpoint = self.context.advance(start, increment)
        return increment, endpoint

    def _initial_trial_candidate(self, accepted):
        self._guard()
        self.context._safe_point()
        if self.initial_trial is None or accepted.completed_targets != 0:
            return accepted.mechanical, None
        if canonical(accepted.histories) != canonical(self.initial.histories):
            raise ValueError('initial native pair trial requires virgin accepted origins')
        increment, candidate = self._trial_increment(accepted.mechanical, self._trial_state)
        gap = self._physical_gap(candidate)
        if gap < -self.gap_tolerance or not self._swept_admissible(
                accepted.mechanical, increment, candidate,
                accepted.contact.conservative_lower_gap_m, gap):
            raise ValueError('initial native pair trial path is not certified')
        self._guard()
        self.context._safe_point()
        # This is not an accepted state/history. Only a converged _record can
        # publish an accepted target, and origins remain accepted.histories.
        return candidate, None

    def _target_trial_candidate(self, accepted, records, target):
        """A prior same-parameter mechanical guess with CURRENT material origins.

        records are the validated accepted prefix owned by solve/_restore.
        No accepted state/history is replaced. The current-to-guess supported
        path must be admitted before the ordinary Newton/material transaction.
        """
        self._guard()
        self.context._safe_point()
        if not self.reuse_target_trials:
            return accepted.mechanical, None
        if (type(target) is not int or target != accepted.completed_targets
                or len(records) != target or not 0 <= target < len(self.program.targets)):
            raise ValueError('repeated native pair trial requires validated preceding prefix')
        parameter = canonical(self.program.targets[target])
        matches = [index for index in range(target)
                   if canonical(self.program.targets[index]) == parameter]
        if not matches:
            return accepted.mechanical, None
        guess = self.context.make(records[matches[-1]]['mechanical'], decoded=True)
        increment, candidate = self._trial_increment(accepted.mechanical, guess)
        gap = self._physical_gap(candidate)
        if gap < -self.gap_tolerance or not self._swept_admissible(
                accepted.mechanical, increment, candidate,
                accepted.contact.conservative_lower_gap_m, gap):
            raise ValueError('repeated native pair trial path is not certified')
        self._guard()
        self.context._safe_point()
        return candidate, None

    def solve(self, *, checkpoint: bytes | None = None,
              expected_sha256: str | None = None,
              stop_after: int | None = None, cancellation_token=None) -> NativeRetainedPairRun:
        if self.native._family=='PHYSICAL_FIBRE_NODAL':
            with self.context.operation(cancellation_token):
                return self._solve(checkpoint=checkpoint,expected_sha256=expected_sha256,
                    stop_after=stop_after,cancellation_token=cancellation_token)
        if cancellation_token is not None:
            raise ValueError('cancellation token requires the physical-fibre pair profile')
        return self._solve(checkpoint=checkpoint,expected_sha256=expected_sha256,stop_after=stop_after)

    def _solve(self, *, checkpoint=None,expected_sha256=None,stop_after=None,cancellation_token=None):
        end = len(self.program.targets) if stop_after is None else stop_after
        if type(end) is not int or not 0 <= end <= len(self.program.targets):
            raise ValueError("native pair bounded stop target")
        if checkpoint is None:
            if expected_sha256 is not None:
                raise ValueError("native pair hash requires checkpoint")
            accepted, records = self.initial, ()
            capsule = self._checkpoint(records)
        else:
            accepted, records = self._restore(checkpoint, expected_sha256)
            capsule = checkpoint
        if accepted.completed_targets > end:
            raise ValueError("native pair accepted target cannot rewind")
        if self.native._family=='PHYSICAL_FIBRE_NODAL':self.context.activate_cancellation(cancellation_token)
        try:
            for index in range(accepted.completed_targets, end):
                parameter = self.program.targets[index]
                trial = accepted.mechanical
                origins = accepted.histories
                prepared_evaluation = None
                if index == 0 and self.initial_trial is not None:
                    trial, prepared_evaluation = self._initial_trial_candidate(accepted)
                elif self.reuse_target_trials:
                    trial, prepared_evaluation = self._target_trial_candidate(accepted, records, index)
                for iteration in range(self.program.max_iterations+1):
                    if prepared_evaluation is None:
                        residual, tangent, metrics, responses, contact = self._assemble(
                            trial, parameter, origins,
                        )
                    else:
                        self._guard()
                        residual, tangent, metrics, responses, contact = prepared_evaluation
                        prepared_evaluation = None
                    step, correction = self.context.step(trial, residual, tangent)
                    if max(*metrics, correction) <= self.equilibrium_tolerance:
                        if contact.conservative_lower_gap_m < -self.gap_tolerance:
                            raise RuntimeError("native pair endpoint gap tolerance not met")
                        previous = (sha(records[-1]) if records else sha(dict(
                            schema=self.checkpoint_schema, identity=self.identity,
                        )))
                        proposed, record = self._record(
                            index+1, trial, origins, previous,
                            prepared=(trial, parameter, origins,
                                      (residual, tangent, metrics,
                                       responses, contact)),
                        )
                        next_records = (*records, record)
                        staged = self._checkpoint(next_records)
                        if self.native._family == 'PHYSICAL_FIBRE_NODAL':
                            self.context._safe_point()
                        accepted, records, capsule = proposed, next_records, staged
                        break
                    if iteration == self.program.max_iterations:
                        raise RuntimeError("native pair Newton iteration limit")
                    angular = [step[6*i+3:6*i+6]
                               for i in range(len(self.context.node_ids))]
                    angular.extend(
                        step[self.context.nodal_count+24*i+3*c:
                             self.context.nodal_count+24*i+3*c+3]
                        for i in range(len(self.context.elements)) for c in (0, 1)
                    )
                    fraction = min(
                        1.0, 0.45*np.pi/max(
                            max(float(np.linalg.norm(value)) for value in angular),
                            np.finfo(float).tiny,
                        ),
                    )
                    rejected = dict(geometry=0, gap=0, swept=0, merit=0)
                    last_rejection = ""
                    trial_fraction = fraction
                    for cut in range(self.program.max_backtracks+1):
                        increment = step*trial_fraction
                        candidate = self.context.advance(trial, increment)
                        try:
                            candidate_contact, candidate_gap = self._endpoint_contact(candidate)
                            if candidate_contact is None:
                                rejected["gap"] += 1
                                last_rejection = f"resolved physical gap lower bound {candidate_gap} m"
                                trial_fraction = (_gap_limited_fraction(
                                    trial_fraction, contact.conservative_lower_gap_m,
                                    candidate_gap, self.gap_tolerance,
                                ) if self.gap_step_control else .5*trial_fraction)
                                continue
                        except (ValueError, MemoryError) as exc:
                            if self._recoverable_trial_contact_error(exc):
                                rejected["geometry"] += 1
                                last_rejection = (f"geometry {type(exc).__name__}: "
                                                  f"{str(exc)[:320]}")
                                trial_fraction *= .5
                                continue
                            raise
                        if (candidate_contact.conservative_lower_gap_m
                                < -self.gap_tolerance):
                            rejected["gap"] += 1
                            last_rejection = (
                                "gap lower bound "
                                f"{candidate_contact.conservative_lower_gap_m} m"
                            )
                            trial_fraction = (_gap_limited_fraction(
                                trial_fraction,contact.conservative_lower_gap_m,
                                candidate_contact.conservative_lower_gap_m,self.gap_tolerance,
                            ) if self.gap_step_control else .5*trial_fraction)
                            continue
                        if not self._swept_admissible(
                                    trial, increment, candidate,
                                    contact.conservative_lower_gap_m,
                                    candidate_contact.conservative_lower_gap_m,
                                ):
                            rejected["swept"] += 1
                            last_rejection = "swept gap not certified"
                            trial_fraction *= .5
                            continue
                        fibre = self.native._family == 'PHYSICAL_FIBRE_NODAL'
                        evaluated = self._assemble(candidate, parameter, origins,
                            prepared_contact=(candidate,candidate_contact) if fibre else None)
                        changed = evaluated[2]
                        if max(changed) < max(metrics):
                            trial = candidate
                            if fibre:prepared_evaluation = evaluated
                            break
                        rejected["merit"] += 1
                        last_rejection = f"merit {max(changed)} >= {max(metrics)}"
                        trial_fraction *= .5
                    else:
                        if type(self.contact) is NativeRetainedWholeCurvePair:
                            raise RuntimeError(
                                "native whole pair line-search/gap limit: "
                                f"iteration={iteration}, rejected={rejected}, "
                                f"last={last_rejection}"
                            )
                        raise RuntimeError("native pair line-search/gap limit")
            status = "completed" if end == len(self.program.targets) else "paused"
            return NativeRetainedPairRun(status, accepted, None, capsule,
                                         self.profile)
        except Exception as exc:
            return NativeRetainedPairRun(
                "failed", accepted, type(exc).__name__+": "+str(exc), capsule,
                self.profile,
            )
