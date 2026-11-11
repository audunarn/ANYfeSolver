"""Common-stage native beam/beam round-line penalty DEVELOPMENT owner.

Source candidate, numerically unverified. No child time steppers or contact
condensation: both complete retained bodies enter one midpoint equilibrium.
"""
from fractions import Fraction
from math import isfinite
import numpy as np

from ._ge_beam3_connected_dynamic import _RetainedMidpointAssembly
from ._ge_beam3_generalized_static_boundary import spatial_jacobian
from ._ge_beam3_p5_seeded.core import canonical
from .general_contact.native_retained_whole_pair import NativeRetainedWholeCurvePair

PROFILE = 'NATIVE_B3_GE_WHOLE_ROUND_PAIR_MIDPOINT_DEV_V1'
SCHEMA = 'NATIVE_B3_GE_WHOLE_ROUND_PAIR_MIDPOINT_STATE_DEV_V1'
TRAJECTORY_PROFILE = 'NATIVE_B3_GE_WHOLE_ROUND_PAIR_MIDPOINT_TRAJECTORY_DEV_V1'
TRAJECTORY_SCHEMA = 'NATIVE_B3_GE_WHOLE_ROUND_PAIR_MIDPOINT_JOURNAL_DEV_V1'
ENDPOINT_POLICY = 'CLOSED_CENTRELINE_DISTANCE_LINE_MEASURE'
PATH_FAILURE = 'native contact swept path unresolved'


class NativePairContactAnalysis(_RetainedMidpointAssembly):
    route_profile = PROFILE
    checkpoint_schema = SCHEMA
    trajectory_profile = TRAJECTORY_PROFILE
    trajectory_schema = TRAJECTORY_SCHEMA
    extra_step_keys = ('native_contact',)
    retryable = ('connected midpoint iteration limit',
                 'connected midpoint backtracking failed', PATH_FAILURE)

    def __init__(self, definitions, boundaries, contact, *, endpoint_policy,
                 maximum_penetration_m, maximum_swept_intervals,
                 inertia_order=24, max_iterations=12, max_backtracks=8):
        if type(contact) is not NativeRetainedWholeCurvePair:
            raise TypeError('exact complete two-cell whole native pair required')
        if not contact.certify_curved_ownership or not contact.pooled_quadrature_budget:
            raise ValueError('resolved pooled V14 whole-pair ownership service required')
        if not isfinite(contact.radius) or contact.radius != contact.radius_a+contact.radius_b:
            raise ValueError('finite exact derived native radius sum required')
        if endpoint_policy != ENDPOINT_POLICY:
            raise ValueError('explicit closed centreline line-measure endpoint policy required')
        if (type(maximum_penetration_m) not in (int, float)
                or not isfinite(maximum_penetration_m)
                or not contact.distance_tolerance < maximum_penetration_m < contact.radius):
            raise ValueError('penetration budget must exceed distance tolerance and be below radius sum')
        if type(maximum_swept_intervals) is not int or not 1 <= maximum_swept_intervals <= 4096:
            raise ValueError('bounded positive swept interval count required')
        self.contact = contact
        self.endpoint_policy = endpoint_policy
        self.maximum_penetration_m = float(maximum_penetration_m)
        self.maximum_swept_intervals = maximum_swept_intervals
        super().__init__(definitions, boundaries, inertia_order=inertia_order,
                         max_iterations=max_iterations, max_backtracks=max_backtracks)
        if self.fixed:
            raise ValueError('first two-body contact profile requires both bodies free')

    def _admit_graph(self):
        if len(self.elements) != 2 or set(self.elements[0].node_ids) & set(self.elements[1].node_ids):
            raise ValueError('two separate complete native bodies required')
        for element, cells, nodes in zip(self.elements,
                (self.contact.cells_a, self.contact.cells_b),
                (self.contact.node_ids_a, self.contact.node_ids_b)):
            if element.node_ids != nodes or any(c.element_id != element.element_id
                    or canonical(c.reference.canonical_data()) != canonical(element.operator.reference.canonical_data())
                    for c in cells):
                raise ValueError('contact/native complete physical curve routing mismatch')
            # Straight nodal coordinates alone do not establish zero physical lift.
            if any(np.any(element.operator.reference.half_cell_lift(c, .5)) for c in (0, 1)):
                raise ValueError('curved physical lift is not admitted by first contact midpoint profile')

    def _definition(self):
        data = super()._definition()
        data['native_contact'] = dict(identity=self.contact.identity,
            radius_sum_m=self.contact.radius,
            checkpoint_schema=self.checkpoint_schema, trajectory_profile=self.trajectory_profile,
            trajectory_schema=self.trajectory_schema, extra_step_keys=self.extra_step_keys,
            retryable=self.retryable,
            endpoint_policy=self.endpoint_policy, maximum_penetration_m=self.maximum_penetration_m,
            maximum_swept_intervals=self.maximum_swept_intervals,
            path='AFFINE_NODE_POSITIONS_EXP_ENDPOINT_CELL_CHART_ZERO_LIFT')
        return data

    def _guard(self, token=None, stage='native_pair_contact.guard'):
        super()._guard(token, stage)
        if self.contact._fingerprint() != self.contact.identity:
            raise ValueError('captured native contact definition changed')
        if not isfinite(self.contact.radius) or self.contact.radius != self.contact.radius_a+self.contact.radius_b:
            raise ValueError('captured native derived radius sum changed')

    def _poses(self, state):
        poses = []
        for k, element in enumerate(self.elements):
            nodes = [self.node_index[n] for n in element.node_ids]
            poses.extend((state.positions[nodes], state.position_low[nodes], state.cell_rotations[k]))
        return tuple(poses)

    def _gap(self, state, threshold):
        return self.contact.geometry_gap_lower(*self._poses(state), minimum_gap_m=threshold)

    def _evaluate(self, state, token=None):
        r, j, fi, mass, dv, dq, account = super()._evaluate(state, token)
        self._guard(token, 'native_pair_contact.evaluate')
        contact = self.contact.evaluate(*self._poses(state))
        block = contact.tangent.copy()
        for first in (0, 42):
            own = slice(first, first+42)
            block[own, own] = spatial_jacobian(contact.residual[own], block[own, own])
        slots = list(self.maps[0]+self.maps[1])
        r[slots] += contact.residual
        j[np.ix_(slots, slots)] += block
        account = dict(account, contact_regularization_storage=contact.potential_j,
                       total_energy=account['total_energy']+contact.potential_j,
                       native_contact=dict(profile=PROFILE,
            contact_kernel_identity=self.contact.identity,
            contact_kernel_service='NATIVE_RETAINED_WHOLE_Q2_PAIR_DEV_V14_POOLED_PHYSICAL_GAP_QUADRATURE',
            derivative_error_chart='30_UNIQUE_PHYSICAL_CONTROL_POINT_COORDINATES_IN_METRES',
            endpoint_policy=self.endpoint_policy, potential_j=contact.potential_j,
            moment_origin_m=(0.,0.,0.),
            force_semantics='PHYSICAL_CONTACT_FORCE_ON_BODY_NEGATIVE_POTENTIAL_GRADIENT',
            assembled_residual_semantics='INTERNAL_POTENTIAL_GRADIENT_ADDED_TO_NATIVE_EQUILIBRIUM',
            beam_a_force_n=contact.beam_a_force_n, beam_b_force_n=contact.beam_b_force_n,
            beam_a_moment_nm=contact.beam_a_moment_nm, beam_b_moment_nm=contact.beam_b_moment_nm,
            conservative_lower_gap_m=contact.conservative_lower_gap_m,
            minimum_sampled_gap_m=contact.minimum_sampled_gap_m,
            energy_error_j=contact.energy_error_j, force_error_n=contact.force_error_n,
            tangent_error_n_per_m=contact.tangent_error_n_per_m,
            search_cells=contact.search_cells, integration_intervals=contact.integration_intervals,
            unresolved_search_intervals=contact.unresolved_search_intervals))
        if not np.isfinite(r).all() or not np.isfinite(j).all() or not isfinite(account['total_energy']):
            raise ValueError('nonfinite assembled contact work map')
        return r, j, fi, mass, dv, dq, account

    def _initialize_state(self, velocity, token, physical_displacement=None):
        state = super()._initialize_state(velocity, token, physical_displacement)
        if self._gap(state, 0.) < 0.:
            raise ValueError('initial native beam pair penetration or unresolved separation')
        return state

    def _increment(self, state, unknown, dt):
        step = np.zeros(self.size)
        p = list(self.p)
        step[p] = dt*state.velocity[p]+.5*dt*dt*unknown[p]
        return step

    def _swept(self, state, unknown, dt, token):
        """Complete-path distance lower bound, including nearest-feature switches.

        Zero lifts mean every half-cell point is an affine convex combination of
        its two moving endpoints. The distance infimum is Lipschitz with constant
        max node displacement A + max node displacement B, independently of
        closest-point ownership. Geometry lowers and scalar subtraction enclose
        the complete closed curve; this is not endpoint sampling as acceptance.
        """
        step = self._increment(state, unknown, dt)
        speeds = []
        for element in self.elements:
            vectors = [step[6*self.node_index[n]:6*self.node_index[n]+3] for n in element.node_ids]
            # L1 is an outward-safe upper for Euclidean point displacement.
            speeds.append(max(sum(abs(Fraction(float(x))) for x in v) for v in vectors))
        speed = sum(speeds, Fraction(0))
        # Uniform range error relative to the exact affine binary64-input path.
        # Each product and each fsum has at most one ulp rounding (two for the
        # platform's possible double rounding). Applying triangle inequalities
        # to high=fsum(h,l,s*d), low=fsum(h,l,s*d,-high) gives the conservative
        # 8*eps*(|h|+|l|+|d|)+12*minsub per coordinate, including underflow.
        # Physical curve control points themselves round their fsum. Their
        # extra error is <=4*eps*A+4*minsub. The Q2 shape absolute sum <=5/4;
        # using 2 gives <=24*eps*A+32*minsub for the complete rounded curve.
        error = Fraction(0)
        eps = Fraction(2)**-52; subnormal = Fraction(2)**-1074
        for element in self.elements:
            errors = []
            for node in element.node_ids:
                index = self.node_index[node]
                errors.append(sum(24*eps*(abs(Fraction(float(state.positions[index,a])))+
                    abs(Fraction(float(state.position_low[index,a])))+
                    abs(Fraction(float(step[6*index+a]))))+32*subnormal for a in range(3)))
            error += max(errors)
        threshold = -self.maximum_penetration_m
        cache = {}
        def gap(s):
            if s not in cache:
                self._guard(token, 'native_pair_contact.swept')
                cache[s] = self._gap(self._advance(state, step*s), threshold)
            return cache[s]
        pending = [(0., 1.)]; checked = 0; lower = float('inf')
        while pending:
            if checked >= self.maximum_swept_intervals:
                raise ValueError(PATH_FAILURE)
            left, right = pending.pop(); checked += 1
            g0, g1 = gap(left), gap(right)
            if not isfinite(g0) or not isfinite(g1) or min(g0, g1) < threshold:
                raise ValueError(PATH_FAILURE)
            # First envelope maps the rounded sampled curve to the exact affine
            # family; the second maps any interior exact curve back to the
            # rounded _advance/retained-cell family actually evaluated by solver.
            bound = Fraction(float(min(g0, g1)))-Fraction(right-left)*speed/2-2*error
            certified = float(np.nextafter(float(bound), -np.inf))
            if certified >= threshold:
                lower = min(lower, certified)
            else:
                middle = .5*(left+right)
                if middle in (left, right):
                    raise ValueError(PATH_FAILURE)
                pending.extend(((left, middle), (middle, right)))
        return dict(path='AFFINE_NODE_POSITIONS_EXP_ENDPOINT_CELL_CHART_ZERO_LIFT',
            minimum_certified_gap_m=lower, maximum_penetration_m=self.maximum_penetration_m,
            certified_geometry='EXACT_AFFINE_AND_ROUNDED_ADVANCE_RETAINED_CURVE_FAMILY',
            endpoint_sample_to_exact_error_m=float(np.nextafter(float(error), np.inf)),
            interior_exact_to_rounded_error_m=float(np.nextafter(float(error), np.inf)),
            arithmetic_range_error_m=float(np.nextafter(float(2*error), np.inf)),
            checked_intervals=checked, evaluated_poses=len(cache))

    def _trial_state(self, accepted, dt, token):
        candidate = super()._trial_state(accepted, dt, token)
        self._swept(accepted, candidate.unknown, dt, token)
        return candidate

    def _step_extra(self, previous, unknown, dt, stage, endpoint, token):
        certificate = self._swept(previous, unknown, dt, token)
        contact = self.contact.evaluate(*self._poses(stage))
        increment = self._increment(previous, unknown, dt)
        increment[list(self.z)] = unknown[list(self.z)]
        certificate['stage_contact_internal_work_j'] = float(contact.residual @ increment[list(self.maps[0]+self.maps[1])])
        certificate['work_semantics'] = 'STAGE_SPATIAL_INTERNAL_FORCE_DOT_ENDPOINT_CHART_INCREMENT'
        return dict(native_contact=certificate)

    def _validate_initial_contact(self, state):
        if self._gap(state, 0.) < 0.:
            raise ValueError('initial native beam pair penetration or unresolved separation')

    def _admit_saved_state(self, state):
        gap = self._gap(state, -self.maximum_penetration_m)
        if not isfinite(gap) or gap < -self.maximum_penetration_m:
            raise ValueError('restored native pair endpoint exceeds penetration budget or is unresolved')
