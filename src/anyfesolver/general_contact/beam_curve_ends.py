"""Physical-end point work on classical quadratic curves, opt-in development.

Distributed work and its curvature-dependent distance Hessian remain owned by
beam_curve_pair. Closed ambiguous projections/ownership are explicit non-success.
"""
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .beam_curve_pair_static import B3PairManager, validate_b3_pair_contact
from .beam_curve_pair import b3_distance_jet, closest_on_b3
from .beam_pair_ends import _line_definition, _EndPairResults
from .beam_end_topology import physical_beam_ends
from .beam_plane import _beam_types
from .beam_end_radial_geometry import regular_curve_path
from .capped_cylinder import _sqrt_bound
from .core import ContactSolverSettings, ContactValidationReport

PROFILE = 'CLASSICAL_B2_B3_ROUND_TUBE_QUADRATIC_PHYSICAL_ENDS_STATIC_DEV_V1'


def validate_curve_beam_ends(model, definition, settings=None):
    if tuple(map(len, (definition.surfaces, definition.rigid_bodies,
                       definition.interactions))) != (2, 0, 1):
        return ContactValidationReport('invalid', ('two single-cell beam surfaces required',), PROFILE)
    if any(s.beam_end_caps != 'hemisphere' for s in definition.surfaces) or (
            definition.interactions[0].endpoint_stiffness is None):
        return ContactValidationReport('invalid', ('hemisphere ends need explicit endpoint_stiffness N/m',), PROFILE)
    report = validate_b3_pair_contact(model, _line_definition(definition), settings)
    return ContactValidationReport(report.status, report.issues, PROFILE)


@dataclass(frozen=True)
class _CurveEndResults(_EndPairResults):
    def to_dict(self):
        return {**super().to_dict(),
                'minimum_gap_kind': 'conservative_full_quadratic_curve_lower_bound',
                'geometry_kinematics': 'quadratic_curve_unique_convex_endpoint_projection_v1'}


class BeamCurveEndManager(B3PairManager):
    def __init__(self, model, definition, settings=None, initial_displacement=None):
        controls = ContactSolverSettings() if settings is None else settings
        report = validate_curve_beam_ends(model, definition, controls)
        if not report.valid:
            raise ValueError('unsupported quadratic physical ends: ' + '; '.join(report.issues))
        super().__init__(model, _line_definition(definition), controls, initial_displacement)
        self.definition = definition
        self.profile = PROFILE
        self.end_caps = 'hemisphere'
        self.endpoint_stiffness = definition.interactions[0].endpoint_stiffness
        self.elements = tuple(model.mesh.elements[s.element_ids[0]] for s in definition.surfaces)
        self.ends = []
        for surface, element in zip(definition.surfaces, self.elements):
            cell = SimpleNamespace(element_id=surface.element_ids[0], node_ids=element.node_ids)
            local = physical_beam_ends(model, (cell,), _beam_types()).values()
            self.ends.append(tuple(sorted(0 if i == 0 else 2 for _, i, _ in local)))
        self.ends = tuple(self.ends)

    def _endpoint_work(self, positions):
        from .beam_curve_end_geometry import endpoint_interval
        E = 0.; f = np.zeros(18); K = np.zeros((18, 18)); fields = []; seen = set()
        certificates = {(side, endpoint): endpoint_interval(
            positions[side], positions[side], positions[1-side], positions[1-side],
            endpoint, self.radius) for side in (0, 1) for endpoint in self.ends[side]}
        for side in (0, 1):
            source, target = positions[side], positions[1-side]
            for endpoint in self.ends[side]:
                proof = certificates[side, endpoint]
                if proof is None:
                    raise ValueError('quadratic physical-end projection/ownership is unresolved')
                node = self.elements[side].node_ids[0 if endpoint == 0 else -1]
                row = {'source_surface': self.definition.surfaces[side].name,
                       'source_element_id': self.definition.surfaces[side].element_ids[0],
                       'source_node_id': int(node), 'state': 'open'}
                if proof.open:
                    projection = closest_on_b3(source[endpoint], target)
                    row['gap'] = projection.distance-self.radius
                    fields.append(row); continue
                station = b3_distance_jet(source, target, endpoint/2, self.radius,
                                         source_is_a=side == 0)
                t = station.target_parameter
                if not float(proof.parameter_lower) <= t <= float(proof.parameter_upper):
                    raise ValueError('quadratic numerical projection leaves its certified bracket')
                if proof.target_endpoint is not None:
                    # Literal IEEE geometry, cancellation-resistant activation.
                    separation = tuple(Fraction.from_float(float(x))-Fraction.from_float(float(y))
                        for x, y in zip(source[endpoint], target[proof.target_endpoint]))
                    squared = sum(value*value for value in separation)
                    radius = Fraction.from_float(self.radius)
                    root = _sqrt_bound(squared, upper=True)
                    difference = squared-radius*radius
                    gap = float(difference/(root+radius))
                    if gap == 0 and difference != 0:
                        raise ValueError('quadratic endpoint activation conversion underflows')
                else:
                    gap = station.gap
                    scale = max(self.radius, np.max(np.abs(source)), np.max(np.abs(target)), 1.)
                    if abs(gap) <= 256*np.finfo(float).eps*scale:
                        raise ValueError('quadratic interior activation sign is unresolved at roundoff')
                row.update(gap=gap, target_coordinate=t,
                           target_element_id=self.definition.surfaces[1-side].element_ids[0])
                if gap >= 0:
                    fields.append(row); continue
                if proof.ownership > 0:
                    row['state'] = 'side_owned'; fields.append(row); continue
                other = proof.target_endpoint
                if other is not None:
                    if other not in self.ends[1-side]:
                        raise ValueError('quadratic cap projects to a nonphysical selected target end')
                    reverse = certificates[1-side, other]
                    if reverse is None or reverse.open or reverse.ownership >= 0 or (
                            reverse.target_endpoint != endpoint):
                        raise ValueError('quadratic active cap/cap partner is not reciprocal')
                    key = tuple(sorted(((side, endpoint), (1-side, other))))
                    row.update(feature_kind='end_end', target_node_id=int(
                        self.elements[1-side].node_ids[0 if other == 0 else -1]))
                else:
                    key = (side, endpoint, 'interior')
                    row['feature_kind'] = 'end_interior'
                if key in seen:
                    row['state'] = 'reciprocal_owned_once'; fields.append(row); continue
                seen.add(key)
                ke = self.endpoint_stiffness
                E += .5*ke*gap*gap
                f += ke*gap*station.gradient
                K += ke*(np.outer(station.gradient, station.gradient)+gap*station.hessian)
                row.update(state='active', endpoint_force_N=-ke*gap)
                fields.append(row)
        return E, f, K, tuple(fields), len(seen)

    def evaluate(self, displacement, *, tangent=True):
        positions = self._positions(displacement)
        if not all(regular_curve_path(p, p, self.settings.maximum_swept_subdivisions)
                   for p in positions):
            raise ValueError('quadratic physical-end curve regularity is unresolved')
        endpoint, ef, eK, fields, active = self._endpoint_work(positions)
        force, K, line = super().evaluate(displacement, tangent=tangent)
        dofs = np.r_[self.dofs[0].reshape(-1), self.dofs[1].reshape(-1)]
        np.add.at(force, dofs, self.assembly_map.T@ef)
        if tangent:
            rows, cols = np.meshgrid(dofs, dofs, indexing='ij')
            local = self.assembly_map.T@eK@self.assembly_map
            K += sparse.coo_matrix((local.ravel(), (rows.ravel(), cols.ravel())),
                                  shape=(self.n_dof, self.n_dof)).tocsr()
        forces = tuple(-ef[i*9:(i+1)*9].reshape(3, 3) for i in (0, 1))
        wrench = [tuple(np.asarray(old)+np.sum(values, axis=0)) for old, values in
                  zip((line.beam_a_force, line.beam_b_force), forces)]
        moments = [tuple(np.asarray(old)+np.sum(np.cross(p, values), axis=0)) for old, p, values
                   in zip((line.beam_a_moment, line.beam_b_moment), positions, forces)]
        result = _CurveEndResults(self.profile, line.minimum_gap, line.contact_energy+endpoint,
            line.active_source_intervals, line.integration_points, line.maximum_integration_relative_error,
            *wrench, *moments, line.gap_tolerance_met, self.stiffness,
            line.contact_energy, endpoint, self.endpoint_stiffness,
            sum(map(len, self.ends)), active, fields)
        return force, K, result

    def _endpoint_path_admissible(self, first, last):
        from .beam_curve_end_geometry import endpoint_interval
        def rational(poses):
            return tuple(np.array([[Fraction.from_float(float(x)) for x in row]
                                   for row in p], dtype=object) for p in poses)
        def certify(a, b, depth):
            proofs = {(side, endpoint): endpoint_interval(a[side], b[side],
                a[1-side], b[1-side], endpoint, self.radius)
                for side in (0, 1) for endpoint in self.ends[side]}
            fixed = all(proof is not None for proof in proofs.values())
            if fixed:
                for (side, endpoint), proof in proofs.items():
                    if proof.open or proof.ownership > 0 or proof.target_endpoint is None:
                        continue
                    other = proof.target_endpoint
                    reverse = proofs.get((1-side, other))
                    if reverse is None or reverse.open or reverse.ownership >= 0 or (
                            reverse.target_endpoint != endpoint):
                        fixed = False; break
                if fixed:
                    return True
            if depth >= self.settings.maximum_swept_subdivisions:
                return False
            mid = tuple((x+y)/2 for x, y in zip(a, b))
            return certify(a, mid, depth+1) and certify(mid, b, depth+1)
        return certify(rational(first), rational(last), 0)

    def _gap_certificate(self, positions):
        from .beam_curve_end_geometry import certify_curve_pair_gap
        return certify_curve_pair_gap(*positions, self.radius,
            accuracy=self.settings.beam_pair_gap_accuracy,
            maximum_cells=self.settings.maximum_beam_pair_gap_cells,
            maximum_depth=self.settings.maximum_beam_pair_gap_depth)

    def _certificate(self, displacement):
        return self._gap_certificate(self._positions(displacement))

    def _swept_admissible(self, start, end):
        first, last = self._positions(start), self._positions(end)
        if not all(regular_curve_path(a, b, self.settings.maximum_swept_subdivisions)
                   for a, b in zip(first, last)):
            return False
        from .beam_curve_end_geometry import _curve_controls
        q0, q1 = [tuple(np.array([[Fraction.from_float(float(x)) for x in row]
                                  for row in p], dtype=object) for p in poses)
                  for poses in (first, last)]
        delta = tuple(b-a for a, b in zip(q0, q1))
        velocities = tuple(np.array(_curve_controls(p), dtype=object).T for p in delta)
        speed = max(_sqrt_bound(sum((x-y)**2 for x, y in zip(a, b)), upper=True)
                    for a in velocities[0] for b in velocities[1])
        tol = Fraction.from_float(self.gap_tolerance)
        def gap(t):
            return self._gap_certificate(tuple(a+t*d for a, d in zip(q0, delta))).conservative_lower_gap
        g0, g1 = gap(Fraction(0)), gap(Fraction(1))
        if g0 < -self.gap_tolerance:
            raise ValueError('committed quadratic physical-end gap exceeds tolerance')
        if g1 < -self.gap_tolerance:
            return False
        def certify(left, right, gl, gr, depth):
            if min(Fraction.from_float(gl), Fraction.from_float(gr))-speed*(right-left)/2 >= -tol:
                return True
            if depth >= self.settings.maximum_swept_subdivisions:
                return False
            middle = (left+right)/2; gm = gap(middle)
            if gm < -self.gap_tolerance:
                return False
            return (certify(left, middle, gl, gm, depth+1)
                    and certify(middle, right, gm, gr, depth+1))
        return certify(Fraction(0), Fraction(1), g0, g1, 0) and self._endpoint_path_admissible(first, last)

    def restart_contract(self):
        return {**super().restart_contract(),
                'endpoint_stiffness': self.endpoint_stiffness,
                'physical_ends': [[int(e.node_ids[0 if i == 0 else -1]) for i in ends]
                                  for e, ends in zip(self.elements, self.ends)],
                'point_law': 'unique_quadratic_owned_end_end_or_end_interior_v1',
                'geometry_revision': 'exact_tensor_gap_convex_projection_ownership_affine_path_v1'}
