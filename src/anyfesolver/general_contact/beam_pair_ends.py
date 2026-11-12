"""Opt-in B2 pair potential with genuine physical endpoint work.

The symmetric distributed line potential is unchanged. Each owned end/interior
point contributes one spring; reciprocal end/end projections contribute once.
Closed feature transitions remain explicit non-success in this development row.
"""
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np
from scipy import sparse

from .beam_pair import (_BeamPairManager, _BeamPairResults, _projection_work,
                        validate_beam_pair_contact)
from .beam_chain_consistent import _windows, _integrate, _check_projection_plateau
from .beam_end_topology import physical_beam_ends
from .beam_plane import _beam_types
from .beam_end_radial_geometry import regular_curve_path
from .capped_cylinder import _sqrt_bound
from .core import ContactSolverSettings, ContactValidationReport

PROFILE = 'CLASSICAL_B2_B2_ROUND_TUBE_PHYSICAL_ENDS_PAIR_STATIC_DEV_V1'


def _line_definition(definition):
    return replace(definition,
        surfaces=tuple(replace(s, beam_end_caps='none') for s in definition.surfaces),
        interactions=tuple(replace(i, endpoint_stiffness=None) for i in definition.interactions))


def validate_beam_pair_ends(model, definition, settings=None):
    if len(definition.surfaces)==2 and any(len(s.element_ids)>1 for s in definition.surfaces):
        from .beam_chain_ends import validate_beam_chain_ends
        return validate_beam_chain_ends(model,definition,settings)
    from anyfesolver.elements import QuadraticBeamElement
    if any(type(model.mesh.elements.get(element)) is QuadraticBeamElement
           for s in definition.surfaces for element in s.element_ids):
        from .beam_curve_ends import validate_curve_beam_ends
        return validate_curve_beam_ends(model, definition, settings)
    from anyfesolver.elements import BeamElement
    if tuple(map(len, (definition.surfaces, definition.rigid_bodies,
                       definition.interactions))) != (2, 0, 1):
        return ContactValidationReport('invalid', ('two explicit B2 surfaces and one pair required',), PROFILE)
    if any(len(s.element_ids) != 1 or
           type(model.mesh.elements.get(s.element_ids[0])) is not BeamElement
           for s in definition.surfaces):
        return ContactValidationReport('invalid', ('physical-end pair row requires one classical B2 per side',), PROFILE)
    if any(s.beam_end_caps != 'hemisphere' for s in definition.surfaces) or (
            definition.interactions[0].endpoint_stiffness is None):
        return ContactValidationReport('invalid', ('both surfaces need hemisphere ends and explicit endpoint_stiffness N/m',), PROFILE)
    report = validate_beam_pair_contact(model, _line_definition(definition), settings)
    return ContactValidationReport(report.status, report.issues, PROFILE)


@dataclass(frozen=True)
class _EndPairResults(_BeamPairResults):
    line_contact_energy: float
    endpoint_contact_energy: float
    endpoint_stiffness: float
    candidate_endpoints: int
    active_endpoints: int
    endpoint_fields: tuple

    def to_dict(self):
        return {**super().to_dict(), 'line_contact_energy': self.line_contact_energy,
            'endpoint_contact_energy': self.endpoint_contact_energy,
            'endpoint_stiffness_N_per_m': self.endpoint_stiffness,
            'candidate_endpoints': self.candidate_endpoints, 'active_endpoints': self.active_endpoints,
            'endpoint_fields': [dict(row) for row in self.endpoint_fields],
            'beam_end_cap_geometry': 'outward_hemisphere_point_penalty',
            'minimum_gap_kind': 'exact_segment_rectangle_outward_lower_bound',
            'moment_origin': [0., 0., 0.], 'moment_units': 'N*m'}


class BeamPairEndManager(_BeamPairManager):
    def __init__(self, model, definition, settings=None, initial_displacement=None):
        if any(len(s.element_ids)>1 for s in definition.surfaces):
            raise ValueError('multi-cell physical ends require the chain analysis contributor')
        controls = ContactSolverSettings() if settings is None else settings
        report = validate_beam_pair_ends(model, definition, controls)
        if not report.valid:
            raise ValueError('unsupported beam physical-end pair: ' + '; '.join(report.issues))
        super().__init__(model, _line_definition(definition), controls, initial_displacement)
        self.definition = definition
        self.profile = PROFILE
        self.end_caps = 'hemisphere'
        self.endpoint_stiffness = definition.interactions[0].endpoint_stiffness
        self.ends = tuple(tuple(sorted(local for _, local, _ in
            physical_beam_ends(model, (cell,), _beam_types()).values())) for cell in self.segments)

    def _minimum_gap(self, u):
        from .beam_pair_end_geometry import minimum_gap
        return minimum_gap(*self._positions(u), self.radius)

    def _integrate_direction(self, source, target, reference_length, source_is_a):
        windows = _windows(source, (target,), (0,), self.radius)
        E = 0.; f = np.zeros(12); K = np.zeros((12, 12)); points = 0; error = 0.
        for _, left, right in windows:
            _check_projection_plateau(source, target, (left + right) / 2)
            work = _integrate(self, source, target, reference_length, left, right)
            E += work[0]; f += work[1]; K += work[2]; points += work[3]
            error = max(error, work[4])
        if not source_is_a:
            order = np.r_[np.arange(6, 12), np.arange(6)]
            f = f[order]; K = K[np.ix_(order, order)]
        return E, f, K, len(windows), points, error

    def _endpoint_work(self, positions):
        from .beam_pair_end_geometry import endpoint_feature
        E = 0.; f = np.zeros(12); K = np.zeros((12, 12)); fields = []; seen = set()
        for side in (0, 1):
            source, target = positions[side], positions[1-side]
            for endpoint in self.ends[side]:
                feature = endpoint_feature(source, target, endpoint, self.radius)
                row = {'source_surface': self.definition.surfaces[side].name,
                    'source_element_id': self.segments[side].element_id,
                    'source_node_id': self.segments[side].node_ids[endpoint],
                    'target_element_id': self.segments[1-side].element_id,
                    'target_coordinate': float(feature.projection), 'gap': feature.gap,
                    'state': 'open'}
                if feature.gap_sign >= 0:
                    fields.append(row); continue
                if feature.projection_tie or feature.ownership == 0:
                    raise ValueError('active physical-end projection/ownership tie is unresolved')
                if feature.ownership > 0:
                    row['state'] = 'side_owned'; fields.append(row); continue
                if feature.target_endpoint is not None:
                    other = feature.target_endpoint
                    if other not in self.ends[1-side]:
                        raise ValueError('active physical end projects to a nonphysical selected target end')
                    key = tuple(sorted(((side, endpoint), (1-side, other))))
                    row['feature_kind'] = 'end_end'
                    row['target_node_id'] = self.segments[1-side].node_ids[other]
                else:
                    key = (side, endpoint, 'interior')
                    row['feature_kind'] = 'end_interior'
                if key in seen:
                    row['state'] = 'reciprocal_owned_once'; fields.append(row); continue
                seen.add(key)
                t = float(feature.projection)
                source_shape = np.array((1.-endpoint, float(endpoint)))
                target_shape = np.array((1.-t, t))
                separation = np.array([float(x) for x in feature.separation])
                _, grad, H = _projection_work(source_shape, target_shape, separation,
                    target[1]-target[0], feature.interior, self.radius, source_is_a=side==0)
                gap = feature.gap; ke = self.endpoint_stiffness
                E += .5 * ke * gap * gap
                f += ke * gap * grad
                K += ke * (np.outer(grad, grad) + gap * H)
                row.update(state='active', endpoint_force_N=-ke*gap)
                fields.append(row)
        return E, f, K, tuple(fields), len(seen)

    def evaluate(self, displacement, *, tangent=True):
        u = np.asarray(displacement, dtype=float).reshape(-1)
        n = self.model.mesh.dof_manager.total_dofs
        if u.size != n or not np.all(np.isfinite(u)):
            raise ValueError('beam physical-end pair displacement is invalid')
        poses = self._positions(u)
        if not all(regular_curve_path(p, p, self.settings.maximum_swept_subdivisions) for p in poses):
            raise ValueError('beam physical-end pair curve regularity is unresolved')
        first = self._integrate_direction(*poses, self.segments[0].length, True)
        second = self._integrate_direction(poses[1], poses[0], self.segments[1].length, False)
        line = .5*(first[0]+second[0]); end, ef, eK, fields, active = self._endpoint_work(poses)
        local = .5*(first[1]+second[1])+ef; block = .5*(first[2]+second[2])+eK
        dofs = np.concatenate([s.dofs.reshape(-1) for s in self.segments])
        f = np.zeros(n); np.add.at(f, dofs, local)
        if tangent:
            r, c = np.meshgrid(dofs, dofs, indexing='ij')
            K = sparse.coo_matrix((block.ravel(), (r.ravel(), c.ravel())), shape=(n,n)).tocsr()
        else: K = None
        physical = (-local[:6].reshape(2,3), -local[6:].reshape(2,3))
        forces = [tuple(np.sum(p,axis=0)) for p in physical]
        moments = [tuple(np.sum(np.cross(x,p),axis=0)) for x,p in zip(poses,physical)]
        gap = self._minimum_gap(u)
        result = _EndPairResults(self.profile, gap, line+end, first[3]+second[3],
            first[4]+second[4], max(first[5],second[5]), *forces, *moments,
            gap >= -self.gap_tolerance, self.stiffness, line, end,
            self.endpoint_stiffness, sum(map(len,self.ends)), active, fields)
        return f, K, result

    def _swept_admissible(self, start, end):
        from .beam_pair_end_geometry import (minimum_gap, admissible_endpoint_path,
                                             admissible_line_projection_path)
        a, b = self._positions(start)
        # Endpoints may be regular while an intermediate physical curve collapses.
        last = tuple(s.positions(end) for s in self.segments)
        if not all(np.all(np.isfinite(p)) for p in last): return False
        if not all(regular_curve_path(p,q,self.settings.maximum_swept_subdivisions)
                   for p,q in zip((a,b),last)): return False
        qa = np.array([[Fraction.from_float(float(x)) for x in p] for p in a], dtype=object)
        qb = np.array([[Fraction.from_float(float(x)) for x in p] for p in b], dtype=object)
        da = np.array([[Fraction.from_float(float(y))-x for x,y in zip(p,q)]
                       for p,q in zip(qa,last[0])], dtype=object)
        db = np.array([[Fraction.from_float(float(y))-x for x,y in zip(p,q)]
                       for p,q in zip(qb,last[1])], dtype=object)
        speed = max(_sqrt_bound(sum((x-y)**2 for x,y in zip(p,q)),upper=True)
                    for p in da for q in db)
        tolerance = Fraction.from_float(self.gap_tolerance)
        def gap(t): return minimum_gap(qa+t*da, qb+t*db, self.radius)
        first, final = gap(Fraction(0)), gap(Fraction(1))
        if first < -self.gap_tolerance:
            raise ValueError('committed beam physical-end pair exceeds gap tolerance')
        if final < -self.gap_tolerance: return False
        def certify(l,r,gl,gr,depth):
            if min(Fraction.from_float(gl),Fraction.from_float(gr))-speed*(r-l)/2 >= -tolerance:
                return True
            if depth >= self.settings.maximum_swept_subdivisions: return False
            mid=(l+r)/2; gm=gap(mid)
            if gm < -self.gap_tolerance: return False
            return certify(l,mid,gl,gm,depth+1) and certify(mid,r,gm,gr,depth+1)
        return (certify(Fraction(0),Fraction(1),first,final,0) and
            admissible_endpoint_path(a,last[0],b,last[1],*self.ends,self.radius,
                                     self.settings.maximum_swept_subdivisions) and
            admissible_line_projection_path(a,last[0],b,last[1],self.radius,
                                             self.settings.maximum_swept_subdivisions))

    def restart_contract(self):
        return {**super().restart_contract(), 'endpoint_stiffness': self.endpoint_stiffness,
            'physical_end_nodes': [[s.node_ids[i] for i in ends] for s,ends in zip(self.segments,self.ends)],
            'point_law': 'unique_owned_end_end_or_end_interior_reduced_projection_v1',
            'geometry_revision': 'exact_affine_projection_ownership_and_regular_curve_v1'}
