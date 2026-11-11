"""Create a new stress-free point reference, never reset an accepted state.

The supplied shell rotations transport point offsets and beam frames. They do
not assign an independent shell director: Q4 computes its reference directors
from the new geometry. Line/footprint correspondence needs a separate extension.
"""
from dataclasses import dataclass, replace
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation

from .control import cancellation_safe_point
from ._stiffened_shell_ge_experimental import (
    LinearDefinition, ShellPanel, CURVED_Q4_POINT_REFERENCE_V1,
    CURVED_Q4_POINT_MATERIAL_REFERENCE_V1, CURVED_Q4_S3_POINT_REFERENCE_V1,
    PROFILE as B3_GE_E4_PL_STIFFENED_SHELL_V1, Q4_ALGORITHMIC_PULLBACK_V1,
)

POLICY = 'GE_STRESS_FREE_POINT_REFERENCE_TRANSPORT_V1_DEV'
MATERIAL_POLICY = 'GE_STRESS_FREE_POINT_MATERIAL_REFERENCE_TRANSPORT_V1_DEV'


def _vector(value):
    if not isinstance(value, (tuple, list, np.ndarray)) or np.shape(value) != (3,):
        raise ValueError('three finite physical reference components required')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in value):
        raise ValueError('numeric physical reference components required')
    answer = tuple(float(v) for v in value)
    if not np.isfinite(answer).all():
        raise ValueError('finite physical reference components required')
    return answer


@dataclass(frozen=True)
class ReferencePosePerturbation:
    """Sparse reference-position change and spatial left frame rotation.

    Unlisted nodes stay unchanged. Beam entries are only for unconnected nodes;
    point-connected beam changes are derived from their shell hosts.
    Shell rotations transport attached ports only; at an unattached shell node
    they have no modeled effect. They do not prescribe Q4 reference directors.
    """
    domain: str
    node: int
    translation_m: tuple
    rotation_vector_rad: tuple = (0., 0., 0.)

    def __post_init__(self):
        if type(self.domain) is not str or self.domain not in ('shell', 'beam'):
            raise ValueError('explicit shell or beam reference domain required')
        if type(self.node) is not int or self.node <= 0:
            raise ValueError('positive exact model node ID required')
        object.__setattr__(self, 'translation_m', _vector(self.translation_m))
        object.__setattr__(self, 'rotation_vector_rad', _vector(self.rotation_vector_rad))


@dataclass(frozen=True)
class StressFreePointReference:
    definition: LinearDefinition
    source_identity: str
    target_identity: str
    stress_free_scaled_max: float
    max_transport_disagreement: float
    policy: str = POLICY
    production_qualified: bool = False


def create_stress_free_point_reference(definition, perturbations, *,
                                      resources=None, cancellation_token=None):
    """Transport an explicitly selected virgin Q4/S3/GE point reference.

    At a point, delta_b = u_s + (Q_s-I)(r_b0-r_s0) and R_bnew = Q_s R_b0.
    This transports the physical eccentricity and orientation once. The shell
    incremental reference frame stays I, so the stored point reference data
    become e_new=Q_s e_old and A_new=Q_s A_old in the new reference basis.
    Multiple shell hosts
    must give the same beam pose within arithmetic roundoff. Supports and model
    IDs retain their reference meaning; no accepted history is supplied/copied.
    Moving a supported node relocates its zero-displacement reference fixture;
    it is not deformation against the old fixture location.

    Existing compilation checks admit both models and reject unsupported new
    facet/beam geometry, including warped Q4, degenerate S3 or triads inconsistent with the
    new native centerline tangent. Frames are never silently realigned. This
    preparation occurs once, outside any solve/trial.
    """
    from .stiffened_shell import create_analysis
    if (type(definition) is not LinearDefinition or
            type(definition.shell) is not ShellPanel or
            definition.profile != B3_GE_E4_PL_STIFFENED_SHELL_V1 or
            definition.shell_reference_policy not in (CURVED_Q4_POINT_REFERENCE_V1,
                                                      CURVED_Q4_POINT_MATERIAL_REFERENCE_V1,
                                                      CURVED_Q4_S3_POINT_REFERENCE_V1) or
            definition.q4_pullback_policy != Q4_ALGORITHMIC_PULLBACK_V1 or
            definition.lines or definition.footprints or
            (definition.shell.s3_elements and definition.shell_reference_policy !=
                CURVED_Q4_S3_POINT_REFERENCE_V1) or
            (definition.shell.j2_curve is not None and
                definition.shell_reference_policy != CURVED_Q4_POINT_MATERIAL_REFERENCE_V1)):
        raise ValueError('explicit admitted curved-Q4/S3 point reference definition required')
    if type(perturbations) is not tuple or any(type(v) is not ReferencePosePerturbation
                                              for v in perturbations):
        raise ValueError('typed immutable reference perturbation tuple required')
    policy = (MATERIAL_POLICY if definition.shell_reference_policy ==
              CURVED_Q4_POINT_MATERIAL_REFERENCE_V1 else POLICY)
    cancellation_safe_point(cancellation_token, 'stiffened-shell.reference.start')
    source = create_analysis(definition=definition, resources=resources)
    target = None
    try:
        original = source._definition  # compiled, owned definition; never state
        updates = {}
        hosts = {}
        for point in original.points:
            hosts.setdefault(point.beam_node, []).append(point.shell_node)
        for row in perturbations:
            key = (row.domain, row.node)
            if key in updates:
                raise ValueError('duplicate reference perturbation node')
            known = source._shell_index if row.domain == 'shell' else source._beam_index
            if row.node not in known:
                raise ValueError('unknown reference perturbation model node')
            if row.domain == 'beam' and row.node in hosts:
                raise ValueError('point-connected beam reference is derived from shell hosts')
            updates[key] = (np.array(row.translation_m),
                            Rotation.from_rotvec(row.rotation_vector_rad).as_matrix())
        if all(not np.any(delta) and np.array_equal(motion,np.eye(3))
               for delta,motion in updates.values()):
            cancellation_safe_point(cancellation_token, 'stiffened-shell.reference.complete')
            info=source.preflight()
            return StressFreePointReference(original,info['identity'],info['identity'],
                                             info['stress_free_scaled_max'],0.,policy=policy)
        zero = (np.zeros(3), np.eye(3))
        shell_delta = {node: updates.get(('shell', node), zero)
                       for node in source._shell_ids}
        shell_xyz = source._shell_xyz + np.array([shell_delta[n][0] for n in source._shell_ids])
        if not np.isfinite(shell_xyz).all():
            raise ValueError('nonfinite perturbed shell reference')
        beam_geometry = {}
        disagreement = 0.
        eps = 64*np.finfo(float).eps
        for node in source._beam_ids:
            old_position, old_frame = source._beam_nodes[node]
            candidates = []
            for shell_node in hosts.get(node, ()):
                delta, motion = shell_delta[shell_node]
                host_position = source._shell_xyz[source._shell_index[shell_node]]
                offset = old_position-host_position
                rotation_delta = (motion-np.eye(3))@offset
                # Only operands participating in this transport set the
                # roundoff scale; distant geometry must not hide a conflict.
                scale = max(float(np.linalg.norm(value)) for value in
                            (old_position,host_position,offset,delta,rotation_delta))
                if not np.isfinite(scale):
                    raise ValueError('nonfinite reference transport scale')
                candidates.append((delta+rotation_delta, motion@old_frame, scale))
            if not candidates:
                delta, motion = updates.get(('beam', node), zero)
                candidates.append((delta, motion@old_frame, 0.))
            delta, frame, local_scale = candidates[0]
            for other_delta, other_frame, other_scale in candidates[1:]:
                scale = max(local_scale,other_scale,np.finfo(float).tiny)
                difference = max(float(np.linalg.norm(delta-other_delta)/scale),
                                 float(np.linalg.norm(frame-other_frame)))
                disagreement = max(disagreement, difference)
                if difference > eps:
                    raise ValueError('inconsistent multiple-host reference transports')
            position = old_position+delta
            if not np.isfinite(position).all() or not np.isfinite(frame).all():
                raise ValueError('nonfinite perturbed beam reference')
            beam_geometry[node] = (position, frame)
        beams = []
        for (element, _, _), inertia in zip(source._beam_parts, source._reference_modal_inertias):
            cancellation_safe_point(cancellation_token, 'stiffened-shell.reference.beam')
            reference = element.operator.reference
            positions=np.array([beam_geometry[n][0] for n in element.node_ids])
            frames=np.array([beam_geometry[n][1] for n in element.node_ids])
            if (np.array_equal(positions,reference.coordinates) and
                    np.array_equal(frames,reference.nodal_triads)):
                beams.append(original.beams[len(beams)])
                continue
            new_reference = type(reference)(
                positions,frames,
                regularity_relative_tolerance=reference._regularity_relative_tolerance,
                rotation_tolerance=reference._rotation_tolerance,
                frame_tolerance=reference._frame_tolerance)
            rebuilt = type(element)(element.element_id, element.node_ids,
                new_reference, element.section, order=element.operator.order,
                arithmetic_policy=element.operator.arithmetic_policy)
            beams.append(type(original.beams[len(beams)]).capture(rebuilt, inertia))
        shell = replace(original.shell,
            coordinates_m=tuple(tuple(float(v) for v in row) for row in shell_xyz))
        changed = replace(original, shell=shell, beams=tuple(beams))
        cancellation_safe_point(cancellation_token, 'stiffened-shell.reference.compile')
        target = create_analysis(definition=changed, resources=resources)
        cancellation_safe_point(cancellation_token, 'stiffened-shell.reference.complete')
        return StressFreePointReference(target._definition,
            source.preflight()['identity'], target.preflight()['identity'],
            target.preflight()['stress_free_scaled_max'], disagreement,
            policy=policy)
    finally:
        if target is not None:
            target.close()
        source.close()
