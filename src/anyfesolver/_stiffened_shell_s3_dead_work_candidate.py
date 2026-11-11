"""Experimental finite S3 director-enhanced reference-dead surface work.

The linear native pressure vector does not uniquely specify a finite surface.
This candidate explicitly lifts its enhanced displacement along the fitted
normal, using the shortest relative director tilt. The rejected full-log lift
is retained only as a counterexample. Explicit reference_dead_director_v2 owner
dispatch uses the director lift; historical reference_dead is unchanged. This
does not qualify finite pressure or alter internal shell equations. See the
development note s3-dead-work-candidate.md and subsequent integration evidence.
"""
import numpy as np

from ._ge_beam3_mixed_ad import (Jet2, constant_matrix, matmul, matvec, transpose,
                               skew, so3_exp, so3_log)
from ._ge_beam3_pose_joint import _exp_terms
from ._ge_beam3_variational_shell import _quaternion_jets, deformation
from ._stiffened_shell_rotation_charts import owned_anchors

POLICY = "S3_COROTATED_ENHANCED_REFERENCE_DEAD_WORK_CANDIDATE_V1"
DIRECTOR_POLICY = "S3_DIRECTOR_ENHANCED_REFERENCE_DEAD_WORK_CANDIDATE_V2"


def evaluate(reference, normal, native_force, coordinates, *, anchors=None,
             director_only=True):
    """Return external work, coordinate gradient/Hessian, spatial rows/Jacobian.

    Inputs are one triangle and its native uniform reference-normal pressure
    covector. Dense arrays are bounded to the element's 18 coordinates. The
    Jacobian has spatial wrench rows and additive-chart increment columns.
    """
    reference, normal, native_force, coordinates = (
        np.asarray(v, dtype=float) for v in
        (reference, normal, native_force, coordinates))
    for value, shape in ((reference, (3, 3)), (normal, (3,)),
                         (native_force, (18,)), (coordinates, (18,))):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("finite triangle dead-work inputs required")
    if abs(np.linalg.norm(normal)-1.) > 1e-12:
        raise ValueError("unit reference normal required")
    # This is a mechanics experiment with a native-vector input contract, not
    # a general load-vector API. The caller obtains that vector from MIN3.
    if type(director_only) is not bool:
        raise ValueError("explicit candidate selection required")
    if anchors is not None:
        anchors = owned_anchors(anchors, 3)
    if not director_only:
        local, first, second = deformation(reference, coordinates,
            stable_translation=True, rotation_anchors=anchors)
    w, x, y, z = _quaternion_jets(reference, None,
        displacement=coordinates.reshape(3, 6)[:, :3])
    fitted = ((w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y)),
              (2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x)),
              (2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z))
    projection = sum(normal[i]*fitted[i][j]*normal[j]
                     for i in range(3) for j in range(3))
    tilts = []
    if director_only:
        for node in range(3):
            theta = [Jet2.variable(coordinates[6*node+3+j], 6*node+3+j, 18)
                     for j in range(3)]
            physical = so3_exp(theta)
            if anchors is not None:
                physical = matmul(physical, constant_matrix(anchors[node], 18))
            director = matvec(matmul(transpose(fitted), physical),
                              [Jet2.constant(v, 18) for v in normal])
            cosine = sum(normal[j]*director[j] for j in range(3))
            if cosine.value <= -1.+1e-7:
                raise ValueError("antipodal relative director outside candidate chart")
            cross = [normal[1]*director[2]-normal[2]*director[1],
                     normal[2]*director[0]-normal[0]*director[2],
                     normal[0]*director[1]-normal[1]*director[0]]
            spin = skew(cross)
            square = matmul(spin, spin)
            shortest = [[float(i == j)+spin[i][j]+square[i][j]/(1.+cosine)
                         for j in range(3)] for i in range(3)]
            tilts.append(so3_log(shortest))
    enhancement = Jet2.constant(0., 18)
    work = Jet2.constant(0., 18)
    for node in range(3):
        for axis in range(3):
            slot = 6*node+axis
            work += native_force[slot]*Jet2.variable(coordinates[slot], slot, 18)
            slot += 3
            value = tilts[node][axis] if director_only else Jet2(
                local[slot], first[slot], second[slot])
            enhancement += native_force[slot]*value
    work += projection*enhancement

    chart = np.eye(18)
    blocks = []
    for node in range(3):
        slots = slice(6*node+3, 6*node+6)
        _, jacobian, derivative = _exp_terms(coordinates[slots])
        singular = np.linalg.svd(jacobian, compute_uv=False)
        if singular[-1] <= 1e-11*max(1., singular[0]):
            raise ValueError("singular additive shell chart; rebase required")
        chart[slots, slots] = jacobian
        blocks.append((slots, derivative))
    force = np.linalg.solve(chart.T, work.gradient)
    correction = np.zeros((18, 18))
    for slots, derivative in blocks:
        correction[slots, slots] = np.einsum('ijk,i->jk', derivative, force[slots])
    tangent = np.linalg.solve(chart.T, work.hessian-correction)
    if not all(np.isfinite(a).all() for a in
               (work.value, work.gradient, work.hessian, force, tangent)):
        raise ValueError("finite dead-work response required")
    return work.value, work.gradient, work.hessian, force, tangent
