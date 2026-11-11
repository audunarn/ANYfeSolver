"""Private reference-linearized B2/finite-shell-support point experiment.

The shell port is the least-squares rigid motion of three or four *physical* vertex
translations. Its finite support, coordinates and weights are explicit; no
isolated drilling rotation, penalty, shell stiffness or public MPC is inferred.
The beam port uses the physical B2 member's sectional station trace. The
constraint differential comes from the existing objective master-frame joint
at its stress-free reference. This is a G0 development operator only.
"""

from dataclasses import dataclass

import numpy as np

from ._b2_physical_linear_definition import PhysicalB2Member
from ._ge_beam3_pose_joint import RigidPoseJoint


LAW_ID = "B2_FINITE_Q4_POINT_REFERENCE_LINEAR_V1_DEV"
S3_LAW_ID = "B2_FINITE_S3_POINT_REFERENCE_LINEAR_V1_DEV"


def _skew(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])


def _finite(value, shape, label):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"finite {label} with shape {shape} required")
    return a.copy()


@dataclass(frozen=True)
class ReferencePointLaw:
    shell_reference_m: np.ndarray
    beam_reference_m: np.ndarray
    shell_port: np.ndarray
    beam_port: np.ndarray
    constraint: np.ndarray
    reference_frames: np.ndarray
    law_id: str = ""
    production_qualified: bool = False
    beam_constraint_differential: np.ndarray | None = None

    def __post_init__(self):
        port_shape = np.asarray(self.shell_port).shape
        if (len(port_shape) != 2 or port_shape[0] != 6 or
                port_shape[1] not in (18, 24)):
            raise ValueError("three- or four-vertex physical shell port required")
        shell_width = port_shape[1]
        expected_id = S3_LAW_ID if shell_width == 18 else LAW_ID
        if self.law_id and self.law_id != expected_id:
            raise ValueError("point law identity does not match shell support")
        object.__setattr__(self, "law_id", expected_id)
        for name, shape in (("shell_reference_m", (3,)),
                            ("beam_reference_m", (3,)),
                            ("shell_port", (6, shell_width)),
                            ("beam_port", (6, 12)),
                            ("constraint", (6, shell_width+12)),
                            ("reference_frames", (2, 3, 3))):
            owned = _finite(getattr(self, name), shape, name)
            # A bytes-backed view cannot be made writable by a caller, so
            # the reference curvature cannot drift from its stored rows.
            sealed = np.frombuffer(owned.tobytes(), dtype=np.float64).reshape(shape)
            object.__setattr__(self, name, sealed)
        if self.beam_constraint_differential is not None:
            owned = _finite(self.beam_constraint_differential, (6, 6), "beam pose differential")
            object.__setattr__(self, "beam_constraint_differential",
                               np.frombuffer(owned.tobytes(), dtype=float).reshape(6, 6))

    def gap(self, shell_displacement, beam_displacement):
        shell = _finite(shell_displacement, (self.shell_port.shape[1],),
                        "shell displacement")
        beam = _finite(beam_displacement, (12,), "beam displacement")
        return self.constraint @ np.r_[shell, beam]

    def multiplier_work(self, shell_displacement, beam_displacement, multiplier):
        lam = _finite(multiplier, (6,), "point multiplier")
        return float(lam @ self.gap(shell_displacement, beam_displacement))

    def forces(self, multiplier):
        lam = _finite(multiplier, (6,), "point multiplier")
        action = self.constraint.T @ lam
        shell_width = self.shell_port.shape[1]
        return action[:shell_width], action[shell_width:]

    def candidate_prestress_curvature(self, multiplier):
        """Reference Hessian of objective multiplier work through affine ports.

        This is a private conservative spectral candidate.  The ordinary
        reference-linear static gap and stiffness remain affine.
        """
        lam = _finite(multiplier, (6,), "point multiplier")
        positions = np.stack((self.shell_reference_m, self.beam_reference_m))
        joint = RigidPoseJoint(positions, self.reference_frames)
        pose = joint.evaluate(positions, self.reference_frames, lam)
        shell_width = self.shell_port.shape[1]
        port = np.zeros((12, shell_width+12))
        port[:6, :shell_width] = self.shell_port
        port[6:, shell_width:] = self.beam_port
        return np.array(port.T @ pose.chart_hessian[:12, :12] @ port,
                        copy=True)


def compile_reference_point(member: PhysicalB2Member, shell_vertices_m, *,
                            station: float, offset_local_m=(0., 0., 0.)):
    """Return six affine rows for one finite shell support and B2 station.

    Shell DOFs are three or four [ux,uy,uz,rx,ry,rz] tuples. The shell port's rigid
    translation/rotation is the equal-vertex-weight fit of the *nodal
    translations*. Rotational shell DOFs still enter the actual shell stiffness;
    they are deliberately not identified with an unsupported nodal drill.
    """
    if not isinstance(member, PhysicalB2Member):
        raise ValueError("owned physical B2 member required")
    shape = np.asarray(shell_vertices_m).shape
    if shape not in ((3, 3), (4, 3)):
        raise ValueError("three or four shell vertices required")
    x = _finite(shell_vertices_m, shape, "shell vertices")
    if type(station) not in (int, float) or not np.isfinite(station) or not 0. <= station <= 1.:
        raise ValueError("normalized B2 station in [0,1] required")
    origin = x.mean(axis=0)
    radius = x-origin
    fit = np.zeros((3*len(x), 6))
    for i, r in enumerate(radius):
        fit[3*i:3*i+3, :3] = np.eye(3)
        fit[3*i:3*i+3, 3:] = -_skew(r)
    if np.linalg.matrix_rank(fit) != 6:
        raise ValueError("finite noncollinear shell point support required")
    # Equal weights identify the actual support vertices. A least-squares
    # fit is a kinematic port; shell constitutive stiffness is supplied outside.
    left = np.linalg.solve(fit.T @ fit, fit.T)
    shell_port = np.zeros((6, 6*len(x)))
    for i in range(len(x)):
        shell_port[:, 6*i:6*i+3] = left[:, 3*i:3*i+3]
    trace = member.trace_global(np.zeros(12), stations=(float(station),),
                                offset_local_m=offset_local_m)
    beam_port = trace.jacobian[0].copy()
    point = trace.reference_points_m[0].copy()
    normal = np.cross(x[1]-x[0], x[-1]-x[0])
    area_scale = np.linalg.norm(normal)
    diameter = max(np.linalg.norm(a-b) for a in x for b in x)
    if area_scale <= 1e-12*diameter**2:
        raise ValueError("nondegenerate shell reference plane required")
    normal /= area_scale
    if max(abs((vertex-origin)@normal) for vertex in x) > 1e-10*diameter:
        raise ValueError("planar shell point support required")
    tangent = x[1]-x[0]
    tangent /= np.linalg.norm(tangent)
    side = np.cross(normal, tangent)
    shell_frame = np.column_stack((tangent, side, normal))
    beam_frame = member.local_frame_rows().T
    reference_positions = np.array((origin, point))
    reference_frames = np.array((shell_frame, beam_frame))
    joint = RigidPoseJoint(reference_positions, reference_frames)
    derivative = joint.evaluate(reference_positions, reference_frames,
                                np.zeros(6)).constraint_jacobian
    constraint = np.column_stack((derivative[:, :6] @ shell_port,
                                  derivative[:, 6:] @ beam_port))
    return ReferencePointLaw(origin, point, shell_port, beam_port, constraint,
                             reference_frames, beam_constraint_differential=derivative[:, 6:])
