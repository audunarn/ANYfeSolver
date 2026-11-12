"""Physical free-sphere coordinates and shared radial contact work maps.

Isotropic inertia is explicitly supplied. Frictionless sphere traction is central,
so its finite spin is an exact, decoupled torque-free motion; translation remains
in the monolithic structural Newmark system. No mass is inferred from radius.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation


FREE_SPHERE_PROFILE = "CLASSICAL_B2_B3_FREE_ISOTROPIC_SPHERE_NEWMARK_DEV_V1"
FREE_SPHERE_HISTORY_PROFILE = "CLASSICAL_B2_B3_FREE_ISOTROPIC_SPHERE_NEWMARK_HISTORY_DEV_V1"
FREE_SHELL_SPHERE_PROFILE = "PLANAR_GUIDED_Q4_FREE_ISOTROPIC_SPHERE_NEWMARK_DEV_V1"
FREE_SHELL_SPHERE_HISTORY_PROFILE = "PLANAR_GUIDED_Q4_FREE_ISOTROPIC_SPHERE_NEWMARK_HISTORY_DEV_V1"


@dataclass(frozen=True)
class ContactFreeRigidBody:
    """A sphere centred at its declared geometry point, initially unrotated.

    ``isotropic_inertia_kg_m2`` is each principal inertia about that centre.
    Velocity and angular velocity are in the spatial frame, SI units.
    """
    body_name: str
    mass_kg: float
    isotropic_inertia_kg_m2: float
    initial_velocity_m_s: tuple[float, float, float] = (0., 0., 0.)
    initial_angular_velocity_rad_s: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        if type(self.body_name) is not str or not self.body_name:
            raise ValueError("free rigid body needs a stable body name")
        for name in ("mass_kg", "isotropic_inertia_kg_m2"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or value <= 0.):
                raise ValueError(f"{name} must be explicitly finite and positive")
            object.__setattr__(self, name, float(value))
        for name in ("initial_velocity_m_s", "initial_angular_velocity_rad_s"):
            try:
                values = tuple(getattr(self, name))
            except TypeError as exc:
                raise ValueError(f"{name} needs three finite components") from exc
            if (len(values) != 3 or any(isinstance(x, (bool, np.bool_))
                    or not isinstance(x, (int, float, np.integer, np.floating))
                    or not np.isfinite(x) for x in values)):
                raise ValueError(f"{name} needs three finite components")
            object.__setattr__(self, name, tuple(float(x) for x in values))

    def contract(self):
        return dict(body_name=self.body_name, mass_kg=self.mass_kg,
                    isotropic_inertia_kg_m2=self.isotropic_inertia_kg_m2,
                    initial_velocity_m_s=list(self.initial_velocity_m_s),
                    initial_angular_velocity_rad_s=list(self.initial_angular_velocity_rad_s))


@dataclass(frozen=True)
class FreeRigidBodyState:
    centre_of_mass_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    acceleration_m_s2: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]
    angular_velocity_rad_s: tuple[float, float, float]
    contact_force_n: tuple[float, float, float]
    kinetic_energy_j: float
    linear_momentum_kg_m_s: tuple[float, float, float]
    angular_momentum_about_origin_kg_m2_s: tuple[float, float, float]


class _FreeSphereManager:
    """Extend the radial contributor by the same potential's centre derivatives."""
    def __init__(self, child, body, *, allow_shell=False):
        from .beam_radial import _BeamRadialManager
        from .sphere import _SphereShellManager
        if type(allow_shell) is not bool:
            raise TypeError('free-sphere shell selection must be explicit boolean')
        expected = _SphereShellManager if allow_shell else _BeamRadialManager
        if type(child) is not expected or (allow_shell and child.kind != 'sphere'):
            raise ValueError('free sphere requires its explicitly selected radial work owner')
        self.child, self.body = child, body
        self.definition, self.settings = child.definition, child.settings
        self.profile = FREE_SHELL_SPHERE_PROFILE if allow_shell else FREE_SPHERE_PROFILE
        self.gap_tolerance = child.gap_tolerance
        self.radius = child.radius
        self.cells = child.faces if allow_shell else child.cells
        self.reference_point = child.reference_center if allow_shell else child.reference_point
        self.shell = allow_shell
        self.structural_size = child.model.mesh.dof_manager.total_dofs
        # Uniform nodal translation is the exact centre work map because the
        # centreline shape functions sum to one, including deformed B3 curves.
        nodes = {node: dofs for cell in self.cells for node, dofs in zip(cell.node_ids, cell.dofs)}
        rows = [int(dof) for dofs in nodes.values() for dof in dofs[:3]]
        columns = list(range(3))*len(nodes)
        self.translation_map = sparse.coo_matrix((np.ones(len(rows)), (rows, columns)),
            shape=(self.structural_size, 3)).tocsr()

    def restart_contract(self):
        return dict(profile=self.profile, contact=self.child.restart_contract(),
                    free_rigid_body=self.body.contract())

    def evaluate(self, displacement, *, tangent=True):
        u = np.asarray(displacement, dtype=float)
        if u.shape != (self.structural_size+3,) or not np.all(np.isfinite(u)):
            raise ValueError("free sphere needs structural and three centre coordinates")
        if self.shell:self.child.center = self.reference_point+u[-3:]
        else:self.child.point = self.reference_point+u[-3:]
        force, stiffness, result = self.child.evaluate(u[:self.structural_size], tangent=tangent)
        A = self.translation_map
        residual = np.r_[force, -np.asarray(A.T @ force).reshape(-1)]
        coupled = None
        if tangent:
            cross = -stiffness @ A
            coupled = sparse.bmat([[stiffness, cross],
                [cross.T, A.T @ stiffness @ A]], format='csr')
        return residual, coupled, replace(result, profile=self.profile)

    def state(self, time_s, displacement, velocity, acceleration, contact):
        centre = self.reference_point+np.asarray(displacement)[-3:]
        v, a = np.asarray(velocity)[-3:], np.asarray(acceleration)[-3:]
        omega = np.asarray(self.body.initial_angular_velocity_rad_s)
        rotation = Rotation.from_rotvec(time_s*omega).as_matrix()
        momentum = self.body.mass_kg*v
        angular = np.cross(centre, momentum)+self.body.isotropic_inertia_kg_m2*omega
        kinetic = .5*(self.body.mass_kg*float(v @ v)
                       +self.body.isotropic_inertia_kg_m2*float(omega @ omega))
        if not all(np.all(np.isfinite(x)) for x in (centre, v, a, rotation, momentum, angular, kinetic)):
            raise RuntimeError("nonfinite free rigid-body state")
        vec = lambda value: tuple(float(x) for x in value)
        return FreeRigidBodyState(vec(centre), vec(v), vec(a), tuple(map(vec, rotation)),
            vec(omega), vec(contact.rigid_force), float(kinetic), vec(momentum), vec(angular))
