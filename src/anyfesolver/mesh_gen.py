"""ANYfeSolver adapters for neutral meshes produced by ANYmesher.

ANYmesher owns geometry, numbering, mesh quality and beam-shell coupling
records.  This module retains the solver decisions: converting those records to
FEModel elements and MPCs, assigning the legacy default material, and
interpreting the historical support labels.  Public 0.1 mesh helpers remain
available through this adapter for the 0.2.x compatibility line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from anymesher import (
    Mesh as NeutralMesh,
    PanelMeshConfig as NeutralPanelMeshConfig,
    StiffenedPanel as NeutralStiffenedPanel,
    StiffenerCrossSection,
    beam_mesh as neutral_beam_mesh,
    panel_edge_nodes,
    simple_panel_mesh as neutral_simple_panel_mesh,
    stiffened_panel_mesh as neutral_stiffened_panel_mesh,
    verify_mesh_quality as verify_neutral_mesh_quality,
)

if TYPE_CHECKING:
    from .fe_core import FEModel, FEMesh, Material

@dataclass
class MeshConfig:
    """Configuration for mesh generation."""

    shell_num_divisions_x: int = 4
    shell_num_divisions_y: int = 4
    beam_num_divisions: int = 1

    use_coupling_elements: bool = True
    coupling_stiffness: float = 0.0
    use_shared_nodes: bool = False

    tolerance: float = 1.0e-6

    default_material: str = "steel"
    plate_material: str = "steel"
    stiffener_material: str = "steel"

    use_8node_shells: bool = False
    align_mesh_to_stiffeners: bool = False
    rotating_eccentricity: bool = True

    def __post_init__(self) -> None:
        _validate_rotating_eccentricity(self.rotating_eccentricity)


def _validate_rotating_eccentricity(value: bool) -> None:
    if type(value) is not bool:
        raise TypeError("rotating_eccentricity must be a bool")


@dataclass
class PanelGeometry:
    """Geometry and load metadata for a rectangular stiffened panel."""

    length: float = 0.0
    width: float = 0.0

    plate_thickness: float = 0.0
    plate_material: str = "steel"

    stiffener_type: str = "T-bar"
    stiffener_spacing: float = 0.0
    stiffener_height: float = 0.0
    stiffener_web_thickness: float = 0.0
    stiffener_flange_width: float = 0.0
    stiffener_flange_thickness: float = 0.0
    stiffener_material: str = "steel"
    num_stiffeners: int = 1

    in_plane_support: str = "Integrated"
    rotational_support: str = "SS"

    axial_stress: float = 0.0
    transverse_stress: float = 0.0
    shear_stress: float = 0.0
    pressure: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "width": self.width,
            "plate_thickness": self.plate_thickness,
            "plate_material": self.plate_material,
            "stiffener_type": self.stiffener_type,
            "stiffener_spacing": self.stiffener_spacing,
            "stiffener_height": self.stiffener_height,
            "stiffener_web_thickness": self.stiffener_web_thickness,
            "stiffener_flange_width": self.stiffener_flange_width,
            "stiffener_flange_thickness": self.stiffener_flange_thickness,
            "stiffener_material": self.stiffener_material,
            "num_stiffeners": self.num_stiffeners,
            "in_plane_support": self.in_plane_support,
            "rotational_support": self.rotational_support,
            "axial_stress": self.axial_stress,
            "transverse_stress": self.transverse_stress,
            "shear_stress": self.shear_stress,
            "pressure": self.pressure,
        }

    @classmethod
    def from_anystructure(cls, anystructure_data: Any) -> "PanelGeometry":
        geometry = cls()
        if hasattr(anystructure_data, "Plate"):
            plate = anystructure_data.Plate
            geometry.length = getattr(plate, "span", 0.0)
            geometry.width = getattr(plate, "spacing", 0.0)
            geometry.plate_thickness = getattr(plate, "t", 0.0)
            geometry.plate_material = "steel"
        if hasattr(anystructure_data, "Stiffener"):
            stiffener = anystructure_data.Stiffener
            geometry.stiffener_type = getattr(stiffener, "stiffener_type", "T-bar")
            geometry.stiffener_spacing = getattr(stiffener, "spacing", 0.0)
            geometry.stiffener_height = getattr(stiffener, "hw", 0.0)
            geometry.stiffener_web_thickness = getattr(stiffener, "tw", 0.0)
            geometry.stiffener_flange_width = getattr(stiffener, "b", 0.0)
            geometry.stiffener_flange_thickness = getattr(stiffener, "tf", 0.0)
            geometry.stiffener_material = "steel"
        if hasattr(anystructure_data, "sigma_x1"):
            geometry.axial_stress = anystructure_data.sigma_x1
        if hasattr(anystructure_data, "sigma_y1"):
            geometry.transverse_stress = anystructure_data.sigma_y1
        if hasattr(anystructure_data, "tau_xy"):
            geometry.shear_stress = anystructure_data.tau_xy
        if hasattr(anystructure_data, "pressure"):
            geometry.pressure = anystructure_data.pressure
        return geometry

# Parent-adopted input-admission threshold for coupling shape weights:
# the weights must form a partition of unity within this absolute
# dimensionless tolerance (rtol=0).  Weights are never renormalized.
COUPLING_PARTITION_OF_UNITY_ATOL = 1.0e-10


class InterpolatedBeamShellMPCElement:
    """
    Duck-typed MPC-only element for eccentric beam-to-shell coupling.

    The first node is the beam slave node.  The remaining nodes are the shell
    master nodes of the shell element underneath the beam node.  Shape weights
    are evaluated at the projected beam-node location in the shell element.
    """

    # Class default keeps legacy/default V1 checkpoint namespaces unchanged.
    connection_trace = "linear"

    def __init__(
        self,
        element_id: int,
        beam_node_id: int,
        shell_node_ids: List[int],
        shape_weights: np.ndarray,
        eccentricity: np.ndarray,
        material_name: str = "steel",
        *,
        rotating_eccentricity: bool = True,
        connection_trace: str = "linear",
    ):
        _validate_rotating_eccentricity(rotating_eccentricity)
        self.rotating_eccentricity = rotating_eccentricity
        if connection_trace not in ("linear", "director_integrated"):
            raise ValueError("connection_trace must be 'linear' or 'director_integrated'")
        if connection_trace == "director_integrated" and not rotating_eccentricity:
            raise ValueError("director_integrated requires rotating_eccentricity=True")
        if connection_trace != "linear":
            self.connection_trace = connection_trace
        self.element_id = element_id
        self.beam_node_id = beam_node_id
        self.shell_node_ids = list(shell_node_ids)
        self.shape_weights = np.asarray(shape_weights, dtype=float)
        self.eccentricity = np.asarray(eccentricity, dtype=float)
        self.material_name = material_name
        self.node_ids = [beam_node_id] + self.shell_node_ids
        self._stiffness_matrix = None
        self._mass_matrix = None
        self._validated_shape_weights()
        self._validated_eccentricity()

    @property
    def num_nodes(self) -> int:
        return 1 + len(self.shell_node_ids)

    @property
    def dofs_per_node(self) -> int:
        return 6

    @property
    def total_dofs(self) -> int:
        return self.num_nodes * self.dofs_per_node

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        coords = []
        for node_id in self.node_ids:
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(f"MPC element {self.element_id} references missing node {node_id}")
            coords.append(node.coords())
        return np.asarray(coords, dtype=float)

    def get_dof_mapping(self, mesh: "FEMesh") -> List[int]:
        dofs: List[int] = []
        for node_id in self.node_ids:
            node = mesh.get_node(node_id)
            if node is not None:
                dofs.extend(node.dofs)
        return dofs

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        # Constraint is enforced exactly by assembly.build_constraint_transformation().
        K = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        self._stiffness_matrix = K
        return K

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

    def compute_geometric_stiffness_matrix(self, mesh: "FEMesh", material: "Material", state: Any = None) -> np.ndarray:
        return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

    def compute_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Any = None,
        num_layers: int = 5,
        tangent: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Any]:
        force = np.zeros(self.total_dofs, dtype=float)
        stiffness = np.zeros((self.total_dofs, self.total_dofs), dtype=float) if tangent else None
        return force, stiffness, state

    def compute_stresses(
        self,
        mesh: "FEMesh",
        displacements: np.ndarray,
        material: "Material",
        return_global: bool = False,
    ) -> Dict[str, np.ndarray]:
        return {}

    def _validated_shape_weights(self) -> np.ndarray:
        """Return the admitted shape weights, revalidated against mutation."""
        weights = np.asarray(self.shape_weights, dtype=float)
        if weights.ndim != 1 or weights.size != len(self.shell_node_ids):
            raise ValueError(
                f"Interpolated MPC element {self.element_id} requires a 1-D "
                f"shape-weight array with one weight per shell master node "
                f"({len(self.shell_node_ids)} expected, got shape {weights.shape})"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError(
                f"Interpolated MPC element {self.element_id} shape weights "
                "contain non-finite values"
            )
        # The reduction itself may overflow to +/-inf or NaN for finite
        # mixed-sign weights; the explicit finiteness check below rejects
        # that, so the overflow warning is deliberately silenced here.
        with np.errstate(over="ignore", invalid="ignore"):
            weight_sum = float(weights.sum())
        if (
            not np.isfinite(weight_sum)
            or abs(weight_sum - 1.0) > COUPLING_PARTITION_OF_UNITY_ATOL
        ):
            raise ValueError(
                f"Interpolated MPC element {self.element_id} shape weights must "
                f"form a partition of unity within "
                f"{COUPLING_PARTITION_OF_UNITY_ATOL}; got sum {weight_sum!r}"
            )
        return weights

    def _validated_eccentricity(self) -> np.ndarray:
        """Return the admitted common offset r, revalidated against mutation."""
        eccentricity = np.asarray(self.eccentricity, dtype=float)
        if eccentricity.shape != (3,):
            raise ValueError(
                f"Interpolated MPC element {self.element_id} eccentricity must "
                f"have shape (3,), got {eccentricity.shape}"
            )
        if not np.all(np.isfinite(eccentricity)):
            raise ValueError(
                f"Interpolated MPC element {self.element_id} eccentricity "
                "contains non-finite values"
            )
        return eccentricity

    def get_mpc_constraints(self, mesh: "FEMesh") -> List[Dict[str, Any]]:
        if getattr(self, "connection_trace", "linear") not in ("linear", "director_integrated"):
            raise ValueError("unsupported connection_trace")
        if getattr(self, "connection_trace", "linear") == "director_integrated":
            geometry = self._director_trace_geometry(mesh)
            if geometry is not None:
                from ._director_trace import director_trace_state
                positions, slave, offset, xi, masters = geometry
                _, B, _ = director_trace_state(positions, slave, offset, xi, np.zeros(12))
                inputs = [d for n in masters for d in mesh.get_node(n).dofs]
                return [{"slave": mesh.get_node(self.beam_node_id).dofs[i],
                         "masters": {d: float(v) for d,v in zip(inputs,B[i]) if v != 0.},
                         "value": 0., "label": f"director_trace_{self.element_id}_{i}"}
                        for i in range(6)]
        # Revalidate the aliased mutable inputs so later corruption cannot
        # silently bypass admission, and fail closed on any missing
        # referenced node (including zero-weight masters) before returning
        # a partial constraint set.
        shape_weights = self._validated_shape_weights()
        rx, ry, rz = self._validated_eccentricity()

        beam_node = mesh.get_node(self.beam_node_id)
        if beam_node is None:
            raise ValueError(
                f"Interpolated MPC element {self.element_id} references missing "
                f"beam node {self.beam_node_id}"
            )

        beam_dofs = beam_node.dofs
        translational_masters = [{}, {}, {}]
        rotational_masters = [{}, {}, {}]

        for shell_node_id, weight in zip(self.shell_node_ids, shape_weights):
            shell_node = mesh.get_node(shell_node_id)
            if shell_node is None:
                raise ValueError(
                    f"Interpolated MPC element {self.element_id} references "
                    f"missing shell master node {shell_node_id}"
                )
            if float(weight) == 0.0:
                continue
            s = shell_node.dofs
            w = float(weight)

            translational_masters[0][s[0]] = translational_masters[0].get(s[0], 0.0) + w
            translational_masters[0][s[4]] = translational_masters[0].get(s[4], 0.0) + w * rz
            translational_masters[0][s[5]] = translational_masters[0].get(s[5], 0.0) - w * ry

            translational_masters[1][s[1]] = translational_masters[1].get(s[1], 0.0) + w
            translational_masters[1][s[3]] = translational_masters[1].get(s[3], 0.0) - w * rz
            translational_masters[1][s[5]] = translational_masters[1].get(s[5], 0.0) + w * rx

            translational_masters[2][s[2]] = translational_masters[2].get(s[2], 0.0) + w
            translational_masters[2][s[3]] = translational_masters[2].get(s[3], 0.0) + w * ry
            translational_masters[2][s[4]] = translational_masters[2].get(s[4], 0.0) - w * rx

            rotational_masters[0][s[3]] = rotational_masters[0].get(s[3], 0.0) + w
            rotational_masters[1][s[4]] = rotational_masters[1].get(s[4], 0.0) + w
            rotational_masters[2][s[5]] = rotational_masters[2].get(s[5], 0.0) + w

        return [
            {"slave": beam_dofs[0], "masters": translational_masters[0], "value": 0.0, "label": f"interp_beam_shell_ux_{self.element_id}"},
            {"slave": beam_dofs[1], "masters": translational_masters[1], "value": 0.0, "label": f"interp_beam_shell_uy_{self.element_id}"},
            {"slave": beam_dofs[2], "masters": translational_masters[2], "value": 0.0, "label": f"interp_beam_shell_uz_{self.element_id}"},
            {"slave": beam_dofs[3], "masters": rotational_masters[0], "value": 0.0, "label": f"interp_beam_shell_rx_{self.element_id}"},
            {"slave": beam_dofs[4], "masters": rotational_masters[1], "value": 0.0, "label": f"interp_beam_shell_ry_{self.element_id}"},
            {"slave": beam_dofs[5], "masters": rotational_masters[2], "value": 0.0, "label": f"interp_beam_shell_rz_{self.element_id}"},
        ]

    def get_nonlinear_mpc_descriptor(self, mesh: "FEMesh") -> Optional[Dict[str, Any]]:
        """Return the finite-rotation master-slave descriptor for corotational solves.

        The admitted intrinsic SO(3) mean ``R_bar`` of shell-master rotations
        satisfies ``sum_i w_i Log(R_bar.T R_i) = 0`` on the supported local
        branch. The eccentric offset rotates with this frame:
            u_b     = sum_i w_i u_{s,i} + (R_bar - I) r_0
            theta_b = Log(R_bar)
        Constructor option ``rotating_eccentricity=False`` disables the finite-
        rotation lever arm and retains the fixed-reference linearized law.
        """
        rotating_eccentricity = getattr(self, "rotating_eccentricity", True)
        _validate_rotating_eccentricity(rotating_eccentricity)
        trace = getattr(self, "connection_trace", "linear")
        if trace not in ("linear", "director_integrated"):
            raise ValueError("unsupported connection_trace")
        geometry = self._director_trace_geometry(mesh) if trace == "director_integrated" else None
        if not rotating_eccentricity:
            return None
        shape_weights = self._validated_shape_weights()
        r = np.asarray(self._validated_eccentricity(), dtype=float).copy()
        if float(np.linalg.norm(r)) == 0.0 and geometry is None:
            return None
        beam_node = mesh.get_node(self.beam_node_id)
        if beam_node is None:
            raise ValueError(
                f"Interpolated MPC element {self.element_id} references missing "
                f"beam node {self.beam_node_id}"
            )
        merged_weights: Dict[int, float] = {}
        merged_dofs: Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        for shell_node_id, weight in zip(self.shell_node_ids, shape_weights):
            shell_node = mesh.get_node(shell_node_id)
            if shell_node is None:
                raise ValueError(
                    f"Interpolated MPC element {self.element_id} references "
                    f"missing shell master node {shell_node_id}"
                )
            w = float(weight)
            if w == 0.0:
                continue
            nid = int(shell_node_id)
            s = shell_node.dofs
            merged_weights[nid] = merged_weights.get(nid, 0.0) + w
            merged_dofs[nid] = (
                tuple(int(d) for d in s[:3]),
                tuple(int(d) for d in s[3:6]),
            )
        masters = tuple(
            (merged_dofs[nid][0], merged_dofs[nid][1], float(w))
            for nid, w in merged_weights.items()
            if abs(float(w)) > 0.0
        )
        beam_dofs = beam_node.dofs
        return {
            "element_id": int(self.element_id),
            "slave_u_dofs": tuple(int(d) for d in beam_dofs[:3]),
            "slave_theta_dofs": tuple(int(d) for d in beam_dofs[3:6]),
            "masters": masters,
            "eccentricity": r,
            "label_prefix": f"interp_beam_shell_{self.element_id}",
            "director_trace_geometry": geometry,
        }

    def _director_trace_geometry(self, mesh):
        """Admit only a geometry-consistent straight two-master edge or endpoint."""
        _validate_rotating_eccentricity(getattr(self, "rotating_eccentricity", True))
        if not getattr(self, "rotating_eccentricity", True):
            raise ValueError("director_integrated requires rotating_eccentricity=True")
        weights = self._validated_shape_weights()
        offset = self._validated_eccentricity().copy()
        nodes = [mesh.get_node(n) for n in self.shell_node_ids]
        beam = mesh.get_node(self.beam_node_id)
        if beam is None or any(n is None for n in nodes):
            raise ValueError("director_integrated references missing nodes")
        if len(set(self.shell_node_ids)) != len(nodes) or len(nodes) not in (1,2) or np.any(weights < 0.):
            raise ValueError("director_integrated requires one or two distinct nonnegative edge masters")
        positions = np.array([n.coords() for n in nodes],dtype=float)
        slave = np.array(beam.coords(),dtype=float)
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(slave)):
            raise ValueError("director_integrated requires finite reference geometry")
        span = float(np.linalg.norm(positions[-1]-positions[0])) if len(nodes)==2 else 0.
        scale = max(span,float(np.linalg.norm(offset)),1.)
        if np.linalg.norm(slave-weights@positions-offset) > 64*np.finfo(float).eps*scale:
            raise ValueError("director_integrated slave reference position disagrees with edge weights and offset")
        if len(nodes)==2 and (not np.isfinite(span) or span <= 0.):
            raise ValueError("director_integrated requires a finite positive edge length")
        if abs(float(weights.sum())-1.) > 64*np.finfo(float).eps:
            raise ValueError("director_integrated requires unit weights to reference roundoff")
        if len(nodes)==1 or np.count_nonzero(weights)==1:
            return None
        return positions.copy(),slave.copy(),offset,float(weights[1]),tuple(self.shell_node_ids)



class RigidLidMPCElement:
    """
    Constraint-only rigid end diaphragm.

    The element ties an end ring to a free center reference node using rigid-body
    kinematics. It adds end-ring coupling without adding lid shell elements, so
    lid stresses and lid pressure loads are not recovered.
    """

    def __init__(
        self,
        element_id: int,
        center_node_id: int,
        ring_node_ids: List[int],
        material_name: str = "steel",
    ):
        self.element_id = int(element_id)
        self.center_node_id = int(center_node_id)
        self.ring_node_ids = [int(node_id) for node_id in ring_node_ids if int(node_id) != int(center_node_id)]
        self.material_name = material_name
        self.node_ids = [self.center_node_id] + self.ring_node_ids

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def dofs_per_node(self) -> int:
        return 6

    @property
    def total_dofs(self) -> int:
        return self.num_nodes * self.dofs_per_node

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        coords = []
        for node_id in self.node_ids:
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(f"Rigid lid element {self.element_id} references missing node {node_id}")
            coords.append(node.coords())
        return np.asarray(coords, dtype=float)

    def get_dof_mapping(self, mesh: "FEMesh") -> List[int]:
        return []

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        return np.zeros((0, 0), dtype=float)

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        return np.zeros((0, 0), dtype=float)

    def compute_geometric_stiffness_matrix(self, mesh: "FEMesh", material: "Material", state: Any = None) -> np.ndarray:
        return np.zeros((0, 0), dtype=float)

    def compute_stresses(
        self,
        mesh: "FEMesh",
        displacements: np.ndarray,
        material: "Material",
        return_global: bool = False,
    ) -> Dict[str, np.ndarray]:
        return {}

    @staticmethod
    def _nonzero_masters(items: Dict[int, float]) -> Dict[int, float]:
        return {int(dof): float(value) for dof, value in items.items() if abs(float(value)) > 0.0}

    def get_mpc_constraints(self, mesh: "FEMesh") -> List[Dict[str, Any]]:
        center = mesh.get_node(self.center_node_id)
        if center is None:
            return []
        center_dofs = center.dofs
        constraints: List[Dict[str, Any]] = []
        for ring_node_id in self.ring_node_ids:
            node = mesh.get_node(ring_node_id)
            if node is None:
                continue
            rx, ry, rz = (node.coords() - center.coords()).tolist()
            node_dofs = node.dofs
            translation_masters = (
                {center_dofs[0]: 1.0, center_dofs[4]: rz, center_dofs[5]: -ry},
                {center_dofs[1]: 1.0, center_dofs[3]: -rz, center_dofs[5]: rx},
                {center_dofs[2]: 1.0, center_dofs[3]: ry, center_dofs[4]: -rx},
            )
            for local_index, masters in enumerate(translation_masters):
                constraints.append(
                    {
                        "slave": node_dofs[local_index],
                        "masters": self._nonzero_masters(masters),
                        "value": 0.0,
                        "label": f"rigid_lid_{self.element_id}_u{local_index + 1}",
                    }
                )
            for local_index in range(3, 6):
                constraints.append(
                    {
                        "slave": node_dofs[local_index],
                        "masters": {center_dofs[local_index]: 1.0},
                        "value": 0.0,
                        "label": f"rigid_lid_{self.element_id}_r{local_index - 2}",
                    }
                )
        return constraints

def _neutral_panel(panel: PanelGeometry) -> NeutralStiffenedPanel:
    """Copy geometry-only panel fields into the canonical mesher contract."""

    return NeutralStiffenedPanel(
        length=float(panel.length),
        width=float(panel.width),
        plate_thickness=float(panel.plate_thickness),
        stiffener_type=str(panel.stiffener_type),
        stiffener_spacing=float(panel.stiffener_spacing),
        stiffener_height=float(panel.stiffener_height),
        stiffener_web_thickness=float(panel.stiffener_web_thickness),
        stiffener_flange_width=float(panel.stiffener_flange_width),
        stiffener_flange_thickness=float(panel.stiffener_flange_thickness),
        num_stiffeners=int(panel.num_stiffeners),
    )


def _neutral_config(config: MeshConfig) -> NeutralPanelMeshConfig:
    """Copy mesh-only options into ANYmesher without analysis metadata."""

    return NeutralPanelMeshConfig(
        shell_num_divisions_x=int(config.shell_num_divisions_x),
        shell_num_divisions_y=int(config.shell_num_divisions_y),
        beam_num_divisions=int(config.beam_num_divisions),
        use_coupling_elements=bool(config.use_coupling_elements),
        tolerance=float(config.tolerance),
        use_8node_shells=bool(config.use_8node_shells),
        align_mesh_to_stiffeners=bool(config.align_mesh_to_stiffeners),
    )


def _add_legacy_steel(
    model: "FEModel",
    *,
    density: float = 0.0,
    yield_stress: float = 0.0,
) -> None:
    """Install the material historically supplied by generated test models."""

    model.add_material(
        name="steel",
        elastic_modulus=210.0e9,
        poisson_ratio=0.3,
        density=float(density),
        yield_stress=float(yield_stress),
    )
    model.current_material = "steel"


def _install_nodes(model: "FEModel", mesh: NeutralMesh) -> None:
    for node_id, coordinates in mesh.nodes.items():
        x, y, z = np.asarray(coordinates, dtype=float).reshape(3)
        model.add_node(int(node_id), float(x), float(y), float(z))


def _install_shells(
    model: "FEModel",
    mesh: NeutralMesh,
    *,
    thickness: float,
    material_name: str = "steel",
) -> None:
    from .elements import create_shell_element

    for element_id, node_ids in {**mesh.quads, **mesh.tris}.items():
        director_kwargs: dict[str, object] = {}
        if len(node_ids) == 3:
            # NeutralMesh panel triangles are generated on the physical +Z
            # plate surface; do not delegate their director to connectivity.
            director_kwargs["reference_normal"] = (0.0, 0.0, 1.0)
        model.add_element(
            int(element_id),
            create_shell_element(
                int(element_id),
                list(node_ids),
                material_name=material_name,
                thickness=float(thickness),
                **director_kwargs,
            ),
        )


def _install_panel_beams(
    model: "FEModel",
    mesh: NeutralMesh,
    panel: PanelGeometry,
    *,
    material_name: str = "steel",
) -> None:
    from .elements import BeamElement, QuadraticBeamElement

    section = StiffenerCrossSection.from_geometry(
        panel.stiffener_type,
        panel.stiffener_height,
        panel.stiffener_web_thickness,
        panel.stiffener_flange_width,
        panel.stiffener_flange_thickness,
    ).as_dict()
    for element_id, node_ids in mesh.beams.items():
        element_type = QuadraticBeamElement if len(node_ids) == 3 else BeamElement
        model.add_element(
            int(element_id),
            element_type(
                int(element_id),
                list(node_ids),
                material_name=material_name,
                cross_section=dict(section),
            ),
        )


def _install_couplings(
    model: "FEModel",
    mesh: NeutralMesh,
    *,
    material_name: str = "steel",
    rotating_eccentricity: bool = True,
) -> None:
    _validate_rotating_eccentricity(rotating_eccentricity)
    for element_id, coupling in mesh.couplings.items():
        model.add_element(
            int(element_id),
            InterpolatedBeamShellMPCElement(
                int(element_id),
                beam_node_id=int(coupling.beam_node),
                shell_node_ids=list(coupling.plate_nodes),
                shape_weights=np.asarray(coupling.weights, dtype=float),
                eccentricity=np.asarray(coupling.eccentricity, dtype=float),
                material_name=material_name,
                rotating_eccentricity=rotating_eccentricity,
            ),
        )


def _unique_node_ids(*node_lists: List[int]) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []
    for node_list in node_lists:
        for node_id in node_list:
            if node_id not in seen:
                seen.add(node_id)
                ordered.append(node_id)
    return ordered


def _add_custom_support(
    model: "FEModel",
    name: str,
    node_ids: List[int],
    dof_constraints: Dict[str, float],
) -> None:
    from .boundary import BoundaryCondition

    unique_ids = _unique_node_ids(node_ids)
    if unique_ids:
        model.add_boundary_condition(BoundaryCondition(name, unique_ids, dof_constraints))


def _add_panel_boundary_conditions(
    model: "FEModel",
    panel: PanelGeometry,
    edges: Dict[str, List[int]],
) -> None:
    """Interpret legacy support labels against ANYmesher's panel node sets."""

    support = (panel.in_plane_support or "").strip().lower()
    rotational = (panel.rotational_support or "").strip().upper()
    longitudinal_edges = _unique_node_ids(edges["y0"], edges["yW"])
    transverse_edges = _unique_node_ids(edges["x0"], edges["xL"])
    all_edge_nodes = edges["all"]

    if support == "integrated":
        _add_custom_support(
            model,
            "Integrated_edge_translations",
            all_edge_nodes,
            {"ux": 0.0, "uy": 0.0, "uz": 0.0},
        )
    elif support == "girder - long":
        _add_custom_support(
            model,
            "Longitudinal_girder_edges",
            longitudinal_edges,
            {"ux": 0.0, "uy": 0.0, "uz": 0.0},
        )
        _add_custom_support(model, "Transverse_reference_ux", edges["x0"][:1], {"ux": 0.0})
    elif support == "girder - trans":
        _add_custom_support(
            model,
            "Transverse_girder_edges",
            transverse_edges,
            {"ux": 0.0, "uy": 0.0, "uz": 0.0},
        )
        _add_custom_support(model, "Longitudinal_reference_uy", edges["y0"][:1], {"uy": 0.0})
    else:
        _add_custom_support(model, "Reference_out_of_plane", all_edge_nodes[:1], {"uz": 0.0})

    if rotational == "CL":
        _add_custom_support(
            model,
            "Clamped_edge_rotations",
            all_edge_nodes,
            {"rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    elif rotational == "FS":
        _add_custom_support(
            model,
            "Fixed_simple_longitudinal_rotations",
            longitudinal_edges,
            {"rx": 0.0, "ry": 0.0, "rz": 0.0},
        )


def generate_stiffened_panel_mesh(
    panel: PanelGeometry,
    config: Optional[MeshConfig] = None,
) -> "FEModel":
    """Build a solver model from ANYmesher's neutral stiffened-panel mesh."""

    from .fe_core import FEModel

    config = config or MeshConfig()
    _validate_rotating_eccentricity(config.rotating_eccentricity)
    if config.use_shared_nodes:
        raise ValueError(
            "use_shared_nodes=True is no longer supported for eccentric stiffeners. "
            "Use separate beam nodes with interpolated beam-shell MPC constraints instead."
        )

    neutral = neutral_stiffened_panel_mesh(_neutral_panel(panel), _neutral_config(config))
    model = FEModel(name=f"StiffenedPanel_{panel.length}x{panel.width}")
    _add_legacy_steel(model, density=7850.0, yield_stress=235.0e6)
    _install_nodes(model, neutral)
    _install_shells(model, neutral, thickness=panel.plate_thickness)
    _install_panel_beams(model, neutral, panel)
    _install_couplings(model, neutral, rotating_eccentricity=config.rotating_eccentricity)
    _add_panel_boundary_conditions(model, panel, panel_edge_nodes(neutral))
    return model


def generate_simple_panel_mesh(
    length: float,
    width: float,
    thickness: float,
    num_divisions_x: int = 4,
    num_divisions_y: int = 4,
    use_8node_elements: bool = False,
) -> "FEModel":
    """Build the legacy solver panel from an ANYmesher neutral primitive."""

    from .fe_core import FEModel

    neutral = neutral_simple_panel_mesh(
        length,
        width,
        thickness,
        num_divisions_x=num_divisions_x,
        num_divisions_y=num_divisions_y,
        use_8node_elements=use_8node_elements,
    )
    panel = PanelGeometry(length=length, width=width, plate_thickness=thickness)
    model = FEModel(name=f"SimplePanel_{length}x{width}")
    _add_legacy_steel(model)
    _install_nodes(model, neutral)
    _install_shells(model, neutral, thickness=thickness)
    _add_panel_boundary_conditions(model, panel, panel_edge_nodes(neutral))
    return model


def generate_beam_mesh(
    length: float,
    num_divisions: int = 10,
    cross_section: Optional[Dict[str, float]] = None,
) -> "FEModel":
    """Build the legacy fixed-end solver beam from an ANYmesher primitive."""

    from .boundary import FixedSupport
    from .elements import BeamElement
    from .fe_core import FEModel

    neutral = neutral_beam_mesh(length, num_divisions=num_divisions)
    model = FEModel(name=f"SimpleBeam_{length}")
    _add_legacy_steel(model)
    _install_nodes(model, neutral)
    section = cross_section or {
        "area": 0.01,
        "Iy": 1.0e-6,
        "Iz": 1.0e-6,
        "J": 1.0e-6,
    }
    for element_id, node_ids in neutral.beams.items():
        model.add_element(
            int(element_id),
            BeamElement(
                int(element_id),
                list(node_ids),
                material_name="steel",
                cross_section=section,
            ),
        )
    model.add_boundary_condition(FixedSupport("Fixed_1", [1]))
    return model


def _neutral_shell_mesh_from_model(model: "FEModel") -> NeutralMesh:
    """Flatten solver shell geometry for canonical ANYmesher quality checks."""

    from .elements import ShellElement

    neutral = NeutralMesh()
    for element_id, element in model.mesh.elements.items():
        if not isinstance(element, ShellElement):
            continue
        node_ids = tuple(int(node_id) for node_id in element.node_ids)
        target = neutral.tris if len(node_ids) in (3, 6) else neutral.quads
        target[int(element_id)] = node_ids
        for node_id in node_ids:
            neutral.nodes[node_id] = np.asarray(model.mesh.nodes[node_id].coords(), dtype=float)
    return neutral


def verify_mesh_quality(model: "FEModel") -> Dict[str, Any]:
    """Return the historical dict facade over ANYmesher's quality report."""

    return verify_neutral_mesh_quality(_neutral_shell_mesh_from_model(model)).as_dict()


__all__ = [
    "InterpolatedBeamShellMPCElement",
    "MeshConfig",
    "PanelGeometry",
    "RigidLidMPCElement",
    "StiffenerCrossSection",
    "generate_beam_mesh",
    "generate_simple_panel_mesh",
    "generate_stiffened_panel_mesh",
    "verify_mesh_quality",
]
