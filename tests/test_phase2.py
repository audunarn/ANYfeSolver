import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement, create_shell_element
from anyfesolver.boundary import FixedSupport, LoadCase, BoundaryCondition
from anyfesolver.assembly import build_constraint_transformation, solve_linear


def test_multilevel_mpc_cantilever():
    """Test cascading MPC constraints: node 3 depends on node 2, which depends on node 1."""
    model = FEModel(name="multilevel_mpc")
    model.add_material("steel", 210e9, 0.3)

    # 3 collinear nodes
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 2.0, 0.0, 0.0)

    # Add elements (beams)
    section = {"area": 0.01, "Iy": 8e-6, "Iz": 1e-6, "J": 1e-6, "orientation": (0.0, 0.0, 1.0)}
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_element(2, BeamElement(2, [2, 3], "steel", section))

    # Clamp node 1
    model.add_boundary_condition(FixedSupport("clamp", [1]))

    # Let's add custom boundary conditions that represent MPCs.
    # To do this, we can define a custom BC or let's create CoupledBeamShellElement-like constraint dictionary.
    # Wait, get_mpc_constraints is called on elements.
    # We can inject a mock element or custom element subclass that returns cascading MPCs.

    class CascadingMPCElement:
        def __init__(self, element_id):
            self.element_id = element_id
            self.material_name = "default"

        def get_dof_mapping(self, mesh):
            return []

        def get_mpc_constraints(self, mesh):
            dofs_node2 = mesh.get_node(2).dofs
            dofs_node3 = mesh.get_node(3).dofs
            dofs_node1 = mesh.get_node(1).dofs
            # uz of node 2 depends on uz of node 1 (e.g. uz2 = uz1 + 1.0)
            # uz of node 3 depends on uz of node 2 (e.g. uz3 = uz2 + 2.0)
            # Thus: uz3 = uz1 + 3.0
            return [
                {
                    "slave": dofs_node2[2],  # uz of node 2
                    "masters": {dofs_node1[2]: 1.0},
                    "value": 1.0,
                    "label": "mpc1"
                },
                {
                    "slave": dofs_node3[2],  # uz of node 3
                    "masters": {dofs_node2[2]: 1.0},
                    "value": 2.0,
                    "label": "mpc2"
                }
            ]

    model.add_element(3, CascadingMPCElement(3))

    # Apply load on node 3
    lc = LoadCase(name="load")
    lc.add_nodal_load(3, [0, 0, 100.0, 0, 0, 0])

    # Solve
    u, info = solve_linear(model, lc)

    # Verify displacements:
    # Node 1 is fixed: uz1 = 0
    # Node 2 uz2 = uz1 + 1.0 = 1.0
    # Node 3 uz3 = uz2 + 2.0 = 3.0
    assert np.isclose(u[model.mesh.get_node(1).dofs[2]], 0.0)
    assert np.isclose(u[model.mesh.get_node(2).dofs[2]], 1.0)
    assert np.isclose(u[model.mesh.get_node(3).dofs[2]], 3.0)


def test_circular_mpc_error():
    """Verify that circular MPC constraints raise ValueError."""
    model = FEModel(name="circular_mpc")
    model.add_material("steel", 210e9, 0.3)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)

    class CircularMPCElement:
        def __init__(self, element_id):
            self.element_id = element_id
            self.material_name = "default"
        def get_dof_mapping(self, mesh):
            return []
        def get_mpc_constraints(self, mesh):
            dofs1 = mesh.get_node(1).dofs
            dofs2 = mesh.get_node(2).dofs
            return [
                {
                    "slave": dofs1[2],
                    "masters": {dofs2[2]: 1.0},
                    "value": 0.0
                },
                {
                    "slave": dofs2[2],
                    "masters": {dofs1[2]: 1.0},
                    "value": 0.0
                }
            ]

    model.add_element(1, CircularMPCElement(1))
    lc = LoadCase("load")

    with pytest.raises(ValueError, match="Circular MPC dependency detected"):
        solve_linear(model, lc)


def test_drilling_stabilization():
    """Verify that physical drilling stabilization has exactly 6 zero energy modes on a free shell."""
    model = FEModel(name="drilling_stabilization_modes")
    model.add_material("steel", 210e9, 0.3)

    # 4-node flat shell element
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, 1.0, 0.0)
    model.add_node(4, 0.0, 1.0, 0.0)

    # Enable physical drilling stabilization parameter
    elem = create_shell_element(
        1,
        [1, 2, 3, 4],
        "steel",
        thickness=0.01,
        drilling_stabilization=1.0e-3,
    )
    model.add_element(1, elem)

    # Compute stiffness matrix
    K = elem.compute_stiffness_matrix(model.mesh, model.materials["steel"])

    # Compute eigenvalues of stiffness matrix to count zero energy modes
    eigenvals = np.linalg.eigvalsh(K)

    # Flat shell element has 24 DOFs. Free element has exactly 6 rigid body modes (near-zero eigenvalues).
    zero_modes = np.sum(eigenvals < 1.0e-5)
    assert zero_modes == 6, f"Expected 6 rigid body modes, got {zero_modes}. Eigenvalues: {eigenvals}"


def test_global_stress_recovery():
    """Verify that return_global=True correctly computes local and global stress components."""
    model = FEModel(name="stress_recovery")
    E = 210e9
    nu = 0.3
    model.add_material("steel", E, nu)

    # 4-node flat shell element rotated 45 degrees in XY plane
    c = np.cos(np.pi / 4.0)
    s = np.sin(np.pi / 4.0)
    # Node coordinates
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, c, s, 0.0)
    model.add_node(3, c - s, s + c, 0.0)  # (0, sqrt(2), 0)
    model.add_node(4, -s, c, 0.0)

    elem = create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.01)
    model.add_element(1, elem)

    # Create displacements vector corresponding to pure uniaxial tension sigma_xx = 100 MPa along global X
    sigma_0 = 100.0e6
    eps_xx = sigma_0 / E
    eps_yy = -nu * eps_xx

    # Total DOFs is model.mesh.num_nodes * 6 = 24
    u = np.zeros(24)
    for node_id in [1, 2, 3, 4]:
        node = model.mesh.get_node(node_id)
        x, y, z = node.x, node.y, node.z
        dofs = node.dofs
        u[dofs[0]] = eps_xx * x  # ux
        u[dofs[1]] = eps_yy * y  # uy

    from anyfesolver.assembly import compute_stresses

    # Compute stresses with return_global=True
    stresses = compute_stresses(model, u, return_global=True)

    assert 1 in stresses
    elem_stresses = stresses[1]

    # Check that all expected keys are present
    expected_keys = [
        "local_xx_top", "local_yy_top", "local_zz_top", "local_xy_top", "local_xz_top", "local_yz_top",
        "local_xx_bot", "local_yy_bot", "local_zz_bot", "local_xy_bot", "local_xz_bot", "local_yz_bot",
        "global_xx_top", "global_yy_top", "global_zz_top", "global_xy_top", "global_xz_top", "global_yz_top",
        "global_xx_bot", "global_yy_bot", "global_zz_bot", "global_xy_bot", "global_xz_bot", "global_yz_bot"
    ]
    for key in expected_keys:
        assert key in elem_stresses

    # Check values at Gauss integration points
    for idx in range(len(elem.gauss_points)):
        # Global stress along X should be close to sigma_0
        assert np.isclose(elem_stresses["global_xx_top"][idx], sigma_0, rtol=1e-5)
        assert np.isclose(elem_stresses["global_xx_bot"][idx], sigma_0, rtol=1e-5)

        # Other global stresses should be close to 0
        assert np.isclose(elem_stresses["global_yy_top"][idx], 0.0, atol=1e-3)
        assert np.isclose(elem_stresses["global_zz_top"][idx], 0.0, atol=1e-3)
        assert np.isclose(elem_stresses["global_xy_top"][idx], 0.0, atol=1e-3)
        assert np.isclose(elem_stresses["global_xz_top"][idx], 0.0, atol=1e-3)
        assert np.isclose(elem_stresses["global_yz_top"][idx], 0.0, atol=1e-3)

        # Local stresses should be rotated:
        # sigma'_xx = sigma_0/2, sigma'_yy = sigma_0/2, tau'_xy = -sigma_0/2
        assert np.isclose(elem_stresses["local_xx_top"][idx], sigma_0 / 2.0, rtol=1e-5)
        assert np.isclose(elem_stresses["local_yy_top"][idx], sigma_0 / 2.0, rtol=1e-5)
        assert np.isclose(elem_stresses["local_xy_top"][idx], -sigma_0 / 2.0, rtol=1e-5)
