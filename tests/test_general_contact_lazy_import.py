"""Keep opt-in public contact imports available without loading contact on ordinary solves."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_no_contact_static_route_leaves_contact_kernel_unloaded():
    source = Path(os.environ.get("ANYSOLVER_TEST_PACKAGE_ROOT",
                                 Path(__file__).resolve().parents[1] / "src"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source), environment.get("PYTHONPATH", "")))
    environment["ANYSOLVER_TEST_SOURCE"] = str(source)
    code = textwrap.dedent("""
        import os
        from pathlib import Path
        import sys

        import anyfesolver
        source = Path(os.environ["ANYSOLVER_TEST_SOURCE"]).resolve()
        assert Path(anyfesolver.__file__).resolve().is_relative_to(source)
        assert "anyfesolver.general_contact" not in sys.modules

        from anyfesolver.boundary import LoadCase
        from anyfesolver.mesh_gen import generate_beam_mesh
        model = generate_beam_mesh(1.0, num_divisions=1)
        load = LoadCase("axial")
        load.add_nodal_load(2, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = anyfesolver.solve_static_nonlinear(
            model, load, num_steps=1, max_iterations=5,
        )
        assert result.status == "completed", result.failure_reason
        assert "anyfesolver.general_contact.core" not in sys.modules

        import anyfesolver.general_contact as package
        from anyfesolver.general_contact import core, runtime
        for name in (
            "ContactDefinition", "ContactInteraction", "ContactPathControl",
            "ContactPathNodalForce", "ContactPathProgram", "ContactResults",
            "ContactSolverSettings", "ContactSurface", "ContactValidationReport",
            "RigidContactBody", "solve_static_contact_path",
            "ContactNodalLoadHistory", "ContactTransientSolverControls",
            "TransientContactResults", "TransientContactSettings",
            "solve_transient_contact", "validate_general_contact",
        ):
            assert getattr(anyfesolver, name) is getattr(package, name), name
        assert core._ACTIVE_CONTACT is runtime._ACTIVE_CONTACT
        assert core.active_contact_manager is runtime.active_contact_manager
        assert core.suspend_contact is runtime.suspend_contact
    """)
    completed = subprocess.run(
        [sys.executable, "-c", code], env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
