"""Focused checks for the additive public S3SlidingFrictionSettings export."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_public_friction_settings_export_identity_and_listing():
    import anyfesolver.general_contact as package
    from anyfesolver.general_contact import S3SlidingFrictionSettings
    from anyfesolver.general_contact import s3_sliding_friction

    assert "S3SlidingFrictionSettings" in package.__all__
    assert "S3SlidingFrictionSettings" in dir(package)
    assert S3SlidingFrictionSettings is s3_sliding_friction.S3SlidingFrictionSettings
    assert package.S3SlidingFrictionSettings is s3_sliding_friction.S3SlidingFrictionSettings


def test_keyword_construction_matches_positional_and_validates():
    from anyfesolver.general_contact import S3SlidingFrictionSettings

    keyword = S3SlidingFrictionSettings(friction_coefficient=.3,
        tangential_stiffness_Pa_per_m=1000., steps_by_dof=(2e-7,)*60,
        maximum_candidate_evaluations=145)
    positional = S3SlidingFrictionSettings(.3, 1000., (2e-7,)*60, 145)
    assert keyword == positional
    assert keyword.steps_by_dof == (2e-7,)*60
    assert keyword.maximum_candidate_evaluations == 145
    with pytest.raises(ValueError):
        S3SlidingFrictionSettings(.3, 1000., (), 145)
    with pytest.raises(ValueError):
        S3SlidingFrictionSettings(.3, 1000., (2e-7,)*60, 0)


def test_friction_example_settings_match_measured_candidate_values():
    from examples.general_contact_s3_friction_physical_path import build_case

    model, _definition, _settings, friction, _program = build_case()
    total_dofs = model.mesh.dof_manager.total_dofs
    assert total_dofs == 60
    assert friction.friction_coefficient == .3
    assert friction.tangential_stiffness_Pa_per_m == 1000.
    assert friction.steps_by_dof == (2e-7,)*total_dofs
    assert len(friction.steps_by_dof) == total_dofs
    assert friction.maximum_candidate_evaluations == 145


def test_public_friction_settings_export_stays_lazy():
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

        from anyfesolver.general_contact import S3SlidingFrictionSettings
        from anyfesolver.general_contact import s3_sliding_friction
        assert S3SlidingFrictionSettings is s3_sliding_friction.S3SlidingFrictionSettings
        settings = S3SlidingFrictionSettings(friction_coefficient=.3,
            tangential_stiffness_Pa_per_m=1000., steps_by_dof=(2e-7,)*60,
            maximum_candidate_evaluations=145)
        assert settings.steps_by_dof == (2e-7,)*60
        assert settings.maximum_candidate_evaluations == 145
    """)
    completed = subprocess.run(
        [sys.executable, "-c", code], env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
