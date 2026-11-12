"""Main-worktree public example routing; no scientific qualification claim."""
import json
from pathlib import Path
import runpy
import numpy as np


def test_public_example_effective_owner_and_imports(capsys):
    root = Path(__file__).resolve().parents[1]
    scope = runpy.run_path(str(root / "examples/native_contact_free_finite_velocity_dev.py"))
    main = scope["main"]
    actual = main.__globals__
    factory = actual["create_analysis"]
    owners = []

    def capture(*args, **kwargs):
        owner = factory(*args, **kwargs)
        owners.append(owner)
        return owner

    actual["create_analysis"] = capture
    try:
        main()
    finally:
        actual["create_analysis"] = factory
    output = json.loads(capsys.readouterr().out)
    assert len(owners) == 1
    owner = owners[0]
    assert owner._context.stage_residual_tolerance == 1e-11
    assert output["epoch"] == 2
    assert output["time_seconds"] == .002
    assert np.isfinite(output["total_energy_j"]) and output["total_energy_j"] > 0
    provenance = output["provenance"]
    assert provenance["status"] == "DEVELOPMENT"
    assert provenance["production_qualified"] is False
    assert provenance["restart_supported"] is True
    assert provenance["profile_id"] == "GE_BEAM3_NATIVE_CONTACT_FREE_FINITE_VELOCITY_DEV_V2"
    import anyfesolver.ge_beam3_native_dynamics as public
    import anyfesolver._ge_beam3_native_dynamic_analysis as wrapper
    import anyfesolver._ge_beam3_native_midpoint as midpoint
    import anyfesolver._ge_beam3_dynamic_elastic_state as elastic
    import anyfesolver._ge_beam3_centered_finite_inertia as inertia
    for module in (public, wrapper, midpoint, elastic, inertia):
        assert Path(module.__file__).resolve().parent == root / "src/anyfesolver"
