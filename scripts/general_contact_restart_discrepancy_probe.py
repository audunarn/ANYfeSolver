"""Development diagnosis of the existing augmented force-control restart case."""

import json
from pathlib import Path
import sys
import time

import numpy as np

from anyfesolver import solve_static_nonlinear


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_general_contact_force_augmented import _loaded_patch  # noqa: E402


def _summary(result):
    history = result.info.get("force_displacement_history", ())
    contact = result.info.get("general_contact", {})
    return {
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "load_factor": float(result.load_factor),
        "displacements": np.asarray(result.displacements, dtype=float).tolist(),
        "augmentation_history": [{
            key: row.get(key) for key in (
                "load_factor", "augmentation_iterations",
                "normal_multiplier_change", "contact",
            )
        } for row in history],
        "pressure_pa": contact.get("normal_multiplier_estimate_pa"),
        "active_pressure_dofs": contact.get("active_pressure_dofs"),
        "minimum_gap_m": contact.get("minimum_gap"),
        "contact_force_n": contact.get("integration_force"),
        "has_checkpoint": result.restart_checkpoint is not None,
    }


def main():
    output = Path(sys.argv[1])
    matched_first_step = "--matched-first-step" in sys.argv[2:]
    model, definition, settings, load = _loaded_patch()
    common = dict(
        contact=definition, contact_settings=settings, load_case=load,
        control="force", kinematics="von_karman", max_iterations=8,
        num_steps=2, min_step_fraction=0.1, emit_restart_checkpoint=True,
    )
    start = time.perf_counter()
    full = solve_static_nonlinear(model, **common)
    half = solve_static_nonlinear(
        model, max_load_factor=0.5,
        **({**common, "num_steps": 1} if matched_first_step else common),
    )
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    delta = np.asarray(resumed.displacements) - np.asarray(full.displacements)
    record = {
        "source_case": "tests/test_general_contact_force_augmented.py:_loaded_patch",
        "matched_first_step": matched_first_step,
        "elapsed_seconds": time.perf_counter() - start,
        "settings": {
            "pressure_tolerance_pa": settings.pressure_tolerance,
            "maximum_augmentations": settings.maximum_augmentations,
            "normal_stiffness": definition.interactions[0].normal_stiffness,
        },
        "full": _summary(full),
        "half": _summary(half),
        "resumed": _summary(resumed),
        "maximum_absolute_displacement_difference_m": float(np.max(np.abs(delta))),
    }
    output.write_text(json.dumps(record, indent=2, default=str) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "elapsed_seconds": record["elapsed_seconds"],
        "statuses": [full.status, half.status, resumed.status],
        "maximum_absolute_displacement_difference_m": (
            record["maximum_absolute_displacement_difference_m"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
