"""Development-only coupled elastic sphere / stiffened panel indentation cycle.

Run ``python -m examples.elastic_sphere_panel_indentation --output indentation.json``.
Path: open (existing 5 mm nominal), geometric first touch, first touch - 1 mm,
first touch, original open. Frictionless corner-to-bilinear-face contact with one
compressive force multiplier per physical sphere corner. Quasi-static; the sphere's
six rigid motions are guided by mass-weighted rows only for these increments (they
do not apply to a future drop). No dynamics, friction, plasticity, impact or restart.

Exit status 0 only for ``indentation_cycle_completed``; 3 for a typed refusal or a
partial/cancelled result (the JSON is still written with the actual state reached).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from anysolver.general_contact.corner_contact_indentation import (
    STATUS_COMPLETED, run_cycle,
)
from examples.elastic_sphere_stiffened_panel import _runtime_identity

SCHEMA = "anysolver.elastic-sphere-panel-indentation-v1"


def _plain(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report, _ = run_cycle()
    report.update(schema=SCHEMA, runtime=_runtime_identity(), is_checkpoint=False,
                  numerical_acceptance="pending until the single cold-installed run")
    args.output.write_text(json.dumps(report, indent=2, default=_plain, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps(dict(status=report["status"], output=str(args.output),
                          impact="unsupported", restart="unsupported")))
    return 0 if report["status"] == STATUS_COMPLETED else 3


if __name__ == "__main__":
    raise SystemExit(main())
