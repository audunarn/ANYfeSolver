"""Development-only capture of the first unresolved moving-patch pose."""

import json
import sys
from pathlib import Path

import numpy as np

from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
from scripts.general_contact_d1_coupled_refinement import main


capture = Path(sys.argv[1])
sys.argv = [sys.argv[0], *sys.argv[2:]]
original = MovingS3PatchProjectedNormal.work


def observed(self, current_source, current_target, multipliers, *, tangent=True):
    try:
        return original(self, current_source, current_target, multipliers,
                        tangent=tangent)
    except (ValueError, MemoryError) as exc:
        if not capture.exists():
            capture.write_text(json.dumps({
                "error": str(exc),
                "source": np.asarray(current_source).tolist(),
                "target": np.asarray(current_target).tolist(),
                "multipliers": np.asarray(multipliers).tolist(),
                "tangent": tangent,
                "shared_edge_classifier": self._only_artificial_shared_edge_ties(
                    current_source, current_target),
            }, indent=2) + "\n", encoding="utf-8")
        raise


MovingS3PatchProjectedNormal.work = observed
main()
