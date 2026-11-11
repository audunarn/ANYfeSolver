"""Follow-up F09 private canonical-SI local energy admission discriminator.

The preserved V1 diagnostic showed that applying an exact relative-energy
guard after separately rounding an N-mm basic stiffness can disagree with SI
for a near-coupled section, even though the exact matrices are congruent.
Question: does conversion of a *declared* N-mm section to canonical SI before
constructing the private operator restore the decision, while failing closed
if the conversion erases a nonzero stiffness?  Keep the same three source
sections and no expanded physical sweep.  This is not graph admission.
"""

from __future__ import annotations

from fractions import Fraction as F
from hashlib import sha256
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from anyfesolver._b2_physical_flexibility_linear import (
    LinearSection, RepresentabilityError,
)


ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = (ROOT / "docs" / "agent_plans" /
               "B3_GE_E4_PL_G0_F09_LOCAL_ENERGY_ADMISSION_DEV_V1.json")
V1_PATH = ROOT / "scripts" / "b2_f09_local_energy_admission_dev.py"
SOURCE = ROOT / "src" / "anyfesolver" / "_b2_physical_flexibility_linear.py"
OUTPUT = (ROOT / "docs" / "agent_plans" /
          "B3_GE_E4_PL_G0_F09_CANONICAL_ENERGY_ADMISSION_DEV_V1.json")
V1 = runpy.run_path(str(V1_PATH))
UNIT = F(1000)


def _canonical_si_from_n_mm(matrix):
    """Convert owned N-mm stiffness to SI by exact unit factors, then round.

    A public definition has canonical SI ownership; this private helper is a
    feasibility check of the boundary, not a new product import route.
    """
    factors = (F(1), F(1), F(1), UNIT, UNIT, UNIT)
    stored = np.asarray(matrix, dtype=float)
    if stored.shape != (6, 6) or not np.isfinite(stored).all():
        raise RepresentabilityError("nonfinite declared-unit stiffness")
    converted = np.empty((6, 6), dtype=float)
    for i in range(6):
        for j in range(6):
            exact = F(float(stored[i, j]))/(factors[i]*factors[j])
            try:
                value = float(exact)
            except OverflowError as exc:
                raise RepresentabilityError("canonical-SI conversion overflow") from exc
            if not math.isfinite(value) or (exact != 0 and value == 0):
                raise RepresentabilityError("canonical-SI conversion erased nonzero stiffness")
            converted[i, j] = value
    return LinearSection.generalized(converted)


def _pair(name, section, start):
    mm_section, mm_length = V1["_convert"](section, 2.)
    canonical = _canonical_si_from_n_mm(mm_section.stiffness)
    si = V1["_case"](name + "_source_SI", 2., section, start)
    reconverted = V1["_case"](name + "_N_mm_to_SI", mm_length/float(UNIT),
                              canonical, start)
    original = np.asarray(section.stiffness)
    restored = np.asarray(canonical.stiffness)
    scale = np.maximum(np.abs(original), np.finfo(float).tiny)
    return {
        "name": name, "source_SI": si, "declared_N_mm_to_canonical_SI": reconverted,
        "max_stored_section_component_relative_change": float(np.max(
            np.abs(restored-original)/scale)),
        "same_energy_guard_decision": (si.get("energy_guard_pass") ==
                                       reconverted.get("energy_guard_pass")),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"immutable diagnostic already exists: {OUTPUT}")
    start = time.monotonic()
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    predecessor_sources_current = all(
        sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
        for item in predecessor["source_graph"])
    record = {
        "status": "running",
        "claim": "private canonical-SI local feasibility; no F09/G0B acceptance",
        "question": "Does declared-unit conversion before private local energy rounding restore equivalent decisions without silently erasing nonzero stiffness?",
        "predecessor": {"path": str(PREDECESSOR),
                        "sha256": sha256(PREDECESSOR.read_bytes()).hexdigest(),
                        "status": predecessor["status"],
                        "source_hashes_current": predecessor_sources_current,
                        "near_coupled_raw_decisions": [
                            (case["name"], case["status"])
                            for case in predecessor["cases"]
                            if case["name"].startswith("near_coupled")]},
        "candidate_rule": ("convert declared physical inputs to canonical SI "
                           "with no nonzero stiffness erasure; verify exact SPD, "
                           "representable local output, and exact 1e-8 relative "
                           "six-basic-mode energy rounding guard in SI"),
        "source_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_graph": [
            {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}
            for path in (Path(__file__), V1_PATH, SOURCE)],
        "python": sys.version.split()[0], "numpy": np.__version__,
        "limits": {"wall_seconds": 120, "rss_bytes": 2*1024**3,
                   "numerical_threads": 1},
        "pairs": [], "negative": None,
    }
    try:
        if not predecessor_sources_current:
            raise RuntimeError("predecessor evidence source hashes no longer match")
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            record["threadpools"] = [
                {"internal_api": item.get("internal_api"),
                 "num_threads": item.get("num_threads")} for item in pools]
            if any(item.get("num_threads") != 1 for item in pools):
                raise RuntimeError("numerical thread limit is not effective")
            for name, width, depth in (
                    ("rectangle_12x3_mm", F(12, 1000), F(3, 1000)),
                    ("rectangle_192x0p1875_mm", F(192, 1000), F(3, 16000))):
                section, metadata = V1["_rectangle"](width, depth)
                pair = _pair(name, section, start)
                pair["section"] = metadata
                record["pairs"].append(pair)
            near = np.eye(6)
            near[0, 4] = near[4, 0] = 1. - 2.**-40
            record["pairs"].append(_pair(
                "near_coupled", LinearSection.generalized(near), start))
            negative = np.eye(6)
            negative[3, 3] = np.nextafter(0., 1.)
            try:
                _canonical_si_from_n_mm(negative)
                record["negative"] = {"status": "unexpected_admission"}
            except RepresentabilityError as exc:
                record["negative"] = {"status": "explicit_rejection",
                                      "type": type(exc).__name__,
                                      "message": str(exc)}
            V1["_budget"](start)
            record["status"] = "completed_development_diagnostic"
    except Exception as exc:
        record["status"] = "failed_development_diagnostic"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        record["elapsed_seconds"] = time.monotonic()-start
        record["peak_rss_bytes"] = (psutil.Process().memory_info().peak_wset
                                    if sys.platform == "win32" else
                                    psutil.Process().memory_info().rss)
        OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": record["status"],
                          "pairs": [(p["name"], p["source_SI"]["status"],
                                     p["declared_N_mm_to_canonical_SI"]["status"],
                                     p["same_energy_guard_decision"])
                                    for p in record["pairs"]],
                          "negative": record["negative"],
                          "elapsed_seconds": record["elapsed_seconds"],
                          "peak_rss_bytes": record["peak_rss_bytes"],
                          "output": str(OUTPUT)}))
    return 0 if record["status"] == "completed_development_diagnostic" else 1


if __name__ == "__main__":
    raise SystemExit(main())
