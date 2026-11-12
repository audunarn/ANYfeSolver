"""One all-channel native BL5 P1 screen against the saved matched P0 cases.

This isolates the multiplier space on one unchanged native body. It cannot
select P1 or replace physical/external and seven-ratio F03 assessment.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import anyfesolver
import numba
import numpy as np
import psutil
from threadpoolctl import threadpool_info

from anyfesolver import stiffened_shell as ss
from scripts import r8_bl5_matched_native_reference as native
from scripts import r8_bl5_saved_local_transfer_dev as transfer


ROOT = Path(__file__).resolve().parents[1]
PRIOR_SHA256 = "84a0280817174df9946f455dd6939842c21d6648bd23796d149906300914fe1b"


def evaluate(prior_path: Path) -> dict:
    if sha256(prior_path.read_bytes()).hexdigest() != PRIOR_SHA256:
        raise ValueError("exact saved P0 local-transfer result required")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (prior.get("status") != "DEVELOPMENT_LOCAL_TRANSFER_NOT_QUALIFIED" or
            set(prior.get("cases", {})) != {"fy", "mx"}):
        raise ValueError("complete saved P0 Fy/Mx cases required")
    pools = threadpool_info()
    if (Path(anyfesolver.__file__).resolve().parent != ROOT / "src" / "anyfesolver" or
            psutil.Process().cpu_affinity() != [0] or numba.config.DISABLE_JIT != 1 or
            numba.get_num_threads() != 1 or not pools or
            any(item["num_threads"] != 1 for item in pools)):
        raise ValueError("effective source/CPU/thread/JIT contract changed")
    p0 = native.definition()
    # The native-fibre BL5 owner admits the coarse-trace P1 pairing; on this
    # matched four-edge fixture it has the same five-channel 25-row space.
    p1_line = replace(p0.lines[0], basis="continuous_p1_coarse")
    p1 = replace(p0, lines=(p1_line,))
    baseline = ss.create_analysis(definition=p0)
    candidate = ss.create_analysis(definition=p1)
    try:
        delta = (candidate._K - baseline._K).tocsr()
        delta.eliminate_zeros()
        if (delta.nnz or candidate._shell_count != baseline._shell_count or
                candidate._beam_ids != baseline._beam_ids or
                not np.array_equal(candidate._fixed, baseline._fixed) or
                baseline._C.shape[0] != 20 or candidate._C.shape[0] != 25):
            raise AssertionError("P1 candidate changed physical body or supports")
        rows = {}
        for case in ("fy", "mx"):
            updated = transfer._native(candidate, case, expected_rows=25)
            original = prior["cases"][case]["native"]
            if max(abs(a["x_m"] - b["x_m"]) for a, b in
                   zip(updated["stations"], original["stations"], strict=True)) > 1e-14:
                raise AssertionError("P0/P1 physical sampling changed")
            old = original["summary"]
            new = updated["summary"]
            rows[case] = {"p0": old, "p1": updated,
                          "translation_rms_ratio_p1_over_p0": new["translation_rms_m"] / old["translation_rms_m"],
                          "director_rms_ratio_p1_over_p0": new["director_rms"] / old["director_rms"],
                          "absolute_local_work_ratio_p1_over_p0": new["absolute_local_work_J"] / old["absolute_local_work_J"],
                          "applied_work_ratio_p1_over_p0": new["applied_work_J"] / old["applied_work_J"]}
    finally:
        candidate.close()
        baseline.close()
    return {"status": "DEVELOPMENT_P1_SCREEN_NOT_QUALIFIED",
            "identity": "R8_BL5_NATIVE_P1_COARSE_CONTRAST_DEV_V2",
            "prior_sha256": PRIOR_SHA256,
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "body_operator_delta_nnz": 0,
            "rows": rows,
            "effective_runtime": {"cpu_affinity": psutil.Process().cpu_affinity(),
                                  "numba_disable_jit": int(numba.config.DISABLE_JIT),
                                  "numba_threads": int(numba.get_num_threads()),
                                  "threadpools": pools,
                                  "anyfesolver_import": str(Path(anyfesolver.__file__).resolve())},
            "limitations": ["One matched 1:1 coarse native body and two loads only.",
                            "A smaller gap under more constraints is not a stability or physical-fidelity proof.",
                            "External P1 matching and all required ratios/refinements remain open."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing development result")
    result = evaluate(args.prior)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({case: {name: row[name] for name in (
        "translation_rms_ratio_p1_over_p0", "director_rms_ratio_p1_over_p0",
        "absolute_local_work_ratio_p1_over_p0", "applied_work_ratio_p1_over_p0")}
        for case, row in result["rows"].items()}))


if __name__ == "__main__":
    main()
