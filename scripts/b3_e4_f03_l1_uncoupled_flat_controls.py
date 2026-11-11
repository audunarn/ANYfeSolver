"""Bounded E4 plate-only and native B3 flat-web-only work controls.

Extract the already assembled diagonal domain blocks from the matched native
development fixture; no L1 constraint participates. This separates element
idealization from the coupled comparison without changing geometry or section.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.sparse import linalg as spla
from threadpoolctl import threadpool_info

from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_l1_assembly_experimental import SPACE_B
from scripts.b3_e4_f03_l1_matched_flat_web import build
from scripts.b3_e4_f03_ccx_matched_t_reference import PLATE_T_M


def evaluate() -> dict:
    # This admitted assembly supplies only K and numbering. Its line rows
    # are not included in either independent domain solve below.
    model, plate_tip, beam_tip, web = build(SPACE_B)
    owner = stiffened_shell.create_analysis(definition=model)
    try:
        k = owner._K
        shell_count = owner._shell_count
        count = k.shape[0]
        fixed = set(int(v) for v in owner._fixed)
        plate_slot = model.shell.node_ids.index(plate_tip)
        beam_slot = owner._beam_index[beam_tip]
        cases = {}
        for name, begin, end, terms in (
            ("plate_only", 0, shell_count,
             ((6*plate_slot+1, 100.),
              (6*plate_slot+3, -100.*PLATE_T_M/2))),
            ("web_only", shell_count, count,
             ((shell_count+6*beam_slot+3, 10.),)),
        ):
            free = np.array([i for i in range(begin, end)
                             if i not in fixed], dtype=int)
            if len(free) == 0 or len(free) > 100_000:
                raise ValueError("bounded supported domain block required")
            load = np.zeros(count)
            for slot, amount in terms:
                load[slot] += amount
            factor = spla.splu(k[np.ix_(free, free)].tocsc())
            q = np.zeros(count)
            q[free] = factor.solve(load[free])
            residual = k@q-load
            relative = float(np.linalg.norm(residual[free]) /
                             max(1., np.linalg.norm(load[free])))
            if not np.isfinite(q).all() or relative > 1e-8:
                raise RuntimeError("uncoupled native equilibrium failed")
            cases[name] = {
                "load_displacement_work_j": float(load@q),
                "free_equilibrium_ratio": relative,
                "matrix_rows": int(len(free)),
                "factorizations": 1,
            }
        return {
            "classification": "uncoupled_flat_web_element_control_not_F03_assessment",
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "section": web,
            "cases": cases,
            "effective_threadpools": [
                {"internal_api": value.get("internal_api"),
                 "num_threads": value.get("num_threads")}
                for value in threadpool_info()],
            "scope": "same E4 and B3 local operators and fixed roots, L1 disabled",
        }
    finally:
        owner.close()


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, sort_keys=True))
