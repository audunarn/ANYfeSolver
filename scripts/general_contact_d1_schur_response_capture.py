"""Retain bounded equilibrated pressure-response matrices from one slide case."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anyfesolver.general_contact import linearized_pivot
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import main


def capture(prefix: Path) -> None:
    result_path = prefix.with_suffix(".json")
    response_path = prefix.with_suffix(".npz")
    if result_path.exists() or response_path.exists():
        raise FileExistsError("pressure-response capture output already exists")
    original = linearized_pivot.propose_supported_schur_step
    snapshots = []

    def observed(pressure, gap, supported, response, **kwargs):
        proposal = original(pressure, gap, supported, response, **kwargs)
        snapshots.append({
            "pressure": np.asarray(pressure).copy(),
            "gap": np.asarray(gap).copy(),
            "supported": np.asarray(supported).copy(),
            "response": np.asarray(response).copy(),
            "proposal": (np.asarray(proposal.multipliers).copy()
                         if proposal is not None else np.full_like(pressure, np.nan)),
        })
        return proposal

    linearized_pivot.propose_supported_schur_step = observed
    try:
        main(8, result_path, 100.0, True, 3)
    finally:
        linearized_pivot.propose_supported_schur_step = original
        np.savez(response_path, **{
            f"{name}_{index}": value
            for index, snapshot in enumerate(snapshots, 1)
            for name, value in snapshot.items()
        })
    print(f"captured {len(snapshots)} responses at {response_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path)
    capture(parser.parse_args().prefix)
