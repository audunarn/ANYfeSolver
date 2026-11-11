"""Disposable full-load check of the best rejected secant-grid fraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anyfesolver.general_contact import static_path
from anyfesolver.general_contact.supported_schur import SupportedSchurProposal
from scripts.general_contact_d1_free_target_press_probe import main as press_main


def main(output: Path) -> None:
    original = static_path.propose_supported_schur_secant_backtrack
    forced = False

    def probe(full: SupportedSchurProposal, actual_full_gap,
              *, diagnostics=None):
        nonlocal forced
        proposal = original(full, actual_full_gap, diagnostics=diagnostics)
        if proposal is not None or forced or diagnostics is None:
            return proposal
        fraction = diagnostics.get(
            "schur_secant_backtrack_minimum_scanned_fraction")
        residual = diagnostics.get(
            "schur_secant_backtrack_minimum_scanned_residual_pa")
        if fraction is None or residual is None:
            return None
        data = full.backtrack_data
        if data is None:
            return None
        modes = data.supported_indices
        direction = full.multipliers-data.base_multipliers
        candidate = data.base_multipliers+fraction*direction
        full_gap = np.asarray(actual_full_gap, dtype=float)[modes]
        linear_gap_change = data.equilibrated_gap_response @ direction[modes]
        predicted_gap = (data.base_gap+fraction*linear_gap_change
                         +fraction*fraction*(full_gap-data.base_gap
                                             -linear_gap_change))
        argument = candidate[modes]-data.normal_stiffness*predicted_gap
        active = tuple(int(modes[index]) for index in
                       np.flatnonzero(argument > 0.0))
        diagnostics["schur_secant_forced_minimum_dev"] = True
        forced = True
        return SupportedSchurProposal(
            candidate, active, full.scaled_condition,
            fraction*full.step_fraction,
            full.predicted_residual_before, float(residual),
        )

    static_path.propose_supported_schur_secant_backtrack = probe
    try:
        press_main(output, columns=24, equilibrium_tolerance=1.0e-8,
                   nonlinear_backtrack=True)
    finally:
        static_path.propose_supported_schur_secant_backtrack = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
