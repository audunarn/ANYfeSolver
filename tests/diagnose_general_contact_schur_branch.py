"""Capture one rejected small-case projection response branch without changing mechanics."""

from pathlib import Path
from dataclasses import replace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from anyfesolver import solve_static_nonlinear
from anyfesolver.general_contact import linearized_pivot
from test_general_contact_supported_schur_path import _case


if __name__ == "__main__":
    model, definition, settings, program = _case()
    if len(sys.argv) == 2:
        settings = replace(settings,
                           augmentation_schur_difference_step_m=float(sys.argv[1]))
    print(f"difference_step_m={settings.augmentation_schur_difference_step_m}")
    original = linearized_pivot._projected_clip_signature
    calls = []

    def record(geometry, source, target):
        signature = original(geometry, source, target)
        calls.append((source.copy(), target.copy(), signature))
        return signature

    linearized_pivot._projected_clip_signature = record
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    print(f"status={result.status} reason={result.failure_reason}")
    trace = result.info.get("contact_failure", {}).get("augmentation_trace", ())
    for index, entry in enumerate(trace):
        if "schur_proposed" in entry:
            print(f"augmentation={index+1} schur_proposed={entry['schur_proposed']} "
                  f"ineligible={entry.get('schur_ineligible')} "
                  f"scaled_condition={entry.get('schur_scaled_condition')}")
    print(f"signature_calls={len(calls)}")
    if calls:
        base_source, base_target, base_signature = calls[0]
        print(f"base_ambiguous={base_signature is None}")
        print(f"base_source_bbox={base_source.min(axis=0).tolist()} "
              f"to {base_source.max(axis=0).tolist()}")
        print(f"base_target_bbox={base_target.min(axis=0).tolist()} "
              f"to {base_target.max(axis=0).tolist()}")
        for index, (source, target, signature) in enumerate(calls[1:], 1):
            if signature != base_signature:
                source_shift = np.max(np.abs(source-base_source))
                target_shift = np.max(np.abs(target-base_target))
                changed = (None if signature is None or base_signature is None
                           else [i for i, (a, b) in enumerate(zip(
                               base_signature, signature,
                           )) if a != b])
                print(f"first_changed_call={index} "
                      f"ambiguous={signature is None} "
                      f"source_shift={source_shift:.12g} "
                      f"target_shift={target_shift:.12g} "
                      f"changed_signature_entries={changed}")
                break
        else:
            print("all_signatures_identical")
