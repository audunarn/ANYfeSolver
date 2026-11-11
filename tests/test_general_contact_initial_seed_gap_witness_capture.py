"""Rigorous gap-witness capture at the actual initial-seed refusal.

Attempt-08 of the coupled scope recorded the production seed decision on
this fixture: the origin-linearized KKT QP is optimal and cheap (scaled
merit 0.5 -> 0.00601028557678257) yet the seed is refused with
``seed_not_swept_admissible`` (admissible fraction 0.25). That rules out
proposal algebra and QP cost as the immediate obstruction, but it does NOT
distinguish a physical penetration inside the represented overlap from
conservative certificate uncertainty. This module resolves exactly that
one discriminator with the unchanged production witness machinery: the
real admission capture runs once, ``s3_patch_static.
overlap_violating_point_witness`` is intercepted with a disposable
diagnostics dict, and every rigorous found-point witness is retained
together with the source/target physical endpoint arrays, the error
boxes, the gap tolerance and the program/settings/source binding. The
solve stops at the existing private ``_INITIAL_SEED_DECISION_HOOK``
before any Newton iteration; no full solve, refinement or sweep runs.

Endpoint and path witnesses are recorded distinctly: an endpoint witness
encloses one static pose (the production endpoint call site passes the
same array object as start and end); a path witness encloses the actual
swept path between two distinct poses. A rigorous found witness whose
``physical_gap_upper_m`` lies below -gap_tolerance (50um on this
fixture) proves a real violating point at that enclosed parametric
location, so the refusal is physical there; NO found witness is NOT a
safety proof — the refusal then remains conservative certificate
uncertainty. The helper's diagnostics expose the enclosed physical gap
and the foot barycentric enclosure; no normal quantity is invented
beyond what the helper provides, and stable represented faces are
reported only through their physical endpoint coordinates because the
witness call sites carry no persistent face IDs. All captured data is
diagnostic evidence only, never an accepted physical prefix; no
acceptance gate beyond the unchanged production ones is evaluated.
"""
import json
import os
import tempfile

import numpy as np
import pytest

from anyfesolver.control import SolveCancelled
from anyfesolver.general_contact import s3_patch_static, semismooth_path
from anyfesolver.general_contact.core import _contact_settings_contract

from test_general_contact_admission_globalization_driver_small_patch import (
    _case,
    _enabled,
    _solve,
)

CAPTURE_NAME = "initial-seed-gap-witness-capture-v1.json"
_GAP_TOLERANCE_M = 5.0e-5


def _seeded(settings):
    from dataclasses import replace
    return replace(_enabled(settings),
                   initial_contact_seed="linearized_kkt_dev")


def _source_manifest():
    runtime = os.environ.get("AUTOMATIC_RUNTIME_RESULT")
    if not runtime:
        return None
    install = os.path.join(os.path.dirname(runtime), "install.json")
    if not os.path.isfile(install):
        return None
    with open(install, encoding="utf-8") as handle:
        record = json.load(handle)
    return {"payload_sha256": record.get("payload_sha256"),
            "mode": record.get("mode")}


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    return repr(value)


def test_real_seed_gap_witnesses_are_captured_at_the_refusal(monkeypatch):
    model, definition, settings, program = _case()
    seeded = _seeded(settings)
    captured = {}
    witnesses = []
    original_witness = s3_patch_static.overlap_violating_point_witness

    def witness_wrapper(*args, **kwargs):
        diagnostics = {}
        kwargs["diagnostics"] = diagnostics
        found, checked = original_witness(*args, **kwargs)
        gap_tolerance = float(kwargs["gap_tolerance"])
        gap_upper = diagnostics.get("physical_gap_upper_m")
        entry = {
            "kind": ("endpoint" if (args[0] is args[1] and args[2] is args[3])
                     else "path"),
            "gap_tolerance_m": gap_tolerance,
            "found": bool(found),
            "samples_checked": int(diagnostics.get("samples_checked", 0)),
            "diagnostics": {key: _plain(value)
                            for key, value in diagnostics.items()},
            "source_start_m": np.asarray(args[0], dtype=float).tolist(),
            "source_end_m": np.asarray(args[1], dtype=float).tolist(),
            "target_start_m": np.asarray(args[2], dtype=float).tolist(),
            "target_end_m": np.asarray(args[3], dtype=float).tolist(),
            "endpoint_errors": _plain(kwargs.get("endpoint_errors")),
            "physical_violation_proven": bool(
                found and gap_upper is not None
                and gap_upper < -gap_tolerance),
        }
        witnesses.append(entry)
        return found, checked

    def hook(payload):
        captured["payload"] = payload
        raise SolveCancelled("initial-seed gap witnesses captured",
                             stage="contact_path.step:1.semismooth."
                                   "initial_seed.decision")

    monkeypatch.setattr(s3_patch_static, "overlap_violating_point_witness",
                        witness_wrapper)
    assert semismooth_path._INITIAL_SEED_DECISION_HOOK is None
    monkeypatch.setattr(semismooth_path, "_INITIAL_SEED_DECISION_HOOK", hook)
    with pytest.raises(SolveCancelled,
                        match="initial-seed gap witnesses captured"):
        _solve(model, definition, seeded, program, max_load_factor=0.5)

    payload = captured["payload"]
    decision = payload["record"]
    assert decision["selection"] == "linearized_kkt_dev"
    assert decision["outcome"] in {"admitted", "refused"}
    for entry in witnesses:
        assert entry["source_start_m"] and entry["target_start_m"]
        source = np.asarray(entry["source_start_m"], dtype=float)
        target = np.asarray(entry["target_start_m"], dtype=float)
        assert source.shape == (4, 3) and target.shape == (3, 3)
        assert np.all(np.isfinite(source)) and np.all(np.isfinite(target))
        if entry["found"]:
            gap_upper = entry["diagnostics"]["physical_gap_upper_m"]
            assert gap_upper is not None
            assert gap_upper < -entry["gap_tolerance_m"]
            lower = entry["diagnostics"]["foot_barycentric_lower"]
            upper = entry["diagnostics"]["foot_barycentric_upper"]
            assert len(lower) == 2 and len(upper) == 2

    endpoint = [entry for entry in witnesses if entry["kind"] == "endpoint"]
    path = [entry for entry in witnesses if entry["kind"] == "path"]
    proven = [entry for entry in witnesses if entry["physical_violation_proven"]]
    if proven:
        classification = ("physical_penetration_proven_at_enclosed_"
                          "witness_point")
    elif witnesses:
        classification = ("no_rigorous_witness_found_refusal_remains_"
                          "certificate_uncertainty")
    else:
        classification = ("no_witness_invocation_observed_refusal_remains_"
                          "certificate_uncertainty")
    record = {"schema": "INITIAL_SEED_GAP_WITNESS_CAPTURE_V1",
              "selection": "linearized_kkt_dev",
              "source_manifest": _source_manifest(),
              "settings_contract": _contact_settings_contract(seeded),
              "program": {"steps_per_segment": program.steps_per_segment,
                          "controls": len(program.controls),
                          "nodal_forces": len(program.nodal_forces)},
              "fixture_gap_tolerance_m": _GAP_TOLERANCE_M,
              "outcome": decision["outcome"],
              "reason": decision.get("reason"),
              "origin_merit": (float(decision["origin_merit"])
                               if "origin_merit" in decision else None),
              "refusal_classification": classification,
              "witness_calls": len(witnesses),
              "endpoint_witness_calls": len(endpoint),
              "path_witness_calls": len(path),
              "witnesses_found": len(proven),
              "endpoint_witnesses": endpoint,
              "path_witnesses": path}
    if "admissible_fraction" in decision:
        record["admissible_fraction"] = float(decision["admissible_fraction"])
    for key in ("seed_merit", "qp_iterations", "qp_terminal", "cost_s",
                "free_gap_min_m"):
        if key in decision:
            record[f"proposal_{key}"] = decision[key]
    with open(os.path.join((os.environ.get("TEMP") or tempfile.gettempdir()), CAPTURE_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
