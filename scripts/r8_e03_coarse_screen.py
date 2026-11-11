"""Deferred, single-phase level-0/+offset E03 B2 feasibility evidence.

No solver, subprocess, supervisor or automatic comparison is launched here.
The root-owned adapter must obtain authority, supervise execution and retain raw
outputs before calling finish_screen, including on failure or timeout.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile


CCX_PATH = Path("C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/calculix-223/CalculiX-2.23.0-win-x64/bin/ccx.exe")
CCX_SHA256 = "27a9426a1ee2f72c38fbe892e0ffaf38e14867628ffd0235770261b6f3ec9c5f"
WHEEL_SHA256 = "fa5507fe14cc09949743adb7d5ae340a2499d9b813d6b80a1b7d4ea414267e82"
MODULE_MANIFEST = ".qualification/b3-ge-e4-pl/r8-e03-curved-source-dev-20261007-v1/inputs-manifest-v1.json"
SOURCE_FILES = (
    "scripts/r8_e03_coarse_screen.py", "scripts/r8_e03_curved_reference.py",
    "scripts/r8_e03_native_reference.py", "scripts/r8_e02_double_output.py",
    "scripts/r8_e02_rigid_tip_reference.py",
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def _bindings(bindings, phase):
    """Check caller-declared identities; this does not grant execution authority."""
    if phase not in ("native", "external"):
        raise ValueError("exactly one native or external phase required")
    from scripts.r8_e03_curved_reference import _detached_receipt

    bindings = _detached_receipt(bindings)
    if bindings.get("phase") != phase:
        raise ValueError("explicit one-phase binding required")
    for key in ("case_id", "supervisor_id", "execution_scope"):
        if not isinstance(bindings.get(key), str) or not bindings[key].strip():
            raise ValueError("explicit "+key+" binding required")
    source_root = Path(bindings["source_root"]).resolve()
    if set(bindings["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("exact runner/builder/adapter/parser source bindings required")
    for name in SOURCE_FILES:
        if _sha((source_root/name).read_bytes()) != bindings["source_sha256"][name]:
            raise ValueError("source identity mismatch: "+name)
    # Imported support scripts must come from the declared source root.
    import scripts.r8_e03_curved_reference as reference
    loaded = [(Path(__file__), SOURCE_FILES[0]),
        (Path(reference.__file__), SOURCE_FILES[1]),
        (Path(sys.modules[reference.read_double_output.__module__].__file__), SOURCE_FILES[3]),
        (Path(sys.modules[reference.ReferenceParseError.__module__].__file__), SOURCE_FILES[4])]
    if any(path.resolve() != source_root/name for path, name in loaded):
        raise ValueError("runner/reference/parser loaded from a different source root")
    wheel = Path(bindings["wheel_path"]).resolve()
    if wheel.suffix != ".whl" or _sha(wheel.read_bytes()) != bindings["wheel_sha256"] or bindings["wheel_sha256"] != WHEEL_SHA256:
        raise ValueError("exact wheel identity required")
    manifest_raw = (source_root/MODULE_MANIFEST).read_bytes()
    if _sha(manifest_raw) != bindings["module_manifest_sha256"]:
        raise ValueError("existing module manifest identity required")
    manifest = json.loads(manifest_raw)
    if manifest["reused_wheel_sha256"] != WHEEL_SHA256 or len(manifest["source_modules"]) != 431:
        raise ValueError("existing 431-module wheel manifest required")
    if any(bindings["source_sha256"][name] != manifest["inputs"][name] for name in SOURCE_FILES[1:]):
        raise ValueError("existing verified builder/adapter/parser source identities required")
    provider = Path(bindings["installed_provider_root"]).resolve()
    # Reuse entry-v1.py's source/wheel/site payload parity, without importing a solver.
    source = {p.relative_to(source_root/"src").as_posix(): _sha(p.read_bytes())
        for p in (source_root/"src/anyfesolver").rglob("*.py")}
    installed = {p.relative_to(provider).as_posix(): _sha(p.read_bytes())
        for p in (provider/"anyfesolver").rglob("*.py")}
    with zipfile.ZipFile(wheel) as archive:
        payload = {name: _sha(archive.read(name)) for name in archive.namelist()
            if name.startswith("anyfesolver/") and name.endswith(".py")}
    if source != manifest["source_modules"] or source != payload or source != installed:
        raise ValueError("source/wheel/installed 431-module payload mismatch")
    command = bindings.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        raise ValueError("exact argv command binding required")
    if phase == "external":
        if Path(bindings["executable_path"]).resolve() != CCX_PATH.resolve():
            raise ValueError("pinned CCX executable path required")
        if _sha(CCX_PATH.read_bytes()) != CCX_SHA256:
            raise ValueError("pinned CCX binary hash mismatch")
        if command != [str(CCX_PATH), "-i", "e03"]:
            raise ValueError("exact CCX -i e03 command required; no -o")
    elif phase == "native":
        if not isinstance(bindings.get("native_capability_law"), str) or not bindings["native_capability_law"].strip():
            raise ValueError("explicit adopted native capability law binding required")
    else:
        raise ValueError("exactly one native or external phase required")
    return bindings


def prepare_screen(phase, evidence_dir, *, bindings):
    """Prepare one immutable case/deck; use a fresh directory per phase attempt."""
    from scripts.r8_e03_curved_reference import build_case, build_deck

    bindings = _bindings(bindings, phase)
    case = build_case(0, 1)
    if bindings["case_id"] != case["case_id"]:
        raise ValueError("explicit exact level-0/+offset case identity required")
    deck, metadata = build_deck(case)
    output = Path(evidence_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"e03.inp").write_bytes(deck.encode("ascii"))
    _json(output/"case.json", case)
    _json(output/"metadata.json", metadata)
    _json(output/"bindings.json", bindings)
    _json(output/"request.json", dict(phase=phase, status="PREPARED_NOT_EXECUTED",
        production_qualified=False))
    return output


def finish_screen(evidence_dir, *, receipt, raw):
    """Retain completed/failed raw output, then validate ONE prepared phase.

    raw is stdout/stderr plus native_result JSON bytes, or frd/sta bytes.
    receipt is the caller's actual process record, never synthesized here.
    A refusal writes failure.json and leaves every received raw byte in place.
    Missing raw files must be supplied as empty bytes with failure in the receipt.
    """
    from scripts.r8_e03_curved_reference import (
        B2, BALANCE_LIMIT, _detached_receipt, balance_measure, build_case, build_deck, digest, parse_result,
    )

    output = Path(evidence_dir).resolve()
    # Exclusive directory prevents retries or replacement of a failed attempt.
    retained = output/"received"
    retained.mkdir(exist_ok=False)
    try:
        (retained/"receipt.raw.txt").write_text(repr(receipt), encoding="utf-8")
        phase = json.loads((output/"request.json").read_text(encoding="utf-8"))["phase"]
        names = {"stdout", "stderr", "native_result"} if phase == "native" else {"stdout", "stderr", "frd", "sta"}
        for name, data in raw.items():
            if name not in {"stdout", "stderr", "native_result", "frd", "sta"} or type(data) is not bytes:
                raise ValueError("declared raw names with byte payloads required")
            (retained/(name+".raw")).write_bytes(data)
        # Retain even malformed JSON-compatible receipts before validating them.
        (retained/"receipt.raw.json").write_text(json.dumps(receipt, allow_nan=True), encoding="utf-8")
        if set(raw) != names:
            raise ValueError("exact one-phase raw inventory required")
        receipt = _detached_receipt(receipt)
        bindings = _bindings(json.loads((output/"bindings.json").read_text(encoding="utf-8")), phase)
        case, metadata = build_case(0, 1), build_deck(build_case(0, 1))[1]
        if json.loads((output/"case.json").read_text(encoding="utf-8")) != case:
            raise ValueError("prepared case changed")
        if digest(json.loads((output/"metadata.json").read_text(encoding="utf-8"))) != digest(metadata):
            raise ValueError("prepared metadata changed")
        if _sha((output/"e03.inp").read_bytes()) != metadata["deck_sha256"]:
            raise ValueError("prepared deck changed")
        for key in ("phase", "case_id", "supervisor_id", "execution_scope", "command", "wheel_sha256", "source_sha256", "module_manifest_sha256"):
            if receipt.get(key) != bindings[key]:
                raise ValueError("process receipt binding mismatch: "+key)
        if receipt.get("deck_sha256") != metadata["deck_sha256"]:
            raise ValueError("process deck identity mismatch")
        resource = Path(receipt["resource_receipt_path"])
        resource_raw = resource.read_bytes()
        (retained/"resource.raw").write_bytes(resource_raw)
        if _sha(resource_raw) != receipt["resource_receipt_sha256"]:
            raise ValueError("actual retained resource receipt identity required")
        for name, data in raw.items():
            if receipt.get(name+"_sha256") != _sha(data):
                raise ValueError("raw output identity mismatch: "+name)
        wall = receipt.get("process_wall_seconds")
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
            raise ValueError("precise actual process-wall duration required")
        if type(receipt.get("exit_code")) is not int or receipt["exit_code"] != 0 or receipt.get("timed_out") is not False:
            raise ValueError("supervised process failed or timed out")
        if phase == "external":
            if receipt.get("executable_path") != str(CCX_PATH) or receipt.get("executable_sha256") != CCX_SHA256:
                raise ValueError("actual pinned executable identity required")
            versions = re.findall(rb"\b(?:CalculiX\s+)?Version\s+([0-9]+\.[0-9]+)(?![.0-9])", raw["stdout"], re.I)
            if not versions or set(versions) != {b"2.23"} or receipt.get("version") != "2.23":
                raise ValueError("CCX version 2.23 must appear in actual execution stdout")
            result = parse_result(case, metadata, raw["frd"], raw["sta"], raw["stdout"], raw["stderr"], receipt)
        else:
            origin = Path(receipt["anyfesolver_origin"]).resolve()
            if not origin.is_relative_to(Path(bindings["installed_provider_root"]).resolve()):
                raise ValueError("native child must load declared installed provider")
            origin_name = origin.relative_to(Path(bindings["installed_provider_root"]).resolve()).as_posix()
            modules = json.loads((Path(bindings["source_root"])/MODULE_MANIFEST).read_bytes())["source_modules"]
            if origin_name != "anyfesolver/__init__.py" or receipt.get("anyfesolver_origin_sha256") != modules[origin_name] or _sha(origin.read_bytes()) != modules[origin_name]:
                raise ValueError("actual native child origin must match installed module manifest hash")
            result = _detached_receipt(json.loads(raw["native_result"]))
            for key, value in dict(status="COMPLETED_UNQUALIFIED_B2_CANDIDATE", profile=B2,
                    case_id=case["case_id"], level=0, offset_sign=1, production_qualified=False).items():
                if type(result.get(key)) is not type(value) or result.get(key) != value:
                    raise ValueError("native result contract mismatch: "+key)
            if type(result.get("external_work_j")) not in (int, float) or result["external_work_j"] <= 0:
                raise ValueError("positive finite native work required")
            for key, count in (("force_balance_n", 3), ("moment_balance_nm", 3),
                    ("root_reaction_wrench_world_origin", 6)):
                values = result.get(key)
                if not isinstance(values, list) or len(values) != count or any(type(v) not in (int, float) for v in values):
                    raise ValueError("complete finite native observable required: "+key)
            balance = balance_measure(case, result["force_balance_n"], result["moment_balance_nm"])
            if type(result.get("scaled_balance")) not in (int, float) or balance > BALANCE_LIMIT or result.get("scaled_balance") != balance:
                raise ValueError("native declared common-origin balance failed")
            tips = result.get("tip_poses")
            if not isinstance(tips, list) or len(tips) != 2 or any(type(t) is not dict for t in tips):
                raise ValueError("two declared native physical tips required")
            if any(type(t.get("rib_index")) is not int for t in tips) or sorted(t["rib_index"] for t in tips) != [0, 1]:
                raise ValueError("distinct native rib identities 0 and 1 required")
            for tip in tips:
                pose = tip.get("world_pose")
                if not isinstance(pose, list) or len(pose) != 6 or any(type(v) not in (int, float) for v in pose):
                    raise ValueError("complete finite native tip pose required")
            if result.get("native_capability_law") != bindings["native_capability_law"]:
                raise ValueError("native capability law mismatch")
            coordinates = dict(point_multiplier_coordinates="native ReferencePointLaw rows; not world spatial wrenches",
                point_gap_coordinates="native ReferencePointLaw rows; not world pose components")
            if any(result.get(key) != value for key, value in coordinates.items()):
                raise ValueError("declared native row coordinate conventions required")
            expected_points = {f"rib{rib}-point{i}" for rib in range(2) for i in range(case["nx"])}
            for key in ("point_multipliers_native_rows", "point_gaps_native_rows"):
                rows = result.get(key)
                if not isinstance(rows, list) or len(rows) != len(expected_points):
                    raise ValueError("complete native point rows required: "+key)
                if any(not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
                        or not isinstance(row[1], list) or len(row[1]) != 6
                        or any(type(v) not in (int, float) for v in row[1]) for row in rows):
                    raise ValueError("finite six-component native point rows required")
                if {row[0] for row in rows} != expected_points:
                    raise ValueError("exact distinct declared native point identities required")
        _json(retained/"receipt.json", receipt)
        _json(output/"observables.json", result)
        _json(output/"outcome.json", dict(status="PROVISIONAL_COARSE_PHASE_COMPLETED",
            phase=phase, production_qualified=False, accuracy_scored=False,
            full_e03="OPEN", ge_point_correspondence="OPEN", converged_accuracy="OPEN"))
        return result
    except Exception as error:
        _json(retained/"failure.json", dict(status="REFUSED_OR_FAILED", error_type=type(error).__name__, reason=str(error)))
        raise
