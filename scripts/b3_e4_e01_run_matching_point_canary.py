"""Execute one pinned B31/S4 matching-point feasibility screen.

This is a bounded development canary. Its result cannot qualify E01 until
section, connection, reaction and refinement contracts are independently met.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import psutil

from _b3_e4_ccx_3d_canary_parser import _HEADER, _NODE, _STEP


EXE = Path("C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/"
           "calculix-223/CalculiX-2.23.0-win-x64/bin/ccx.exe")
EXE_SHA256 = "27a9426a1ee2f72c38fbe892e0ffaf38e14867628ffd0235770261b6f3ec9c5f"
MAX_SECONDS = 120.
MAX_TREE_BYTES = 8*1024**3
EXPECTED_NODES = frozenset((*range(1, 7), *range(101, 106)))


def digest(path: Path):
    return sha256(path.read_bytes()).hexdigest()


def parse_completed(stdout: str, status: str, dat: str):
    if stdout.count("Job finished") != 1:
        raise ValueError("one completed CalculiX job required")
    if re.search(r"(?i)\b(error|fatal|divergence|not converged)\b", stdout):
        raise ValueError("CalculiX reported a numerical error")
    rows = [line for line in status.splitlines()
            if re.match(r"^\s*\d+\s+\d+\b", line)]
    matches = [_STEP.match(line) for line in rows]
    if len(matches) != 1 or matches[0] is None or any(
            abs(float(value)-1.) > 1e-8 for value in matches[0].groups()):
        raise ValueError("one full accepted linear increment required")
    lines = dat.splitlines()
    headers = [(i, match) for i, line in enumerate(lines)
               if (match := _HEADER.match(line))]
    if len(headers) != 1 or abs(float(headers[0][1].group(1))-1.) > 1e-8:
        raise ValueError("one full-step DAT displacement block required")
    displacements = {}
    rotations = {}
    for line in lines[headers[0][0]+1:]:
        if not line.strip():
            continue
        if re.match(r"^\s*(forces|stresses|displacements)\s*\(", line, re.I):
            break
        match = _NODE.match(line)
        if match is None:
            raise ValueError("malformed CalculiX displacement row")
        node = int(match.group(1))
        values = tuple(float(value) for value in match.group(2).split())
        if node in displacements or node not in EXPECTED_NODES or len(values) not in (3, 6):
            raise ValueError("unexpected CalculiX displacement inventory")
        displacements[node] = values[:3]
        if len(values) == 6:
            rotations[node] = values[3:]
    if set(displacements) != EXPECTED_NODES:
        raise ValueError("incomplete CalculiX displacement inventory")
    if not all(abs(value) < float("inf") for row in displacements.values()
               for value in row):
        raise ValueError("finite nodal response required")
    tip = displacements[105]
    if not any(abs(value) > 1e-15 for value in tip):
        raise ValueError("zero tip response in loaded canary")
    return {"printed_nodes": len(displacements), "accepted_steps": 1,
            "tip_translation_m": tip, "tip_rotation_rad": rotations.get(105),
            "output_precision": "CalculiX DAT, about seven digits"}


def _tree_bytes(process: psutil.Process):
    total = 0
    try:
        children = process.children(recursive=True)
    except psutil.NoSuchProcess:
        children = ()
    for item in (process, *children):
        try:
            total += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _kill_tree(process: psutil.Process):
    try:
        children = process.children(recursive=True)
    except psutil.NoSuchProcess:
        children = ()
    for item in reversed(children):
        try:
            item.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        process.kill()
    except psutil.NoSuchProcess:
        pass


def run(deck: Path, output: Path, expected_sha256: str):
    if digest(EXE) != EXE_SHA256:
        raise ValueError("pinned CalculiX binary changed")
    if digest(deck) != expected_sha256:
        raise ValueError("input deck differs from inert evidence")
    if output.exists():
        raise FileExistsError("preserve prior CalculiX evidence")
    output.mkdir(parents=True)
    name = deck.stem
    shutil.copyfile(deck, output/f"{name}.inp")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    started = time.perf_counter()
    maximum = 0
    reason = None
    with (output/"stdout.log").open("wb") as stdout, (
            output/"stderr.log").open("wb") as stderr:
        child = subprocess.Popen((str(EXE), "-i", name), cwd=output,
                                 env=env, stdout=stdout, stderr=stderr)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            maximum = max(maximum, _tree_bytes(process))
            elapsed = time.perf_counter()-started
            if maximum > MAX_TREE_BYTES:
                reason = "PROCESS_TREE_MEMORY_LIMIT"
            elif elapsed > MAX_SECONDS:
                reason = "TIMEOUT_120_SECONDS"
            if reason is not None:
                _kill_tree(process)
                break
            time.sleep(.05)
        code = child.wait()
    elapsed = time.perf_counter()-started
    parsed = None
    stdout_text = (output/"stdout.log").read_text(errors="replace")
    stderr_text = (output/"stderr.log").read_text(errors="replace")
    warnings = [line[:250] for line in (stdout_text+"\n"+stderr_text).splitlines()
                if re.search(r"(?i)\bwarning\b", line)][:20]
    if reason is None and code != 0:
        reason = f"CALCULIX_EXIT_{code}"
    if reason is None:
        try:
            parsed = parse_completed(
                stdout_text,
                (output/f"{name}.sta").read_text(errors="replace"),
                (output/f"{name}.dat").read_text(errors="replace"))
        except (OSError, ValueError) as exc:
            reason = f"PARSE_{type(exc).__name__}: {exc}"
    if reason is None and warnings:
        reason = "CALCULIX_WARNING_REVIEW"
    record = {"classification": "E01 development canary; no comparison acceptance",
              "input": str(deck.resolve()), "input_sha256": expected_sha256,
              "executable": str(EXE), "executable_sha256": EXE_SHA256,
              "command": [str(EXE), "-i", name], "omp_threads": 1,
              "time_limit_s": MAX_SECONDS, "memory_limit_bytes": MAX_TREE_BYTES,
              "peak_observed_tree_bytes": maximum, "elapsed_s": elapsed,
              "exit_code": code, "failure": reason,
              "warning_lines": warnings, "parsed": parsed}
    (output/"result.json").write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("expected_sha256")
    args = parser.parse_args()
    result = run(args.deck, args.output, args.expected_sha256)
    print(json.dumps({"failure": result["failure"], "parsed": result["parsed"],
                      "elapsed_s": result["elapsed_s"]}))
