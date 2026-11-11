"""Bounded deterministic G6 rehearsal/formal confirmation."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests/test_ge_beam3_g6_completion.py"
NODES = tuple(f"{TEST.as_posix()}::test_{index:02d}_" for index in range(1, 9))


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def strict(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value: raise ValueError("duplicate evidence key")
            value[key] = item
        return value
    value = json.loads(raw.decode("ascii"), object_pairs_hook=pairs,
        parse_constant=lambda _token: (_ for _ in ()).throw(ValueError("nonfinite evidence")))
    if canonical(value) != raw: raise ValueError("canonical evidence required")
    return value


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream: stream.write(canonical(value))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run(output, revision, package_receipt, package_sha256, lane):
    output = Path(output).resolve(); receipt_path = Path(package_receipt).resolve()
    if output.is_relative_to(ROOT.resolve()) or output.exists():
        raise ValueError("new external G6 output required")
    if git("status", "--porcelain", "--untracked-files=all") or git("rev-parse", "HEAD") != revision:
        raise ValueError("exact clean frozen G6 candidate required")
    receipt_raw = receipt_path.read_bytes()
    if sha256(receipt_raw).hexdigest() != package_sha256:
        raise ValueError("installed-artifact receipt hash mismatch")
    receipt = strict(receipt_raw)
    if receipt.get("revision") != revision or receipt.get("schema") != "GE_BEAM3_G6_INSTALLED_ARTIFACT_V1":
        raise ValueError("installed-artifact candidate mismatch")
    output.parent.mkdir(parents=True, exist_ok=True); output.mkdir()
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      NUMEXPR_NUM_THREADS="1", GE_BEAM3_G6_SCALE_ELEMENTS="1024")
    import pytest
    class Recorder:
        def __init__(self): self.collected=[]; self.passed=[]; self.bad=[]; self.module=None
        def pytest_collection_modifyitems(self, items):
            self.collected=[item.nodeid for item in items]; self.module=items[0].module
        def pytest_runtest_logreport(self, report):
            if report.failed or report.skipped: self.bad.append(report.nodeid)
            if report.when == "call" and report.passed: self.passed.append(report.nodeid)
    # Select by discovered exact names while retaining the frozen 01..08 order.
    import importlib.util
    spec=importlib.util.spec_from_file_location("g6_inventory", TEST)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    selected=[str(TEST)+"::"+name for name in sorted(name for name in vars(module)
        if name.startswith("test_") and name[5:7].isdigit())]
    if len(selected) != 8 or [Path(item.split("::")[0]).name for item in selected] != [TEST.name] * 8:
        raise ValueError("exact eight-node G6 inventory required")
    recorder=Recorder()
    code=pytest.main(["-q", "-p", "no:cacheprovider", "--basetemp", str(output / "pytest"), *selected],
                     plugins=[recorder])
    expected_names = [item.split("::")[-1] for item in selected]
    if ([item.split("::")[-1] for item in recorder.collected] != expected_names
            or code or recorder.bad or recorder.passed != recorder.collected):
        raise RuntimeError("G6 scientific inventory failed")
    records=recorder.module.SCIENTIFIC_RECORDS
    from anyfesolver._ge_beam3_g6_domain import PARITY_DISPOSITIONS, adjudicate
    result=adjudicate(records, parity_rows=dict(PARITY_DISPOSITIONS),
                      installed_artifact=receipt["installed_artifact"])
    science={"schema":"GE_BEAM3_G6_CONFIRMATION_V1", "candidate_commit":revision,
        "lane":lane, "selected":expected_names, "records":records,
        "adjudication":result, "package_receipt_sha256":package_sha256,
        "production_default_qualified":False}
    write_new(output / "result.json", science)
    write_new(output / "completion.json", {"result_sha256":sha256(canonical(science)).hexdigest(),
        "all_children_terminal":True, "automatic_retry":False})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",required=True)
    parser.add_argument("--revision",required=True); parser.add_argument("--package-receipt",required=True)
    parser.add_argument("--package-sha256",required=True)
    parser.add_argument("--lane",choices=("rehearsal","formal"),required=True)
    args=parser.parse_args(); run(args.output,args.revision,args.package_receipt,args.package_sha256,args.lane)
    return 0


if __name__ == "__main__": raise SystemExit(main())
