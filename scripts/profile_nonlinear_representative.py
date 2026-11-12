"""Profile one frozen representative case after an unmeasured warm solve."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import pstats
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_RUNNER = ROOT / "scripts/qualify_nonlinear_representative.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function_sha256(function) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _load_campaign_module(
    expected_runner_sha256: str,
    expected_build_case_sha256: str,
):
    runner_sha256 = _sha256(CAMPAIGN_RUNNER)
    if runner_sha256.lower() != expected_runner_sha256.lower():
        raise RuntimeError(
            "campaign runner SHA-256 mismatch: "
            f"expected {expected_runner_sha256}, observed {runner_sha256}"
        )
    spec = importlib.util.spec_from_file_location(
        "representative_campaign", CAMPAIGN_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CAMPAIGN_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_case_sha256 = _function_sha256(module._build_case)
    if build_case_sha256.lower() != expected_build_case_sha256.lower():
        raise RuntimeError(
            "campaign _build_case SHA-256 mismatch: "
            f"expected {expected_build_case_sha256}, observed {build_case_sha256}"
        )
    return module, runner_sha256, build_case_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--build-provenance", type=Path, required=True)
    parser.add_argument("--expected-campaign-runner-sha256", required=True)
    parser.add_argument("--expected-build-case-sha256", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    sys.path.insert(0, str(args.site.resolve()))
    import anyfesolver
    from anyfesolver import nonlinear_static

    imported = Path(anyfesolver.__file__).resolve()
    imported.relative_to(args.site.resolve())
    campaign, campaign_runner_sha256, build_case_sha256 = (
        _load_campaign_module(
            args.expected_campaign_runner_sha256,
            args.expected_build_case_sha256,
        )
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    provenance = json.loads(args.build_provenance.read_text(encoding="utf-8"))
    cases = {str(case["id"]): case for case in manifest["performance_cases"]}
    spec = cases[args.case]
    nonlinear_static._ensure_nonlinear_acceleration()

    warm_model, warm_options, _warm_probes = campaign._build_case(spec, "never")
    warm = nonlinear_static.solve_static_nonlinear(warm_model, **warm_options)
    if str(warm.status) != "completed":
        raise RuntimeError(f"warm solve failed: {warm.status}")

    model, options, _probes = campaign._build_case(spec, "never")
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    result = nonlinear_static.solve_static_nonlinear(model, **options)
    profiler.disable()
    elapsed = time.perf_counter() - started
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(args.profile))

    stats = pstats.Stats(profiler)
    entries = []
    ordered = sorted(
        stats.stats.items(),
        key=lambda item: float(item[1][3]),
        reverse=True,
    )
    for (filename, line, function), (primitive, calls, total, cumulative, _callers) in ordered[: args.top]:
        entries.append(
            {
                "file": str(Path(filename).resolve()),
                "line": int(line),
                "function": str(function),
                "primitive_calls": int(primitive),
                "calls": int(calls),
                "total_seconds": float(total),
                "cumulative_seconds": float(cumulative),
            }
        )
    payload = {
        "schema": "anyfesolver.nonlinear_static.representative_profile",
        "version": 1,
        "case": args.case,
        "status": str(result.status),
        "load_factor": float(result.load_factor),
        "steps": len(result.steps),
        "profiled_solve_wall_seconds": elapsed,
        "profile_total_calls": int(stats.total_calls),
        "profile_primitive_calls": int(stats.prim_calls),
        "profile_total_seconds": float(stats.total_tt),
        "profile": str(args.profile.resolve()),
        "identity": {
            "anyfesolver_module": str(imported),
            "anyfesolver_version": importlib.metadata.version("anyfesolver"),
            "manifest_sha256": campaign._sha256(args.manifest),
            "candidate_revision": provenance["candidate_revision"],
            "candidate_wheel_sha256": provenance["candidate_wheel_sha256"],
            "build_provenance_sha256": campaign._sha256(args.build_provenance),
            "campaign_runner": str(CAMPAIGN_RUNNER.resolve()),
            "campaign_runner_sha256": campaign_runner_sha256,
            "build_case_sha256": build_case_sha256,
        },
        "top_cumulative": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if str(result.status) == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
