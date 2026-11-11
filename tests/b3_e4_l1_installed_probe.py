"""Fresh-process installed-wheel smoke for experimental L1 weak-line wiring."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        while part := stream.read(1024*1024):
            value.update(part)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    site = args.site.resolve(strict=True)
    wheel = args.wheel.resolve(strict=True)
    if (len(args.source_commit) != 40 or
            any(c not in "0123456789abcdef" for c in args.source_commit)):
        raise ValueError("exact source commit required")
    sys.path.insert(0, str(site))

    import anyfesolver
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info
    from anyfesolver._stiffened_shell_l1_assembly_experimental import SPACE_A, SPACE_B

    imported = Path(anyfesolver.__file__).resolve(strict=True)
    if site not in imported.parents:
        raise RuntimeError("installed ANYfeSolver was not imported")
    test_file = Path(__file__).with_name(
        "test_stiffened_shell_l1_assembly_experimental.py")
    spec = importlib.util.spec_from_file_location("l1_installed_checks", test_file)
    checks = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = checks
    spec.loader.exec_module(checks)

    for space, rows in ((SPACE_A, 28), (SPACE_B, 20)):
        checks.test_sparse_weak_line_matches_independent_dense_solution_and_restart(
            space, rows)
        checks.test_sparse_line_rows_match_independent_sampled_weak_work(space)
    checks.test_weak_line_rejects_mismatched_reference_correspondence()
    checks.test_weak_line_rejects_repeated_material_connection()
    checks.test_line_owner_rejects_finite_checkpoint_before_point_driver()

    report = {
        "status": "passed",
        "scope": "experimental L1 reference-linear installed wiring",
        "source_commit": args.source_commit,
        "wheel_sha256": digest(wheel),
        "installed_package": str(imported),
        "runner_sha256": digest(Path(__file__).resolve()),
        "test_sha256": digest(test_file),
        "checks": 7,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {key: os.environ.get(key) for key in (
            "NUMBA_DISABLE_JIT", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "effective_threadpools": [
            {key: item.get(key) for key in ("internal_api", "num_threads",
                                             "filepath")}
            for item in threadpool_info()],
        "qualification": "not accepted",
    }
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2),
                           encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "wheel_sha256": report["wheel_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
