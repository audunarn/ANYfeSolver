"""Public example check: the initial-contact-seed example runs end to end.

The owned example ``examples/general_contact_initial_contact_seed_dev.py``
is executed as a real public entry point against the guard's immutable
source snapshot (derived from the attempt's ``install.json``; the
repository ``src`` tree is the fallback outside the guard). The example
reports its trajectory truthfully and exits 0 exactly when the accepted
prefix completes, so this check verifies that contract: valid JSON on
stdout with the effective seed selection and the full seed record
(outcome, counts, refusal reason, cost), and an exit code consistent
with the reported status. No acceleration or nonlinear feasibility is
assumed; the physical outcome is recorded as it occurs.

The current 4x4 example's source-bound Windows public entry completed in a
recorded 12.32-second process wall interval. The earlier 2x2 fixture's
167-second divergence and 180-second cap remain historical failures; the
mesh change prevents using those timings to claim a production speedup.
"""
import json
import os
import subprocess
import sys

_EXAMPLE_NAME = "general_contact_initial_contact_seed_dev.py"


def _snapshot_site():
    """The guard's immutable source snapshot site, when running under it."""
    runtime = os.environ.get("AUTOMATIC_RUNTIME_RESULT")
    if not runtime:
        return None
    install = os.path.join(os.path.dirname(runtime), "install.json")
    if not os.path.isfile(install):
        return None
    with open(install, encoding="utf-8") as handle:
        record = json.load(handle)
    site = record.get("site")
    if site and os.path.isdir(os.path.join(site, "anyfesolver")):
        return site
    return None


def _run_example(seed):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example = os.path.join(repo, "examples", _EXAMPLE_NAME)
    assert os.path.isfile(example), example
    env = os.environ.copy()
    site = _snapshot_site()
    if site is not None:
        # The subprocess must import the same immutable snapshot as this
        # test process; the guard clears PYTHONPATH, so bind it here.
        env["PYTHONPATH"] = site
    completed = subprocess.run(
        [sys.executable, example, "--seed", seed],
        cwd=repo, env=env, capture_output=True, text=True, timeout=170.)
    return completed


def test_public_example_seeded_half_prefix_runs_and_reports_truthfully():
    completed = _run_example("linearized_kkt_dev")
    assert "initial_contact_seed" in completed.stdout, (
        "example produced no report\nstdout:\n" + completed.stdout
        + "\nstderr:\n" + completed.stderr)
    payload = json.loads(completed.stdout)
    assert payload["initial_contact_seed"] == "linearized_kkt_dev"
    assert payload["status"] in {"completed", "diverged", "stopped_at_limit"}
    # The public exit contract: 0 exactly when the accepted prefix
    # completed; a truthful non-completed trajectory exits nonzero.
    assert (completed.returncode == 0) == (payload["status"] == "completed")
    assert isinstance(payload["failure_reason"], (str, type(None)))
    seed = payload["seed_record"]
    assert seed["selection"] == "linearized_kkt_dev"
    assert seed["outcome"] in {"admitted", "refused"}
    assert seed["constrained_factorizations"] == 1
    assert seed["constrained_solves"] >= 1
    if seed["outcome"] == "admitted":
        assert seed["admissible_fraction"] == 1.0
        assert seed["seed_merit"] < seed["origin_merit"]
    else:
        assert isinstance(seed["reason"], str) and seed["reason"]
