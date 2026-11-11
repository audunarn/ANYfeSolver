"""Pure supervisor checks; these tests never start a numerical child."""
import importlib.util
import base64
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1] / "scripts/b2_large_scale_development.py"
_SPEC = importlib.util.spec_from_file_location("b2_large_scale_development", _PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def run_args(tmp_path, *, run_profile="source_dev", size="large", cases=1,
             installed_site=None, budget=None):
    return SimpleNamespace(run_profile=run_profile, size=size, cases=cases,
                           installed_site=installed_site, budget=budget,
                           out=str(tmp_path / "attempt"), profile=False)


def make_site(tmp_path):
    site = tmp_path / "site"
    package = site / "anyfesolver"
    package.mkdir(parents=True)
    for name in ("__init__.py", "stiffened_shell.py",
                 "_stiffened_shell_b2_q4_panel_experimental.py",
                 "_stiffened_shell_b2_physical_rigid_footprint.py",
                 "dependency.py"):
        (package / name).write_text(f"# installed {name}\n")
    metadata = site / "anysolver-0.4.7.dist-info"
    metadata.mkdir()
    rows = []
    for file in package.iterdir():
        data = file.read_bytes()
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append(f"anyfesolver/{file.name},sha256={encoded},{len(data)}")
    rows.append("anysolver-0.4.7.dist-info/RECORD,,")
    (metadata / "RECORD").write_text("\n".join(rows) + "\n")
    return site


def bind_fake_wheel(tmp_path, site, monkeypatch, *, run_profile="installed_large_dev"):
    wheel = tmp_path / "frozen.whl"
    with ZipFile(wheel, "w") as archive:
        for file in (site / "anyfesolver").iterdir():
            archive.write(file, f"anyfesolver/{file.name}")
        archive.write(site / "anysolver-0.4.7.dist-info/RECORD",
                      "anysolver-0.4.7.dist-info/RECORD")
    if run_profile in ("installed_large_refinement_dev", "installed_beam_rich_dev"):
        monkeypatch.setattr(runner, "INSTALLED_REFINEMENT_WHEEL", wheel)
        monkeypatch.setattr(runner, "INSTALLED_REFINEMENT_WHEEL_SHA256", runner.digest(wheel))
        monkeypatch.setattr(runner, "INSTALLED_REFINEMENT_SITE", site)
        monkeypatch.setattr(runner, "INSTALLED_REFINEMENT_BUDGET", tmp_path / "refinement-large-budget.json")
        if run_profile == "installed_beam_rich_dev":
            monkeypatch.setattr(runner, "INSTALLED_BEAM_RICH_BUDGET", tmp_path / "beam-rich-budget.json")
            monkeypatch.setattr(runner, "INSTALLED_BEAM_RICH_OUT", tmp_path / "beam-rich-attempt")
    else:
        monkeypatch.setattr(runner, "INSTALLED_LARGE_WHEEL", wheel)
        monkeypatch.setattr(runner, "INSTALLED_LARGE_WHEEL_SHA256", runner.digest(wheel))
    return wheel


def test_default_run_retains_source_limits_and_historical_budget(tmp_path):
    name, limits, budget, site = runner.resolve_run(run_args(tmp_path))
    assert name == "source_dev" and site is None
    assert limits == runner.RunProfile(2 * 1024**3, 90., 240., 60.)
    assert budget == runner.DEFAULT_BUDGET.resolve()
    assert runner.settings("large", 1)["nx"] == 100
    assert runner.settings("large", 1)["ny"] == 100


def test_installed_profile_requires_large_size_site_and_separate_budget(tmp_path):
    site = make_site(tmp_path)
    budget = tmp_path / "installed-budget.json"
    args = run_args(tmp_path, run_profile="installed_large_dev", cases=100, installed_site=str(site),
                    budget=str(budget))
    name, limits, selected_budget, selected_site = runner.resolve_run(args)
    assert name == "installed_large_dev"
    assert limits == runner.RunProfile(4 * 1024**3, 240., 300., 60.)
    assert selected_budget == budget and selected_site == site
    assert runner.settings("large", 100)["cases"] == 100
    with pytest.raises(AttributeError):
        limits.child_seconds = 999
    with pytest.raises(TypeError):
        runner.RUN_PROFILES["installed_large_dev"] = limits

    for changes, message in (
        ({"size": "scale"}, "requires --size large"),
        ({"cases": 10}, "requires exactly 100 cases"),
        ({"installed_site": None}, "requires --installed-site"),
        ({"budget": None}, "requires a separate --budget"),
        ({"budget": str(runner.DEFAULT_BUDGET)}, "cannot use the source development budget"),
    ):
        with pytest.raises(ValueError, match=message):
            runner.resolve_run(SimpleNamespace(**(vars(args) | changes)))
    with pytest.raises(ValueError, match="--installed-site requires"):
        runner.resolve_run(run_args(tmp_path, installed_site=str(site)))


def test_refinement_profile_requires_bound_site_budget_and_wheel(tmp_path, monkeypatch):
    site = make_site(tmp_path)
    wheel = bind_fake_wheel(tmp_path, site, monkeypatch,
                            run_profile="installed_large_refinement_dev")
    budget = runner.INSTALLED_REFINEMENT_BUDGET
    args = run_args(tmp_path, run_profile="installed_large_refinement_dev", cases=100,
                    installed_site=str(site), budget=str(budget))
    name, limits, selected_budget, selected_site = runner.resolve_run(args)
    assert name == "installed_large_refinement_dev"
    assert limits == runner.RunProfile(4 * 1024**3, 240., 300., 60.)
    assert runner.child_time_limit(limits, 0., installed=True)[0] == 230.
    assert selected_budget == budget and selected_site == site
    identity = runner.installed_site_identity(site, run_profile=name)
    assert identity["installed_wheel"] == str(wheel)
    assert identity["installed_wheel_sha256"] == runner.digest(wheel)
    elsewhere = tmp_path / "elsewhere/anyfesolver"
    elsewhere.mkdir(parents=True)
    (elsewhere / "__init__.py").write_text("# other site\n")
    for changes, message in (
        ({"size": "scale"}, "requires --size large"),
        ({"cases": 10}, "requires exactly 100 cases"),
        ({"installed_site": None}, "requires --installed-site"),
        ({"installed_site": str(tmp_path / "elsewhere")}, "bound installed site"),
        ({"budget": None}, "requires a separate --budget"),
        ({"budget": str(tmp_path / "other.json")}, "requires its separate budget"),
    ):
        with pytest.raises(ValueError, match=message):
            runner.resolve_run(SimpleNamespace(**(vars(args) | changes)))
    (site / "anyfesolver/dependency.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="differs from frozen wheel"):
        runner.installed_site_identity(site, run_profile=name)


def test_refinement_large_budget_is_distinct_from_focused_installed_checks():
    focused = runner.ROOT / "docs/agent_plans/development/r8-b2-refinement-installed-20261009/budget.json"
    large = runner.ROOT / "docs/agent_plans/development/r8-b2-refinement-large-20261009/budget.json"
    assert runner.INSTALLED_REFINEMENT_BUDGET == large
    assert runner.INSTALLED_REFINEMENT_FOCUSED_BUDGET == focused
    assert large.resolve(strict=False) != focused.resolve(strict=False)
    runner.require_separate_budget("installed_large_refinement_dev", large.resolve(strict=False))
    runner.require_separate_budget("installed_large_dev", runner.INSTALLED_LARGE_BUDGET.resolve(strict=False))
    for profile, budget in (
        ("installed_large_refinement_dev", focused),
        ("installed_large_dev", large),
        ("source_dev", focused),
        ("source_dev", large),
        ("source_dev", runner.INSTALLED_LARGE_BUDGET),
    ):
        with pytest.raises(ValueError, match="different registered activity"):
            runner.require_separate_budget(profile, budget.resolve(strict=False))


def test_installed_site_identity_and_import_origin_guard(tmp_path, monkeypatch):
    site = make_site(tmp_path)
    wheel = bind_fake_wheel(tmp_path, site, monkeypatch)
    package = site / "anyfesolver"
    assert runner.verify_installed_import_origin(package / "__init__.py", site) == (package / "__init__.py")
    source_package = tmp_path / "source/anyfesolver"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text("# source\n")
    with pytest.raises(RuntimeError, match="outside installed site"):
        runner.verify_installed_import_origin(source_package / "__init__.py", site)
    identity = runner.installed_site_identity(site)
    assert identity["installed_site"] == str(site)
    assert identity["installed_module_sha256"]["stiffened_shell.py"] == runner.digest(package / "stiffened_shell.py")
    assert identity["installed_site_record_sha256"] == runner.digest(site / "anysolver-0.4.7.dist-info/RECORD")
    assert identity["installed_wheel"] == str(wheel)
    assert identity["installed_wheel_sha256"] == runner.digest(wheel)
    assert identity["installed_product_file_count"] == 5
    (package / "stiffened_shell.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="differs from frozen wheel"):
        runner.installed_site_identity(site)
    (package / "stiffened_shell.py").write_text("# installed stiffened_shell.py\n")
    (package / "dependency.py").write_text("tampered dependency\n")
    with pytest.raises(ValueError, match="differs from frozen wheel"):
        runner.installed_site_identity(site)
    (package / "dependency.py").write_text("# installed dependency.py\n")
    record_file = site / "anysolver-0.4.7.dist-info/RECORD"
    original_record = record_file.read_text()
    record_file.write_text(original_record.replace("sha256=", "sha256=changed", 1))
    with pytest.raises(ValueError, match="RECORD differs"):
        runner.installed_site_identity(site)
    record_file.write_text(original_record)
    wheel.write_bytes(wheel.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="frozen SHA256"):
        runner.installed_site_identity(site)


def test_installed_runtime_refuses_wrong_cpu_thread_or_jit():
    runtime = dict(affinity=[0], NUMBA_DISABLE_JIT="1",
                   thread_env={name: "1" for name in runner.THREAD_ENV},
                   threadpools=[{"num_threads": 1}])
    runner.verify_installed_runtime(runtime)
    for changes in ({"affinity": [0, 1]}, {"NUMBA_DISABLE_JIT": "0"},
                    {"threadpools": [{"num_threads": 2}]}):
        with pytest.raises(RuntimeError):
            runner.verify_installed_runtime(runtime | changes)


def test_installed_inventory_requires_exact_generated_and_compiled_counts():
    config = runner.settings("large", 100)
    counts = dict(runner.INSTALLED_EXPECTED_COUNTS)
    definition = SimpleNamespace(shells=[None] * 10_000, members=[None] * 1_000,
                                 physical_footprints=[None] * 100)
    owner = SimpleNamespace(_shell_formulations=[None] * 10_000,
                            _members=[None] * 1_000,
                            footprints=[None] * 100,
                            _footprint_incidence=[SimpleNamespace(row_count=5)] * 400,
                            _C=SimpleNamespace(shape=(2_000, 1)))
    assert runner.verify_installed_inventory(config, counts, definition=definition) == counts
    assert runner.verify_installed_inventory(config, counts, owner=owner) == dict(runner.INSTALLED_COMPILED_COUNTS)
    with pytest.raises(RuntimeError, match="exact full-size"):
        runner.verify_installed_inventory(runner.settings("large", 10), counts)
    with pytest.raises(RuntimeError, match="generated inventory"):
        runner.verify_installed_inventory(config, counts | {"attached_patches": 99})
    owner._members.pop()
    with pytest.raises(RuntimeError, match="compiled inventory"):
        runner.verify_installed_inventory(config, counts, owner=owner)
    owner._members.append(None)
    owner._footprint_incidence.pop()
    with pytest.raises(RuntimeError, match="compiled inventory"):
        runner.verify_installed_inventory(config, counts, owner=owner)


def test_installed_zero_exit_needs_bound_child_record(tmp_path, monkeypatch):
    import json
    site = make_site(tmp_path)
    bind_fake_wheel(tmp_path, site, monkeypatch)
    identity = runner.installed_site_identity(site)
    config = runner.settings("large", 100)
    outdir = tmp_path / "attempt"
    outdir.mkdir()
    runtime = dict(python_optimize=0, affinity=[0], NUMBA_DISABLE_JIT="1",
        thread_env={name: "1" for name in runner.THREAD_ENV},
        threadpools=[{"num_threads": 1}],
        anyfesolver_file=str(site / "anyfesolver/__init__.py"),
        anyfesolver_sha256=runner.digest(site / "anyfesolver/__init__.py"))
    record = dict(status="passed_development_probe",
        config=json.loads(json.dumps(config)),
        inventory=dict(runner.INSTALLED_EXPECTED_COUNTS),
        compiled_inventory=dict(runner.INSTALLED_COMPILED_COUNTS), runtime=runtime)
    with pytest.raises(RuntimeError, match="invalid installed child record"):
        runner.verify_installed_child_record(outdir, config, site, identity)
    (outdir / "child.json").write_text(json.dumps(record))
    runner.verify_installed_child_record(outdir, config, site, identity)
    record["compiled_inventory"]["attached_patches"] = 99
    (outdir / "child.json").write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="compiled inventory"):
        runner.verify_installed_child_record(outdir, config, site, identity)


def test_installed_budget_leaves_cleanup_margin_and_preserves_reserve():
    profile = runner.RUN_PROFILES["installed_large_dev"]
    assert runner.child_time_limit(profile, 0., installed=True) == (230., 240., 10.)
    assert runner.child_time_limit(profile, 220., installed=True) == (10., 20., 10.)
    with pytest.raises(RuntimeError, match="cleanup margin"):
        runner.child_time_limit(profile, 231., installed=True)
    assert runner.child_time_limit(runner.RUN_PROFILES["source_dev"], 0.) == (90., 180., 0.)


@dataclass
class FakePhysicalCase:
    case_id: str
    selected_displacements: tuple = (0.,)
    constraint_gaps: tuple = (0.,)
    internal_strain_energy: float = 1.


def make_refinement_batch(*, corrected=("case-7", "case-99"), extra_correction=False):
    case_ids = tuple(f"case-{i+1}" for i in range(100))
    capture = runner.BatchSolveCapture(tuple((case_id, np.array([1.])) for case_id in case_ids))
    for case_id in case_ids:
        capture.solve(lambda rhs: rhs, np.array([1.]))
        if case_id in corrected:
            capture.solve(lambda rhs: rhs, np.array([.1]))
        if extra_correction and case_id == "case-7":
            capture.solve(lambda rhs: rhs, np.array([.01]))
        capture.on_case(case_id)
    result = SimpleNamespace(status="completed", failure_reason=None,
        factorization_count=1, maximum_rhs_chunk=1,
        solve_count=capture.primary_solve_attempts + capture.corrective_solve_attempts,
        cases=tuple(FakePhysicalCase(case_id) for case_id in case_ids),
        displacement_envelopes=(), affine_reaction_envelopes=(),
        member_resultant_envelopes=())
    return capture, result


def test_refinement_work_counts_and_refuses_extra_or_nonfinite_work():
    config = runner.settings("large", 100)
    capture, result = make_refinement_batch()
    work = runner.verify_refinement_work(capture, result, config, solves_before=0)
    assert work["primary_solves"] == 100
    assert work["corrective_solves"] == 2
    assert work["total_solves"] == 102
    assert dict(work["corrections_by_case"])["case-7"] == 1
    assert dict(work["corrections_by_case"])["case-99"] == 1
    result.solve_count += 1
    with pytest.raises(RuntimeError, match="unaccounted"):
        runner.verify_refinement_work(capture, result, config, solves_before=0)
    result.solve_count -= 1
    result.cases[6].selected_displacements = (float("nan"),)
    with pytest.raises(RuntimeError, match="nonfinite"):
        runner.verify_refinement_work(capture, result, config, solves_before=0)
    result.cases[6].selected_displacements = (0.,)
    result.status = "failed"
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.verify_refinement_work(capture, result, config, solves_before=0)
    result.status = "completed"
    result.factorization_count = 2
    with pytest.raises(RuntimeError, match="work contract"):
        runner.verify_refinement_work(capture, result, config, solves_before=0)
    over_capture, over_result = make_refinement_batch(extra_correction=True)
    with pytest.raises(RuntimeError, match="excessive"):
        runner.verify_refinement_work(over_capture, over_result, config, solves_before=0)


def test_beam_rich_profile_is_bound_and_cannot_charge_other_ledgers(tmp_path, monkeypatch):
    site = make_site(tmp_path)
    wheel = bind_fake_wheel(tmp_path, site, monkeypatch,
                            run_profile="installed_beam_rich_dev")
    args = run_args(tmp_path, run_profile="installed_beam_rich_dev", size="beam_rich",
                    cases=100, installed_site=str(site),
                    budget=str(runner.INSTALLED_BEAM_RICH_BUDGET))
    args.out = str(runner.INSTALLED_BEAM_RICH_OUT)
    name, limits, budget, selected_site = runner.resolve_run(args)
    assert name == "installed_beam_rich_dev"
    assert limits == runner.RunProfile(4 * 1024**3, 240., 300., 60.)
    assert runner.child_time_limit(limits, 0., installed=True) == (230., 240., 10.)
    assert budget == runner.INSTALLED_BEAM_RICH_BUDGET and selected_site == site
    assert runner.installed_site_identity(site, run_profile=name)["installed_wheel_sha256"] == runner.digest(wheel)
    for changes, message in (
        ({"size": "large"}, "requires --size beam_rich"),
        ({"cases": 10}, "requires exactly 100 cases"),
        ({"installed_site": None}, "requires --installed-site"),
        ({"budget": str(runner.INSTALLED_REFINEMENT_BUDGET)}, "different registered activity"),
        ({"out": str(tmp_path / "other-attempt")}, "unique registered output"),
    ):
        with pytest.raises(ValueError, match=message):
            runner.resolve_run(SimpleNamespace(**(vars(args) | changes)))
    with pytest.raises(ValueError, match="different registered activity"):
        runner.resolve_run(run_args(tmp_path, budget=str(runner.INSTALLED_BEAM_RICH_BUDGET)))
    with pytest.raises(ValueError, match="requires installed_beam_rich_dev"):
        runner.resolve_run(run_args(tmp_path, size="beam_rich"))


def test_beam_rich_source_generation_has_five_to_one_mesh_and_physical_pads(monkeypatch):
    monkeypatch.syspath_prepend(str(runner.ROOT / "src"))
    config = runner.settings("beam_rich", 100)
    definition, inventory, _ = runner.model(config)
    assert config["nx"] == config["ny"] == 20
    assert inventory["shell_nodes"] == 441
    assert inventory["beam_nodes"] == 1010
    assert inventory["dofs"] == 8706
    assert inventory["fixed_dofs"] == 126
    assert inventory["beam_segments_per_shell"] == 5
    assert runner.verify_installed_inventory(config, inventory, definition=definition,
        run_profile="installed_beam_rich_dev") == dict(runner.BEAM_RICH_EXPECTED_COUNTS)
    assert definition.members[0].reference_points_m == ((0., .5, .2), (.2, .5, .2))
    assert definition.members[4].reference_points_m[-1][0] == 1.
    assert definition.physical_footprints[0].beam_node == definition.members[4].node_ids[1]
    assert len({pad.beam_node for pad in definition.physical_footprints}) == 100
    with pytest.raises(RuntimeError, match="generated inventory"):
        runner.verify_installed_inventory(config, inventory | {"attached_patches": 99},
            run_profile="installed_beam_rich_dev")


def test_beam_rich_recovery_counts_selected_work_and_refuses_omission():
    cases = tuple(SimpleNamespace(selected_displacements=(0., 0.),
        beam_resultants_local=(("b2-0-0", ((0.,)*6, (0.,)*6, (0.,)*6)),),
        footprint_wrenches=(SimpleNamespace(footprint_id="pad-0-1"),))
        for _ in range(100))
    result = SimpleNamespace(cases=cases, member_resultant_envelopes=(0.,)*18)
    expected = dict(cases=100, selected_member_stations=300,
        selected_member_components=1800, selected_footprint_wrenches=100,
        selected_displacement_values=200, member_resultant_envelopes=18)
    assert runner.verify_beam_rich_recovery(result, selected_member="b2-0-0",
        selected_pad="pad-0-1") == expected
    cases[5].beam_resultants_local = ()
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.verify_beam_rich_recovery(result, selected_member="b2-0-0",
            selected_pad="pad-0-1")


def test_beam_rich_frozen_registration_precedes_budget_spend(tmp_path, monkeypatch):
    import json
    frozen = runner.INSTALLED_BEAM_RICH_REGISTRATION
    if not frozen.exists():
        pytest.skip(
            "exact historical frozen beam-rich registration bytes are absent "
            "from this evidence-free source export: " + str(frozen) + " "
            "(docs/agent_plans/development/r8-b2-beam-rich-20261009/"
            "registration.json); the frozen-registration refusal contract "
            "exercises only the exact historical artifact, so a supplied "
            "corrupted or changed registration still runs and refuses")
    original = json.loads(frozen.read_text())
    registration = tmp_path / "registration.json"
    monkeypatch.setattr(runner, "INSTALLED_BEAM_RICH_REGISTRATION", registration)
    identity = {"installed_wheel_sha256": runner.INSTALLED_REFINEMENT_WHEEL_SHA256}
    args = (runner.settings("beam_rich", 100), identity,
        runner.RUN_PROFILES["installed_beam_rich_dev"],
        runner.INSTALLED_BEAM_RICH_BUDGET, runner.INSTALLED_BEAM_RICH_OUT)
    with pytest.raises(FileNotFoundError):
        runner.verify_beam_rich_registration(*args)
    # The restored registration is a historical frozen artifact: a runner
    # identity frozen against an earlier runner must not authorize the
    # current request's budget spend.
    registration.write_text(json.dumps(original))
    if original["candidate"]["runner_sha256"] != runner.digest(runner.__file__):
        with pytest.raises(ValueError, match="frozen registration"):
            runner.verify_beam_rich_registration(*args)
    current = json.loads(json.dumps(original))
    current["candidate"]["runner_sha256"] = runner.digest(runner.__file__)
    registration.write_text(json.dumps(current))
    assert runner.verify_beam_rich_registration(*args)["registration_sha256"] == runner.digest(registration)
    for key, value in (("runner_sha256", "0"*64), ("wheel_sha256", "0"*64)):
        changed = json.loads(json.dumps(original))
        changed["candidate"][key] = value
        registration.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="frozen registration"):
            runner.verify_beam_rich_registration(*args)
    changed = json.loads(json.dumps(original))
    changed["run"]["output"] = "docs/agent_plans/development/r8-b2-beam-rich-20261009/attempt-v2"
    registration.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="frozen registration"):
        runner.verify_beam_rich_registration(*args)


def test_beam_rich_child_receipt_requires_recovery_and_work(tmp_path, monkeypatch):
    import json
    site = make_site(tmp_path)
    bind_fake_wheel(tmp_path, site, monkeypatch, run_profile="installed_beam_rich_dev")
    identity = runner.installed_site_identity(site, run_profile="installed_beam_rich_dev")
    config = runner.settings("beam_rich", 100)
    capture, result = make_refinement_batch()
    work = runner.verify_refinement_work(capture, result, config, solves_before=0)
    recovery = dict(cases=100, selected_member_stations=300,
        selected_member_components=1800, selected_footprint_wrenches=100,
        selected_displacement_values=200, member_resultant_envelopes=18)
    batch = dict(status="completed", failure_reason=None, case_count=100,
        factorization_count=1, maximum_rhs_chunk=1, solve_count=102,
        captured_primary_solve_attempts=100, captured_corrective_solve_attempts=2,
        accepted_case_solve_counts=capture.completed_case_solve_counts,
        selected_displacements=[[0., 0.] for _ in range(100)],
        maximum_constraint_gap=0.)
    runtime = dict(python_optimize=0, affinity=[0], NUMBA_DISABLE_JIT="1",
        thread_env={name: "1" for name in runner.THREAD_ENV},
        threadpools=[{"num_threads": 1}],
        anyfesolver_file=str(site / "anyfesolver/__init__.py"),
        anyfesolver_sha256=runner.digest(site / "anyfesolver/__init__.py"))
    record = dict(status="passed_development_probe",
        run_profile="installed_beam_rich_dev", config=json.loads(json.dumps(config)),
        inventory=dict(runner.BEAM_RICH_EXPECTED_COUNTS),
        compiled_inventory=dict(runner.BEAM_RICH_COMPILED_COUNTS),
        runtime=runtime, refinement_work=work,
        beam_rich_recovery=recovery, batch=batch)
    outdir = tmp_path / "attempt"
    outdir.mkdir()
    child_file = outdir / "child.json"
    child_file.write_text(json.dumps(record))
    runner.verify_installed_child_record(outdir, config, site, identity,
        run_profile="installed_beam_rich_dev")
    record["beam_rich_recovery"]["selected_member_stations"] = 299
    child_file.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="selected recovery"):
        runner.verify_installed_child_record(outdir, config, site, identity,
            run_profile="installed_beam_rich_dev")
    record["beam_rich_recovery"]["selected_member_stations"] = 300
    record["refinement_work"]["total_solves"] = 103
    child_file.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="work or physical response"):
        runner.verify_installed_child_record(outdir, config, site, identity,
            run_profile="installed_beam_rich_dev")


def test_finite_response_accepts_real_physical_footprint_result_types(monkeypatch):
    monkeypatch.syspath_prepend(str(runner.ROOT / "src"))
    from anyfesolver._stiffened_shell_b2_physical_rigid_footprint import (
        B2PhysicalRigidFootprintWrench, PhysicalFootprintNodeAction)
    from anyfesolver._stiffened_shell_b2_q4_panel_experimental import (
        B2PhysicalRigidFootprintBatchResult, B2PhysicalRigidFootprintSelectedCase,
        ExperimentalB2Q4PanelResponseEnvelope,
        ExperimentalB2Q4PanelMemberResultantEnvelope)
    zeros3 = (0., 0., 0.)
    action = PhysicalFootprintNodeAction(1, 11, 0, zeros3, zeros3,
                                         zeros3, zeros3, (0.,) * 5)
    wrench = B2PhysicalRigidFootprintWrench("pad-1", zeros3, zeros3,
                                            zeros3, zeros3, zeros3, (action,))
    case = B2PhysicalRigidFootprintSelectedCase("case-1", (0.,), (0.,), (), (),
        (("b2-1", ((0.,) * 6,)),), 0., footprint_wrenches=(wrench,),
        constraint_multipliers=(0.,), constraint_gaps=(0.,))
    displacement = ExperimentalB2Q4PanelResponseEnvelope(
        0, 0., "case-1", 0., "case-1")
    resultant = ExperimentalB2Q4PanelMemberResultantEnvelope(
        "b2-1", 0, 0., "N", 0., "case-1", 0., "case-1")
    batch = B2PhysicalRigidFootprintBatchResult(
        "completed", (0,), (), (case,), 1, 1,
        displacement_envelopes=(displacement,),
        member_resultant_envelopes=(resultant,))
    assert runner._finite_physical_response(batch)
    assert not runner._finite_physical_response(
        replace(batch, cases=(replace(case, internal_strain_energy=float("inf")),)))


def test_refinement_child_record_rejects_unaccounted_work(tmp_path, monkeypatch):
    import json
    site = make_site(tmp_path)
    bind_fake_wheel(tmp_path, site, monkeypatch,
                    run_profile="installed_large_refinement_dev")
    identity = runner.installed_site_identity(site,
        run_profile="installed_large_refinement_dev")
    config = runner.settings("large", 100)
    capture, result = make_refinement_batch()
    work = runner.verify_refinement_work(capture, result, config, solves_before=0)
    outdir = tmp_path / "attempt"
    outdir.mkdir()
    batch = dict(status="completed", failure_reason=None, case_count=100,
        factorization_count=1, maximum_rhs_chunk=1, solve_count=102,
        captured_primary_solve_attempts=100, captured_corrective_solve_attempts=2,
        accepted_case_solve_counts=capture.completed_case_solve_counts,
        selected_displacements=[[0.] for _ in range(100)],
        maximum_constraint_gap=0.)
    runtime = dict(python_optimize=0, affinity=[0], NUMBA_DISABLE_JIT="1",
        thread_env={name: "1" for name in runner.THREAD_ENV},
        threadpools=[{"num_threads": 1}],
        anyfesolver_file=str(site / "anyfesolver/__init__.py"),
        anyfesolver_sha256=runner.digest(site / "anyfesolver/__init__.py"))
    record = dict(status="passed_development_probe",
        run_profile="installed_large_refinement_dev",
        config=json.loads(json.dumps(config)),
        inventory=dict(runner.INSTALLED_EXPECTED_COUNTS),
        compiled_inventory=dict(runner.INSTALLED_COMPILED_COUNTS),
        runtime=runtime, refinement_work=work, batch=batch)
    child_file = outdir / "child.json"
    child_file.write_text(json.dumps(record))
    runner.verify_installed_child_record(outdir, config, site, identity,
        run_profile="installed_large_refinement_dev")
    batch["solve_count"] = 103
    child_file.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="work or physical response"):
        runner.verify_installed_child_record(outdir, config, site, identity,
            run_profile="installed_large_refinement_dev")


def test_python_optimization_refusal_and_child_receipt(tmp_path):
    runner.verify_python_optimization(0)
    with pytest.raises(RuntimeError, match="optimization must be disabled"):
        runner.verify_python_optimization(1)
    (tmp_path / "child.json").write_text('{"runtime":{"python_optimize":0}}')
    receipt = {}
    runner.record_child_optimize(receipt, tmp_path)
    assert receipt["effective_child_python_optimize"] == 0


def test_installed_parent_uses_site_and_separate_ledger_without_child(tmp_path, monkeypatch):
    site = make_site(tmp_path)
    bind_fake_wheel(tmp_path, site, monkeypatch)
    budget = tmp_path / "installed-budget.json"
    args = run_args(tmp_path, run_profile="installed_large_dev", cases=100, installed_site=str(site),
                    budget=str(budget))
    class FakePsutil:
        Error = OSError
        @staticmethod
        def Process():
            return FakeSupervisor()
    monkeypatch.setitem(__import__("sys").modules, "psutil", FakePsutil)
    def refuse_spawn(command, **kwargs):
        assert command[-4:] == ["--run-profile", "installed_large_dev",
                                "--installed-site", str(site)]
        assert kwargs["env"]["PYTHONPATH"] == str(site)
        assert kwargs["env"]["NUMBA_DISABLE_JIT"] == "1"
        assert kwargs["env"]["PYTHONOPTIMIZE"] == "0"
        assert all(kwargs["env"][name] == "1" for name in runner.THREAD_ENV)
        raise OSError("source-only refusal")
    monkeypatch.setattr(runner.subprocess, "Popen", refuse_spawn)
    assert runner.run_parent(args) == 1
    import json
    receipt = json.loads((tmp_path / "attempt/receipt.json").read_text())
    assert receipt["status"] == "spawn_refused"
    assert receipt["limits"]["max_rss_bytes"] == 4 * 1024**3
    assert receipt["limits"]["child_seconds"] == 230.
    assert receipt["limits"]["configured_child_seconds"] == 240.
    assert receipt["limits"]["cleanup_margin_seconds"] == 10.
    assert receipt["limits"]["total_seconds"] == 300.
    assert receipt["budget_file"] == str(budget)
    assert receipt["effective_parent_python_optimize"] == 0
    assert receipt["requested_child_python_optimize"] == 0
    assert receipt["effective_child_python_optimize"] is None
    assert not budget.exists()


def test_source_parent_retains_source_import_and_legacy_limits(tmp_path, monkeypatch):
    args = run_args(tmp_path, budget=str(tmp_path / "source-budget.json"))
    class FakePsutil:
        Error = OSError
        @staticmethod
        def Process():
            return FakeSupervisor()
    monkeypatch.setitem(__import__("sys").modules, "psutil", FakePsutil)
    def refuse_spawn(command, **kwargs):
        assert "--run-profile" not in command
        assert kwargs["env"]["PYTHONPATH"] == str(runner.ROOT / "src") + runner.os.pathsep + str(runner.ROOT)
        assert kwargs["env"]["PYTHONOPTIMIZE"] == "0"
        raise OSError("source-only refusal")
    monkeypatch.setattr(runner.subprocess, "Popen", refuse_spawn)
    assert runner.run_parent(args) == 1
    import json
    receipt = json.loads((tmp_path / "attempt/receipt.json").read_text())
    assert receipt["limits"]["max_rss_bytes"] == 2 * 1024**3
    assert receipt["limits"]["child_seconds"] == 90.
    assert receipt["limits"]["total_seconds"] == 240.
    assert receipt["limits"]["reserve_seconds"] == 60.
    assert "run_profile" not in receipt and "installed_site" not in receipt


@pytest.mark.parametrize("failure_at", ["constructor", "children"])
def test_post_spawn_monitor_error_reaps_and_charges(tmp_path, monkeypatch, failure_at):
    import json
    child = SimpleNamespace(pid=123, returncode=None, killed=False, waited=False)
    def poll():
        return child.returncode
    def kill():
        child.killed = True
        child.returncode = -9
    def wait(timeout=None):
        child.waited = True
        return child.returncode
    child.poll, child.kill, child.wait = poll, kill, wait
    class Watched:
        def cpu_affinity(self):
            return [0]
        def children(self, recursive=False):
            raise OSError("children unavailable")
        def kill(self):
            child.kill()
    class FakePsutil:
        Error = OSError
        calls = 0
        @classmethod
        def Process(cls, pid=None):
            cls.calls += 1
            if cls.calls == 1:
                return FakeSupervisor()
            if failure_at == "constructor":
                raise OSError("process unavailable")
            return Watched()
    monkeypatch.setitem(__import__("sys").modules, "psutil", FakePsutil)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_k: child)
    budget = tmp_path / "budget.json"
    args = run_args(tmp_path, budget=str(budget))
    assert runner.run_parent(args) == 1
    receipt = json.loads((tmp_path / "attempt/receipt.json").read_text())
    ledger = json.loads(budget.read_text())
    assert receipt["status"] == "supervision_failed"
    assert "unavailable" in receipt["supervision_error"]
    assert child.killed and child.waited
    assert receipt["process_wall_s"] > 0.
    assert ledger["used_s"] == receipt["process_wall_s"]


class FakeSupervisor:
    def __init__(self, *, refuse=False, fail_restore=False):
        self.value = [0, 1]
        self.events = []
        self.refuse = refuse
        self.fail_restore = fail_restore

    def cpu_affinity(self, value=None):
        if value is None:
            self.events.append(("read", tuple(self.value)))
            return list(self.value)
        self.events.append(("set", tuple(value)))
        if self.fail_restore and value == [0, 1] and self.value == [0]:
            raise OSError("restore failed")
        if not self.refuse or value != [0]:
            self.value = list(value)


def test_child_is_spawned_after_cpu0_is_effective_and_parent_is_restored():
    supervisor = FakeSupervisor()
    def create(command, **kwargs):
        assert supervisor.value == [0]
        assert command == ["python", "child.py"]
        assert kwargs == {"cwd": "repo"}
        return "child"
    assert runner.spawn_with_cpu0(create, supervisor, ["python", "child.py"], cwd="repo") == "child"
    assert supervisor.value == [0, 1]
    assert supervisor.events == [
        ("read", (0, 1)), ("set", (0,)), ("read", (0,)), ("set", (0, 1))]


def test_affinity_refusal_prevents_spawn_and_restores_parent():
    supervisor = FakeSupervisor(refuse=True)
    with pytest.raises(RuntimeError, match="did not take effect"):
        runner.spawn_with_cpu0(lambda *_args, **_kwargs: pytest.fail("spawned"),
                               supervisor, ["python", "child.py"])
    assert supervisor.value == [0, 1]


def test_spawn_error_restores_parent():
    supervisor = FakeSupervisor()
    def fail(*_args, **_kwargs):
        raise OSError("spawn failed")
    with pytest.raises(OSError, match="spawn failed"):
        runner.spawn_with_cpu0(fail, supervisor, ["python", "child.py"])
    assert supervisor.value == [0, 1]


def test_later_case_failure_keeps_its_own_solve_and_accepted_prefix():
    cases = (("case-1", np.array([1.])), ("case-2", np.array([2.])))
    capture = runner.BatchSolveCapture(cases)
    capture.solve(lambda rhs: 2 * rhs, np.array([3.]))
    capture.on_case("case-1")
    capture.solve(lambda rhs: 2 * rhs, np.array([4.]))
    assert capture.next_index == 1
    assert capture.latest["case_id"] == "case-2"
    np.testing.assert_array_equal(capture.latest["applied"], [2.])
    np.testing.assert_array_equal(capture.latest["scaled_rhs"], [4.])
    assert runner.failed_solve_diagnostic(None, "case-3", capture.latest) == {
        "available": False, "expected_case_id": "case-3",
        "captured_case_id": "case-2",
        "reason": "no solve snapshot for the failed case"}


def test_accepted_correction_keeps_primary_and_counts_each_case():
    cases = (("case-1", np.array([1.])), ("case-2", np.array([2.])))
    capture = runner.BatchSolveCapture(cases)
    capture.solve(lambda rhs: rhs + 1., np.array([3.]))
    capture.solve(lambda rhs: rhs + 2., np.array([.5]))
    assert capture.latest["raw_answer"].tolist() == [4.]
    assert capture.latest["scaled_rhs"].tolist() == [3.]
    assert capture.latest["corrective_solves"][0]["raw_answer"].tolist() == [2.5]
    capture.on_case("case-1")
    capture.solve(lambda rhs: rhs + 1., np.array([4.]))
    capture.on_case("case-2")
    assert capture.primary_solve_attempts == 2
    assert capture.corrective_solve_attempts == 1
    assert capture.completed_case_solve_counts == [
        dict(case_id="case-1", primary_solve_attempts=1, corrective_solve_attempts=1),
        dict(case_id="case-2", primary_solve_attempts=1, corrective_solve_attempts=0)]


def test_corrective_capture_is_bounded_and_keeps_latest_attempt():
    capture = runner.BatchSolveCapture((("case-1", np.array([1.])),))
    capture.solve(lambda rhs: rhs, np.array([1.]))
    for ordinal in range(1, capture.MAX_CORRECTIVE_SNAPSHOTS + 3):
        capture.solve(lambda rhs: rhs, np.array([float(ordinal)]))
    assert capture.latest["corrective_solve_count"] == capture.MAX_CORRECTIVE_SNAPSHOTS + 2
    assert len(capture.latest["corrective_solves"]) == capture.MAX_CORRECTIVE_SNAPSHOTS
    assert capture.latest["corrective_solves"][-1]["ordinal"] == capture.MAX_CORRECTIVE_SNAPSHOTS + 2
    assert capture.latest["raw_answer"].tolist() == [1.]


def test_failed_case_accurate_kkt_row_uses_unscaled_production_rhs(monkeypatch):
    from scipy import sparse
    monkeypatch.syspath_prepend(str(runner.ROOT / "src"))
    from anyfesolver._stiffened_shell_b2_q4_panel_experimental import _bounded_accurate_residual

    selector = sparse.csr_matrix(([1.], ([0], [0])), shape=(6, 1))
    constraint = sparse.csr_matrix(([1.], ([0], [0])), shape=(1, 6))
    stiffness = sparse.diags([2., 0., 0., 0., 0., 0.], format="csr")
    kkt = sparse.csr_matrix([[2., 1.], [1., 0.]])
    calls = []
    owner = SimpleNamespace(_factor_scale=np.ones(2), _ku0=np.zeros(6),
        _cu0=np.zeros(1), _T=selector, _u0=np.zeros(6), _C=constraint,
        _K=stiffness, _abs_C=abs(constraint), _abs_K=abs(stiffness),
        _abs_Ct=abs(constraint.T), _abs_Tt=abs(selector.T),
        _footprint_incidence=(), _compatibility_length=1.,
        _resolution_kkt=None, _project=lambda values: np.asarray(values)[:1])
    def build_kkt(answer, rhs, explicit_rhs):
        calls.append((answer.copy(), rhs.copy(), explicit_rhs.copy()))
        owner._resolution_kkt = kkt
        return False
    owner._explicit_load_resolved = build_kkt
    applied = np.array([1., 0., 0., 0., 0., 0.])
    capture = runner.BatchSolveCapture((("case-7", applied),))
    capture.solve(lambda _rhs: np.array([0., 1.0000002]), np.array([1., 0.]))
    with pytest.raises(RuntimeError, match="correction failed"):
        capture.solve(lambda _rhs: (_ for _ in ()).throw(RuntimeError("correction failed")),
                      np.array([.1, 0.]))
    captured = capture.latest
    result = runner.failed_solve_diagnostic(owner, "case-7", captured,
                                            accurate_residual=_bounded_accurate_residual)
    row = result["accurate_kkt_worst_equilibrium_row"]
    assert result["worst_equilibrium_dof"] == row["row"] == 0
    assert len(calls) == 1 and calls[0][1].tolist() == [1., 0.]
    assert row["available"] and row["unscaled_rhs"] == 1.
    assert row["raw_projected"] == pytest.approx(2e-7)
    assert row["signed_defect"] == pytest.approx(2e-7)
    assert row["outward_abs_upper"] >= abs(row["signed_defect"])
    assert row["unchanged_tolerance"] == result["worst_equilibrium_tolerance"]
    assert row["within_unchanged_tolerance"] is False
    assert result["diagnostic_solution"] == "primary_solve_only"
    assert result["solve_capture"]["primary_status"] == "returned"
    assert result["solve_capture"]["corrective_solve_attempts"] == 1
    assert result["solve_capture"]["last_corrective_ordinal"] == 1
    assert result["solve_capture"]["last_corrective_status"] == "raised"
    import json
    assert len(json.dumps(result)) < 3000
    owner._explicit_load_resolved = lambda *_args: pytest.fail("recomputed cached KKT")
    cached = runner.failed_solve_diagnostic(owner, "case-7", captured,
                                            accurate_residual=_bounded_accurate_residual)
    assert cached["accurate_kkt_worst_equilibrium_row"] == row


def test_created_child_restore_failure_is_distinct_and_charged(tmp_path):
    class FakeChild:
        pid = 123
        returncode = -9
    child = FakeChild()
    supervisor = FakeSupervisor(fail_restore=True)
    with pytest.raises(runner.SpawnedAffinityRestoreError) as raised:
        runner.spawn_with_cpu0(lambda *_args, **_kwargs: child,
                               supervisor, ["python", "child.py"])
    assert raised.value.process is child
    out = tmp_path / "attempt"
    out.mkdir()
    budget = tmp_path / "budget.json"
    prior = {"used_s": 5., "attempts": []}
    receipt = {"status": "started"}
    status = runner.record_spawn_failure(receipt, prior, budget, out, 1.25,
        raised.value, child_created=True, exit_code=child.returncode)
    assert status == "post_spawn_affinity_restore_failed"
    assert prior["used_s"] == 6.25
    assert prior["attempts"][0]["wall_s"] == 1.25
    assert receipt["process_wall_s"] == 1.25
    assert receipt["exit_code"] == -9
    assert budget.exists() and (out / "receipt.json").exists()


def test_post_spawn_cleanup_kills_descendants_before_accounting():
    class FakeProcess:
        pid = 123
        def __init__(self):
            self.killed = False
            self.waited = False
        def kill(self):
            self.killed = True
        def poll(self):
            return -9 if self.killed else None
        def wait(self, timeout=None):
            self.waited = True
        def children(self, recursive=False):
            assert recursive
            return [descendant]
    root = FakeProcess()
    descendant = FakeProcess()
    class FakePsutil:
        Error = OSError
        @staticmethod
        def Process(pid):
            assert pid == root.pid
            return root
    runner.terminate_process_tree(root, FakePsutil)
    assert descendant.killed and root.killed and root.waited


def test_pre_spawn_refusal_charges_no_child_time(tmp_path):
    out = tmp_path / "attempt"
    out.mkdir()
    budget = tmp_path / "budget.json"
    prior = {"used_s": 5., "attempts": []}
    receipt = {"status": "started"}
    status = runner.record_spawn_failure(receipt, prior, budget, out, .25,
        RuntimeError("no CPU0"), child_created=False)
    assert status == "spawn_refused"
    assert prior == {"used_s": 5., "attempts": []}
    assert receipt["process_wall_s"] == 0.
    assert receipt["attempt_wall_s"] == .25
    assert not budget.exists()
