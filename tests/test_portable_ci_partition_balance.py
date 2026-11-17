"""Pool balance of the portable CI matrix under predicted CI seconds.

Weights (``scripts.run_portable_ci._module_weight``, milliseconds) are the
predicted CI time of a whole module, derived from measured GitHub-runner
durations.  They only decide assignment; this test also proves that every
manifest module is still assigned exactly once.
"""
import pytest

from scripts import run_portable_ci as runner

SHARDS, SUBSHARDS, WORKERS = 8, 4, 2
# A pool's predicted time may exceed the best conceivable makespan
# L = max(heaviest module, half the cell total) by at most 25 %, and must stay
# at or below 1800 s (75 % of the 2400 s shared limit) unless one single module
# alone is heavier: a module is never split, so such a pool is the module itself
# (documented in docs/agent_plans/CI_SHARD_TIMEOUT_20261116.md).  The old
# size-based weights gave ratios up to 1.66 and pools of 1782-1891 s.
MAKESPAN_BOUND = 1.25
MAX_POOL_SECONDS = 1800.0


def _seconds(module):
    return runner._module_weight(module) / 1000.0


@pytest.fixture(scope='module')
def cells():
    manifest = runner.merge_test_modules()
    result = {}
    for shard in range(SHARDS):
        outer = runner.matrix_modules(manifest, shard, SHARDS)
        for sub in range(SUBSHARDS):
            selected = runner.matrix_modules(outer, sub, SUBSHARDS)
            result[(shard, sub)] = (selected, runner.execution_partitions(selected, WORKERS))
    return manifest, result


def test_every_manifest_module_is_assigned_exactly_once(cells):
    manifest, result = cells
    critical = set(runner.PORTABLE_CURRENT_S3_SUCCESSOR_MODULES)
    general = [m for m in manifest if m not in critical]
    seen = []
    for (shard, sub), (selected, partitions) in result.items():
        pooled = [m for pool in partitions for m in pool]
        assert len(pooled) == len(set(pooled)) and set(pooled) == set(selected)
        assert critical <= set(selected)
        seen.extend(m for m in selected if m not in critical)
    assert sorted(seen) == sorted(general)


def test_pools_are_balanced_and_far_below_the_shared_limit(cells):
    _, result = cells
    isolated = set(runner.PORTABLE_ISOLATED_MODULES)
    failures = []
    for cell, (_, partitions) in sorted(result.items()):
        ordinary = [p for p in partitions if not (len(p) == 1 and p[0] in isolated)]
        times = [sum(_seconds(m) for m in p) for p in ordinary]
        modules = [_seconds(m) for p in ordinary for m in p]
        lower_bound = max(max(modules), sum(modules) / len(ordinary))
        if max(times) > MAKESPAN_BOUND * lower_bound or max(times) > max(MAX_POOL_SECONDS, max(modules)):
            failures.append((cell, round(max(times)), round(lower_bound)))
    assert not failures


def test_measured_modules_exist_and_weights_follow_measurements():
    for module, seconds in runner.PORTABLE_MODULE_CI_SECONDS.items():
        assert (runner.ROOT / module).is_file(), module
        assert runner._module_weight(module) == round(1000 * seconds)


def test_module_duration_summary_is_cheap_and_never_fails(tmp_path, monkeypatch, capsys):
    pool = tmp_path / 'P01'
    pool.mkdir()
    (pool / 'junit.xml').write_text(
        '<testsuites><testsuite>'
        '<testcase classname="tests.test_a" name="t1" time="1.5"/>'
        '<testcase classname="tests.test_a.TestC" name="t2" time="2.0"/>'
        '<testcase classname="tests.test_b" name="t3" time="0.5"/>'
        '</testsuite></testsuites>', encoding='utf-8')
    summary = tmp_path / 'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    runner._record_module_durations(tmp_path, tmp_path)
    import json
    record = json.loads((tmp_path / 'portable-ci-module-durations.json').read_text())
    assert record == {'tests/test_a.py': 3.5, 'tests/test_b.py': 0.5}
    assert 'tests/test_a.py | 3.5' in summary.read_text()
    (pool / 'junit.xml').write_text('<broken', encoding='utf-8')
    runner._record_module_durations(tmp_path, tmp_path)
    assert 'skipped' in capsys.readouterr().out
