from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from scripts import development_test_runtime as runtime
from scripts import run_portable_ci as runner


def test_portable_native_deadline_headroom_is_fixed_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv('ANYSOLVER_GE_BEAM3_DEADLINE_SCALE', '1000')
    environment = runner._worker_environment(tmp_path)
    assert environment['ANYSOLVER_GE_BEAM3_DEADLINE_SCALE'] == '4'
    assert environment['OPENBLAS_NUM_THREADS'] == '1'
    assert environment['MKL_NUM_THREADS'] == '1'
    # Preparing CI execution settings does not mutate the calling application.
    assert __import__('os').environ['ANYSOLVER_GE_BEAM3_DEADLINE_SCALE'] == '1000'


def test_explicit_selection_preserves_node_and_rejects_empty_missing_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    (tmp_path/'tests').mkdir()
    (tmp_path/'tests/test_x.py').write_text('')
    assert runner._development_targets(['tests/test_x.py::test_x[units-m]']) == ('tests/test_x.py::test_x[units-m]',)
    for targets in ([], ['tests/test_x.py::'], ['tests/test_x.py']*2, ['-k']):
        with pytest.raises(ValueError): runner._development_targets(targets)
    with pytest.raises(FileNotFoundError): runner._development_targets(['tests/test_missing.py'])
    (tmp_path/'other.py').write_text('')
    with pytest.raises(ValueError): runner._development_targets(['other.py'])


def test_development_does_not_collect_or_deselect_ci_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path/'temp'))
    (tmp_path/'tests').mkdir()
    (tmp_path/'tests/test_x.py').write_text('')
    def forbidden(*args, **kwargs): raise AssertionError('broad inventory/partition reached')
    monkeypatch.setattr(runner, 'merge_test_modules', forbidden)
    monkeypatch.setattr(runner, 'execution_partitions', forbidden)
    commands = []
    def launch(command, *, cwd, env, **kwargs):
        commands.append(command)
        assert cwd == tmp_path and env['NUMBA_DISABLE_JIT'] == '1'
        assert env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
        Path(env['ANYSOLVER_DEVELOPMENT_RECORD']).write_text('{}')
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    assert runner.run(workers=8, timeout_seconds=1, development_targets=['tests/test_x.py::test_x']) == 0
    assert len(commands) == 1 and commands[0][-1] == 'tests/test_x.py::test_x'
    assert not any(c.startswith('--deselect') for c in commands[0])


def test_missing_runtime_record_cannot_report_success(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path/'temp'))
    (tmp_path/'tests').mkdir()
    (tmp_path/'tests/test_x.py').write_text('')
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs:
                        SimpleNamespace(poll=lambda: 0, returncode=0))
    assert runner.run(workers=1, timeout_seconds=1, development_targets=['tests/test_x.py']) == 1


@pytest.mark.parametrize('code', [1, 2, 4, 5])
def test_child_non_success_is_never_accepted(tmp_path, monkeypatch, code):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path/'temp'))
    (tmp_path/'tests').mkdir()
    (tmp_path/'tests/test_x.py').write_text('')
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs:
                        SimpleNamespace(poll=lambda: code, returncode=code))
    assert runner.run(workers=1, timeout_seconds=1, development_targets=['tests/test_x.py']) == 1


def test_runtime_reports_observed_settings_and_rejects_wrong_source(tmp_path, monkeypatch):
    monkeypatch.setenv('NUMBA_DISABLE_JIT', '1')
    numba = SimpleNamespace(config=SimpleNamespace(DISABLE_JIT=True), get_num_threads=lambda: 1)
    modules = {'anyfesolver': SimpleNamespace(__file__=str(tmp_path/'__init__.py')), 'numba': numba}
    result = runtime.runtime_record(modules, tmp_path, [{'num_threads': 1}])
    assert not result['errors'] and result['jit_disabled'] is True and result['numba_threads'] == 1
    assert runtime.runtime_record(modules, tmp_path/'other')['errors']
    assert runtime.runtime_record(modules, tmp_path, [{'num_threads': 2}])['errors']
    numba.config.DISABLE_JIT = False
    assert runtime.runtime_record(modules, tmp_path)['errors']


def test_focused_mode_cannot_replace_matrix_or_set_ci_runtime():
    with pytest.raises(ValueError):
        runner.run(workers=1, timeout_seconds=1, development_targets=[], matrix_shard_index=0)
    with pytest.raises(ValueError): runner.run(workers=1, timeout_seconds=1, jit='off')


def test_development_deadline_terminates_owned_process(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path/'temp'))
    (tmp_path/'tests').mkdir()
    (tmp_path/'tests/test_x.py').write_text('')
    process = SimpleNamespace(poll=lambda: None, returncode=-1)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: process)
    ticks = iter((0., 0., 2., 2.))
    monkeypatch.setattr(runner.time, 'monotonic', lambda: next(ticks))
    terminated = []
    monkeypatch.setattr(runner, '_terminate_tree', terminated.append)
    assert runner.run(workers=1, timeout_seconds=1, development_targets=['tests/test_x.py']) == runner.TIMEOUT_EXIT_CODE
    assert terminated == [process]
    record = json.loads(next((tmp_path/'temp').glob('*/P01/resource.json')).read_text())
    assert record['status'] == 'timeout' and record['child_exit_codes'] == [-1]


def test_nested_matrix_preserves_parent_allocation_and_complete_coverage(monkeypatch):
    critical = tuple(runner.PORTABLE_CURRENT_S3_SUCCESSOR_MODULES)
    modules = (*critical, *(f"tests/test_general_{i}.py" for i in range(64)))
    monkeypatch.setattr(runner, '_module_weight', lambda module: 1)
    general = set(modules) - set(critical)
    all_general = []
    for parent in range(8):
        outer = runner.matrix_modules(modules, parent, 8)
        assigned = []
        for sub in range(4):
            selected = runner.matrix_modules(outer, sub, 4)
            assert set(critical) <= set(selected)
            assert set(selected) <= set(outer)
            assigned.extend(m for m in selected if m in general)
        assert len(assigned) == len(set(assigned))
        assert set(assigned) == set(outer) - set(critical)
        all_general.extend(assigned)
    assert len(all_general) == len(set(all_general)) == len(general)
    assert set(all_general) == general


@pytest.mark.parametrize('parent,inner', [
    ((None,None),(0,4)), ((0,8),(None,4)), ((0,8),(0,None)),
    ((0,8),(-1,4)), ((0,8),(4,4)), ((0,8),(0,0)),
    ((0,8),(True,4)), ((0,8),(0,True))])
def test_nested_matrix_rejects_invalid_configuration(monkeypatch, parent, inner):
    monkeypatch.setattr(runner, 'merge_test_modules',
                        lambda: (*runner.PORTABLE_CURRENT_S3_SUCCESSOR_MODULES,
                                 *(f'tests/test_general_{i}.py' for i in range(64))))
    monkeypatch.setattr(runner, '_module_weight', lambda module: 1)
    with pytest.raises(ValueError, match='matrix'):
        runner.run(workers=2, timeout_seconds=1,
                   matrix_shard_index=parent[0], matrix_shard_count=parent[1],
                   matrix_subshard_index=inner[0], matrix_subshard_count=inner[1])


def test_development_refuses_nested_ci_selection():
    with pytest.raises(ValueError, match='focused development'):
        runner.run(workers=2, timeout_seconds=1, development_targets=[],
                   matrix_subshard_index=0, matrix_subshard_count=4)


@pytest.mark.parametrize('failure', [False, True])
def test_ci_full_success_or_failed_child_stops_live_sibling(tmp_path, monkeypatch, failure):
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    monkeypatch.setattr(runner, 'merge_test_modules', lambda: ('a.py','b.py'))
    monkeypatch.setattr(runner, 'execution_partitions', lambda *args: (('a.py',),('b.py',)))
    launched = []
    terminated = []
    def launch(*args, **kwargs):
        index = len(launched)
        values = iter((None, 1) if failure and index == 0 else
                      (None, None) if failure else (None, 0))
        process = SimpleNamespace(returncode=None)
        def poll():
            process.returncode = next(values, process.returncode)
            return process.returncode
        process.poll = poll
        launched.append(process)
        return process
    def terminate(process):
        if process.returncode is None:
            terminated.append(process)
            process.returncode = -15
    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    monkeypatch.setattr(runner, '_terminate_tree', terminate)
    assert runner.run(workers=2, timeout_seconds=1) == int(failure)
    assert len(launched) == 2
    assert terminated == ([launched[1]] if failure else [])


def test_ci_does_not_launch_more_work_after_immediate_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    monkeypatch.setattr(runner, 'merge_test_modules', lambda: ('a.py','b.py'))
    monkeypatch.setattr(runner, 'execution_partitions', lambda *args: (('a.py',),('b.py',)))
    launched = []
    def launch(*args, **kwargs):
        process = SimpleNamespace(poll=lambda: 1, returncode=1)
        launched.append(process)
        return process
    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    monkeypatch.setattr(runner, '_terminate_tree', lambda process: None)
    assert runner.run(workers=2, timeout_seconds=1) == 1
    assert len(launched) == 1


def test_ci_worker_stops_at_first_failure(tmp_path):
    assert '-x' in runner._worker_command(('tests/test_x.py',), tmp_path)
