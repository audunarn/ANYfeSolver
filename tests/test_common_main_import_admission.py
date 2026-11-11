"""The combined public entry points retain ordinary load-vector admission."""
def test_combined_public_namespaces_preserve_builtin_gravity():
    import anyfesolver
    from anyfesolver import stiffened_shell, general_contact
    from test_static_batch_coupled_acceleration_guards import _model, _gravity, _owned, _public
    assert stiffened_shell.create_analysis is not None
    assert general_contact.report_general_contact_capability is not None
    assert anyfesolver.iter_linear_batches is not None
    model = _model()
    case = _gravity()
    owned = _owned(model, case)
    assert owned is not None
    assert owned.tobytes() == _public(model, case).tobytes()


def test_foreign_warning_facade_is_refused_without_observation():
    from anyfesolver import boundary
    import warnings
    import pytest
    if boundary._WARNING_IMPLEMENTATION is warnings:
        pytest.skip('This stdlib does not split its warnings implementation')
    class Foreign:
        def __getattribute__(self, name):
            raise AssertionError('Foreign warning facade must not be observed')
    namespace = vars(boundary._WARNING_IMPLEMENTATION)
    original = namespace['_wm']
    namespace['_wm'] = Foreign()
    try:
        assert not boundary._numpy_error_routes_are_builtin()
    finally:
        namespace['_wm'] = original


def test_foreign_warning_context_getter_is_refused(monkeypatch):
    from anyfesolver import boundary
    import warnings
    import pytest
    if boundary._WARNING_IMPLEMENTATION is warnings:
        pytest.skip('This stdlib does not split its warnings implementation')
    seen = []
    monkeypatch.setattr(warnings, '_get_context', lambda: seen.append('foreign'))
    assert not boundary._numpy_error_routes_are_builtin()
    assert seen == []


def test_missing_or_foreign_warning_context_is_refused(monkeypatch):
    from anyfesolver import boundary
    import warnings
    import pytest
    if boundary._WARNING_IMPLEMENTATION is warnings:
        pytest.skip('This stdlib does not split its warnings implementation')
    namespace = vars(boundary._WARNING_IMPLEMENTATION)
    original = namespace.pop('_wm')
    try:
        assert not boundary._numpy_error_routes_are_builtin()
    finally:
        namespace['_wm'] = original
    monkeypatch.setattr(boundary._WARNING_IMPLEMENTATION, '_global_context', object())
    assert not boundary._numpy_error_routes_are_builtin()


def test_preimport_warning_context_getter_cannot_authorize_itself():
    import subprocess, sys
    if sys.version_info < (3, 14):
        import pytest
        pytest.skip('This stdlib does not split its warnings implementation')
    code = '''import warnings
seen = []
original = warnings._get_context
def foreign():
    seen.append('foreign')
    return original()
warnings._get_context = foreign
import anyfesolver.boundary as boundary
seen.clear()
assert not boundary._numpy_error_routes_are_builtin()
assert not seen
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ordinary_warning_writer_and_context_refusal(monkeypatch, tmp_path):
    import io, sys, warnings
    from anyfesolver import boundary
    if boundary._WARNING_IMPLEMENTATION is warnings:
        import pytest
        pytest.skip('This stdlib does not split its warnings implementation')
    original_impl = next(function for name, function, _ in boundary._WARNING_FUNCTIONS
                         if name == '_showwarnmsg_impl')
    with io.TextIOWrapper(io.FileIO(tmp_path / 'warnings.log', 'w'),
                         encoding='utf8', errors='backslashreplace') as stream:
        monkeypatch.setattr(sys, 'stderr', stream)
        monkeypatch.setattr(warnings, '_showwarnmsg_impl', original_impl)
        assert boundary._numpy_error_routes_are_builtin()
        monkeypatch.setattr(boundary._WARNING_IMPLEMENTATION, '_use_context', 1)
        assert not boundary._numpy_error_routes_are_builtin()


def test_preimport_warning_context_descriptor_is_not_authorized():
    import subprocess, sys
    if sys.version_info < (3, 14):
        import pytest
        pytest.skip('This stdlib does not split its warnings implementation')
    code = '''import _py_warnings
seen = []
_py_warnings._GlobalContext.log = property(lambda self: seen.append('foreign'))
import anyfesolver.boundary as boundary
seen.clear()
assert not boundary._numpy_error_routes_are_builtin()
assert not seen
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
