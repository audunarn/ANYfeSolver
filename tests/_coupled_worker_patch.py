"""Test helper (owner decision OD14): run the real certificate worker with test code executed inside the
child after its arithmetic module is loaded, e.g. to inject singular values or a MemoryError there.
Only the test's child is changed; production code is not patched."""

from __future__ import annotations

import sys
import textwrap


def patch_worker(monkeypatch, cc, code: str) -> None:
    """Replace the worker command for this test by the real worker whose ``_load_math`` also runs
    ``code`` (names available: ``m`` the arithmetic module, ``np``, ``w`` the worker module)."""
    original = cc._svd_worker_command()
    worker_path, arguments = original[3], original[4:]
    script = textwrap.dedent('''
        import sys
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location('_worker_under_test', {path!r})
        w = module_from_spec(spec)
        spec.loader.exec_module(w)
        load = w._load_math

        def patched():
            m = load()
            import numpy as np
            exec({code!r}, {{'m': m, 'np': np, 'w': w}})
            return m
        w._load_math = patched
        sys.exit(w._main([{path!r}] + {arguments!r}))
    ''').format(path=worker_path, code=textwrap.dedent(code), arguments=list(arguments))
    monkeypatch.setattr(cc, '_svd_worker_command', lambda: [sys.executable, '-I', '-S', '-c', script])
