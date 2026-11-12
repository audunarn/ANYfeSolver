"""Opt-in pytest record for focused source development; never artifact acceptance."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def runtime_record(modules, source, pools=None):
    errors = []
    imported = {name: str(Path(module.__file__).resolve()) for name, module in modules.items()
                if (name == "anyfesolver" or name.startswith("anyfesolver.")) and getattr(module, "__file__", None)}
    if any(not Path(path).is_relative_to(source) for path in imported.values()):
        errors.append("ANYfeSolver imports escaped the selected source checkout")
    numba = modules.get("numba")
    actual_jit = bool(numba.config.DISABLE_JIT) if numba is not None else None
    actual_threads = int(numba.get_num_threads()) if numba is not None else None
    if numba is not None and (actual_threads != 1 or actual_jit != (os.environ["NUMBA_DISABLE_JIT"] == "1")):
        errors.append("effective Numba settings differ from requested focused mode")
    if pools is not None and any(pool["num_threads"] != 1 for pool in pools):
        errors.append("effective numerical pools are not single-threaded")
    return {"scope": "source development only", "imports": imported, "effective_pools": pools,
            "numba_threads": actual_threads, "jit_disabled": actual_jit, "errors": errors,
            "native_deadline_scale": os.environ.get("ANYSOLVER_GE_BEAM3_DEADLINE_SCALE"),
            "plugin_autoload_disabled": os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"}


def pytest_sessionfinish(session, exitstatus):
    pools = None
    if "numpy" in sys.modules or "scipy" in sys.modules:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    record = runtime_record(dict(sys.modules), Path(os.environ["ANYSOLVER_DEVELOPMENT_SOURCE"]).resolve(), pools)
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    record.update(collected=session.testscollected, failed=session.testsfailed, pytest_exit_code=int(exitstatus),
                  outcomes={name: len(rows) for name, rows in reporter.stats.items()
                            if name in ("passed", "failed", "skipped", "error", "xfailed", "xpassed", "deselected")}
                  if reporter is not None else {})
    if record["errors"] and int(exitstatus) == 0:
        session.exitstatus = 1
    record["exit_code"] = int(session.exitstatus)
    Path(os.environ["ANYSOLVER_DEVELOPMENT_RECORD"]).write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    for error in record["errors"]:
        if reporter is not None:
            reporter.write_line("[development runtime failure] "+error)
