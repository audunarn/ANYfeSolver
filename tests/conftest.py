"""Shared pytest configuration.

PyCharm may launch pytest with ``tests`` as the process working directory,
while several integration tests intentionally execute repository scripts by
relative path.  Normalize the working directory once for the full session so
those tests behave identically from PyCharm, PowerShell and CI.
"""

from __future__ import annotations

import os
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_REPOSITORY_ROOT)


import pytest

import sys

if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts import retirement_policy


def pytest_configure(config):
    """Refuse explicit retired selectors before any collection or import.

    pytest 9 does not consult ``pytest_ignore_collect`` for explicit initial
    paths, and dropping items only in ``pytest_collection_modifyitems``
    would import the retired modules first.  Scan the invocation selectors
    here, resolved against the ORIGINAL invocation directory (this conftest
    normalizes the process working directory to the repository root at
    import time), and refuse the run before collection starts.  Ordinary
    directory collection stays with ``pytest_ignore_collect``; retired nodes
    inside active modules stay with the deselection hook below.
    """
    config.addinivalue_line(
        "markers",
        "slow: single test far longer than any ordinary CI test; skipped unless "
        "--run-slow / ANYSOLVER_RUN_SLOW=1 (runs in the separate slow-tests workflow)",
    )
    if retirement_policy.opt_in_enabled():
        return
    refused = retirement_policy.retired_explicit_selectors(
        config.args, config.invocation_params.dir
    )
    if refused:
        report = retirement_policy.retirement_report()
        raise pytest.UsageError(
            "refusing to run retired programme selector(s): "
            + ", ".join(refused)
            + f" (policy {report['policy']}; explicit future-use opt-in: "
            + f"{report['opt_in_env']}=1)"
        )

@pytest.fixture(scope="module")
def f03_q8_sided_faces():
    """Manufactured face geometry for bounded algebra, not external evidence.

    Four by eight serendipity cells on each face, with distinct plate/bar
    ownership even at their coincident interface. No saved deck, solver output,
    physical supports or historical qualification files are required.
    """
    import numpy as np

    coords = {}
    domains = []
    for heights in ((-.0175, .0175), (.0175, .1175)):
        domain = set()
        for z in heights:
            for i, x in enumerate(np.linspace(.9, 1., 9)):
                for j, y in enumerate(np.linspace(-.05, .05, 17)):
                    if i % 2 and j % 2:  # Q8 omits cell-centre nodes.
                        continue
                    node = len(coords) + 1
                    coords[node] = np.array((x, y, z))
                    domain.add(node)
        domains.append(domain)
    assert len(coords) == 484 and not domains[0] & domains[1]
    return coords, *domains


def pytest_ignore_collect(config, collection_path):
    """Keep retired programme modules out of ordinary collection.

    Returning True prevents collection (and therefore import) of the
    retired experimental stiffened-shell / GE beam-shell programme during
    default directory collection.  Explicit command-line paths bypass
    this hook, so pytest_collection_modifyitems below is the second,
    node-level guard.
    """
    if retirement_policy.opt_in_enabled():
        return None
    relative = retirement_policy.relative_path(collection_path)
    if relative is None:
        return None
    return retirement_policy.is_retired_module(relative)


SLOW_ENV = "ANYSOLVER_RUN_SLOW"


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="also run tests marked slow (opt-in tier; see .github/workflows/slow-tests.yml)",
    )


def _slow_enabled(config):
    return bool(config.getoption("--run-slow")) or os.environ.get(SLOW_ENV) == "1"


def pytest_collection_modifyitems(config, items):
    """Skip opt-in slow tests by default, then deselect retired programme items collected via explicit paths.

    Retired nodes inside otherwise active modules are removed here with
    truthful reporting through the pytest deselection hook; their sibling
    nodes keep running.  Whole retired modules never reach this point:
    directory collection ignores them before import and explicit selectors
    are refused in ``pytest_configure``.
    """
    if not _slow_enabled(config):
        skip_slow = pytest.mark.skip(
            reason="opt-in slow tier: run with --run-slow or ANYSOLVER_RUN_SLOW=1"
        )
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip_slow)
    if retirement_policy.opt_in_enabled():
        return
    kept = []
    dropped = []
    for item in items:
        if retirement_policy.is_retired_collected_node(item.path, item.nodeid):
            dropped.append(item)
        else:
            kept.append(item)
    if dropped:
        items[:] = kept
        config.hook.pytest_deselected(items=dropped)
        reporter = config.pluginmanager.getplugin("terminalreporter")
        if reporter is not None:
            report = retirement_policy.retirement_report()
            reporter.write_line(
                f"retired programme: deselected {len(dropped)} collected item(s) "
                f"(policy {report['policy']}; explicit future-use opt-in: "
                f"{report['opt_in_env']}=1)"
            )
