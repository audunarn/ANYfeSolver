"""CI coverage of both batched Q4 paths: the batched fast path and the forced per-element fallback.

Each case is solved and recovered twice in this process, once with the batched
numerics available and once inside ``_forced_fallback()`` (the per-element path).
The displacement vector and the complete ``compute_stresses`` result must be
bit-identical (dtype, shape and bytes), and the batch must have run only in the
fast path.
"""
import hashlib
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4_batch_cases as cases  # noqa: E402

from anyfesolver import _q4_planar_batch as B  # noqa: E402
from anyfesolver import solve_linear  # noqa: E402
from anyfesolver.assembly import compute_stresses  # noqa: E402

# Sizes above both batch minimum row counts (96 formation, 128 recovery Q4 rows).
CASES = [("plate", 12), ("sl", 12), ("pinch", 12), ("rplate", 12),
         ("mplate", 16), ("msl", 16), ("stiff", 12)]


def _run(case, n, monkeypatch, forced):
    rows = {"form": 0, "recover": 0}
    form, recover = B.form, B.recover

    def counting_form(*args, **kwargs):
        rows["form"] += args[0].shape[0]
        return form(*args, **kwargs)

    def counting_recover(*args, **kwargs):
        rows["recover"] += args[0].shape[0]
        return recover(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(B, "form", counting_form)
        patch.setattr(B, "recover", counting_recover)
        model, load = cases.CASES[case](n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if forced:
                with B._forced_fallback():
                    u, _ = solve_linear(model, load)
                    stresses = compute_stresses(model, u)
            else:
                u, _ = solve_linear(model, load)
                stresses = compute_stresses(model, u)
    hu, hs = hashlib.sha256(), hashlib.sha256()
    cases.feed(hu, np.asarray(u))
    cases.feed(hs, stresses)
    return hu.hexdigest(), hs.hexdigest(), rows


def test_batch_available_on_ci():
    assert B.available() is True
    with B._forced_fallback():
        assert B.available() is False
    assert B.available() is True


@pytest.mark.parametrize("case,n", CASES)
def test_fast_path_equals_forced_fallback(case, n, monkeypatch):
    fast = _run(case, n, monkeypatch, forced=False)
    fallback = _run(case, n, monkeypatch, forced=True)
    assert fast[:2] == fallback[:2]
    assert fallback[2] == {"form": 0, "recover": 0}
    assert fast[2]["recover"] > 0
    if case == "rplate":
        # Distinct geometries: every element misses the shared-geometry cache.
        assert fast[2]["form"] == n * n
