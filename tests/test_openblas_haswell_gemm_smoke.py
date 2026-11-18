"""The CI runner pins OPENBLAS_CORETYPE=Haswell (scripts/run_portable_ci.py).

numpy >= 2.5.2 bundles OpenBLAS 0.3.34, whose forced-Haswell dgemm kernel
segfaults (Windows: access violation 3221225477) for matrices of roughly 175x175
or larger on AVX-512 capable CPUs (reproduced: Ryzen 9 7950X, numpy 2.5.5; numpy
<= 2.5.1 with OpenBLAS 0.3.33 is fine).  This smoke test fails closed in a
subprocess so the crash is a clear failure instead of a random one in a solver test.
"""
import os
import subprocess
import sys

_CODE = (
    "import numpy as np\n"
    "a = np.ones((400, 400)); b = np.ones((400, 400))\n"
    "for _ in range(20):\n"
    "    c = a @ b\n"
    "assert c[0, 0] == 400.0\n"
)


def test_forced_haswell_large_gemm_does_not_crash():
    env = dict(os.environ, OPENBLAS_CORETYPE="Haswell", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", _CODE], env=env, capture_output=True, text=True, timeout=120)
    import numpy

    assert result.returncode == 0, (
        f"forced-Haswell {numpy.__version__} dgemm crashed (exit {result.returncode}): numpy >= 2.5.2 bundles "
        "OpenBLAS 0.3.34 whose Haswell kernel segfaults on large matrices on AVX-512 CPUs. "
        "Fix: install numpy<2.5.2 (CI does so via .github/ci-constraints.txt / PIP_CONSTRAINT) or drop "
        "OPENBLAS_CORETYPE=Haswell. " + result.stderr[-300:]
    )
