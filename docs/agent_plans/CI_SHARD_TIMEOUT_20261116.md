# CI shard 2400 s timeouts: findings (2026-11-17)

Evidence only; no code change. Runs: 43950995536 (commit 322411bbe), 44048316984 (HEAD 4f4655ec4, still running when written; 2 failures so far), earlier 43861722899, 43188937230 (not re-read).

## What stalls
In run 43950995536, 5 jobs hit "shared 2400-second limit reached" (win 3.12 2/1, 2/3; ubuntu 3.12 3/1; 3.13 5/2; 3.14 7/3). Pool partitions were rebuilt with `scripts/run_portable_ci.py` functions at the run's commit (module counts match the logs: 15/17, 16/16, 14/16, 15/16, 15/15).
- The pytest "passed" summary comes from the *other, finished* pool (e.g. 148 passed in 407 s). The stalled pool never printed a summary: it is not "passed".
- Counting progress dots (total dots minus finished-pool tests) puts the stalled pool at ~74-103 of 134-217 tests when output stopped. The test after that point is a heavy contact test: enriched_path full364 (2/1), native_fibre_pair_static (2/3), s3_friction_replay partial_overlap (3/1), s3_q4_patch_augmented (5/2), ge_beam3_native_definition native transaction (7/3). No single module is common to all (intersection empty); the six always-repeated S3 successor modules sit before the stall point.
- Hence not a deadlock in one module: it is a heavily imbalanced pool. Finished pool 2-7 min, other pool >40 min (up to 6x).

## Why only CI
- `_module_weight` balances by file size (plus few overrides), not runtime. The `general_contact_*` modules are small files with multi-minute tests.
- Measured locally (py3.13 venv, cold numba cache, scale 8, OMP=1): `test_general_contact_native_fibre_pair_static.py` alone = 723 s (staged 200 s, weak_normal 198 s, actual 121 s, refined 114 s). Commit 6f3a37233 recorded 488 s for one of them on a shared runner (~2.4x slower than here). CI passing jobs of run 44048316984 show 100-225 s tests (curved_s3_multifacet 225 s, coupled_tool_staging 185 s).
- Deadline scale 8 does not wait: guards raise when exceeded, so it only turns "fails at 480 s" into "passes slowly"; tests that were failing before (hiding the timeout) now run to completion and exceed the pool budget. Scale 4 -> 8 therefore unmasked, not caused, the limit.
- Not found: pipe backpressure (workers inherit stdout), numba/numpy version mismatch (local venv numba present, numpy 2.5.5).

## Proposal (not implemented: needs CI to prove before/after)
1. Add time-based `PORTABLE_MODULE_WEIGHT_OVERRIDES` for the multi-minute contact modules (fibre_pair_static, curved_s3_multifacet, coupled_tool_staging, curved_pair_lifecycle, enriched_path, s3_friction_replay, s3_q4_patch_augmented), from measured durations; the existing mechanism, no test changes.
2. Or raise `--matrix-subshard-count` 4 -> 8 for these OS/python cells, or lift the 2400 s limit for the job (cost ~ minutes). A unit check can assert the max pool weight ratio per shard.
3. Add `--durations` per module summary (`junit`/`--durations=0` file) to learn true module times from CI.

## Second item: displacement-hash test
In run 43950995536 it failed on ubuntu-latest 3.13 shard 5/0 (not Windows): `test_protected_linear_and_corotational_displacement_hashes` ('linear', 8) got 3f1deac1... instead of 0be6e35e.... Not yet failed in 44048316984. Same family in that run: `test_native_fibre_enforcement_successive_response[k8e7]` (ubuntu 3.14 7/0) and an exact-bytes replay mismatch (win 3.11 0/1) differ only in last digits (e.g. 0.005414355840998849 vs ...8505, ~1 ulp): runner CPU/BLAS bitwise differences (FMA/AVX dispatch) are the likely cause. The pin was not weakened. Options: pin per-runner CPU flags (`OPENBLAS_CORETYPE=...`, NUMBA_CPU_NAME=generic), or restrict bitwise pins to a reference runner class with a tolerance fallback elsewhere (owner decision).

## Not run
Full pool/shard reproduction locally; per-module local timings of all stalled-pool modules; reading 43861722899/43188937230 logs; the hash test locally.

## Implemented (follow-up, owner-approved)
- `scripts/run_portable_ci.py`: weights are predicted CI milliseconds. `PORTABLE_MODULE_CI_SECONDS` (86 modules >= 60 s) = per module, sum over tests of the median CI duration (call/setup/teardown separately) in the "slowest 20 durations" blocks of runs 44048316984, 43950995536, 43861722899 (96 pytest job logs; lower bound, since only each pool's 20 slowest tests appear). Unmeasured modules: 2 s + bytes/50 kB. Only assignment changes; no test selection change.
- Predicted max pool seconds per cell, old -> new weights (new time model): worst 1891 (6,3) / 1782 (4,3) / 1495 (2,3) -> 1432 (0,0). Worst makespan/lower-bound 1.66 -> 1.05.
- Remaining risk: indivisible heavy modules (s3_q4_patch_augmented, q4_s3_patch_static, d1_interior_profile, fibre_pair_static, ...) alone fill a pool (1.0-1.4 ks median); single tests vary 3x between runs (q4_s3_patch_static::...four_phase_path_and_restart: 536/1293/1756 s). Splitting such modules by node is the next option.
- CI records per-module seconds (junit -> `portable-ci-module-durations.json` artifact + step summary).
- Test: `tests/test_portable_ci_partition_balance.py`.

## Hash test `test_protected_linear_and_corotational_displacement_hashes` (item 5)
- Failures with the same wrong hash 3f1deac1... (expected 0be6e35e..., case ('linear', 8)): run 43861722899 windows 3.13 shard 4/1 and run 43950995536 ubuntu 3.13 shard 5/0. The module runs only on one shard per OS/Python; both failures are Python 3.13. Run 44048316984 windows 3.13 passed (cell with the module at that commit: 5/2, after other predecessors; ubuntu 3.13 not in its failure list).
- Local, py3.13 venv, CI environment (scale 8, OMP=1, NUMBA threads 1): alone PASS (183 s); in the exact failing pool of run 43950995536 (12 modules incl. v6g and fast_total before it, in pool order) PASS (the test took 289 s there). Not reproduced; no order dependence found locally. No env-var or global-state mutation found in the predecessor modules (grep).
- Evidence leans against runner CPU noise (same value on two OSes) and for a deterministic alternative path that is Python 3.13 / dependency-version dependent on CI (e.g. a different SciPy/BLAS or solver code path chosen in the CI installation), or state left by modules that precede it in the pool in CI only (run 43861722899 pool preceded by v2d_linear_native_parity, v2d_native_state_corotational, v6e/v6g/v6t, ...).
- Options (pin untouched): (a) run the test in a fresh interpreter (add to PORTABLE_ISOLATED_MODULES) to separate order effects; (b) print scipy/numpy/BLAS and `solve_linear` backend info in this test's failure message and CI job log to identify the alternative path; (c) in CI, run only this test alone on 3.13 once to classify order vs environment. Weights now change pool order, so a rerun will also show whether the failure moves with its predecessors.

## Bitwise mismatches: cause established (CPU baseline experiment, 2026-11-17)
Reading: in (a) `test_native_fibre_enforcement_successive_response[k8e7]` and (b) `test_curved_material_authentic_split_restart_and_atomic_failure`, `stored` is a committed reference checkpoint (`tests/data/*.checkpoint`, written on the author's machine) and `replayed` is computed now (`native_multi_s3_static.py:819`). The hash test compares against stored pre-R1 constants. So all three are cross-machine bitwise comparisons, not in-process nondeterminism.

Machine here: AMD Zen4 (numba host cpu znver4), AVX-512, numpy 2.5.5, numba 0.68.1, scipy 1.18.1 (bundled scipy-openblas 0.3.34, `architecture: SkylakeX`), all CI env as in the runner (OMP/OPENBLAS threads 1). Each case run alone, same cold numba cache:

| setting | (a) | (b) | hash test |
|---|---|---|---|
| default (SkylakeX kernels) | pass 40 s | pass 131 s | pass 295 s |
| OPENBLAS_CORETYPE=Haswell | FAIL, identical to CI (stored 9.123885834601848e-07, replayed ...846e-07) | FAIL | FAIL, hash 3f1deac1... = the CI hash |
| OPENBLAS_CORETYPE=Zen | not run | not run | FAIL, same 3f1deac1... |
| NPY_DISABLE_CPU_FEATURES=AVX512* | pass | pass | not run |
| NUMBA_CPU_NAME=generic | pass 39 s | pass 132 s | pass 281 s |
| NUMBA_CPU_NAME=haswell | pass 36 s | pass 126 s | not run |
| Haswell + no AVX512 + numba haswell | FAIL | FAIL | FAIL (15 s) |

Conclusion: the references were produced with OpenBLAS's AVX-512 (SkylakeX) kernels; CI runners (no AVX-512 or different kernel choice: Haswell/Zen kernels) produce 1-ulp-different BLAS results. This reproduces the exact CI values for all three tests and explains the same wrong hash on two OSes. numpy SIMD and numba code generation are not involved (no effect); `MKL_ENABLE_INSTRUCTIONS` is a no-op (wheels use OpenBLAS). Cost of the numba/numpy variables: none measurable (+-5 %); OPENBLAS_CORETYPE=Haswell: none measurable.

No runner-environment baseline can make CI match the references: forcing `OPENBLAS_CORETYPE=SkylakeX` crashes on runners without AVX-512. Not implemented (and no pin/tolerance changed). Options for the owner:
1. Regenerate the affected references (replay checkpoints, hash constants) once under a pinned kernel `OPENBLAS_CORETYPE=Haswell` (valid on every AVX2 x86-64 runner and on AVX-512 machines) and set the same variable in the runner worker environment and in the generation instructions. Gives bitwise cross-machine reproducibility; re-pin = owner decision.
2. Make the replay-critical path BLAS-independent (compensated/pure NumPy reductions): product change, preserves pins but changes code.
3. Contract change: replay equality up to a documented ulp tolerance for non-identity leaves, hash tests replaced by tolerance band + exact hash on a pinned-kernel lane.
Proof run: shards containing the tests are 7/0 (a), 0/1 (b) and the hash test's shard (changes with the new weights; compute with `scripts/run_portable_ci.py`'s partition: 5/2 at HEAD); with option 1 each must pass on windows and ubuntu.
