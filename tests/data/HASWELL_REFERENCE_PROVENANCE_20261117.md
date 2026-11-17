# Haswell-kernel reference provenance (2026-11-17)

Why: references were written with OpenBLAS SkylakeX (AVX-512) kernels; AVX2 CI runners use Haswell/Zen kernels
and differ in the last bits. CI/runner now pins `OPENBLAS_CORETYPE=Haswell` (scripts/run_portable_ci.py).
Platform: Windows 11, AMD Zen4 host, Python 3.13 venv, numpy 2.5.5, scipy 1.18.1 (scipy-openblas 0.3.34), numba 0.68.1.
Env: OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
ANYSOLVER_GE_BEAM3_DEADLINE_SCALE=8 PYTHONHASHSEED=0. Helper scripts: reseal.py, gen.py, hashgen.py, cmp.py (kept with the session notes).

## Regenerated
`general_contact_native_fibre_s3_plastic_certified_k4e7_v1.checkpoint` ("re-seal"): the stored accepted state
(mechanical, shell displacement) is replayed unchanged under Haswell and the derived leaves re-sealed
(`owner._record` per stored record, then `owner._checkpoint`); identity unchanged.
- old file sha256 a88d56a9ec6b1347794ff08a50fa72793865d1f28c9ff189249aa4f8719c0c83
- new file sha256 1370cbdd346ffa490c5b60309b9ce5f0eafe284b521af32915c962c4dd5c07aa
- changed: 2 float leaves (contact.tangent_error_n_per_m and its pair copy, 9.123885834601848e-07 -> ...846e-07), max relative difference 2.3e-16; plus the internal checkpoint_sha256 string.
- Under Haswell: test_native_fibre_enforcement_successive_response[k8e7] and ..._seeded_successive_response pass; under SkylakeX they now fail (expected).

## Regenerated, second round (scaled criterion: per file max|new-old|/max|old| <= 1e-12)
Same platform/env as above. Commands: `gen.py` (plane_v7: `_analysis((1.,0.,1.)).solve().checkpoint`, verified to reproduce the committed bytes exactly under OPENBLAS_CORETYPE=SkylakeX), `hashgen.py` (the three `_solve_plate` cases; a SkylakeX run reproduces the three old hashes exactly), `cmp.py`.

### tests/data/general_contact_native_curved_plane_v7.checkpoint
- old sha256 3a75bff0d2bc0201e867c4d98011e4db8edfad0b1d288590404e886d68d8a96f, new fdae61b92e492069620b7c8b592469a02e28f636376e3bbb7db81ac346891675
- 139 float leaves changed; max|new-old|/max|old| = 1.6e-14; leaves above 1e-3 in magnitude: max relative 3.4e-16.
- Leaves with relative change > 1e-13 (57), all residual/diagnostic or numerical-zero leaves: `records[i]/contact/energy_error_j` (mag <= 1.5e-17), `contact/force_error_n` (<= 6e-13), `contact/tangent_error_n_per_m` (1.8e-7; abs change 3.2e-14), `mechanical/cell_rotations` off-diagonals (<= 6e-17), `mechanical/nodal_frames` (<= 4e-17), `mechanical/position_low` (<= 1e-16), `mechanical/positions` zero-offset entries (<= 4e-19), `mechanical/resultants` (<= 8e-19).
- Test: test_curved_material_authentic_split_restart_and_atomic_failure passes under Haswell.

### tests/test_s3_r1_qualification_regression.py hash constants (structure and assertions unchanged)
| case | old hash (SkylakeX) | new hash (Haswell) | max abs du / max abs u |
|---|---|---|---|
| linear,8 | 0be6e35ec37efac23e1c79a7f6e8b681b158d0d54994d2910242c79664b17c45 | 3f1deac1f8742a2d347ec232685e1d6e0ede6c3c073b4094f29ca1080251f5c1 | 3.2e-15 |
| corotational,4 | 9620e80d96933d835146adc585e2d71e6c35d0e3b1099a6d1f6ff9535f18ce1d | 1c3f160e0d760741fad496bda447ca2f069ac2d3fb0f0b9d3ffeb2299ebed64c | 5.7e-14 |
| corotational,8 | 23fe778428522ca78b4cb75ffa40854510405625d2c608f1ecb1af88a64c9355 | 6d915aa9647226cd82fa63f061e21104a777f7fb39c52d87fb321dcb66cdeb37 | 3.9e-14 |
Entries with relative change > 1e-13 (|u| > 1e-6 max): linear,8 none above 6.1e-15; corotational,4: 15 of 150 (ux 5, uy 4, rx 3, ry 3; |u| <= 6.0e-5 m); corotational,8: 99 of 486 (ux 34, uy 25, rx 10, ry 7, rz 23; |u| <= 8.8e-3 m, a few above 1e-3). Absolute change <= 1.8e-15 m on all of them (max |u| 3.1e-2 / 4.7e-2 m, uz max 1.2e-2 m). The worst relative changes (2.4e-8 / 2.0e-8) are at entries of magnitude ~7e-8..1e-7 (in-plane displacement / rotation components ~2e-6 of max|u|, with 1.7e-15 absolute change). Convergence tolerance: `solve_static_nonlinear(tolerance=1e-6)`, residual_norm <= tolerance * reference (default; nonlinear_static.py); the changes are ~8 orders below it, i.e. Newton roundoff, not a different solution. Note: some entries with |u| up to 8.8e-3 change by 2e-13 relative, above the per-leaf 1e-13 clause; scaled difference criterion is met.

## NOT regenerated (held for owner decision)
### tests/data/general_contact_native_multi_line_46ea65f6.checkpoint
Proof run (mlproof.py / mlcmp.py): under SkylakeX the current digest code reproduces the committed identity 2831aea349b0... and a fresh `analysis().solve(stop_after=1)` reproduces the committed 6163 bytes exactly. All 1113 hashed descriptor objects of the owner construction were recorded under SkylakeX and Haswell: identical structure; 192 float leaves + 300 hex-float leaves differ (max abs 6.1e-16, max relative 2.5e-15 on |x|>1e-6, scale 1e6); 569 digest strings differ and every one is the digest of a changed input (0 unexplained); 0 other differences. 20 of the hashed objects change. Haswell candidate checkpoint vs old: 100 float leaves differ (max|d|/max|old| 7.1e-15), changed non-float leaves are only `identity`, `checkpoint_sha256`, `records[0].previous_sha256`, `records[0].histories[0].cell_identity`. Not regenerated because physical contact force/moment leaves change by up to 2.5e-13 / 2.6e-12 relative (absolute <= 9.0e-16, components 1.8e-3 / 3.6e-3), above the per-leaf 1e-13 clause. If accepted it is a re-sealed derivative of the wheel fixture 46ea65f6 (wheel_sha256 d654f456..., old checkpoint sha256 6d8f704d43a0c9c7aed71493330612ead89bd68964b3e10d290407c04597a08a): same schema/profile/targets/structure, different float leaves at roundoff and the four digest strings above; candidate in the session notes folder.
