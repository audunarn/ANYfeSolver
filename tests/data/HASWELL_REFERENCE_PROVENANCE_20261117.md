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

## NOT regenerated (differences exceed the 1e-15 rule; owner decision)
- `general_contact_native_curved_plane_v7.checkpoint` (test_curved_material_authentic_split_restart_and_atomic_failure). Old recipe verified: `_analysis((1.,0.,1.)).solve().checkpoint` under SkylakeX reproduces the committed bytes exactly. Haswell candidate (same recipe): 139 float leaves differ; leaves above 1e-3 in magnitude max relative 3.4e-16; the residual diagnostic `contact.tangent_error_n_per_m` (magnitude 1.8e-7) differs by 3.2e-14 absolute (relative 1.8e-7 of itself); max abs difference / file max = 1.6e-14. Candidate in the session notes folder.
- `tests/test_s3_r1_qualification_regression.py` hashes (SkylakeX run reproduces the three old hashes exactly). Haswell: linear,8 -> 3f1deac1f8742a2d347ec232685e1d6e0ede6c3c073b4094f29ca1080251f5c1 (max |du|/max|u| 3.2e-15, max rel on |u|>1e-6 max 6.1e-15); corotational,4 -> 1c3f160e0d760741fad496bda447ca2f069ac2d3fb0f0b9d3ffeb2299ebed64c (5.7e-14, rel on significant 2.4e-8); corotational,8 -> 6d915aa9647226cd82fa63f061e21104a777f7fb39c52d87fb321dcb66cdeb37 (3.9e-14, 2.0e-8). Newton iteration amplifies roundoff; old hashes (0be6e35e..., 9620e80d..., 23fe7784...).
- `general_contact_native_multi_line_46ea65f6.checkpoint` (test_pre_area_multiple_line_checkpoint_remains_replayable): under Haswell the owner *identity* itself changes (components native/shell/pairs digests), a hash-level change not a last-bit change; fixture has wheel provenance (46ea65f6). Not regenerated.
