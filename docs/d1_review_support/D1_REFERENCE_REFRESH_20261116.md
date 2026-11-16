# D1 two-body reference refresh (2026-11-16)

`d1_two_body_reference_compare_full_20260928.json` (sealed, unchanged) holds the production result recorded
before commit 3a32ab711 (2026-10-23, "Add experimental S3 V2D von Karman response"), which made the S3 V2D
element geometrically nonlinear under `von_karman`. Since then the current production solution differs from
that record by 1.68e-9 (max displacement, m or rad) and 1.3e-8 Pa (pressure), so
`test_saved_slide_matches_independent_current_contact_kkt` failed on every platform at its 1e-9 limit.
Found by `git bisect` (good 7f4bf4a3, first bad 3a32ab711).

`d1_two_body_reference_compare_full_20261116.json` is `d1_two_body_reference_compare.py` re-run on source
a8d401ae8 (package alias `anysolver` -> `anyfesolver`, Python 3.13, 1 BLAS thread). Same keys; oracle_* arrays
agree with the 20260928 file to 1e-17; production_* arrays moved by the sizes above. Only
`tests/fixtures/d1_nonlinear_kkt.py` reads the new file; test limits are unchanged.
