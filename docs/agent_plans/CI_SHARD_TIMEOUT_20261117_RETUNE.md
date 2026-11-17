# Retune from run 44100447361 (2026-11-17)

Run 44100447361 (combined branch 6aa6fe89d): 4 cells still end at the shared limit.
Source data: 27 `module-durations-*` artifacts (607 modules measured; the 4 stalled
cells and 1 failed cell upload nothing because their pools never wrote junit),
logs of the 4 stalled jobs, and local `--collect-only` per pool to map progress
percentages to modules (selection after the runner's `--deselect` list).

| cell (log) | stalled pool | last progress | stalled module (evidence) |
|---|---|---|---|
| windows 3.11, shard 0/0 (7 modules) | P01 = `s3_q4_patch_augmented` alone, pred 1432 s | none | that module alone did not finish in 2400 s (P02, 6 S3 modules: pred 13 s, run 35 s) |
| windows 3.13, shard 4/1 (43) | P02, 42 modules, pred 180 s (P01 = `q4_q4_static`: pred 582 s, actual 795 s) | 73 of 445 tests at 14:16 | `enriched_path` (tests 125-127; same module stalled the previous run) |
| windows 3.12, shard 2/2 (50) | P02, 43 modules, pred 338 s (P01 finished 14:23) | 428 of 775 tests at 14:26:48, silent 29 min | window 428-500 contains `transient_clipped_s3`; its test 8 `test_moving_clipped_overlap_with_explicit_enforcement_matches_coupled_reference` runs >1500 s locally |
| ubuntu 3.12, shard 3/3 (53) | P01, 38 modules, pred 359 s (P02 finished 14:39:51) | 523 of 845 tests at 14:39:50, silent 30 min | `s3_friction_replay` (tests 517-572); `test_partial_overlap_coupled_workflow_and_split_restart[False]` runs >1500 s locally |

Cause: single tests that need far more than 2400 s of a pool on a runner (the friction
fix and Haswell pin make them run to the end instead of failing early). The old file-size
weights rated all of these modules at 2-3 s, so they sat in ordinary pools.

Measured vs predicted (module sums, CI seconds): `s3_mixed_mesh_manifest` 1021 vs 145,
`q4_source_augmented` 1116 vs 637, `admission_globalization_final_checks` 623 vs 256,
`retained_body_pair` 531 vs 246, `q4_q4_augmented` 510 vs 231; 45 modules had a proxy of
2-3 s and measured 25-113 s.

Fix (this commit): weights now come from `scripts/portable_ci_module_seconds.json`
(237 modules: measured value of run 44100447361, merged with the earlier estimates by
taking the larger; measured value replaces the byte proxy). Lower bounds for the four
unfinished modules: `s3_q4_patch_augmented` 2400, `enriched_path` 1950,
`s3_friction_replay` 1900, `transient_clipped_s3` 1750 (time from last progress to the
limit). The runner prints each test (`-v`) and records durations also on timeout/failure,
so the next run names every stalled test and measures the finished pools.

Predicted pool seconds per cell after retuning are in the handoff; all cells except
(0,0), (1,0), (2,0) and (3,0) stay at or below 1330 s per pool. Those four cells each
hold one long module that no balancing can split; the following commit gives only
these cells a 5400 s limit (`PORTABLE_LONG_RUNNING_MODULES`).
