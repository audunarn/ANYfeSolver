# S3: coupled specimen, references R0–R2 and timing (2026-11-19)

## Update 2026-11-21 (OD16, review r8 fixes): results bit-identical, timing PASS

Re-run of the same six coupled entries (shared machine, load 8–74 %; raw/runs.log was locked by another process
and not appended). Every non-timing field is identical to 3601a0dcb; summary.json changes in its timing block
only. OD10: R0 PASS, R1 FAIL (unchanged, nothing tuned), R2 PASS. Whole certificate at N = 5000 including the
child (svd entry, median of 3 after a warm-up) **29.3 s** ≤ 45 s (runs 34.6, 28.5, 29.3 s; the spread is in the
child's SVD, 27.4–33.5 s, under machine load; the dense-certificate entry 26.7 s); specimen h = 0.05 accepted,
median wall **34.1 s** ≤ 60 s (default threads 38.6 s); peak parent 307.3 MiB above post-import + child
780.5 MiB = **1188.5 MiB**.

## Update 2026-11-21 (OD15, review r7 non-hook fixes): results bit-identical, timing PASS

Re-run of the same six coupled entries (shared machine, load 43–94 %). Every non-timing field is identical to
6f7e67ce2; summary.json changes in its timing block only. OD10: R0 PASS, R1 FAIL (p 1.17 on 40/80/160, unchanged,
nothing tuned), R2 PASS. Whole certificate at N = 5000 including the child (svd entry, median of 3) **27.1 s**
≤ 45 s (child SVD 26.1 s); specimen h = 0.05 accepted, median wall **32.5 s** ≤ 60 s (default threads 32.8 s);
peak parent 308.6 MiB above post-import + child 780.6 MiB = **1190.4 MiB** (was 1189.6).

## Update 2026-11-21 (OD14, review r6): whole certificate in the child process — results bit-identical, timing PASS

Re-run of the coupled entries (aw/timing_coupled_t1, 	iming_coupled_default, 	iming_oneway_t1, 	iming_svd_t1,
peak_coupled_t1, 0_coupled_n10; one BLAS thread unless noted; shared machine, load 5–85 %).

- **Bit identity.** Every non-timing field of the coupled entries is identical to 3d1227284 (observables, bounds,
  kappa_up, E, V4/V5, reaction). summary.json changes in its timing block only. OD10: R0 PASS (h = 0.05 centre
  0.150 %), R1 FAIL (order difference 0.418, e_c/e_s 1.54 at h = 0.1; unchanged, nothing tuned), R2 PASS 0.75 %.
- **Whole certificate at N = 5000 including the child** (svd entry, median of 3 after one warm-up):
  certified_solve on a banded N = 5000 system (601 340 nonzeros; candidate overlapped with the child's SVD;
  then exact residual, V1–V3, E and V6 in the child) **26.9 s** ≤ 45 s; child SVD 25.9 s, equilibration 0.30 s,
  exact residual 0.19 s, verification 0.28 s, candidate 1.24 s (overlapped). Certificate alone of the dense random
  symmetric matrix of the OD13 measurement: 26.9 s (child SVD 25.8 s, request read 0.16 s, start-up 0.09 s).
- **Specimen h = 0.05** (N = 4969) accepted, median wall **32.7 s** ≤ 60 s (OD13: 35.3 s).
- **Peak commit at N = 4969:** parent 308 MiB above post-import (OD13: 830 MiB; the parent no longer holds a dense
  copy), child 780 MiB (the dense matrix, its equilibrated copy and LAPACK's), sum **1190 MiB** (OD13: 1362 MiB).
## Update 2026-11-20 (OD13, review r5): certificate SVD in a worker process — results bit-identical, timing PASS

Owner decision OD13: the certificate SVD runs in a child process (`docs/SOLID_COUPLING.md`, "Certificate SVD
process"). The same coupled entries were re-run on the OD13 tree (`raw/runs.log`, load 27–87 %; the runner did not
capture exit codes, every `.err` is empty and every JSON is complete). `timing_svd_t1` now times
`_svd_in_worker` (start-up, transfer and SVD) with one warm-up and 3 runs. Nothing was tuned; gates unchanged.

**Results.** Every non-timing field of every re-run JSON (observables, bounds, certificates, forward errors,
reactions, κ samples) is bit-identical to the OD12 runs, and so are R0, R1 and R2 in `summary.json`: the worker's
singular values equal the in-process one-thread SVD bit for bit (same NumPy and OpenBLAS, one thread).

| Gate | Outcome |
|---|---|
| R0 at h = 0.05 (gated) | **PASS**, unchanged: centre 0.150 %, quarter 0.049 %, reaction 1.9e-11 |
| R1 (OD10) | **FAIL**, unchanged (finding R1-4): order difference 0.418 > 0.2; e_c/e_s 1.54 at h = 0.1 |
| R2 (radius 0.25a, 2.7 %) | **PASS**, unchanged: 0.75 % |
| Certificate SVD N = 5000, incl. worker | **PASS**: median 28.2 s (27.8 warm-up; 28.3, 28.0, 28.2 s) ≤ 45 s; the worker's own SVD 27.6–28.0 s |
| Worker overhead (N = 5000) | median 0.26 s per run: start-up to NumPy imported 0.10 s, reading the 200 MB matrix 0.04 s, the rest pipe writing, result and exit |
| Specimen h = 0.05 ≤ 60 s | **PASS**: accepted, median total 35.3 s (34.7–36.1 s; certificate 28.9 s), N 4969 |
| Peak commit | parent 830 MiB above post-import (unchanged), worker 431 MiB at N 4969; sum 1362 MiB total commit (the two peaks bound the simultaneous use) |

The worker holds the matrix it read and LAPACK's working copy (2 × 188 MiB at N 4969); the parent's peak is in
the equilibration (original and scaled copy), before the worker starts. The timings vary with the shared
machine's load (35.3 s here against 33.2 s for OD12 at load 14–81 %).

## Update 2026-11-20 (OD12, review r4): refinement 5 — R0 PASS, R1 FAIL, R2 PASS, timing PASS

Owner decision OD12: the route refines the candidate 5 times (2 before). The coupled entries of `run_all.sh`
(`timing_coupled_t1`, `timing_coupled_default`, `timing_oneway_t1`, `timing_svd_t1`, `peak_coupled_t1`,
`r0_coupled_n10`) were re-run on the OD12 tip (`raw/runs.log`, load 14–81 %). The shell and solid references do not
touch the route or the certificate and were not re-run. The OD11 coupled runs (2 steps, and the 5-step diagnostic
runs) are in `raw_od11/`, their analysis in `summary_od11.json`; `summary.json` `od12_changed_results` compares them.
Nothing was tuned; gates unchanged (OD10).

| Gate | Outcome |
|---|---|
| Both meshes | accepted: h = 0.1 (N 1417) κ_up 5.35e6, E 8.70e-9; h = 0.05 (N 4969) κ_up 6.07e6, E 1.02e-8 |
| R0 at h = 0.05 (gated) | **PASS**: centre 0.150 %, quarter 0.049 %, reaction 1.9e-11 |
| R0 at h = 0.1 (reported) | centre 0.79 %, quarter 0.29 % and 0.15 %, reaction 3.1e-11 |
| R1 (OD10) | **FAIL** (finding R1-4, unchanged): p coupled 0.957, p all-shell 0.539, difference 0.418 > 0.2; e_c/e_s 1.54 at h = 0.1 (fail), 1.15 and 0.96 at h = 0.05 (pass) |
| R2 (radius 0.25a, 2.7 %) | **PASS**: 0.75 %; the 0.05a set is empty (R2-1) |
| Single-thread SVD N = 5000 | **PASS**: median 25.8 s (25.8, 25.8, 25.8, 26.0, 26.3, 25.8 s) ≤ 45 s |
| Specimen h = 0.05 ≤ 60 s | **PASS**: accepted, median total 33.2 s (certificate 27.1 s, candidate 0.47 s) |
| Peak commit | 830 MiB above post-import (reported) |

**Changed accepted results (OD12).** h = 0.1, the only specimen result accepted before: centre w 2.629919678085051e-05
→ …0512e-05 (+3.4e-21, 1.3e-16 relative, bound 1.7e-14), quarter x = 0.3 +6.8e-21 (3.1e-16), quarter x = 0.2
unchanged, reaction_z −999.9999999686454 → …6386 (+6.8e-12), forward error 9.46e-9 → 8.70e-9. h = 0.05 is now
accepted (it refused `forward_error_above_limit` before). All current numbers are bit-identical to the OD11
5-step diagnostic runs. Cause: the extra steps change the candidate's last bits; every change is far inside the
certified bounds.


## Update 2026-11-20 (review r3, OD10 R1 (a)): widened specimen — R0, R1, R2 FAIL (the route refuses at h = 0.05)

The owner's decision for R1-2: widen the solid so that h = 0.1 is resolved, and re-run with the gates unchanged.
Code under test: the r3 tip (this commit). `raw/` now holds the widened runs; the earlier a = 0.2 runs are in
`raw_a02/` and their analysis in `summary_a02.json`. Machine load 4–79 % during the runs (`raw/runs.log`).

**Specimen change.** The footprint is [0.3, 0.7]², a = 0.4, so h = 0.1 has 4 × 4 lattice cells (was 2 × 2). Equation
counts measured first, at h = 0.05: two layers give N = 6703 > 5000 (and the off-centre [0.4, 0.7]² gives 5027),
so the only centred widening under the cap has **one P2 layer** through the thickness. That is a second change to
the specimen, forced by the cap. R2 sample point and radii follow a: (x0 + 0.25a, 0.5) = (0.4, 0.5), radii 0.05a and
0.25a.

| h | n_s | n_v | m | N | Outcome (production route) |
|---|---|---|---|---|---|
| 0.1 | 628 | 729 | 60 | 1417 | accepted: κ_up 5.35e6, E 9.46e-9 |
| 0.05 | 2268 | 2601 | 100 | 4969 | **refused** `COUPLED_UNCERTIFIED forward_error_above_limit`: κ_up 6.07e6 (certificate accepted), E = 0.445 |

**Finding R1-3 (route, fail closed).** At h = 0.05 the SuperLU candidate converges slowly under refinement on this
system: the D-scaled residual falls by about 1e-3 per step. With the route's 2 steps, E / max|x/d| = 0.064, far
above V6's 1e-6, so the solve refuses. Measured (`<scratch>\r3\probe_cand.py`): steps 0–5 give E / scale = 5.3e4,
34, 0.064, 2.2e-4, 1.2e-7, 1.5e-9; a dense LAPACK solve gives 1.7e-7. The refusal is correct (V6 is the authority).
Accepting this specimen would need a production change to the candidate (for example more refinement steps), which
changes the bits of accepted results elsewhere: an owner decision, not made here.

**Gates (OD10, unchanged), production route: FAIL.**

| Gate | Outcome |
|---|---|
| R0 at h = 0.05 (gated) | **FAIL**: no coupled result (refusal R1-3) |
| R0 at h = 0.1 (reported only) | centre 0.79 % (was 0.66 % with a = 0.2); quarter points 0.29 % and 0.15 %; reaction 3.1e-11 |
| R1 | **FAIL**: no coupled result at h = 0.05 |
| R2 (radius 0.25a, 2.7 %) | **FAIL**: no coupled result at h = 0.05 |
| Single-thread SVD at N = 5000 | **pass**: median 25.9 s (26.0, 27.8, 26.6, 25.9, 25.7 s) ≤ 45 s |
| Specimen ≤ 60 s | **FAIL** as a gate (no accepted solve); time to the refusal 32.1 s; the diagnostic accepted solve 33.0 s |
| Peak commit | 829 MiB above post-import (reported; N = 4969, was 505 MiB at N = 3783) |

**Diagnostic only (not a gate): the candidate refined 5 times in the measurement process** (`--diagnostic-refinement
5`, runs `diag_coupled_n10_ref5`, `diag_coupled_n20_ref5`; `summary.json` `diagnostic_refinement_5`). Both meshes are
then accepted (h = 0.05: E 1.02e-8). With that:
- R0 at h = 0.05: pass (centre 0.150 %, quarter 0.049 %, reaction 1.9e-11).
- **R1 still fails**: p coupled 0.957, p all-shell 0.539, difference 0.418 > 0.2; e_c/e_s = 1.54 at h = 0.1 (fail),
  1.15 at h = 0.05 (pass), quarter 0.96 (pass). The Richardson sensitivity also fails (0.885 vs 0.487).
- R2 at radius 0.25a: pass, 0.75 % ≤ 2.7 %. The 0.05a set is empty again (finding R2-1).

**Finding R1-4.** Widening did not remove R1-2: with 4 × 4 cells at h = 0.1 the coupled centre error relative to the
all-shell error grew (1.45 → 1.54) and the order difference grew (0.33 → 0.42). So the 2 × 2-cell resolution was not
the cause. The coupled h = 0.1 centre is read on the solid mid-plane, where the 3-D field differs from the shell
model; the coupled error at h = 0.05 stays within 1.2× of the all-shell error in both specimens. R1 needs a new
owner decision (for example whether the order and ratio gates should apply at h = 0.1, or a different observable).

---

Branch `claude/coupled-dense-20261119`. The code under test was f9382e7ce (OD8) for the first runs. Everything below
refers to the a = 0.2, two-layer specimen; its raw runs are now in `raw_a02/` and its analysis in `summary_a02.json`.

**Update 2026-11-20 (review r2, OD9/OD10).** The runs that use the coupled route or the certificate SVD were
re-run on the r2 tip: `timing_coupled_t1`, `timing_coupled_default`, `timing_svd_t1` (now the production
`_svd_values`, on one verified BLAS thread), `peak_coupled_t1` and `r0_coupled_n10`. Observables, certificates and
every other non-timing field are bit-identical to the first runs, except `verification.forward_error`, which is now
the bound rounded up (review r1 class 5): 1.0152929536163088e-08 → …09e-08 and 1.0058900569309186e-08 → …88e-08,
one ulp each. The shell and solid reference runs do not touch the route and were not re-run. The gates below are
those of owner decision OD10 (2026-11-20); the earlier protocol outcomes are kept for the record.

To reproduce, run `bash run_all.sh`, which starts one fresh process per run and writes `raw/*.json` and
`raw/runs.log`. Then run `python analyse.py > summary.json`. The ANYsolid venv is `C:\Github\ANYsolid\.venv` with
Python 3.11.9, numpy 2.4.6, scipy 1.17.1, scikit-fem 12.0.2, ANYsolid 0.1.0 and scipy-openblas 0.3.31 (64-bit
ints, DYNAMIC_ARCH). The CPU has 32 logical cores.

**The machine was shared.** Other sessions' Python jobs were running. Windows CPU load was 20–71 % during the runs
(per-run values are in `raw/runs.log`). Times are therefore upper-side and not quiet-machine numbers.

**Specimen (spec §7).**
- Plate: L = 1, t = 0.02, E = 210 GPa, ν = 0.3, simply supported (uz = 0 on the edge, a corner pin, a roller).
  Elements are E4-PL Q4.
- Load: uniform p = 1000 Pa on the whole plate, carried by `skin='end'` inside the footprint.
- Footprint: [0.4, 0.6]², a = 0.2, on element edges. It holds a floating 2-layer Kuhn P2 lattice with the same
  in-plane spacing as the shell.

## Equation counts

| h | n_s | n_v | m | N |
|---|---|---|---|---|
| 0.1 | 676 | 375 | 40 | 1091 |
| 0.05 | 2508 | 1215 | 60 | 3783 |

h = 0.025 would exceed the 5000 cap (OD5).

Both solves were accepted:
- h = 0.1: κ_up = 6.89e6, E = 1.01e-8, V1–V3 max ratio 1.6e-27, V4 balance 1.8e-15, V5 2.6e-13.
- h = 0.05: κ_up = 7.05e6, E = 1.02e-8, V4 compatibility 1.2e-16, V4 balance 3.1e-15, V5 2.5e-12.

## R0: same-mesh all-shell (gates: deflection 0.5 %, reaction 1e-9, both relative; OD10: gated at h = 0.05, h = 0.1 reported only)

**R0 gate (h = 0.05): PASS.**

| h | Observable | Coupled | All-shell | Relative difference | Gate |
|---|---|---|---|---|---|
| 0.1 | centre w (solid, at (0.5, 0.5, 0)) | 2.633386e-05 | 2.650742e-05 | 6.55e-3 | reported only (above 0.5 %) |
| 0.1 | w at (0.2, 0.5)\* | 1.603500e-05 | 1.602548e-05 | 5.9e-4 | reported only |
| 0.1 | w at (0.3, 0.5)\* | 2.176425e-05 | 2.174575e-05 | 8.5e-4 | reported only |
| 0.1 | total reaction | −1000.0000000009 | −1000.0000000000 | 8.4e-13 | reported only |
| 0.05 | centre w | 2.658745e-05 | 2.662860e-05 | 1.55e-3 | pass |
| 0.05 | quarter w (0.25, 0.5) | 1.926522e-05 | 1.926220e-05 | 1.6e-4 | pass |
| 0.05 | total reaction | −999.99999999 | −1000.00000000 | 8.2e-12 | pass |

\* At h = 0.1 the quarter point is not a node, so both neighbouring nodes are reported.

**Finding R0-1.** At h = 0.1 the centre deflection differs by 0.66 %, above the 0.5 % gate. That mesh has 2 × 2
lattice cells across the footprint. The centre value is read from the solid mid-plane, where the 3-D field and the
shell field differ in modelling (transverse shear and through-thickness strain). The difference falls to 0.16 % at
h = 0.05, which is the gate mesh. No ill-posed reference occurred: every reference is above 1e-12 of `pL⁴/D` = 6.50e-3
and of `pL²`.

## R1 (OD10): coupled against all-shell at the same h — **FAIL**

Gate: errors against the finest all-shell run R(h/8) (n = 160); the observed order `p = log2(e(0.1)/e(0.05))`
of the coupled centre deflection within 0.2 of the all-shell order, and the coupled error at most 1.2× the all-shell
error at each compared h and observable. The centre is the only observable that is a node at both h; the quarter
point (0.25, 0.5) exists at h = 0.05 only. `analyse.py` → `summary.json` `R1_od10`.

| Reference | p coupled | p all-shell | Difference (≤ 0.2) | e_c/e_s centre h = 0.1 | centre h = 0.05 | quarter h = 0.05 | Outcome |
|---|---|---|---|---|---|---|---|
| R(h/8), the gate | 0.866 | 0.539 | 0.326 **fail** | 1.447 **fail** | 1.154 pass | 0.986 pass | **FAIL** |
| Richardson with p = 1.17 from n = 40/80/160 (sensitivity) | 0.799 | 0.487 | 0.312 fail | 1.410 fail | 1.136 pass | – | fail |

**Finding R1-2.** The coupled error at h = 0.05 is within the gate (1.15× and 0.99×), but at h = 0.1 the coupled
centre error is 1.45× the all-shell error, so the coupled observed order between the two meshes is steeper than
the all-shell one by 0.33. The h = 0.1 coupled centre is read in a solid of only 2 × 2 lattice cells (finding R0-1,
the modelling difference of the 3-D field against the shell), which inflates e_c(0.1) and with it p_c. The outcome
does not depend on the reference choice (the Richardson sensitivity also fails). Under the cap (OD5) only h = 0.1
and 0.05 can be run, so the order rests on that coarse mesh. This needs an owner decision.

## R1 as first specified (superseded by OD10): fine all-shell at h/2, h/4 and h/8 of 0.05 (n = 40, 80, 160)

| Observable | R(h/2) | R(h/4) | R(h/8) | d1 | d2 | p = log2(d1/d2) | Outcome |
|---|---|---|---|---|---|---|---|
| centre w | 2.67549e-05 | 2.68529e-05 | 2.68964e-05 | 9.8e-08 | 4.3e-08 | 1.17 | **FAIL: no asymptotic range** |
| quarter w | 1.93676e-05 | 1.94457e-05 | 1.94795e-05 | 7.8e-08 | 3.4e-08 | 1.19 | **FAIL: no asymptotic range** |

**Finding R1-1.** The protocol fails as specified: d1 and d2 have the same sign, but the observed order is 1.17–1.19,
not within [1.5, 2.5]. E4-PL Q4 deflections on this plate converge at about first order up to n = 160. So the second-order
Richardson premise of rev2 §5.2 does not hold for this element. For information only: against R(h/8), the coupled
h = 0.05 values are 1.15 % (centre) and 1.10 % (quarter) low. The all-shell h = 0.05 values are 1.00 % and 1.12 % low,
so the coupled error is the shell's own discretisation error. The reaction is exact on every mesh (d = 0), so it has
no R1 test.

## R2: all-solid plate (ANYsolid P2, w = 0 on the mid-surface edge lines, uniform traction on `end`)

The R2 run details are in `summary.json`.
- Sizes run from 25 215 to 177 147 DOFs.
- Every reaction equals −1000 to within 1.3e-9 relative.
- After 2 refinement steps, every relative residual is at most 1.3e-8.

OD10 makes the 0.25a sample set the gate with a limit of 2.7 % of |κ_ref|. **R2: PASS.**

| Sample set | Runs that admit it | κ_ref (h_s/2, 4 layers) | U | Coupled κ_xx | Deviation | Gate | Outcome |
|---|---|---|---|---|---|---|---|
| Radius 0.25a = 0.05 (OD10 gate) | all (8 or 26 triangles per face) | 2.397830e-04 | 1.59e-06 (reported) | 2.381479e-04 | 1.64e-06 (0.68 %) | 2.7 % = 6.47e-06 | **pass** |
| Radius 0.05a = 0.01 (first protocol, reported only) | none (0 triangles in every run) | – | – | – | – | – | empty (finding R2-1) |

The diagnostic κ_xx values by run:

| Layers | h_s = 0.05 | h_s = 0.025 |
|---|---|---|
| 2 | 2.38076e-4 | 2.39687e-4 |
| 4 | 2.38188e-4 | 2.39783e-4 |

The extraction is the area-weighted ε_xx = ∂u_x/∂x at the skin-triangle centroids. It uses a central difference along
x inside the owning face, which is exact for the quadratic field. It is applied identically to the coupled solid.

**Finding R2-1.** The rev2 sample set cannot hold any triangle on a structured lattice. The sample point (0.45, 0.5)
is a lattice vertex for every h_s in the protocol. Kuhn skin-triangle centroids sit at least h·√2/3 from every vertex:
0.0236 at h = 0.05 and 0.0118 at h = 0.025. Both distances exceed the 0.01 radius, so each set is empty and the
protocol fails by construction. The non-gating diagnostic with a radius of 0.25a passes with a 3.9× margin.

## Timing on the r2 tip (2026-11-20; certificate SVD on one verified BLAS thread)

One warm-up and 5 runs, median, coupled at h = 0.05 (N = 3783). Machine load 78 % (coupled) and 87 % (SVD) at the
end of the runs (`raw/runs.log`): shared machine, upper-side numbers.

| Phase | Median (s) |
|---|---|
| native assembly and reduction | 1.87 |
| provenance | 0.02 |
| connectors | 1.42 |
| C1 formation and checks | 0.02 |
| certificate (equilibration + single-thread SVD) | 11.49 |
| candidate | 0.15 |
| exact residual | 0.05 |
| verification | 0.14 |
| recovery, V4, V5 | 0.47 |
| **total** | **16.76** (runs 16.23–17.25) |

- **SVD gate at N = 5000** (production `_svd_values`, one warm-up and 5 runs): median **27.2 s** (26.1–28.0 s) ≤ 45 s,
  **pass**.
- **Specimen gate**: **16.8 s** ≤ 60 s, **pass**.
- "Default threads" run: 16.1 s; the certificate SVD is now single-threaded in every run, so it no longer differs.
- Ratio to native shell plus one-way solid (0.269 s, not re-run): 62.4×, reported, not gated (OD2).
- Peak commit 505 MiB above post-import (reported).

## Timing (first runs, f9382e7ce, superseded): one warm-up and 5 runs, median, threads = 1, coupled at h = 0.05 (N = 3783)

| Phase | Median (s) |
|---|---|
| native assembly and reduction | 1.26 |
| provenance (lattice, plane) | 0.01 |
| connectors | 0.96 |
| C1 formation and checks | 0.01 |
| certificate (equilibration + dense SVD) | 10.86 |
| candidate `splu` and refinement | 0.10 |
| exact residual | 0.03 |
| V1–V3 and V6 verification | 0.08 |
| recovery, V4, V5 | 0.33 |
| **total** | **14.39** (runs 14.16–14.88) |

Comparisons and gates:
- **Default threads**, one run: 16.93 s in total, with the certificate taking 13.27 s.
- **Native shell plus the one-way solid** on the same plate (threads = 1, median of 5): 0.238 + 0.030 = 0.269 s. The
  coupled route takes **53.6×** as long. This is reported, not gated (OD2).
- **SVD gate at N = 5000** (threads = 1, one warm-up and 5 runs): median **25.7 s** (24.9–27.6 s) ≤ 45 s, pass. This
  is faster than the earlier 40–45 s session-1 numbers, which were taken under higher load.
- **Specimen gate**: **14.4 s** ≤ 60 s, pass.
- **Peak commit**: 504 MiB above the post-import value (PeakPagefileUsage, fresh process, one coupled solve at
  h = 0.05, threads = 1). This is reported, not gated. The O17a pre-check estimate 4·N²·8 B is 458 MB.

## Summary

| Item (OD10 gates, r2 tip) | Outcome |
|---|---|
| R0 at h = 0.05 (gated) | **pass** |
| R0 at h = 0.1 (reported only) | centre deflection 0.66 % (finding R0-1) |
| R1 coupled vs all-shell | **FAIL**: order difference 0.33 > 0.2, and e_c/e_s = 1.45 > 1.2 at h = 0.1 (finding R1-2); h = 0.05 ratios 1.15 and 0.99 pass |
| R2 at radius 0.25a, limit 2.7 % | **pass** (0.68 %) |
| Timing gates (OD2) | **pass**: single-thread SVD 27.2 s ≤ 45 s; specimen 16.8 s ≤ 60 s |

R1-2 needs an owner decision (for example whether the error-ratio and order gates apply at h = 0.1, whose centre is
read in a 2 × 2-cell solid).
