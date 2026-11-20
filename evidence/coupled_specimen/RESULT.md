# S3: coupled specimen, references R0–R2 and timing (2026-11-19)

Branch `claude/coupled-dense-20261119`. The code under test was f9382e7ce (OD8) for the first runs.

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
