# S3: coupled specimen, references R0–R2 and timing (2026-11-19)

Branch `claude/coupled-dense-20261119`. The code under test is f9382e7ce (OD8). This evidence and the docs are
committed on top of it.

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

## R0: same-mesh all-shell (gates: deflection 0.5 %, reaction 1e-9, both relative)

| h | Observable | Coupled | All-shell | Relative difference | Gate |
|---|---|---|---|---|---|
| 0.1 | centre w (solid, at (0.5, 0.5, 0)) | 2.633386e-05 | 2.650742e-05 | 6.55e-3 | **FAIL** |
| 0.1 | w at (0.2, 0.5)\* | 1.603500e-05 | 1.602548e-05 | 5.9e-4 | pass |
| 0.1 | w at (0.3, 0.5)\* | 2.176425e-05 | 2.174575e-05 | 8.5e-4 | pass |
| 0.1 | total reaction | −1000.0000000009 | −1000.0000000000 | 8.4e-13 | pass |
| 0.05 | centre w | 2.658745e-05 | 2.662860e-05 | 1.55e-3 | pass |
| 0.05 | quarter w (0.25, 0.5) | 1.926522e-05 | 1.926220e-05 | 1.6e-4 | pass |
| 0.05 | total reaction | −999.99999999 | −1000.00000000 | 8.2e-12 | pass |

\* At h = 0.1 the quarter point is not a node, so both neighbouring nodes are reported.

**Finding R0-1.** At h = 0.1 the centre deflection differs by 0.66 %, above the 0.5 % gate. That mesh has 2 × 2
lattice cells across the footprint. The centre value is read from the solid mid-plane, where the 3-D field and the
shell field differ in modelling (transverse shear and through-thickness strain). The difference falls to 0.16 % at
h = 0.05, which is the gate mesh. No ill-posed reference occurred: every reference is above 1e-12 of `pL⁴/D` = 6.50e-3
and of `pL²`.

## R1: fine all-shell at h/2, h/4 and h/8 of 0.05 (n = 40, 80, 160)

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

| Sample set | Runs that admit it | κ_ref (h_s/2, 4 layers) | U | Coupled κ_xx | Deviation | Gate 0.02·|κ_ref| + U | Outcome |
|---|---|---|---|---|---|---|---|
| Protocol: radius 0.05a = 0.01 | none (0 triangles in every run) | – | – | – | – | – | **FAIL (protocol)** |
| Diagnostic: radius 0.25a = 0.05 | all (8 or 26 triangles per face) | 2.397830e-04 | 1.59e-06 | 2.381479e-04 | 1.64e-06 (0.68 %) | 6.39e-06 | pass (not a gate) |

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

## Timing (spec §8 S3, OD2): one warm-up and 5 runs, median, threads = 1, coupled at h = 0.05 (N = 3783)

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

| Item | Outcome |
|---|---|
| R0 at h = 0.05 (the gate mesh) | pass |
| R0 at h = 0.1 | centre deflection 0.66 % > 0.5 % (finding R0-1) |
| R1 | protocol fails: order about 1.2 (finding R1-1) |
| R2 | protocol fails: empty sample set (finding R2-1); the diagnostic passes |
| Timing gates (OD2) | pass: SVD 25.7 s ≤ 45 s; specimen 14.4 s ≤ 60 s |

R1-1 and R2-1 are protocol-definition findings that need an owner decision. R0-1 is reported against the coarse mesh.
