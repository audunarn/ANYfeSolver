# V2D-OPT opt-in triangle (Felippa OPT membrane), 2026-11-26

Status: unregistered, direct-class opt-in, linear static and recovery only. No default change. Not reachable from the
factory or ANYfem (registration needs a successor seal on `elements.py`; deferred to the default-flip programme by owner decision).

## What it is
- `src/anyfesolver/e4_pl_opt_membrane.py`: pure kernel, a port of Felippa, CMAME 192 (2003) 2125-2168, Fig. 10 (template), Fig. 12
  (optimal signature: alpha_b = 3/2, beta1..9 = 1,2,1,0,1,-1,-1,-1,-2, beta0 = max(2/E11C11avg - 3/2, 0.01), isotropic (1-4 nu^2)/2) and eq. (48)
  (strain recovery, e0 = 3/2). Reproduces the paper's Fig. 11 matrices/eigenvalues, Table 3, Table 5 (OPT) and Table 6 (Cook) to the printed digits.
  Standalone prototype note: the earlier fixed beta0 = 1/2 was wrong for nu != 0.
- `src/anyfesolver/e4_pl_s3_v2d_opt_element.py`: `NativeParityE4PLS3V2DOptShellElement`, not a V2D subclass (the default's exact-type guards stay
  untouched). V2D/V2C geometry, MIN3 bending/shear and pressure loads are reused; the membrane block (CST + PL drill) is replaced by the OPT membrane with its
  own drilling stiffness.

## Measured (spot-check suite, default triangle -> OPT; MacNeal numbers are measured, not paper-matched)
Better: MacNeal in-plane shear 6x1/30x2/120x8 -97.4/-65.7/-10.8 % -> +6.5/+0.9/-0.1 %, twisted beam, Scordelis-Lo, LE1, LE5 32, pinched 16.
Worse: hemisphere 8x8 (R/t 250) -47.5 -> -49.0 % (OPT does NOT cure coarse curved-mesh locking), MacNeal 6x1 overshoot +6.5 %, 30x2 extension -0.17 -> +0.41 %,
LE5 64 web/flange (nodal average) -8.4 -> -9.2 %, pinched 32 +0.20 -> +0.69 %. Identical: plates, LE6, out-of-plane shear.

## Trust contract and known limits
See the module docstring of `e4_pl_s3_v2d_opt_element.py` (owner decision 2026-11-26, trust boundary A''): hostile data, instance attributes and subclasses are in
scope; mutation/rebinding of shared library state (module arrays, caches, write flags, `__class__`, metaclasses, `__dict__` writes) is outside the supported
surface. Known v1 limits: legacy `contact.py` admits every `ShellElement`; `recover_stress_result` accepts caller-supplied `element_states`/`kinematics`;
explicit unbound `ShellElement.method(element)` base dispatch bypasses the typed refusals until registration.

## Review history (Codex GPT 6.2 Sol, high)
r1: 2 P1 / 5 P2 / 2 P3; r2: 2 P1 / 3 P2 / 2 P3; r3: 1 P1 / 2 P2; r4: 2 P1 / 1 P2 (same class, tampering with shared library arrays: closed by the contract above, not fixed).

## Default-flip programme (not started)
Registration + seal link, mass/modal, nonlinear current-state route, geometric stiffness, general_contact, mixed Q4 scope, then the flip itself (about 5-7 PRs).
The flip is an owner decision after the benchmark table.
