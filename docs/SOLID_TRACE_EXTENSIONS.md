# Planar production S3/Q4 research traces

Early implementation checkpoint: current S3 default is exact
`NativeParityE4PLS3V2DShellElement`, not the historical hierarchical-bubble S3.
V2D borrows V2C's reviewed MIN3 shear/bending operators. Geometric barycentric
weights alone do not give its linear static transverse displacement field.

Production anchors inspected:

- `e4_pl_s3_v2d_element.py`: native surface shapes, geometry/frame/external
  permutation, retained eighteen global coefficients and physical director.
- `e4_pl_s3_v2c_element.py`: `_operators` differentiates l_i/m_i for shear;
  `compute_dead_transverse_pressure_load` evaluates the actual displacement
  covector of `w_h=sum(N_i*w_i-l_i*rx_i+m_i*ry_i)`.
- Production bending is the gradient of linearly retained nodal rotations.
  The l_i/m_i values vanish at vertices, fixing the integration constants.

On a straight edge a->b with alpha in[0,1], this gives the extra midsurface
normal displacement

`0.5*alpha*(1-alpha)*n*(n dot ((theta_a-theta_b) cross (x_b-x_a)))`.

Add this to barycentric translations, then transport linearly retained global
rotations with their cross product against actual point minus interpolated
reference coordinate. This includes trace offset once. The normal projection
is invariant to director reversal. This is a linear static production-derived
trace; it does not select a finite enhanced surface or kinetic approximation.
The production von-Karman affine-w gradient is a separate declared route.

Plan: exact-type S3/Q4 admission and per-element edge layouts; production MIN3
edge correction only; stronger finite/degenerate geometry and stale guards;
tests of actual production l/m pressure-field values, shear derivatives,
retained bending rotations, rigid fields, work/DOF maps, mixed chains,
frames/director signs/offsets and mutation/refusal. No mechanics kernels or
other sources will be changed. Remaining: implement and focused bounded checks.
No scientific blocker found for this small-linear field. Parent/reviewer
adjudication and independent qualification remain separate; W1.4 is not accepted.

## Implemented contract

`ShellEdgeTrace` now admits exact production Q4 and exact current-policy S3 V2D
classes, including their connected mixed straight chains. Historical S3,
subclasses and other shell formulations remain outside admission. Q4 edge
indices stay (0,1),(1,2),(2,3),(3,0); S3 indices are (0,1),(1,2),(2,0).
The actual production geometric surface weights are used for geometry and
translations. S3 alone adds the MIN3 boundary displacement above, with linear
retained global rotations. It is not a geometric-weight-only approximation.

For an oriented local triangle define
`a_i=x_k-x_j`, `b_i=y_j-y_k`,
`l_i=N_i*(b_k*N_j-b_j*N_k)/2`,
`m_i=N_i*(a_j*N_k-a_k*N_j)/2`, with cyclic(i,j,k). These are the explicit
production pressure-work interpolation functions, not merely antiderivatives
chosen to match shear. Their zero values at all vertices fix the integration
constants to the actual nodal transverse displacements. Restriction to an
edge gives the implemented global normal-projected cross-product expression.
Production B_s verifies its derivative, and production B_b verifies that
retained rotations remain linear. No hidden bubble coordinate is introduced.

Physical director sign, its S3 polarity, reference-surface offset and actual
global DOF maps remain explicit. Transport uses actual target minus nodal
reference coordinate; an element material/reference offset is not added again.
Trace offset locates the physical interface and is counted once in that arm.
The explicit S3 normal projection is invariant to normal reversal. Geometry
must remain planar with one physical director, one thickness, a common inward
side and a monotone straight, nonoverlapping edge chain. Curved facets,
corners, overlaps/backtracking and mixed directors/thicknesses remain refused.

Trace state and geometric vectors are immutable. Stale checks cover geometry,
thickness, physical director/polarity, S3 reference-surface offset, actual DOF
layout and element replacement/removal; missing referenced state is a clean
`CoupledSolveError` (code `COUPLED_TRACE_INVALID`). Existing Q4 behavior and constructor/operation API remain.

## Scientific boundary

This is a research-only linear static displacement trace. The displacement
field's explicit production pressure-work anchor is essential: matching B_s
derivatives alone would not uniquely select integration constants or a finite
surface. No finite-rotation director surface is adopted here; the repository's
finite-work candidate and separate nodal kinetic interpolation do not broaden
this trace. Generalized/nonlinear/dynamic/stability/engineering coupling
qualification is not established. W1.4 and production coupling remain NO_GO
pending parent adjudication and independent gates.

## Verification and handoff

Exact final bounded command:

`C:/Github/ANYsolid/.venv/Scripts/python.exe C:/Github/ANYsolid/tools/run_check.py --name solid-trace-complete --cwd C:/Github/ANYfeSolver --timeout 180 -- python -m pytest tests/test_solid_trace_extensions.py tests/test_solid_coupling.py tests/test_submodel_transfer.py -q`

The supervisor supplies the required ANY ecosystem source PYTHONPATH and
single-thread numerical runtime. Final result: 96 passed / zero failed,
including 38 extension checks and 58 existing coupling/submodel regressions;
process wall 10.343999999997322 s. Evidence:
`C:/Github/ANYsolid/evidence/w1/solid-trace-complete-92199fa00a`.
Full stdout/stderr, JUnit, source/input identities, process receipt and verified
owned-process cleanup are preserved there. Earlier incremental clean runs are
retained in `solid-trace-extensions-d498c4c643` (32 cases),
`solid-trace-q4-regression-090b1f6fd1` (58), and
`solid-trace-extensions-final-7e36c574a5` (36). Counts overlap and are not added
to the final 96 distinct cases. No failed test attempt occurred.

Final trace source SHA-256:
`f7e2e64f85ceb2442511b50d4e005cf2d98ad0b9ee6ca6e76a62c5ec7ed0b164`.
Extension-test SHA-256:
`487d36ac9c417e21497c4f2c4993ae17c6b1566a8cb055b0ec6c7deff067df61`.

The new checks use actual production V2D geometry/order and explicit pressure-field
l_i/m_i formulas as independent value oracles, actual B_s/B_b operators as derivative
and retained-rotation oracles, and nonconstant rotations to expose the MIN3
edge correction explicitly. All three triangle edges, tilted global frames,
normal signs/director polarities, material and trace offsets, nodal integration
constants, scrambled actual global DOFs, mixed chains, six physical rigid modes,
transpose work and total physical force/moment are covered. Mutation and
refusal checks cover corners, overlapping/backtracking traces, curved Q4,
mixed directors/thicknesses, nonfinite thickness/reference offsets, unavailable/replaced elements,
geometry, actual DOFs, legacy admission and out-of-domain points.

Only `src/anyfesolver/solid_trace.py`, the new extension test and this new
document were edited. Mechanics kernels, host assembly, existing tests,
protected result files, ANYsolid source and other repositories were untouched.
No convergence/stability/qualification campaign, installed-artifact check or
publication was run. State: implemented and verified against focused assigned
criteria; independent review/adjudication and W1.4 acceptance remain with parent.
