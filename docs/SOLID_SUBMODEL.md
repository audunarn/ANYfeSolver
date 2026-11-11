# One-way shell -> solid submodel (`anysolver.solid_submodel`)

A converged **linear** shell solution drives the cut boundary of a prepared ANYsolid
`PreparedSolid` (linear tetrahedra, P1 or P2).  The solid is solved once; nothing is fed
back to the shell.  This is the one-way mechanics only: it does not import a regional
mesher (the host supplies `locate`), and it is not the held two-way workflow.

```python
from anysolver.solid_submodel import solve_shell_driven_solid

result = solve_shell_driven_solid(
    fe_model, source_u, prepared,
    cut_groups=["cut_a", "cut_b"],     # boundary (or node) groups, every coefficient driven
    locate=locate,                     # world point (3,) -> iterable of (element_id, natural)
    skin_tractions={"skin": (0.0, 0.0, -1.0e6)},   # external group -> global traction vector
    cancel=None,                       # None | callable() -> bool | object with is_set()
)
```

## Source field

* Admitted shells are the exact classes of `submodel_transfer` (qualified Q4,
  `NativeParityE4PLS3V2DShellElement`); any other shell in the model refuses the source.
  Q4 facets must be planar and strictly convex, V2D triangles non-degenerate with a
  facet-normal reference normal.
* `source_u` must be finite, have one entry per model DOF, and satisfy every support,
  constraint equation and element coupling row of the model (`SOURCE_CONSTRAINT_INCONSISTENT`).
* `locate(x)` receives a world coordinate `ndarray (3,)` and returns `(element_id, natural)`
  pairs on the shell **reference midsurface**: Q4 `(xi, eta)` in `[-1, 1]^2`, V2D `(r, s)` with
  `N = (1-r-s, r, s)`.  Geometry ownership (projection, tolerance, which side) stays with the
  host.  Several hits (a shared edge/corner) are all evaluated and must agree to
  `1e-8` of the source field scale, else `LOCATE_HITS_DISAGREE`.
* Physical displacement at the actual target `x` (all three components, not a reference-surface
  value): with `p = sum_i N_i X_i`, `u_ref = sum_i N_i u_i`, `theta_bar = sum_i N_i theta_i`,
  `u(x) = u_ref + theta_bar x (x - p)`.  The whole displacement is transported from `p` to `x`
  once, so the face/thickness offset of the target is counted exactly once and a varying
  rotation field (including drilling `theta_z`) does not displace the midsurface (`x = p`).
* V2D transverse term.  The normal component of `u_ref` is replaced by the element's own MIN3
  dead-pressure interpolation `w` (`compute_native_pressure_density`, which owns the node
  permutation and the physical-director chart): `u_ref += (w_MIN3 - n . u_ref) n`, with `n` the
  reference normal; then the transport above is applied.  Hand-checked at the centroid of a
  unit right triangle: `w = x*y` gives `1/9` (interpolated nodal `w` is `0`), `w = x^2` gives
  `1/9` (interpolated nodal `w` is `1/3`).  The Q4 needs no correction: its bilinear
  translation reproduces the twist `w = x*y` exactly on a rectangle.  The correction magnitude is
  reported (`diagnostics["source"]["v2d_transverse_correction_max"]`).

## Solid solve

* **All** DOFs of every cut group are prescribed: vertices and, for P2, edge midsides.  A DOF
  shared by several groups (edges/corners) is prescribed once; the number of duplicates is
  `diagnostics["constraints"]["duplicate_dof_count"]`.  Each cut group must carry all three
  components at each of its locations.
* The prescribed set must remove every rigid mode of the solid (`prepared.rigid_modes()`);
  otherwise `UNDERCONSTRAINED` (e.g. a single driven vertex).
* Single-case constrained statics on the free block with symmetric Jacobi scaling, sparse LU and one
  refinement step.  Certified only if the scaled backward error is `<= 1e-10`, the 1-norm condition
  estimate `<= 1e12` and `kappa * eta <= 1e-6`; otherwise `SOLVE_UNCERTIFIED` / `POOR_CONDITIONING`.
  Non-positive free diagonals or a singular factorization give `SOLVE_SINGULAR`; non-finite values
  `SOLVE_NONFINITE`.
* `skin_tractions` maps an external boundary group (not a cut group, sharing no facet with one) to a
  constant global traction vector, applied with `prepared.traction`.

## Result

`SolidSubmodelResult`:

| field | meaning |
| --- | --- |
| `coefficients` | all solid coefficients (cut values prescribed) |
| `nodal_displacements` | `(N, 3)`, aligned to `prepared.mesh.node_ids` |
| `recovery` | `prepared.recover(coefficients)` (true element quadrature) |
| `interface_force`, `interface_moment` | force and moment, global frame, **about the origin**, that the driven boundary exerts on the solid: `r = K q - f` on the prescribed DOFs, `F = sum r_i e_c`, `M = sum x_i x r_i e_c` |
| `skin_force`, `skin_moment` | same resultants of the applied skin load; `interface + skin = 0` |
| `group_resultants` | per cut group `GroupResultant(force, moment, dof_count, shared_dof_count)` |
| `shared_attribution`, `attribution_rule` | DOF -> groups sharing it; a shared DOF's reaction is split **equally** among its groups (`equal_split_among_groups_sharing_the_dof`), so group resultants add up to the total |
| `prescribed_dofs`, `prescribed_values`, `reactions` | the driven set, its values and its reactions |
| `diagnostics` | `certification` (backward error, condition, `kappa*eta`, thresholds, rigid-mode singular values), `residual` (free-block residual, force/moment balance, attribution closure), `constraints`, `source` (hit spread, correction, offsets, element kinds, constraint rows checked), `prepared`, `skin` |

The interface resultants are work-conjugate to the coefficient DOFs, so they are exact for rigid
translation and rotation of P1/P2; they are generalized nodal forces, not a traction field.  The
force/moment balance and free-block residual are checked against the internal-force magnitude
`|K||q|` and refuse with `EQUILIBRIUM_VIOLATED` above `1e-6`.

## Failure and cancellation

All refusals raise `SolidSubmodelError` with a stable `code` (`SOURCE_*`, `LOCATE_*`,
`PREPARED_*`, `CUT_GROUP_INVALID`, `SKIN_INVALID`, `UNDERCONSTRAINED`, `SOLVE_*`,
`POOR_CONDITIONING`, `EQUILIBRIUM_VIOLATED`, `RECOVERY_FAILED`, `CANCEL_INVALID`) and `details`.
`cancel` is polled before every stage and for every driven location; a trip raises
`SolidSubmodelCancelled` (`code == "CANCELLED"`) and returns nothing.  No input is mutated.

## Limits

* Linear statics only; small strain; one load case; no contact, no nonlinear shell or solid.
* Constant isotropic solid material and tet4/P1-P2 as `PreparedSolid` provides; only constant
  skin tractions.
* The shell-to-solid transport is kinematic (displacement and rotation of the reference surface);
  it does not model through-thickness warping, transverse-shear distribution or a stress transfer.
  St. Venant decay: the cut must lie where the shell field is smooth and away from the region of
  interest.
* V2D transverse correction is the MIN3 interpolation of the element, valid for planar triangles;
  non-planar Q4 or V2D facets are refused.  `locate` must return reference-midsurface points: for
  every hit (primary and shared) the target must lie on the element's reference normal line through
  `p`.  The tangential residual `|(x - p) - n n.(x - p)|` must not exceed
  `1e-7 * (largest node-to-node distance of the element) + 64 eps * (largest coordinate magnitude of
  the element nodes and target)`, else `LOCATE_TANGENTIAL_MISMATCH` (details: `element_id`, `target`,
  `natural`, `reference_point`, `tangential_residual`, `allowed`).  The signed normal offset is not
  limited (any face/thickness offset is admissible), and the allowance is independent of unrelated
  point distances.
* The sign convention of `interface_*` is the force on the solid; the force on the shell is the
  negative, and no shell reaction is computed or applied (one-way).
* Hit agreement across a nonconforming V2D/Q4 interface can legitimately exceed the `1e-8`
  tolerance and then refuses; this is deliberate.
