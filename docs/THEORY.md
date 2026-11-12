# FE Solver Theory Notes

This note records the theoretical basis, equation audit, and validity limits
for the production `anyfesolver` package. The implementation target is an
ANYfeSolver beam-shell solver, not a general-purpose nonlinear FE code. A cited
formulation is a design anchor; qualification still comes from the executable
tests and comparison evidence described in
[`QUALITY_CONTROL.md`](QUALITY_CONTROL.md).

## Units and DOFs

- SI units are used internally: m, N, Pa, kg.
- Every node has six DOFs ordered as `ux, uy, uz, rx, ry, rz`.
- Shells, beams, boundary conditions and MPCs share the same global DOF space.
- Fixed DOFs and beam-shell MPC slave DOFs are eliminated by the transformation
  `u = T q + u0` before static, buckling and transient solves.

## Shell Element

`ShellElement` is a 3/6-node triangular or 4/8-node quadrilateral
Mindlin-Reissner shell. At each integration point the element builds a local
orthonormal frame from the surface tangents:

```text
local x = projected xi tangent
local z = shell normal
local y = local z cross local x
```

Global nodal translations and rotations are transformed to that local frame
before evaluating membrane strain, bending curvature and transverse shear.
For an isotropic material the elastic constitutive blocks are:

```text
D_membrane = E h / (1 - nu^2) * [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu)/2]]
D_bending  = E h^3 / (12 (1 - nu^2)) times the same plane-stress block
D_shear    = kappa G h I, kappa = 5/6
```

For a homogeneous orthotropic material, the 3-D engineering compliance is
supplied in material axes and Voigt order
`[11, 22, 33, 23, 13, 12]`, with engineering shear strains. Plane stress uses
the `[11, 22, 12]` compliance submatrix:

```text
Q = inverse(S[[11,22,12],[11,22,12]])
D_membrane = h Q
D_bending  = h^3 Q / 12
D_shear    = kappa h diag(G13, G23)
```

Engineering strain/stress transformations rotate `Q` and the transverse-shear
block from material axes into the integration-point shell frame, retaining
off-axis extension/shear coupling. Drilling stabilization scales with `G12`.
`material_direction` is projected into the shell plane; a parallel-to-normal
projection fails validation. `material_angle_deg` is then applied
right-handed about the positive shell normal. If no direction is supplied,
the angle is measured from shell-local x.

A pre-integrated generalized shell section may instead provide

```text
[N]   [ A   B ][epsilon]
[M] = [B^T  D ][ kappa ]

Q_s = As gamma_s
```

in engineering membrane order `[11,22,12]` and shear order `[13,23]`.
`A` and `As` have units N/m, `B` has units N, and `D` has units N*m. The
assembled `ABD` and `As` laws must be symmetric positive definite. The same
`material_direction`/`material_angle_deg` convention rotates section axes into
the shell frame. This is a linear, already-integrated section law: it supports
unbalanced membrane-bending coupling but contains no ply locations or ply
constitutive data. Recovery therefore returns exact generalized
strains/resultants and deliberately omits fabricated surface, von Mises, Hill,
or ply stresses.

The 4-node shell uses an MITC4-style assumed natural shear interpolation,
following the continuum-mechanics basis of
[Dvorkin and Bathe's four-node shell formulation](https://web.mit.edu/kjb/www/Publications_Prior_to_1998/A_Continuum_Mechanics_Based_Four-Node_Shell_Element_for_General_Nonlinear_Analysis.pdf).
The covariant shear strains are sampled at the four edge midpoints and
interpolated to the 2x2 integration points. This avoids the excessive
thin-plate shear stiffness of a fully integrated displacement Q4 without
introducing the one-point reduced-shear hourglass mode.

The 8-node shell uses full 3x3 membrane/bending integration and reduced 2x2
transverse shear integration. When
`ShellElement(..., reduced_integration=True)` is used for S8R/Q8R,
membrane/bending, mass, geometric stiffness, and stress recovery are evaluated
at the reduced 2x2 rule. The S8R/Q8R path adds a stiffness-scaled
nullspace-projection hourglass term: the reduced-integration free-element
nullspace is computed, the six rigid-body modes are projected out, and the
remaining modes receive positive stabilization. This is an experimental
implementation, not a production-qualified Q8R formulation. The current
stabilization can dominate very thin bending and its rotary mass behavior needs
further work. Q8R is therefore excluded from the qualified capability matrix
and from nonlinear shell batching. Future work should replace it with a
bending-aware stabilization and verify distorted-element, modal, and
thin-limit convergence against the established hourglass-control literature,
including [Flanagan and Belytschko](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620170504)
and [Belytschko and Bindeman](https://www.sciencedirect.com/science/article/pii/004578259390124G).

All shell topologies include a small drilling stabilization strain:

```text
theta_z - 0.5 * (dv/dx - du/dy)
```

This gives the otherwise free drilling rotation a finite stiffness while
leaving rigid rotation about the shell normal strain-free.

### Stress output frames

`compute_stresses` returns membrane/bending/shear components in the
**per-integration-point local shell frame** (local x follows the projected xi
tangent) by default.  On rectangular and parallelogram elements that frame is
constant; on skewed or distorted elements it rotates from point to point, so
default components must not be compared against global-axis expectations.
Frame-invariant outputs (von Mises, principal values) are unaffected.  Pass
`return_global=True` to additionally obtain `global_*_top/bot` surface stress
components in the global frame; the S4 constant-strain patch metrics compare in
that frame.  The 3-node/6-node triangle shells follow the same convention.

## Beam elements

`BeamElement` is a 2-node and `QuadraticBeamElement` a 3-node Timoshenko beam.
Both include axial, two transverse-shear, two bending, and torsional response
in a local orthonormal frame. Their geometric stiffness is based on the
reference axial force. The quadratic element evaluates quadratic interpolation
along a straight reference axis: its middle node must lie at the chord
midpoint, within numerical tolerance. A displaced midside node is not a curved
beam definition and is rejected. Curved rings and arches must be discretized
as straight 2-node segments until a true curved-isoparametric beam with
objective frame interpolation and matching mass/geometric stiffness is
implemented and qualified.

For orthotropic beams, material axes 1/2/3 coincide with beam-local x/y/z.
Axial and both bending terms use `E1`; local-xy and local-xz transverse shear
use `G12` and `G13`. Because a scalar material shear modulus and section `J`
cannot represent general orthotropic torsion, the section must supply positive
`torsional_rigidity` directly in N*m^2. Isotropic beams continue to use `G*J`.

A generalized beam section replaces the uncoupled scalar rigidities with

```text
[N, Vy, Vz, T, My, Mz]^T
  = K_section
    [eps_x, gamma_xy, gamma_xz, kappa_x, kappa_y, kappa_z]^T
```

where `K_section` is a finite symmetric positive-definite 6x6 matrix in
beam-local axes. Off-diagonal terms represent extension-twist, bend-twist,
shear-bending, and other sectional couplings. Beam2 and straight Beam3 use the
law in linear and elastic von Karman response; Beam2 also participates in the
existing corotational formulation. Fiber plasticity and physical fiber-stress
recovery are unavailable because a generalized matrix carries no fiber
geometry or material history. Recovery is explicitly resultants-only.
Likewise, the generalized stiffness does not define the centroidal polar
radius needed by the Wagner torsional geometric-stiffness term. That term is
omitted unless the element supplies
`cross_section["geometric_polar_radius_squared"]` explicitly in m^2; legacy
`area`, `Iy`, and `Iz` keys are not inferred for a generalized section.

## Mass and Pressure Loading

Shell mass is integrated consistently with the shell shape functions.  The
translational mass scales with `rho h`; rotary inertia scales with
`rho h^3 / 12`. The 2-node beam uses translational and rotary lumping by
default, or a consistent interpolation-based mass matrix when
`consistent_mass=True` is set in its section data. The straight-sided 3-node
beam always integrates a consistent translational and section-rotary mass
matrix. Explicit point masses contribute to the assembled mass matrix and to
total-mass, center-of-mass, and inertia diagnostics.

Generalized shell sections may independently supply `mass_per_area` and
`rotary_inertia_per_area`; either omitted value falls back to its corresponding
homogeneous expression above. Generalized beam sections may supply a complete
local 6x6 inertia matrix per unit length; otherwise legacy density/section
mass is retained.

Total mass, center of mass, and origin/center inertia diagnostics are projected
from the assembled mass matrix with rigid-body translation and rotation
fields. They therefore include the same section overrides, rotary inertia,
beam orientation, point masses, and translation-rotation coupling as modal and
dynamic analysis. For a generalized beam inertia to define these scalar
summaries, its local block form in `[v, omega]` order must be

```text
M_section = [[mu I, -skew(h)],
             [skew(h), J_axis]],   h = mu c
```

where `mu` is mass per length and `c` is the sectional center-of-mass offset
from the beam reference axis. `J_axis` is the rotary inertia per length about
that reference axis and may contain products of inertia. An arbitrary positive-
definite 6x6 matrix remains usable as a generalized dynamic inertia, but if its
translational block is anisotropic or its coupling block is not skew-symmetric,
there is no unique scalar mass or center of mass; `calculate_mass_properties`
rejects that diagnostic explicitly.

Shell pressure is assembled as a consistent nodal load. In natural
coordinates, with current or reference covariant surface tangents
`a_xi = dx/dxi` and `a_eta = dx/deta`,

```text
f_i(u) = integral N_i p (a_xi cross a_eta) dxi deta
```

where the sign follows the element node order. Pressure is a dead load by
default. With `LoadCase(..., follower_pressure=True)`, nonlinear static and
arc-length solves evaluate the area vector from the current midsurface. The
exact translational `3 x 3` block of its external-force derivative is

```text
d f_i / d u_j
  = integral p N_i (-N_j,xi [a_eta]_x + N_j,eta [a_xi]_x) dxi deta
```

where `[a]_x b = a cross b`. Therefore the Newton operator is

```text
K_eff = K_internal - K_external
```

and is generally nonsymmetric for an open pressure patch. No independent
rotational pressure moment is introduced because pressure virtual work is
conjugate to midsurface translation. This follows the standard distinction
between dead and follower distributed loads described in the
[Abaqus distributed-load reference](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEPRCRefMap/simaprc-c-loaddistributed.htm).
The implementation is checked by analytical load resultants, rigid-rotation
objectivity, finite differences of the load tangent, nonlinear equilibrium,
and the
[Abaqus pressurized-ring buckling benchmark](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-ringbuckling.htm),
which illustrates why pressure-load stiffness cannot be omitted.

## Linear equilibrium and vibration

With the affine constraint map `u = T q + u0`, linear static equilibrium is
reduced without penalty terms:

```text
(T.T K T) q = T.T (F - K u0)
r = K u - F
```

where `r` is the full residual from which supported-DOF reactions and MPC-force
diagnostics are derived. Free-free models use a rigid-body/nullspace treatment
instead of artificial grounding.

Undamped vibration solves the constrained generalized eigenproblem

```text
(T.T K T) phi = omega^2 (T.T M T) phi
f = omega / (2 pi)
```

with rigid-body modes retained or filtered according to the requested free-free
analysis. Modal mass and normalization use the same assembled mass matrix,
including explicit point masses.

## Linear Transient Dynamics and Slamming V1

The transient solver advances the constrained/reduced linear system:

```text
M qdd + C qd + K q = F(t)
C = alpha M + beta_R K
```

The default method is
[Newmark average acceleration](https://ascelibrary.org/doi/10.1061/JMCEA3.0000098):

```text
beta_N = 1/4
gamma_N = 1/2
```

`TransientConfig(hht_alpha=...)` activates the
[Hilber-Hughes-Taylor alpha method](https://onlinelibrary.wiley.com/doi/abs/10.1002/eqe.4290050306)
with `-1/3 <= alpha <= 0` (`alpha = 0` reproduces plain Newmark). The
equilibrium is enforced in the alpha-weighted form

```text
M a_{n+1} + (1+alpha) (C v_{n+1} + K q_{n+1}) - alpha (C v_n + K q_n)
    = (1+alpha) F_{n+1} - alpha F_n
```

with the HHT-optimal Newmark parameters `gamma = 1/2 - alpha` and
`beta = (1 - alpha)^2 / 4` derived automatically when `beta`/`gamma` are left
at their defaults. For linear problems with the standard HHT assumptions this
parameterization is second-order accurate and unconditionally stable in the
stated range, and it introduces controlled high-frequency numerical
dissipation. Those properties are not, by themselves, a convergence guarantee
for nonlinear contact, material softening, or changing topology. The same
alpha-weighting is applied to the sphere-impact solvers: the linear path
weights the structural external+contact load, and the nonlinear path weights
the internal force and damping terms in the Newton residual with the tangent
scaled by `(1+alpha)`.  The rigid sphere itself keeps the plain Newmark
kinematics; the alpha-weighting acts on the structure side of the contact.

For constant `dt`, `K`, `M` and Rayleigh damping, the solver reuses the sparse
factorization of the effective stiffness:

```text
K_eff = K + a0 M + a1 C
```

The slamming v1 load model is prescribed pressure over selected shell elements.
`PressurePatch` supports explicit element IDs, a centroid-selected rectangular
or circular area, or a custom selector callback.  Pressure magnitude is a
constant, a time table or a callable.  This is a structural response model to a
given pressure history; it is not fluid-structure interaction and does not add
hydrodynamic added mass.

Validity limits for v1 slamming:

- linear transient response about the undeformed structure,
- prescribed shell-normal pressure only,
- centroid inclusion for patch area selection,
- fixed time-step integration,
- no cavitation, water-entry kinematics or pressure feedback.

The limited rigid-sphere collision mode is a structural transient contact
extension, not a general contact solver. A single rigid sphere has user
supplied mass, radius, initial point, travel vector and speed. Contact is
detected against shell surfaces and, when explicitly enabled, beam-axis
segments using the current deformed translational geometry. The normal contact
force is frictionless, compression-only penalty contact:

```text
F_n = max(k_p g - c_n v_n, 0)
```

where `g` is penetration, `v_n` is relative normal velocity from the contact
target to the sphere center, `k_p` is penalty stiffness and `c_n` is optional
contact damping. The equal and opposite shell load is distributed with shell
shape functions at the closest projected surface point. Beam contact splits
the reaction between the segment end nodes.

Production controls add internal event substepping for fast sphere travel,
automatic penalty recommendation from impact inertia and representative shell
stiffness, optional top/bottom/signed shell-thickness contact surfaces, and
duplicate adjacent-element reduction to the deepest active contact point by
default.  The linear impact solver additionally auto-substeps while contact is
active to keep the step below a fraction of the penalty period
(`0.2 sqrt(m/k)`), capped by `max_event_substeps`, so a coarse `dt` with a
stiff penalty stays accurate and convergent instead of producing spurious
forces; free flight and post-separation are not refined.  The nonlinear
implicit path relies on its time-step cutbacks for the same protection.
`validate_contact_configuration` provides structured preflight diagnostics for
unsupported target meshes, missing density, excessive travel per step and
time-step/contact-period warnings. After nonlinear Newton cutbacks, a bounded
distress carryover can pre-subdivide subsequent base steps and then decay as
convergence recovers; `preemptive_substep_count` makes that behavior explicit.

`SphereContactConfig(beam_contact=True)` additionally enables direct
sphere-to-beam contact: each 2-node beam contributes one contact segment and
each 3-node quadratic beam two, with a circular section proxy radius
(`cross_section["contact_radius"]`, defaulting to the equivalent-area circle).
Contact is closest-point-on-segment penalty contact against the deformed beam
axis, with the reaction split linearly between the segment end nodes.  This
covers edge-on strikes on stiffeners and girders; it remains a line-contact
engineering model, not profile-resolved surface contact.

Validity limits for sphere impact:

- one rigid sphere,
- shell midsurface targets plus (opt-in) beam axis segments,
- frictionless normal contact only,
- no rigid-lid, MPC or shell-shell contact targets,
- no spin, rolling, friction or crack propagation.

`NonlinearTransientConfig(kinematics="corotational")` runs the nonlinear
impact equilibrium with the corotational element response (large rigid
rotations with plasticity in the corotated frame); the default remains von
Karman.  The nonlinear impact energy diagnostics account for structural
kinetic energy, an internal work measure from the committed internal force
(exact strain energy while elastic), and the sphere kinetic energy; the
instantaneous penalty-spring energy during contact is the only term outside
the balance.

Material-nonlinear impact:

- `NonlinearTransientConfig(enabled=True)` activates an opt-in implicit
  Newmark/Newton impact path.  The equilibrium uses structural inertia,
  damping, nonlinear element internal force, and iterated sphere-shell penalty
  contact in each time step.
- Shells use the existing layered plane-stress J2 plasticity and beams use the
  existing nonlinear/fiber response where configured.  The geometric scope is
  the current solver nonlinear element theory, not full updated-Lagrangian
  shell impact.
- `PlasticImpactDamageConfig` drives softening/erosion from committed plastic
  state after a converged nonlinear substep.  Failed Newton trials do not
  commit plastic state or damage.  Three criteria are available:
  - `"fixed"`: element fails when max equivalent plastic strain exceeds the
    manual `threshold`.
  - `"mesh_scaled_gl"`: same trigger, but the critical strain is computed per
    shell element as `0.056 + 0.54 * (thickness / element_length)` (GL /
    RP-C208-style linear mesh scaling with `element_length = sqrt(area)`), so
    coarse elements that cannot resolve necking fail earlier.
  - `"rtcl"`: RTCL (Rice-Tracey / Cockcroft-Latham,
    [Tornqvist 2003](https://backend.orbit.dtu.dk/ws/portalfiles/portal/5443674/rt.pdf))
    ductile damage accumulation on top of the same mesh-scaled critical strain. Per
    integration point (Gauss point x thickness layer), the damage increment is
    `dD = w(eta) * d(eps_p) / eps_cr` where `eta` is the stress triaxiality of
    the return-mapped plane-stress state and the weight `w` is 0 for
    `eta <= -1/3` (compression), the Cockcroft-Latham branch for shear-
    dominated states (`w(0) ~ 0.577`), 1 at uniaxial tension (`eta = 1/3`)
    and `exp(1.5 eta - 0.5)` beyond (`~1.65` at equibiaxial tension).  The
    element softens/erodes on `max_point D`.  Beam fiber sections use the
    uniaxial limits (tension fibers weigh 1, compression fibers 0).  This is
    the criterion validated in ship-collision studies; compared to plain
    plastic-strain thresholds it does not erode the compressed side of a dent
    and accumulates monotonically, which is markedly more stable.
  - `"rtcl_modified"`: RTCL with the critical strain recalibrated by the
    plane-strain weight `w_ps = exp(sqrt(3)/2 - 1/2) ~ 1.4424`, so plane-
    strain tension (`eta = 1/sqrt(3)`, the governing state for plate necking
    and the state the GL curve derives from) reaches utilization 1 exactly at
    the mesh-scaled limit.  Uniaxial tension then fails later than plain RTCL
    by the same factor.
- The contact tangent is currently approximate: active contact forces are
  updated within Newton iterations and convergence is protected by line search
  and cutback rather than an exact contact active-set tangent.

Impact fracture / contact erosion:

- `ImpactFractureConfig` optionally erodes contact targets in
  `solve_transient_sphere_impact` after a converged contact substep.  Its
  triggers are contact observables (normal force, penetration ratio or a
  sphere-area contact-pressure proxy) and are therefore geometry-agnostic, so
  both shell midsurface targets and (with `SphereContactConfig(beam_contact=
  True)`) beam segment targets can fracture.  The `max_deleted_fraction` guard
  counts all erodible contact targets.
- `ImpactDamageConfig` is the capacity-based impact screening layer.  It
  estimates local contact patch area from sphere radius, penetration, shell
  thickness and a configured minimum area, then evaluates contact pressure,
  impulse per area and an equivalent-plastic-strain demand proxy against
  yield, ultimate-proxy or user capacity.  This is an area/contact-pressure
  model and therefore screens shell midsurface contact only; beam line contact
  is out of scope (a preflight warning is emitted when beam contact targets are
  active) — use `ImpactFractureConfig` or the nonlinear
  `PlasticImpactDamageConfig` for beam erosion.
- Damage may be accumulated across repeated contact substeps or applied as an
  instant threshold.  Above `softening_start`, element stiffness, mass and
  contact participation are reduced toward the residual fraction; elements are
  eroded when damage reaches `delete_at`.  Optional neighbor smoothing holds a
  one-step isolated spike until repeated contact evidence is present.
- Eroded elements (shells or beams) are removed from later contact searches and
  contribute only residual stiffness/mass to later transient solves; the
  residual-scaling matrix rebuild is element-agnostic.  Nodes, DOFs, MPCs and
  connected members remain in the model.
- Diagnostics separate simple threshold fracture triggers from capacity-based
  damage triggers, while `erosion_summary` reports the combined eroded/softened
  element set used by the transient matrices and contact search.
- This is an engineering erosion model for impact screening.  It is not ductile
  fracture mechanics, crack growth, cohesive damage, tearing, remeshing or
  arbitrary-contact impact.

## Buckling and Nonlinear Statics

Linear buckling solves:

```text
K phi = lambda (KG + K_load) phi
```

with positive `KG` representing destabilizing compression in the supplied
reference stress/resultant state. The beam contribution is driven by axial
force. For shells, the implemented total-Lagrangian initial-stress operator
uses the local Mindlin director field

```text
r(x, y, z) = [u + z ry, v - z rx, w]
delta^2 Pi_sigma = integral_V (grad delta r)^T sigma (grad Delta r) dV
```

and the in-plane through-thickness stress moments

```text
N = integral sigma dz
M = integral z sigma dz
H = integral z^2 sigma dz
```

at each midsurface integration point, expressed in the package's
compression-positive convention. `N` acts on gradients of all three midsurface
translations, `H` acts on the `rx` and `ry` director gradients, and `M`
provides the signed translation/director coupling implied by
`u(z) = u + z ry` and `v(z) = v - z rx`. For uniform through-thickness stress,
`H = N h^2 / 12`. The construction is consistent with the
Reissner-Mindlin director kinematics surveyed in this
[geometrically nonlinear Reissner-Mindlin review](https://doi.org/10.1007/s11831-021-09702-7).

This is fuller than a transverse-displacement-only membrane `KG`, but it is not
a claim of a complete geometrically exact finite-rotation shell. It includes
no drilling-director or transverse-normal-stress contribution. `K_load` is the
current-area follower-pressure tangent when a reference follower load is
supplied. The present symmetric eigensolver accepts it only when the
*constrained* follower tangent satisfies the requested symmetry tolerance;
otherwise it returns `unsupported_nonsymmetric_follower_pencil`. General
complex nonconservative eigenanalysis remains outside scope. The thin-ring
pressure qualification follows the sensitivity highlighted by the
[Abaqus ring benchmark](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-ringbuckling.htm).

The incremental nonlinear static path uses von Karman shell kinematics,
beam-column geometric coupling and optional layered isotropic J2 or
orthotropic Hill-48 plane-stress plasticity with DNV-RP-C208-style material
curves. It is suitable for restrained
plate/stiffened-panel response and pre/post-buckling capacity checks in the
implemented range.

### Plane-stress J2 plasticity

At each shell integration point and thickness layer, the membrane/bending
strain is evaluated as

```text
epsilon(z) = epsilon_m + z kappa
```

and passed to a plane-stress J2 return map. The local nonlinear solve enforces
the yield condition and `sigma_zz = 0`; its discrete algorithmic tangent is
used by both scalar and vectorized assembly. The implementation follows the
plane-stress return-mapping framework of
[Simo and Taylor](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620220310)
and the consistent-tangent principle of
[Simo and Taylor](https://www.sciencedirect.com/science/article/pii/0045782585900702).

For the converged plastic multiplier `dl`, define

```text
t = C (epsilon - epsilon_p,n)
A = I + dl C P
M = inverse(A)
sigma = M t
m = P sigma
g = sqrt((2/3) sigma.T P sigma)
beta = (2/3) sigma_y H
v = M C m
a = 1 - 2 beta dl / (3 g)
```

Exact implicit differentiation of the same discrete consistency equation gives

```text
C_alg = M C - a (v outer v) / (a m.T v + beta g)
```

on a smooth yielding branch; elastic points return `C`. This analytical
consistent tangent is the production default. A central-difference derivative
of the identical return map remains available as the qualification oracle and
is selected automatically only when an analytical row is non-finite or has
pathological amplification. A large elastic condition number alone does not
replace an otherwise finite exact tangent. Numerical perturbations are scaled
by the elastic-column magnitude and advanced to representable floating-point
neighbors when required.

Ordinary consistency iterations use Newton's method. Any point that does not
meet the scaled yield-residual tolerance is completed with a bracketed
bisection for the supported monotonic isotropic-hardening curves. The wrapper
raises `PlaneStressConvergenceError` if the safeguarded solve still cannot
satisfy the local residual; an unconverged stress update is never returned as
valid merely because its tangent is finite. At the elastic/plastic switch and
the piecewise DNV hardening-curve corners the derivative is directional, so
centered-difference agreement is not claimed exactly at those nonsmooth
points.

For a layer tangent `C_alg`, membrane-bending coupling retains both transpose
terms:

```text
K_layer = B_m.T C_alg B_m
        + z B_m.T C_alg B_b
        + z B_b.T C_alg B_m
        + z^2 B_b.T C_alg B_b
```

The scalar and accelerated paths use the same return-map contract; reduced
Q8R shells remain on the scalar path because their hourglass contribution is
not part of the accelerated local tangent.

### Plane-stress Hill-48 plasticity

`Hill48Yield(X, Y, Z, S12, S13, S23)` defines symmetric directional strengths
in material axes. The standard quadratic Hill coefficients form

```text
F(s2-s3)^2 + G(s3-s1)^2 + H(s1-s2)^2
  + 2L t23^2 + 2M t13^2 + 2N t12^2 = 1
```

and the shell reduction acts on `[s11, s22, t12]`. The implementation performs
a backward-Euler associated-flow update in material axes, stores material-axis
plastic strain and Hill equivalent plastic strain, and rotates converged
physical stress back to the shell frame for resultants and recovery. A scalar
hardening curve scales every directional strength by
`flow_stress(alpha)/flow_stress(0)`, preserving their ratios. The exact local
Newton derivative is the production tangent; a central-difference derivative
qualifies it and replaces invalid rows, while an unconverged consistency solve
fails closed. In the isotropic-strength limit, the stress-valued Hill
equivalent measure reduces to von Mises.
When no hardening curve is supplied, the Hill strengths define an
elastic-perfectly-plastic response. Initial residual stresses are transformed
to material axes and checked against the same current Hill surface.

Orthotropic beam fiber plasticity remains opt-in. Fibers use `E1` and the
longitudinal strength `X`; transverse beam shear and torsion remain elastic.
For orthotropic results, physical `von_mises` retains the elastic transverse
shell shear or beam shear/torsion envelope. The separate
`equivalent_stress_measure="hill48"`, `equivalent_stress`, and
`hill_utilization` identify the constitutive Hill measure; utilization uses the
current hardening-scaled strength. Beam Hill recovery is explicitly scoped to
the return-mapped longitudinal fibers, with the mixed elastic components
identified in provenance.

### Corotational kinematics (large rigid rotations)

`solve_static_nonlinear(..., kinematics="corotational")` activates an opt-in
element-independent corotational formulation for large rigid rotations with
small elastic strains.  Per element, the rigid-body rotation `R_rig` is
extracted from the deformed geometry (shell midsurface frame at the element
center; beam axis alignment plus mean axial twist), the nodal displacements
are pulled back to the reference configuration,

```text
u_d (translation) = R_rig^T (x - x_c) - (X - X_c)
u_d (rotation)    = rotvec(R_rig^T exp(skew(theta)))
```

the element's own nonlinear response acts on the deformational part, and
forces are rotated forward: `f = E f_local(u_d)` with
`E = blockdiag(R_rig, ...)`. Verified anchors: internal-force invariance
under rigid rotations up to 170 degrees (machine precision, where von Karman
produces GN-scale spurious forces), and the cantilever roll-up under end
moment matching the analytic circle through 180 degrees of tip rotation.

Two tangent modes are available:

```text
rotated:    K = E k_local E^T
consistent: K = E k_local D + (S + E k_local U) G
```

For the consistent tangent, `D` is the fixed-frame pull-back derivative, `U`
is the deformational pull-back sensitivity to rigid rotation, `S` rotates the
internal force with the frame, and `G = d omega / d u` is the extracted-frame
sensitivity. `D`, `U`, and `S` use the exact rotation-vector Jacobians; `G` is
evaluated by bounded central differences of the shell or beam frame. The
result is deliberately not symmetrized because additive rotation coordinates
and frame terms produce a generally nonsymmetric Jacobian. Element-level
finite-difference checks verify the complete force derivative. This is an
application of the consistent-Jacobian principle represented by
[Simo and Taylor](https://doi.org/10.1016/0045-7825(85)90070-2); it should not
be read as adopting that paper's constitutive model as the corotational
formulation.

`corotational_tangent="auto"` keeps the lower-cost rotated tangent for ordinary
corotational solves and selects `"consistent"` when follower pressure is
active. Users may request `"consistent"` explicitly for demanding
large-rotation Newton solves. An explicit `"rotated"` tangent is rejected for
the follower-pressure/corotational combination because the complete
nonsymmetric equilibrium Jacobian is then required.

Validity limits:

- small strains and small deformational rotations per element; the
  deformational displacements are routed through the elements' own nonlinear
  local responses, so layered shell J2 plasticity, beam fiber plasticity and
  the local von Karman coupling are active in the corotated frame (plastic
  state is objective under rigid rotation); fiber shear/torsion stay elastic
  as in the von Karman path, and fracture/erosion remains unsupported in
  corotational mode;
- the consistent tangent includes the extracted-frame derivative, but its
  numerical `G` evaluation is more expensive and depends on a finite-difference
  scale; the rotated tangent therefore remains the default when follower
  pressure is absent. The automatic/rescue residual-norm line search is still
  disabled in corotational mode because the first frame-rotation excursion can
  otherwise be rejected;
- the pull-back has an intrinsic residual roundoff floor of roughly
  `eps * ||K_e|| * L` per element; use relative tolerances of 1e-5 to 1e-6 and
  realistic load magnitudes;
- eccentric beam-shell MPC couplings keep fixed eccentricity directions and
  should not be used across strongly rotating regions;
- the default `kinematics="von_karman"` path is unchanged and remains the
  route for plastic capacity analyses.

### DNV-RP-C208 Capacity Workflow Anchors

The nonlinear capacity workflow is aligned with
[DNV-RP-C208](https://www.dnv.com/energy/standards-guidelines/dnv-rp-c208-determination-of-structural-capacity-by-non-linear-finite-element-analysis-methods/):

- key parameters such as element type, mesh, material curve, imperfections and
  residual-stress representation should be selected conservatively or
  calibrated against a comparable standard/test case,
- true stress / true plastic strain material curves should be used consistently
  with the element formulation,
- permanent loads should be applied before environmental/proportional loads in
  nonlinear analyses,
- buckling capacity checks should include equivalent imperfections and may use
  scaled eigenmodes or standard imperfection patterns,
- displacement control is useful when the force-controlled run must be driven
  beyond the load limit to identify the peak response.

Implemented interfaces:

- `dnv_c208_steel_properties(grade, thickness, thickness_class="auto",
  fractile="low")` is the canonical validated table lookup.
  `dnv_c208_steel_curve(...)` builds the corresponding low-fractile
  S235/S275/S355/S420/S460 section 4.6.6 curve. Automatic thickness selection
  fails closed outside the tabulated range. Mean curves are deliberately not
  guessed; supply explicit data through `curve_from_properties` when a
  mean-capacity study is required.
- `ImperfectionField`, `EigenmodeImperfection`, `StandardImperfection` and
  `CompositeImperfection` describe stress-free nodal reference-geometry offsets.
  `apply_imperfection()` modifies coordinates before the nonlinear solve, so
  zero displacement in the imperfect model has zero internal force.
- Standard deterministic imperfections include member bow (default `L/300`),
  plate sinusoidal half-wave (default `s/200`) and flange/outstand twist
  (default `0.02 rad`).  The defaults correspond to the reviewed DNV table, but
  users should still calibrate or override amplitudes when the failure mode or
  tolerance class requires it.
- `NonlinearLoadProgram` applies ordered stages. The common DNV sequence is a
  permanent stage first and an environmental/pressure/compression stage second.
  Adaptive increments are capped at every cumulative stage boundary so each
  endpoint is a converged material-state commit.
- `DisplacementControl` augments the Newton system with a scalar displacement
  constraint and a load proportionality factor unknown, allowing monotonic
  capacity tracing past a simple force-control limit. For a multi-stage load
  program, all preceding stages are first equilibrated under force control and
  committed. The final controlled stage restarts from that displacement and
  material state, retains the preceding loads as constant terms, and
  interpolates from the preloaded control value to the requested absolute
  target.

### Arc-length continuation

`anyfesolver.arc_length.solve_static_arc_length()` adds a bounded
[Crisfield-style](https://doi.org/10.1016/0045-7949(81)90108-5) spherical
constraint to the same nonlinear element, material-state, load, and constraint
machinery:

```text
R(q, lambda) =
    F_constant(q) + lambda F_reference(q) - F_internal(q)
K_eff = K_internal - K_constant - lambda K_reference
dq.T W dq + alpha^2 dlambda^2 = ds^2
```

It is intended to cross a first limit point and retain a small, guarded
descending branch. Controls limit the load increment, number of steps,
post-peak count or load fraction, and maximum nodal translation. It does not
perform automatic bifurcation branch switching, nonlinear free-free
continuation, or unrestricted collapse tracing. Current-area follower pressure
is supported for the constant and/or proportional load cases; its exact load
tangent is included in `K_eff`, and a general factorization is used for the
resulting nonsymmetric system. See
[`ARC_LENGTH.md`](ARC_LENGTH.md) for the API and acceptance limits.

Material modelling:

- Isotropic shells use layered plane-stress J2 plasticity and orthotropic
  shells use material-axis Hill-48 plasticity through Gauss-Lobatto thickness
  layers. Result diagnostics include equivalent plastic strain, compressed-side
  plastic strain, layer strain extrema, and the applicable equivalent-stress
  measure when plastic state is available.
- Beam/stiffener plasticity is opt-in through `FiberSectionPlasticityConfig`.
  The beam fiber model integrates uniaxial axial/bending stress over a section
  grid scaled to the supplied `A`, `Iy` and `Iz`; shear and torsion stay
  elastic. If `web_height`, `web_thickness`, `flange_width`, and
  `flange_thickness` are supplied, the grid follows web/flange strips and is
  recentered/rescaled to preserve the section constants. Otherwise a generic
  rectangular grid is used. Both the 2-node and 3-node quadratic beam implement
  the von Karman beam-column coupling and fiber response: the 2-node element
  uses end-difference strain measures, while the 3-node element evaluates the
  same measures at its Gauss points from quadratic interpolation.

### Material-history and patch stress recovery

`recover_stress_result()` is the unified recovery entry point. It accepts a
nonlinear or arc-length result, or explicitly supplied committed element
states. When a result is supplied, recovery defaults to that result's
displacement vector and rejects a mismatching explicit vector. Provenance is
reported per element and per component, so a caller can distinguish
`committed_shell_layer_state`, `committed_beam_fiber_state`, and explicitly
labelled elastic fallback.

For shells, committed return-mapped physical layer stresses are integrated with the
same Gauss-Lobatto layer positions and weights used by the nonlinear
formulation. The output includes membrane and bending stress measures, exact
top/bottom layer values, conventional physical von Mises stress, and the
labelled Hill equivalent/utilization for orthotropic state. For fiber beams,
the stored fiber coordinates, weights, and
stations give axial force and biaxial bending resultants; the material-history
isotropic von Mises envelope remains the uniaxial fiber-stress envelope when a
plastic constitutive history is active for backward compatibility; orthotropic
recovery reports the conventional mixed physical invariant. Transverse shell
shear and beam shear/torsion
are matching elastic reconstructions rather than return-mapped state variables.
They remain available as individual components and in the explicitly labelled
`mixed_reconstruction_von_mises` diagnostic, but are not folded into the
hardening-curve-consistent `von_mises` field. A purely elastic nonlinear state
instead retains the full mixed elastic equivalent stress as primary. In
corotational analyses, recovery uses the objective deformational pull-back,
current coordinates, and current corotated frame. Material history can only
be used when compatible committed states were retained; missing or invalid
state is a visible elastic fallback, never a silent plastic-stress
reconstruction.

Passing `PatchRecoveryConfig` enables a guarded linear least-squares,
[Zienkiewicz-Zhu-style](https://doi.org/10.1002/nme.1620330702) shell surface
patch fit. Qualification is deliberately narrower than the general
superconvergent-patch theory: only full-integration Q4 or Q8 neighborhoods with
consistent topology, material, thickness, provenance, orientation, local
planarity, sufficient rank, and bounded conditioning are accepted. Rank or
conditioning failures fall back to Gauss-to-node extrapolation and averaging
inside the same continuity region. Material, thickness, topology, provenance,
or geometric discontinuities produce separate `nodal_regions`; values are
never cross-averaged across the discontinuity. Q8R, warped/curved
neighborhoods, and incomplete neighborhoods remain outside the qualified fit.
The optional indicator is a normalized global top/bottom surface-stress L2
discrepancy between raw and recovered stresses. It is explicitly *not* a
compliance-weighted ZZ energy-norm error estimate and must not be used as one.

The material-point scope is homogeneous isotropy or orthotropy. General 3-D
anisotropic materials, ply-stack integration, ply stresses,
tension/compression-asymmetric strengths, progressive composite failure, and
shear/torsion plastic interaction are not implemented. Linear laminate effects
may be supplied only as an already-integrated generalized shell section.

Simplified fracture / element erosion:

- `FractureConfig` activates a v1 nonlinear-static damage model based on the
  maximum equivalent plastic strain stored in committed shell/beam plastic
  states.
- Elements are softened only after a converged increment.  The default is
  residual stiffness (`1e-6`), not true topology deletion, so nodes, DOFs,
  MPCs and beam-shell coupling constraints remain in the model.
- Pressure loads attached to deleted shell elements are removed from later
  increments.  Ordinary nodal loads are unchanged and are reported as not
  fracture-aware.
- This is an engineering erosion screen for monotonic capacity studies.  It is
  not crack propagation, cohesive fracture, remeshing, impact fracture, fracture
  toughness assessment or a validated ductile-fracture model.

### Residual stress and prestrain fields

`ShellInitialField` and `BeamInitialField` provide explicit manufacturing-state
inputs to `solve_static_nonlinear(..., initial_fields=...)`. Shell engineering
vectors are in element-local `[xx, yy, xy]` order:

```text
sigma_0(z)   = sigma_membrane + (2 z / t) sigma_bending_positive_face
epsilon_0(z) = epsilon_membrane + z kappa_0
```

Beam fields are scalar, one-section-fiber, or Gauss-point/fiber arrays on a
configured fiber-plasticity section. Initial stress is represented by its
equivalent elastic strain offset, while prestrain is an immutable eigenstrain:

```text
sigma = C (epsilon_kinematic - epsilon_prestrain
           + inverse(C) sigma_initial - epsilon_p)
```

Before any external load is applied, the reduced free-DOF residual is solved to
equilibrium and only the converged plastic state is committed. Requested fields
may therefore redistribute during this initialization phase. Field provenance,
the equilibration history, and geometric-imperfection provenance are reported
separately. Imported stress must lie on or inside the flow surface associated
with the supplied hardening state; a new field cannot be superposed on nonzero
plastic history.

Supplying a new field replaces the complete previous field definition for that
element; multiple components must be supplied together under one source.
Field-bearing restart state must be accompanied by the matching converged
displacement vector. Re-equilibration without that vector requires the
explicit `equilibrate_initial_state=True` opt-in. Failed initialization or
displacement-control attempts restore the last converged displacement, load
factor, and material-state checkpoint.

This implements an initial-state model, not a reconstruction of welding,
forming, or thermal manufacturing history. The stress/prestrain convention is
qualified only in element-local reference coordinates with von Karman
kinematics. Corotational initial fields fail closed pending a verified
objective field transformation. The treatment follows the initial-strain and
nonlinear initial-stress conventions documented in the
[CalculiX 2.22 manual](https://www.dhondt.de/ccx_2.22.pdf) and the explicit
initial-equilibrium treatment in the
[Abaqus initial-conditions documentation](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEKEYRefMap/simakey-r-initialconditions.htm).

## Substantial theory work marked for future implementation

The 2026-07 theory sweep identified the following formulation work. These are
explicit backlog items, not current capabilities:

1. Replace experimental Q8R stabilization with a bending-aware formulation,
   correct and verify its rotary mass, and qualify distorted, thin-limit,
   modal, buckling, and nonlinear behavior before restoring it to production
   or accelerated-batch scope.
2. Implement true curved 3-node beam interpolation and a general curved-shell
   reference geometry, including objective frames, consistent mass, geometric
   stiffness, loads, and recovery.
3. Extend the current `N`/`M`/`H` Mindlin initial-stress operator to a
   geometrically exact finite-rotation shell/director formulation, and add a
   general complex eigensolver for nonsymmetric nonconservative follower-load
   pencils.
4. Make contact force, contact moment, and contact work fully consistent for
   offset shell surfaces and profile-resolved members; retain the present
   midpoint/beam-axis penalty models as engineering approximations.
5. Develop an equilibrated or compliance-weighted recovery estimator for
   adaptive refinement. The current guarded patch result and normalized
   stress-L2 discrepancy intentionally make no energy-norm or guaranteed-error
   claim.
6. Extend shell plasticity to transverse shear and beam-section plasticity to
   shear/torsion interaction. Until then, large
   `mixed_reconstruction_von_mises` values are model-scope warnings: the
   underlying equilibrium used elastic shear/torsion and those components must
   not be capped or described as plastically redistributed.
7. Split shell patch-recovered nodal equivalent stress into the same
   constitutive in-plane and explicitly mixed invariants used by element
   recovery. The current patch tensor retains transverse shear and therefore
   represents the mixed elastic/plastic diagnostic.
8. Recover quadratic-beam shear and torsion at the same Gauss stations as each
   fiber block before forming a mixed diagnostic. The current B3 diagnostic
   combines station-dependent fiber stress with one element-level elastic
   shear/torsion envelope; the constant-response B2 case is unaffected.

## Verification anchors

The test and report suites protect, among other checks:

- DOF ordering, model revisions, separated K/M/C/KG/F assembly, matrix
  symmetry/finiteness, sparse factorization reuse, and provenance;
- fixed/prescribed supports, interpolated/eccentric MPCs, reactions, and
  connected-component rigid-body nullspaces;
- T3/T6/Q4/Q8/Q8R interpolation, rigid modes, membrane/bending/shear patches,
  pressure and mass resultants, distortion checks, and nodal stress recovery;
- linear/quadratic Timoshenko response, axis conventions, torsion, mass,
  geometric stiffness, nonlinear tangent, and fiber yielding;
- modal, Euler/Wagner/plate buckling, repeated roots, and sparse shift-invert;
- DNV curve anchors, plane-stress return mapping, imperfections, staged loads,
  displacement control, corotational objectivity and consistent-tangent
  finite differences, current-area follower pressure, and arc-length limit
  points;
- Newmark/HHT-alpha response, pressure-patch impulse, sphere-contact momentum
  and energy, beam contact, nonlinear impact, RTCL weighting, and committed-state
  damage;
- material-history and guarded patch recovery, discontinuity separation,
  selected/envelope recovery, memory policy, normalized geometry conversion,
  capacity workflow, SESAM round trip/import, and SIF load-case isolation.

CalculiX input decks and reference-case discovery support reproducible external
comparison against the official CalculiX references for
[buckling analysis](https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node128.html)
and [`*DLOAD`](https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node190.html).
A deck-only run has status `not_executed` and is not evidence of numerical
agreement. With explicit execution enabled, each case runs in an isolated
directory, records executable version/hash provenance and logs, rejects stale
or missing output, parses the requested FRD/DAT observables, and passes only
when every declared analytical comparison meets tolerance. Verification
commands and evidence interpretation are in
[`QUALITY_CONTROL.md`](QUALITY_CONTROL.md).
