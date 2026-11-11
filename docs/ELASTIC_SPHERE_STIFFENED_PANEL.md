# Elastic sphere stiffened-panel initial state (2026-10-20)

## S11 accepted static candidate: complete installed cycle and release gates

The opt-in static sphere-panel workflow is accepted on local Windows and packaged
as 0.4.7+contact.candidate.20261021.s11. A fresh uninterrupted installed public
CLI completed open equilibrium, first touch, loading through 1 mm, complete
unloading and original-open recovery: **12 accepted states, one permitted
bisection, 1455.663050299976 process seconds**. All ten prefix array archives match
the earlier saved states exactly. The original componentwise open-recovery bound
1e-10 + 1e-8*abs(reference) passed; maximum excess was
-9.999932291215377e-11. Peak contact force was 2107.500297906452 N, residual
2.3859415008929422e-6 N, and whole-face lower bound
-19.978252498586724 micrometres against the unchanged -50 micrometre requirement.
Final contact force was zero. No owned process survived.

Independent Sonnet review found no blocking defect in the changed connected
solver, geometry, exact inactive-branch proof, certificate and bridge code.
Retained unchanged bound/skin/mechanics dependencies keep their prior scope;
the review is not an independent mechanical accuracy reference. Fifty-seven
installed focused checks pass, including rejection, cancellation and gate-failure
rollback/no-bisection paths. Initial test-double/fixture failures are preserved.

The retained contact-candidate CI passed its build, Windows and Ubuntu jobs,
including transferred checkpoint and declared cross-host parity checks:
https://github.com/audunarn/ANYsolver/actions/runs/39614951215
This is regression coverage for retained workflows, not paired-host S11 response
qualification. The broader matrix is separate and has pre-existing failure nodes;
its exact current/baseline disposition is retained with the release evidence and
must not be reported as all green.

Follow the bundled installation guide, then run from the bundle directory:

    python -m examples.elastic_sphere_panel_indentation --output indentation.json

The final six-wheel candidate was cold-installed and its origins/pins/JIT-off/
single-thread settings verified. All 735 solver files match release source;
relative to the scientifically exercised s11unload1 artifact only trailing
whitespace in the new proof helper differs, with identical parsed code.
Delivered examples are byte-identical. Independent review, scientific result,
wheel/source identities, original failures, budget closure and CI disposition:
C:/ANYworkspaceWork/contact-incremental-delivery-20261020/indentation-s11-acceptance-release-v1/DELIVERY_RECEIPT.json.

This remains a faceted, frictionless, elastic static development candidate.
Falling, velocity/acceleration, impact/rebound, friction/plasticity, public restart,
general accuracy and paired-host S11 qualification are not established.
Earlier status sections below are retained history.


## S11 unloading ambiguity repair: restored local recovery passed

Both actual refused geometries are preserved: corner 395 against faces 44/45
and corner 875 against faces 54/55. The strict closest-point comparison remains
unchanged. An ambiguous row can be eliminated only when its current multiplier
is exactly zero and exact rational bounds prove positive separation beyond the
existing 1e-9 m distance threshold, including position error boxes, positive
oriented side and completeness against every omitted face. No closest owner,
normal, gradient, force or work is invented. Every trial repeats the proof and
retains an audit. Active, touching, penetrating and unproved cases still refuse.

Thirty focused tests passed. The repaired s11unload1 package was built and
cold-installed; all 735 solver files matched source. Pinned runtime, installed
origins, disabled JIT and one thread were verified. Restoration from original
accepted 0.25 mm state again passed exact elastic force/tangent/history and
saved geometry/context checks plus original gates before and after commit.

Final unloading then passed in six Newton iterations; original-open recovery
passed with zero further iterations and no additional bisections. Contact force
was zero at both endpoints. Their equilibrium residuals were 1.874e-6 N and
1.885e-6 N. Whole-face lower bounds were -0.137113 micrometres and zero against
the unchanged -50 micrometre requirement. Maximum full-q difference from the
original open reference was 2.328757846281615e-15, within the unchanged
1e-10 + 1e-8*abs(reference) componentwise bound. Seventeen certified inactive
rows were audited across twenty evaluations, covering both captured corners.
Models and protected inputs remained unchanged. Process wall was
401.31939699998475 seconds; no owned process survived.

Evidence and source checkpoint:
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/indentation-s11-unload-ambiguity-repair-v1/SLICE_RESULT.json`.
This is local Windows restored-continuation evidence. It does not establish a
public restart capability, an uninterrupted full-cycle CLI result, independent
mechanical accuracy, paired-host qualification or release acceptance. Independent
acceptance and delivery gates remain pending. Earlier failures below are retained
as historical evidence. No merge, commit, push or publication was performed.


## S11 final-release check: restoration verified; geometry refusal remains

The saved accepted 0.25 mm unloading state was reconstructed using the exact
installed s11cert2 artifact. Its forces, sparse tangents and generated elastic
histories agreed exactly across gravity-history, empty-history and regenerated
history evaluations. Saved full/reduced coordinates, pose error radii, guide
context, multipliers, residual and minimum gap were verified. Every original
convergence, whole-face and balance gate passed before and after commit.

Final unloading to the saved first-touch height then refused distinct closest
points tied within 1e-9 m on panel faces 45 and 44. All three remaining original
bisections were consumed (four over the complete saved path). No new increment
was accepted. The accepted 0.25 mm state remained exactly unchanged across all
four failed attempts; canonical models and protected inputs also remained
unchanged. Open recovery was never reached. This is a geometry refusal, not a
budget timeout or a failed restoration. The run used 310.88211559999036 seconds
of precise child process wall; no owned process survived.

Evidence, exact commands, runtime pins, three-way history comparisons, accepted
state/history arrays and full failure trace are retained in
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/indentation-s11-final-recovery-v1`.
The actual refused trial query/face-candidate arrays were not captured; use the
saved accepted state and trace for a bounded follow-up with explicit geometry
capture. Do not infer a mathematically proven physical tie from the refusal or
relax its distance/uniqueness rule to force a pass. Restoration evidence does not
create a public restart facility or an uninterrupted full-cycle result.

S11 remains not delivered. Its local installed 1 mm loading evidence is retained,
while final separation/open recovery, independent acceptance and delivery gates
remain incomplete. No product numerical code, contact law, tolerance, target or
iteration bounds changed in this check; no scientific retry followed refusal.

## S11 cert2 installed loading passed; cycle limited during unloading

The exact two-level certificate candidate was built and cold-installed as
`0.4.7+contact.candidate.20261021.s11cert2`. All 734 solver package files matched
the source. Installed private origins, declared numerical pins, disabled JIT and
single-thread runtime checks passed on local Windows.

The one installed cycle accepted all loading targets through **1 mm**. Peak
contact resultant was **2107.5003 N**, equilibrium residual **2.386e-6 N**, and
whole-face lower bound **-19.9783 micrometres** against the unchanged -50 um
requirement. Unloading passed 0.75 mm, 0.5 mm, a permitted 0.375 mm midpoint,
and 0.25 mm. Repeated nonunique-closest-point trial refusals triggered that
existing increment bisection; no contact or tolerance rule was relaxed.

The supervised process reached its **1450 s limit** after the accepted 0.25 mm
unloading state. Its precise process wall was 1450.0779968000134 s. This is a
resource-limited partial result: no acceptance gate reported scientific failure,
but final separation and open recovery were not verified and no final CLI cycle
report was written. Do not interpret it as a completed cycle or a formulation
failure. All ten accepted states, enclosures and full traces are retained at
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/indentation-s11-installed-level2-v1`.

The 11 focused certificate checks from the preceding slice remain valid for
this unchanged source. The bounded Sonnet independent Read review obtained
source/test/bound tool results but timed out without a final verdict; review
is incomplete. S11 remains off main and not delivered; S10 remains the usable
candidate. No scientific retry, merge or push followed the timeout. Impact,
the 10 m drop, friction/plasticity, restart and paired-host S11 acceptance remain
future work. The captured arrays are development evidence, not a public restart.

## S11 loaded iteration repaired; delivery blocked by the fixed certificate

The loaded contact iteration now resolves tensile linearized working-set forces
before nonlinear line search. Corner identities, contact law, structural mechanics,
final criteria and the fixed certificate level are unchanged. Nineteen focused
checks pass. The actual-model 0.25 -> 0.5 mm transition passed all gates in five
Newton iterations; cold-installed execution reproduced that accepted 0.5 mm state.
Its contact resultant is 1053.6558 N, residual 2.365e-6 N, and whole-face lower
bound -39.7630 micrometres against the unchanged -50 micrometre requirement.

The one cold-installed full-cycle attempt then converged at 0.75 mm in five
Newton iterations: residual 2.429e-6 N, guide error 2.724e-16 m, minimum sampled
corner gap -1.777e-17 m, four positive contact multipliers. Its fixed whole-face
lower bound was **-59.9132 micrometres**, so acceptance correctly refused before
commit. No certificate was unavailable; the bound itself failed. This is not a
measurement establishing actual penetration of 59.9 micrometres. Zero increment
bisections were used. The 1 mm target, unloading and open recovery were not reached.

S11 remains **not delivered** and off main. S10 remains the usable candidate.
The bounded batch stopped at this first new scientific nonpass: no refinement
sweep, tolerance change, scientific retry, unrelated qualification or push.
Exact accepted 0.5 mm and converged/rejected 0.75 mm states, runtime identities,
full trace and original failures are retained in
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/indentation-s11-loaded-iteration-repair-v1`.
The next useful action, only if separately commissioned, is a bounded certificate
repair on the saved 0.75 mm geometry while retaining the 50 micrometre requirement;
do not restart the now-resolved loaded-solve investigation. This evidence is local
Windows only. Remaining delivery checks/CI are blocked, and the full 10 m drop
remains a future target.

## S11 cold-installed full-cycle result: delivery blocked

The single cold-installed Windows/Python 3.12 full-cycle attempt returned
`indentation_refused` / `line_search_failed` (no merit reduction after eight
reductions). All four allowed increment bisections were exhausted. Open equilibrium,
first touch and 0.25 mm indentation passed acceptance; the next loading step could
not advance, even after subdivision down to a further 15.625 micrometres.
The 1 mm target, unloading and open recovery were not reached. The accepted loaded
whole-face bound remains -19.61 micrometres against the unchanged -50 micrometre
criterion. Larger-step certificates were not assessed because Newton did not converge.

The installed wheel matched frozen source bytes; exact installed origins, pinned
dependencies, JIT-disabled and one-thread settings passed admission. The first
admission lacked declared Shapely 2.2.0; that preparation failure and its correction
are retained. The full cycle took 657.3027709 process seconds, with sampled aggregate
RSS peak 265.914 MiB, normal process exit 3 and no survivors.

No scientific retry, criterion change, refinement sweep, source repair, integration
or push followed. S11 remains off main and is not delivered; s10 remains usable.
Remaining delivery checks and CI were not launched after this blocking nonpass.
Exact gate states, report, runtime and failed trace are retained in
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/indentation-s11-delivery-v1`.
The trace shows four active multipliers at the accepted state and six in the next
linear system, with increasing trial merit; this is a localization clue, not a
proven cause. Any follow-up must address this transition rather than rerun the
unchanged cycle or broaden qualification. Full 10 m drop remains the target.

## Slice 11: coupled corner-contact indentation (development-only, full cycle pending)

`python -m examples.elastic_sphere_panel_indentation --output indentation.json`
implements the frozen path open (5 mm), geometric first touch, first touch
-1 mm, first touch, original open. Law: frictionless signed closest-point gap of each
face-local sphere corner to the ORIGINAL bilinear panel faces (g >= 0, lambda >= 0,
lambda g = 0, lambda a force in N; no penalty, no pressure). Seam/edge points share
one owner; nonunique, inadmissible (warped), zero-distance-boundary or one-sided-
unresolved geometry refuses. Six mass-weighted rigid guide rows of the sphere (m, rad)
hold only the sphere's free rigid motion during these quasi-static increments; elastic
deformation is free and the guides do not apply to a future drop. Acceptance requires
the existing whole-face certificate (>= -50 um, no unavailable pair; below-panel faces
are never excluded), equilibrium 1e-8*max(1,|load+guide|), active gaps 1e-12 m,
nonnegative multipliers, whole physical force/moment balance with the frozen case
constants, and independent support-reaction agreement (1e-9). Status values:
`indentation_cycle_completed`, `indentation_partial`, `indentation_refused`. Measured
local Windows discriminator: open equilibrium, touch and 0.25 mm indentation pass,
including force/moment balance, reaction recovery, cancellation and exact accepted
state-vector rollback. Four positive multipliers produce 526.8911 N; nonrigid
sphere displacement is 2.907 micrometres and panel change from gravity is
0.247421 mm. One fixed dyadic S3/Q4 proof partition tightens the conservative
whole-face lower bound from -78.51 to -19.61 micrometres, within the unchanged
-50 micrometre limit. Physical/contact faces and bilinear interpolation do not
change; unavailable certificates still refuse. No refinement-level sweep is used.

The complete cold-installed CLI was checked and refused after 0.25 mm, as recorded
above. Unloading recovery was not reached, no s11 bundle has been delivered, and main is unchanged. New numerical
evidence is local Windows only. Source, retained failures, accepted geometry and
measurements are recorded in
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/indentation-s11-fixed-certificate-v1/SLICE_RESULT.json`.

## Slice 10: structural-state bridge

Run `python -m examples.elastic_sphere_panel_state_bridge --output state-bridge.json`
from the s10 bundle. The output reports `structural_state_bridge_prepared`:
972 unrestrained sphere DOFs, 990 panel DOFs and 486 independent panel coordinates,
using the original supports and rotating beam-shell ties. Gravity is prepared
once; the accepted displacement and material history remain separate from trial
copies. The existing constraint Jacobian pulls both bodies' work into the combined
reduced space. Failure and cancellation leave the accepted copy unchanged.

The report includes authoritative body/node/DOF ownership, full and reduced
diagnostic work, reconstruction error, virtual-work differences, gravity checks
and actual runtime identity. It is not a restart checkpoint. The supplied 1 Pa
integrated-gap work and unit-norm generalized probe do not predict contact forces.
Local Windows checks cover state mapping, constraint derivatives and rollback;
new paired-host s10 numerical evidence is pending. No contact law, equilibrium
iteration, physical approximation or tolerance change is introduced.

See the current handoff and
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/state-bridge-s10-v1/DELIVERY_RECEIPT.json`
for bundle identity, installed measurements and retained candidate CI results.
Coupled indentation/unloading needs an explicitly adopted contact law and loading
contract; low-height elastic dynamics and the full 10 m drop remain later targets.

## Slice 9: prescribed multi-face geometry and work

The 2026-10-21 s9 bundle includes a bounded skin-report/CI cleanup repair.
Face-centre normals use the same arithmetic, with whole-model validation before
and after the batch. All scientific assertions and execution limits remain
unchanged. The repaired cold-installed skin CLI passed in 64.4 seconds; the
current receipt is under
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/s9-ci-completion-20261021-v1/`.

From the s9 bundle root, using its pinned Python 3.12, JIT-disabled,
single-thread installation profile:

```console
python -m examples.elastic_sphere_panel_contact_work --output contact-work.json
```

The example prepares panel gravity once and reports all 320 S3 sphere and 100
original bilinear Q4 panel faces. It queries the fixed six-facet lower cap within
20 mm of the lowest reference physical corner against every panel face. Release,
+5 mm separation and -1 mm vertical overclosure are prescribed kinematic probes;
vertical clearance differs from the signed normal gap. Positive gap means
separation, negative means overclosure.

The declared 1 Pa probe differentiates reference-area integrated signed gap.
Separate coordinate and spatial-spin work vectors map to actual structural DOFs.
These are virtual-work probes, not predicted reactions or an adopted pressure law.
Physical offsets and distinct facet directors remain separate; the exterior is
not a watertight smooth sphere. Structural mass, materials and ties are unchanged.

An isolated source-vertex seam permits values and first derivatives only. Strict
Hessian requests still refuse it. Scalar values use independent zero-tolerance
floating-point clipping; the existing mortar law retains its prior clipping path.
No positive sliver is discarded. Successful status is `contact_geometry_prepared`;
the JSON is a report, not a checkpoint. Contact equilibrium and impact remain
unsupported. New numerical acceptance is local Windows only; retained candidate
CI does not establish paired-host s9 evidence.

Local verification passed 53 unique kernel checks, 15 actual-model checks and the
complete cold-installed CLI. The queried reference area is 0.0599124624454 m2;
integrated signed gap is 0.00125945516687 m3 at the near pose and
0.000894174761162 m3 at overclosure. A negative corner gap can coexist with a
positive integrated gap. Full results and runtime/origins are retained in
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/s9-clipping-reconciliation-v1/DELIVERY_RECEIPT.json`.

Remaining milestones: combine body DOFs, accepted preload and rotating constraints
with a declared contact law; demonstrate indentation/unloading; add elastic
dynamics and low-height impact before the full 10 m drop.

Declared benchmark case and delivered initial state for candidate
`0.4.7+contact.candidate.20261020.s6`. The authoritative case and acceptance
criteria live in the active living note
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/elastic-sphere-panel-initial-v1/PLAN.md`;
this document describes the delivered construction and workflow.

## Case

- Hollow deformable elastic sphere, 1 m outside diameter, 10 mm wall,
  represented as a faceted elastic S3 shell skin (162 nodes, 320 triangles,
  fixed case data in `examples/elastic_sphere_stiffened_panel_mesh.py`).
- Clamped 5 x 5 m x 10 mm panel (10 x 10 Q4 shell grid) stiffened on its
  underside by four upright 150 x 10 mm flat bars at 1 m spacing
  (10 beam segments per bar), beam centroids at z = -0.080 m, plate
  midsurface at z = 0, plate top at z = +0.005 m.
- All six DOFs clamped on the four plate edges (Integrated edge translations
  + CL edge rotations, 40 edge nodes each).
- Gravity 9.81 m/s^2; both bodies purely elastic
  (E = 210 GPa, nu = 0.3, rho = 7850 kg/m^3, yield stress 0, no hardening).
- The 10 m drop of the sphere onto the panel is retained as the target; it
  is NOT executed by this candidate.

## Construction

- The panel is built from the canonical ANYmesher neutral stiffened-panel
  geometry through public solver APIs (`generate_stiffened_panel_mesh`,
  `FEModel`, public element/section/constraint services). The generator's
  legacy 235 MPa material is explicitly replaced by the elastic case material
  before assembly; no penalty connection or invented inertia is used.
- The upright 150 x 10 mm flat bar is installed through the canonical public
  section constructor `StiffenerCrossSection.from_geometry("T-bar", 0.150,
  0.010, 0.0, 0.0)` — a zero-flange rectangular upright web under the
  canonical thin-wall convention — which provides the exact area
  (0.0015 m^2), bending inertias (Iy = 2.8125e-6 m^4, Iz = 1.25e-8 m^4),
  extreme-fibre distances (c_y = 0.005 m, c_z = 0.075 m) and the canonical
  thin-wall open-section St Venant model J = h*t^3/3 = 5e-8 m^4 with its
  torsion modulus. J is the canonical thin-wall section model for the open
  rectangular web, not the exact 3D rectangular-torsion value. The beam
  elements carry this canonical section before any assembly; no section
  formula is overridden by hand. The physical shape is a 150 x 10 mm flat
  bar, not a flanged T.
- The bars sit below the plate (beam neutral line and tie eccentricity at
  z = -0.080 m through the mesher's centroid placement parameter, which
  locates only the neutral line; the physical section metadata stay
  h = 0.150 m, t = 0.010 m) through the interpolating beam-shell MPC with
  rotating eccentricity preserved.
- The sphere is fixed explicit case data, not a new generic mesher: a
  frequency-4 icosphere on the midsurface radius 0.495 m, closed with two
  faces per edge, outward winding, radius deviation below 1e-12 m. It is
  built as default S3 shells with per-triangle outward reference normals,
  carries no supports, keeps all DOFs deformable, uses no mass rescaling,
  and is never statically solved.

## Delivered initial state

`examples/elastic_sphere_stiffened_panel.py` (CLI: `--output <path>`):

1. builds panel and sphere and verifies every element carries the declared
   elastic material;
2. solves the PANEL ONLY gravity equilibrium (corotational kinematics,
   work-conjugate rotating-MPC tangent, initial four-way gravity partition (existing adaptive stepping), max 20 Newton
   iterations, tolerance 1e-8);
3. recovers per-node support reactions with the same public corotational
   work-conjugate element force law as the solve (the interpolated beam-shell
   MPC contributes no element force; its constraint force enters through the
   solver's rotating-MPC slave-to-master imbalance transfer), compares the
   recovered named support resultants with the solver-recorded per-step
   resultants (agreement tolerance 1e-9 relative), and checks the global
   force and moment balance against 1e-8 * max(1, 2198*9.81) N and
   1e-8 * max(1, 2198*9.81*5) N m;
4. checks the raw unconstrained translation mass of each body per axis
   against its reference (sphere: rho*t*faceted area; panel: 2198 kg) within
   relative 1e-10;
5. places the stress-free sphere at rest with a 10 m clearance between the
   lowest point of its outer surface (outside radius 0.500 m; midsurface
   radius 0.495 m) and the equilibrated panel top at the impact centre
   (x = y = 0), i.e. sphere centre z = equilibrated panel top + 10.0 + 0.500;
6. writes the case metadata, mesh/node/element/support sets, physical
   masses, runtime identity, accepted panel nodal fields and sphere pose as
   one JSON document with `status=initial_state_prepared` and
   `impact_status=unsupported`. The exported mesh data is the real model
   inventory, not a documentary claim: reference node coordinates and
   element connectivity/type/body sets for both bodies (sphere in local
   coordinates plus its pose), the actual support node ids, and the
   verified bar layout (beam neutral lines at y = -1.5, -0.5, 0.5, 1.5 m,
   z = -0.080 m; plate midsurface nodes at z = 0).

The unconstrained sphere is never solved under gravity: no supports are
invented for it. Mass references are reported distinctly: the assembled
faceted FE skin mass (rho*t*faceted area), the smooth midsurface shell
reference (241.707384343 kg) and the hollow-solid volume reference
(241.715604844 kg) are different quantities; only the faceted value is the
assembly target. The panel target is 2198 kg (plate 1962.5 kg + bars
235.5 kg).

## Impact is unsupported

The gravity-driven impact with both bodies deformable has no admitted
solver route in this candidate. Any impact request is refused clearly:
`request_impact()` raises `ImpactUnsupportedError`, and the CLI `--impact`
prints a refusal document (`status=impact_refused`), writes no output file
and exits 3. No impact history, transient checkpoint or restart is produced
or implied. The analytic free-fall quantities in the output document
(1.427843123 s, 14.007141036 m/s, 23.7115 kJ) are labelled planning
references, not solver results.

## Checks

Focused appended coverage in `tests/test_elastic_sphere_stiffened_panel.py`
(candidate selectors in `scripts/contact_candidate_slice2.json`): fixed
sphere case-data topology, exact metadata/supports/unconstrained sphere,
exported-mesh geometry (real node coordinates, element connectivity and
body sets for both bodies, bars at y = -1.5, -0.5, 0.5, 1.5 m and
z = -0.080 m, clamped-support nodes on the plate edges), per-axis masses
against both references, completed panel gravity solve with the fixed
settings, global force/moment balance tolerances with recovered-vs-recorded
support-resultant agreement, sphere pose and placement, and the impact
refusal (exception and CLI exit 3 with no output file). One session-scoped
fixture performs the single panel gravity solve and shares the verified
payload across the checks.

## Measured initial-state result

Local Python 3.12.13, JIT disabled, one thread: all nine case checks and 14
candidate-tool checks pass. The module CLI writes a complete initial-state JSON.
The assembled panel translation mass is 2197.9999999999995 kg per axis;
the faceted sphere is 237.14289586353 kg per axis, matching rho*t*facet area.
The faceted sphere is about 1.89% lighter than the smooth-shell reference;
this is a reported discretization difference, with no mass or density scaling.
Centre panel displacement is -0.000598370542 m. Global force imbalance is
4.826253e-7 N (limit 2.156238e-4 N), and moment imbalance is 2.156873e-10 Nm
(limit 1.078119e-3 Nm). Recovered and solver-recorded support resultants agree.
The initial four-way partition uses the existing adaptive solver: actual
accepted load factors are 0.25, 0.625 and 1.0, each with three Newton iterations.
The configured relative tolerance remains 1e-8; recorded residual norms have
force units and are not an absolute 1e-8 N criterion.

`panel.mesh.nodes` holds reference SI coordinates; `panel.mesh.elements`
contains element types, body labels and connectivity. `panel.nodal_fields`
contains [ux, uy, uz, rx, ry, rz], with translations in metres and rotations
in radians. Sphere coordinates are local to its reported centre; its initial
stress, velocity and spin are zero. The JSON is an initial-state report,
not a transient restart file. No impact result is claimed.

The s6 bundle uses the existing Contact candidate workflow to cold-install the
same wheel bytes on Windows and Ubuntu. Installed results and final bundle
identity are recorded in the slice-6 delivery receipt referenced by the contact
handoff. No broader qualification follows this slice automatically.


## Slice-7 curved contact-skin geometry (development-only)

The s7 slice adds a development-only face-local curved shell skin geometry
service (`anysolver.general_contact.curved_shell_skin`) and the example
`examples/elastic_sphere_panel_contact_skins.py`. Skin points are keyed
(body, element, local corner) and preserve each element's own physical
reference director and offset, so a shared sphere node keeps one distinct
exterior corner per incident triangle; the sparse coordinate/spin work maps
pull forces and moments back onto the shared authoritative structural DOFs.
The kinematics reuse the existing corotational Exp(theta) director field,
`director_jets`/`director_value_error` and the existing 1.2 rad
total-rotation bound; triangles stay linear and Q4s bilinear; no contact
law, solver admission or existing finite map behavior changed.

The example reuses the single existing gravity/model preparation once
(`prepare_case_models`), reports the 320 S3 sphere top exterior faces and
the 100 original Q4 panel top faces with reference and recovered physical
geometry (panel at the accepted gravity-preloaded state, sphere at its
zero-pose reference and at the declared placement, a prescribed
rigid-translation probe), and measures shared-edge offset-facet separations,
the offset-corner radius range and nominal versus measured centre-referenced
vertical clearance honestly.
The status is `skin_geometry_prepared` with `contact_response=unsupported`;
the offset facets are not claimed to form a watertight exterior and no
contact gap or force history is reported. Focused coverage lives in
`tests/test_general_contact_curved_shell_skin.py` (candidate selectors in
the separate named `curved-skins` batch of
`scripts/contact_candidate_slice2.json`, run by the check adapter after the
static base batch and nine-assertion initial-state batch); the s7 candidate version is
`0.4.7+contact.candidate.20261020.s7`.


Run the prepared geometry report from the installed candidate bundle:

```console
python -m examples.elastic_sphere_panel_contact_skins --output contact-skins.json
```

The local s7 report measured corner radii of 0.4999121147 to 0.4999395352 m
and a maximum separation of 0.0009158812 m between offset copies of shared
sphere corners. These are facet-skin measurements, not whole-face radial
extrema or contact-gap accuracy. The centre-referenced vertical clearance
is 10.0000767534 m against the nominal 10 m placement. No skin stitching or
mass scaling was applied. The final s7 receipt binds the installed results,
source and bundle at
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/curved-skins-s7-v1/DELIVERY_RECEIPT.json`.

CI scheduling note: the first s7 Windows run completed all thirteen new skin
checks and the new installed CLI, but the combined older 35-assertion batch
hit its 120-second process limit. Those same assertions are now partitioned
into 26 static and nine initial-state selectors. The existing named-batch
runner retains every assertion and the unchanged 120-second case, 600-second
host and 953 MiB limits. Failed run 39435678124 is preserved in the s7 evidence.

## Slice 8: elastic panel continuation

Run the development example from the s8 bundle root after installing its wheels
with the pinned Python 3.12, JIT-disabled, single-thread profile:

```console
python -m examples.elastic_sphere_panel_continuation --output panel-continuation.json
```

It prepares the same panel gravity equilibrium once, then applies a 1,000 N
downward global-Z force at centre plate node 61 over two increments and removes
it over two increments. Gravity remains applied throughout. The ordinary public
nonlinear solver receives copies of each accepted displacement and element state,
using corotational kinematics and the work-conjugate rotating beam-shell ties.
Reference coordinates, mass, supports, stiffeners and elastic properties stay
unchanged. The sphere remains unsolved; its exported skin uses body-reference
coordinates. Its 10 m release and impact remain subsequent targets.

The example also solves a fresh gravity-then-probe load program. Peak versus
fresh displacements and unloaded versus original gravity displacements use the
existing componentwise `1e-10 + 1e-8 * abs(reference)` comparison bound. Existing
mass, force/moment balance and reaction-agreement bounds apply. It checks exact
input/model nonmutation and independently reconstructs the accepted panel skins
with the existing `1e-13 m` geometry tolerance. This is a structural continuation
check using the same solver, not an independent mechanical accuracy reference.

Successful output reports `gravity_probe_completed`, per-leg accepted histories,
centre displacements, force balance, comparisons and recovered skins. The JSON is
a result report and cannot restart a solve. Sphere-panel contact response and
impact remain unsupported; this example does not integrate preload into the
contact driver.

The s8 numerical acceptance scope is **local Windows only**. Paired-host
continuation verification is pending. Existing candidate CI selectors and time
limits remain unchanged; their results cover the retained static workflows.
The final tested source, measurements, checks, installed origins and bundle are
recorded in
`C:/ANYworkspaceWork/contact-incremental-delivery-20261020/panel-continuation-s8-v1/DELIVERY_RECEIPT.json`.

The local s8 run measured centre vertical displacement of
`-0.0005983705419914141 m` under gravity, `-0.00106863297724715 m` at peak,
and `-0.0005983705422591824 m` after unloading. Maximum full-vector differences
were `3.7115173e-13` (peak versus uninterrupted) and `1.1541515e-12`
(unloaded versus original gravity), within the componentwise bounds above.
Both endpoint force imbalances were below `6.8e-7 N`; recovered support
resultants agreed with the solver. Twenty-four focused local checks passed,
followed by the cold-installed complete CLI. These measurements apply to this
fixed elastic probe and do not establish general contact or impact accuracy.
