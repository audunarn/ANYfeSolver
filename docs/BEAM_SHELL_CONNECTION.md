# Beam–shell connection options

`CoupledBeamShellElement` connects a beam node to one shell node.
`InterpolatedBeamShellMPCElement` connects it to weighted shell masters. Both
accept the keyword-only Python boolean `rotating_eccentricity`, defaulting to
`True`. This exposes the existing behavior without changing the default.

For corotational analysis, `True` rotates the reference offset with the master
frame: `u_beam = u_master + (R_master - I) r0`. Use
`kinematics="corotational", corotational_tangent="auto"` (which selects
`"work_conjugate"` for rotating connections), or request `"work_conjugate"`
explicitly. `False` keeps the fixed-reference linearized relation
`u_beam = u_master + theta_master × r0`, including in corotational solves.
Linear analyses continue to use that linearized constraint in either mode.
Zero offset retains the existing linear coincident connection in both modes.
The option represents a kinematic tie; it adds no connection stiffness or
connection strength/failure law.

## Direct and interpolated construction

The following complete example uses an elastic beam and prescribed shell-master
motion to show the connection law. Node coordinates and eccentricities use SI
units. Here the shell masters are represented by boundary nodes; a panel model
would also contain its shell elements.

```python
import numpy as np
from anyfesolver import (FEModel, BoundaryCondition, LoadCase,
                       CoupledBeamShellElement, solve_static_nonlinear)
from anyfesolver.elements import BeamElement

model = FEModel("connected_beam")
model.add_material("steel", 210e9, 0.3)
for nid, xyz in [(1, (0, 0, 0)), (2, (1, 0, 0)),
                 (3, (0, 0, 0.1)), (4, (1, 0, 0.1))]:
    model.add_node(nid, *xyz)
model.add_element(1, BeamElement(1, [3, 4], "steel",
    dict(area=0.001, Iy=1e-6, Iz=1e-6, J=1e-6)))
for master, slave in [(1, 3), (2, 4)]:
    model.add_element(10 + slave, CoupledBeamShellElement(
        10 + slave, slave, master, "steel",
        eccentricity=np.array([0., 0., 0.1]), rotating_eccentricity=True))
    conditions = dict(ux=0., uy=0., uz=0., rx=0., ry=0.1, rz=0.)
    if master == 2:
        del conditions["ux"]  # One free elastic equilibrium DOF.
    model.add_boundary_condition(BoundaryCondition(str(master), [master], conditions))
model.apply_boundary_conditions()
load = LoadCase("probe")
load.add_nodal_load(4, forces=(0., 0., -1.))
model.add_load_case(load)
result = solve_static_nonlinear(model, load, kinematics="corotational",
    corotational_tangent="auto", num_steps=1, max_iterations=10)
assert result.status == "completed"
```

For interpolation, supply existing shell-master IDs and shape weights. The beam
node must lie at the weighted reference master position plus the declared
offset. For example, for a beam node at that position:

```python
from anyfesolver import InterpolatedBeamShellMPCElement
connection = InterpolatedBeamShellMPCElement(
    20, beam_node_id=3, shell_node_ids=[1, 2],
    shape_weights=np.array([0.5, 0.5]), eccentricity=np.array([0., 0., 0.1]),
    material_name="steel", rotating_eccentricity=False)
```

Weights must be finite and sum to one within the existing admission tolerance;
they are not renormalized. Nonzero rotating offsets additionally require nonnegative
weights. With rotating offsets, the interpolated
frame is the admitted intrinsic SO(3) mean, satisfying
`sum(w_i Log(R_mean.T R_i)) = 0`, rather than an arithmetic average of rotation
vectors. Two masters require a unique shortest geodesic. Larger means require
the supported open π/4 anchor ball and principal rotation chart. Unsupported
rotations are refused; this option does not widen that domain.

## Generated panels

```python
from anyfesolver import MeshConfig, PanelGeometry, generate_stiffened_panel_mesh
panel = PanelGeometry(length=1., width=0.5, plate_thickness=0.01,
    stiffener_type="T-bar", stiffener_height=0.1,
    stiffener_web_thickness=0.005, stiffener_flange_width=0.05,
    stiffener_flange_thickness=0.005, num_stiffeners=1)
config = MeshConfig(shell_num_divisions_x=2, shell_num_divisions_y=2,
                    rotating_eccentricity=True)
model = generate_stiffened_panel_mesh(panel, config)
```

The solver adapter passes this option to every generated interpolated connection.
It is not a neutral ANYmesher option. Only actual Python booleans are accepted;
integers, strings, `None` and NumPy booleans are rejected before mesh generation.
Existing positional constructor and `MeshConfig` arguments retain their order.

## Standalone pre-yield example

`examples/beam_shell_coupling.py` is a self-contained public-API example of the
registered pre-yield S355 case: the 1 m x 0.24 m, 2x1-shell panel with one
10x40 mm stiffener at e = +0.04 m, canonical S355 curve, -2000 N uz on every
shell-tip node, 6 load steps, at most 30 iterations per step, tolerance 1e-8,
corotational kinematics. `--coupling direct` uses beam nodes on shell edge
nodes; the default `--coupling interpolated` uses genuinely interpolated
mid-span beam nodes.

```
python examples/beam_shell_coupling.py --coupling interpolated
```

The example writes a JSON summary to stdout (and optionally to `--output`):
`status`, `stop_reason`, `requested_load_factor`, `achieved_load_factor`,
`probe_displacement`, `accepted_increments`, `constraint_postcheck` and
`completed`. `completed` is true only when the solver status is
`completed`, the achieved load factor is within 1e-8 of the requested one and
the existing constraint postcheck passed; a cancelled solve reports the
accepted increments seen so far and never claims completion. The process exit
status is 0 only for such a valid completion.

The refreshed private Windows Python 3.12 candidate, built from `6fb907a1c`,
includes both this pre-yield workflow and the completed-prefix checkpoint
workflow below. Its hashes, pinned dependencies and installed verification are
recorded in the
[delivery receipt](../.development/nonlinear-master-slave-coupling-20261017/incremental-20261020/slice3/refresh-6fb907a1c-v1/DELIVERY.json).
All 77 installed focused tests and eight actual CLI runs passed. Use the bundled
README's measured runtime recipe: JIT disabled, one effective numerical thread
and retired-programme selection disabled. The earlier pre-yield-only artifact
is preserved with the same private version and a different source/hash; install
the explicit local wheel from the selected bundle. These checks do not establish
a passing full suite or general scientific qualification.

## Reading a nonlinear result

`result.status == "completed"` means the solver reached its requested target
(force load factor or displacement target). Read the achieved response from
`result.displacements` at the probe node's DOFs. Under displacement control
the achieved controlled value is `result.steps[-1].control_value`; compare it
against the requested target. `result.converged` alone is not completion: it
also covers `stopped_at_limit` results that stopped before the target.
Checkpoint continuation requires a terminal `completed` status. A checkpoint
carrying a stopped terminal state preserves information for inspection but is
refused for continuation. Retain a completed intermediate target as the prefix
for a later restart; this is the intended recovery workflow.

## Completed-prefix plastic checkpoint workflow

The source workflow passes 12 checkpoint tests and 113 shared-contract and
compatibility tests. The four default-continuation checks left unfinished by
the combined run's timeout subsequently passed unchanged. Their comparison
reference reuses the completed, hash-verified uninterrupted path; prefixes and
resumes run fresh. This establishes the registered workflow, not general
scientific qualification. See the task handover for the retained evidence.

The source example's `--case plastic` uses the same S355 model with tip
displacement control. The full target is the case-specific revised endpoint
-11/60 m with 11 initial increments. A completed -0.10 m prefix uses 6 initial
increments; restarting it uses 5. All retain at most 40 Newton iterations and
tolerance 1e-8. Adaptive subdivision can add accepted increments.
Run each command in its own process, with the same coupling selection:

```text
python examples/beam_shell_coupling.py --case plastic --coupling interpolated --checkpoint-out full.json
python examples/beam_shell_coupling.py --case plastic --coupling interpolated --stage prefix --checkpoint-out prefix.json
python examples/beam_shell_coupling.py --case plastic --coupling interpolated --restart prefix.json --checkpoint-out resumed.json
```

Use `--coupling direct` for the matching two-element beam; `interpolated` uses
three beam elements. JSON reports `requested_control`, `achieved_control`,
`control_error`, achieved load factor and constraint outcome. Exit 0 requires
completed status, displacement error at most 1e-8 m and passing constraints.
`accepted_increments` includes the restored prefix on a completed restart;
on cancellation it counts new increments observed in the current process.

Checkpoints use the existing `restart_checkpoint_bytes()` API and are replaced
atomically only on successful completion. Refusal, incomplete solves and
cancellation preserve existing recovery files. Model/settings validation and
failed-terminal checkpoint refusal remain enforced. This workflow introduces
no checkpoint format or manual-state bypass. The original failed attempt and
timeout remain preserved in the handover.

This source workflow is included in the refreshed `6fb907a1c` installed candidate.
The earlier pre-yield-only artifact remains unchanged as historical evidence.
Neither the original -0.20 m target nor director continuation or analytical
accuracy is established by the completed-prefix workflow.

## Evidence and limits

This is a public selection of existing kinematics, not general scientific
qualification. On the default rotating route, the registered S355 case has a
demonstrated completed displacement-control run to the revised -11/60 m
endpoint, including continuation from an accepted -0.10 m state through the
public initial-state route; the original -0.20 m target has not been
completed. The experimental director-integrated trace stops at -1/6 m on the
same case, and the registered analytical accuracy gate remains open; the
failing director continuation tests are the record of that. Nonmatching
interpolation accuracy and mesh convergence require case-specific
verification, including comparisons with suitable independent references. A
nonzero offset and finite rotations can make the two options give different
physical predictions.

Check `result.status == "completed"`, the actual controlled displacement against
the requested target, equilibrium, constraints and physical reactions for the
intended case. `result.converged` alone also covers some stopped-at-limit results
and does not establish target completion. Do not infer collapse or absence of an
equilibrium from a solver stop.

## Experimental director-integrated edge trace

`InterpolatedBeamShellMPCElement` additionally accepts the keyword-only
`connection_trace="director_integrated"`. The default `"linear"` retains every
existing translation trace, including the default and zero-offset behavior
described above. The alternative is an experimental discretization; it is not a
repair that silently changes the meaning of the existing law. It is not yet
available through `MeshConfig`.

For a declared straight reference edge with two distinct master endpoints and
material coordinate `xi`, the new trace integrates the same shortest-geodesic
director used for rotation:

```
R(xi) = R0 Exp(xi Log(R0.T R1))
A(xi) = integral_0^xi R(s) ds
Q(xi) = A(xi) inverse(A(1))
p_beam = p0 + Q(xi) (p1 - p0) + R(xi) eccentricity
```

`Q` is a matrix endpoint weight, not a rotation. Translation derivatives,
endpoint moments and translation/rotation cross terms are included in the full
Jacobian, work map and geometric tangent. Linear/static MPCs use this same
trace's derivative at the reference state. The nonlinear route requires
corotational kinematics and `auto` or `work_conjugate` pullback.

For example, the following constructs an interior connection on a declared edge;
physical shell/beam elements, supports and loads must then be added as usual:

```python
import numpy as np
from anyfesolver import FEModel, InterpolatedBeamShellMPCElement
trace_model = FEModel("experimental_edge")
trace_model.add_material("steel", 210e9, 0.3)
trace_model.add_node(1, 0., 0., 0.)
trace_model.add_node(2, 1., 0., 0.)
trace_model.add_node(3, 0.4, 0., 0.03)
trace_model.add_element(10, InterpolatedBeamShellMPCElement(
    10, 3, [1, 2], np.array([0.6, 0.4]), np.array([0., 0., 0.03]),
    rotating_eccentricity=True, connection_trace="director_integrated"))
```

The reference slave must equal the weighted master position plus the declared
offset to reference floating-point roundoff, and the edge must have finite,
positive length. Weights must be nonnegative and unit sum to reference roundoff;
they are not fitted or renormalized. Three or more masters, duplicate endpoints,
inconsistent geometry and `rotating_eccentricity=False` are refused. One-master
endpoints reduce to the existing direct law. Interior new-mode connections remain
nonlinear at zero offset; this is an explicit difference from the legacy mode.
Input, relative and output rotations retain the existing principal-chart bounds.
This defines an edge reconstruction, with no arbitrary bilinear-surface extension.

The map passes focused bending/shear/axial, finite circular-trace, objectivity,
work/wrench, derivative and small elastic restart/cancellation checks. This does
not establish scientific accuracy: the tested nonmatching eccentric 30-degree
nx8 arc equilibrated but failed the registered accuracy bound. The first nx16
attempt timed out without a terminal result. After a same-map displacement
arithmetic repair, nx16 completed and passed independent equilibrium, constraint
and wrench checks, but still missed the analytical response by about 5.3%.
No broader accuracy or plastic continuation acceptance follows. The earlier
limits and explicit completion/target checks still apply.
