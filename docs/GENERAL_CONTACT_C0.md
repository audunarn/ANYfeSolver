# General contact C0: planar Q4 against a rigid plane

This is the first coupled implementation slice of the general-contact
programme. It is a **prototype**, not an accepted general-contact capability.
It joins the existing nonlinear static equilibrium solve and rejects every
contact combination outside the stated profile.

## Admitted profile

- Qualified planar E4-PL Q4 shell, with an explicit list of element IDs.
- One prescribed, one-sided rigid plane with a unit separation normal.
- One frictionless penalty interaction. `normal_stiffness` has units Pa/m;
  `gap_tolerance` is a positive length in metres.
- Initial physical penetration is rejected, independently of the solve gap
  tolerance. Exact first touch uses the closed-side generalized tangent.
- Small-rotation `von_karman` kinematics for midsurface, top or bottom skin;
  `corotational` kinematics for the midsurface with its consistent tangent.
- Static force or displacement control. The small-rotation penalty profile
  supports a version-2 contact checkpoint; corotational contact restart is
  still unavailable. No fracture, imperfection, shell–shell pairing, beam
  pairing, friction or dynamics.

The plane is fixed during the solve. For this profile the point-to-plane gap
is affine along a linear trial displacement path, so a path with positive gaps
at both ends cannot tunnel through the plane. This observation does not apply
to moving or curved obstacles, rotating skins or other future pair kernels.
Newton trial steps are cut back analytically before their corner gaps exceed
the declared tolerance; the result reports `swept_trial_cutbacks`.

## Discrete law and verification target

Let `N_a` be the nonnegative Q1 pressure functions, shared by adjacent Q4
faces at common nodes, and `dA0` the reference surface measure. Integrals
below extend over the named shell surface. Define

```text
m_a = integral N_a dA0
B_ai = integral N_a N_i dA0
g_i = n · (X_i + u_i + z*d_i - x_plane)
gbar_a = sum_i B_ai*g_i / m_a
Pi_c = (k/2) sum_a m_a * min(gbar_a, 0)^2
```

For top/bottom skins in the small-rotation route,
`d_i = d_ref + theta_i × d_ref` and `z = ±thickness/2`. On a flat Q4,
`d_ref` follows its declared physical reference director or, when none is
declared, its oriented reference surface normal.

The implementation differentiates this potential for the contact residual
and tangent. The rotational gap derivative is
`z*(d_ref × n)`; rotation around the physical director performs no contact
work. The plane reaction is the negative shell wrench about the global origin.

The pressure-space gap is weak. It can be acceptable while a corner gap is
too negative, especially when an edge is the true contact feature. The C0
diagnostic evaluates quadrature and corner gaps. A requested tolerance that
cannot be met stops advancement before committing the violating increment.
Such a case needs the later edge kernel or a better resolved contact model;
raising stiffness alone cannot repair this pressure-space limitation.

The current static profile assumes the shell remains mechanically constrained
when contact opens. An otherwise free normal translation can stop with a
generic nonlinear failure rather than a dedicated rigid-mechanism diagnostic;
that diagnostic is still required by the programme.

## Example

Run the [guided plate example](../examples/general_contact_planar_plate.py)
from an environment containing this checkout's package. Its independent
scalar reaction is 0.4 N and its predicted gap is -0.4 micrometres. The
example stops if the solver does not finish or the gap budget is exceeded.

```python
from anyfesolver import (
    ContactDefinition, ContactInteraction, ContactSurface,
    RigidContactBody, solve_static_nonlinear,
)

contact = ContactDefinition(
    surfaces=(ContactSurface("plate_skin", (1,), side="midsurface"),),
    rigid_bodies=(RigidContactBody("floor", "plane", (0, 0, -0.01),
                                    (0, 0, 1)),),
    interactions=(ContactInteraction("plate_floor", "plate_skin", "floor",
                                     normal_stiffness=1.0e7,
                                     gap_tolerance=0.001),),
)
result = solve_static_nonlinear(model, load_case, contact=contact,
                                kinematics="corotational")
contact_summary = result.info["general_contact"]
```

`model` and `load_case` are ordinary ANYfeSolver objects. Check `result.status`,
`result.failure_reason`, `result.load_factor` and
`contact_summary["gap_tolerance_met"]` before using the result. A stopped
analysis retains its last accepted structural state.

## Evidence state

The C0 profile has focused force, tangent, initial-overclosure, offset-couple,
coupled-solve, rollback, gap-failure and small-rotation restart tests. A
version-2 contact checkpoint binds the contact definition and settings in its
analysis fingerprint; its committed contact-state object is empty because this
frictionless penalty profile has no physical history. No-contact checkpoints
retain their version-1 format. The profile has not passed the programme's
independent review or registered engineering qualification. It must remain
labelled **implemented and locally tested**.

A separate [prescribed translating-plane development profile](GENERAL_CONTACT_C1_RIGID_MOTION.md)
uses force or prescribed displacement control and binds its linear tool path to restart. It does not
change the fixed C0 capability or its evidence status.
