# Floating-body response

The floating response modules accept neutral numerical inputs and keep solver
mechanics in ANYfeSolver. They were integrated on local main on 2026-10-28. Their
tests are development evidence; they do not confer broad marine qualification.

## Frequency response

[`anyfesolver.floating_response`](../src/anyfesolver/floating_response.py) provides
`solve_floating_response`, mean-frame operator/force transformations, restoring
audits, rigid-motion reconstruction and waterline/rotation-envelope checks.
The model is a linear six-DOF floating body at zero forward speed. Coordinates
are translations and incremental rotations in mean-body axes through the mean
centre of gravity, with harmonic convention `exp(-i omega t)`.

Inputs retain full coupled matrices. The implementation refuses unacceptable
operators rather than repairing, symmetrizing or adding damping to them. The
module itself does not import ANYmarine, ANYhydro or Capytaine.

## Transient wind feedback

[`anyfesolver.floating_transient`](../src/anyfesolver/floating_transient.py) exposes
`FloatingTransientSpec`, `FloatingTransientResourceConfig`,
`FloatingTransientResult`, and `solve_floating_transient`.
It uses fixed-step Newmark average acceleration with causal radiation memory
in the first-order mean-body frame. A single turbine with a fixed body-axis
and hub offset contributes scalar piecewise-linear thrust feedback.

Each step requires a unique in-domain thrust root. No root, multiple roots,
an interval of roots, cancellation, or an exceeded rotation envelope produces
an explicit non-success. The default rotation envelope is five degrees;
finite-rotation mechanics and flexible turbine aerodynamic feedback are outside
this route. Nonsymmetric external operators retain their skew parts.

## Structural pressure and mass

[`pressure_regions`](../src/anyfesolver/pressure_regions.py) and
[`floating_pressure_regions`](../src/anyfesolver/floating_pressure_regions.py)
integrate shell pressure over supplied overlap regions using the element's own
virtual-work density. Physical quadrature points are inverted through the actual
element mapping; native S3 rotational work is retained. Geometry ownership stays
outside the solver.

[Concentrated mass](concentrated_mass.md) supplies the structural attached-mass
and inertia operator. [Inertia relief](INERTIA_RELIEF.md) is the separate linear
free-body structural route. These APIs do not broaden each other's admission.

## Regression entry points

The focused source tests are `tests/test_floating_response.py`,
`tests/test_floating_transient.py`,
`tests/test_floating_transient_native_admission.py`,
`tests/test_floating_pressure_regions.py`, and
`tests/test_wave_surface_regions.py`. Use the development runner described in
[quality control](QUALITY_CONTROL.md). Listing these tests does not claim that
they were rerun during documentation maintenance.
