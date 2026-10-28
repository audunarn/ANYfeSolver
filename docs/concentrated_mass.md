# Concentrated physical mass

`FEModel.add_concentrated_mass(node_id, mass, offset=(0,0,0), inertia_cg=J)`
attaches a rigid mass to an existing six-DOF node. Offset is from the node to the
mass centre; J is the symmetric inertia about that centre, all in model axes/SI.
With S(r) v = r cross v, B = [[I,-S(r)],[0,I]], its matrix is
B.T diag(m I,J) B. Thus kinetic energy, offset gravity moments, rigid mass
properties and inertia relief use one canonical FE operator. Both consistent and
lumped element mass formulations retain this exact concentrated contribution.

The tensor must be finite, symmetric and physically realizable (nonnegative
principal moments satisfying triangle inequalities); nonzero inertia requires
positive mass. No stiffness or support is introduced. Scalar point masses remain
compatible. Concentrated records are immutable; assembly revalidates records and
session mass/inertia-relief cache identities include dictionary contents.

This addition is verified for linear structural mass, gravity, floating rigid
response inputs and inertia relief. It adds no flexible turbine aerodynamic
feedback or nonlinear/finite-rotation qualification.
