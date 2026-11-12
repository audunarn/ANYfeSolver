"""Small explicitly restricted enriched-skin development model (SI units).

This is not an installed/public D1 qualification or a replacement for the
registered 174/300-coordinate cases.  Original nodal coordinates are first,
followed by four physical transverse bubble coordinates per source Q4.
"""
from anyfesolver import FEModel, BoundaryCondition, create_shell_element
from anyfesolver.general_contact.enriched_overlap import QuadraturePolicy
from anyfesolver.general_contact.enriched_static import EnrichedStaticSystem, PairDefinition


def build(*, target_z=0., target_shift=0., source_offset=0., target_offset=0., rotation=None,
          cut_fraction=None, target_yield_stress=0., target_hardening_curve=None,
          fixed_nodal=True, target_options=None):
    import numpy as np
    model = FEModel("enriched_reference_linear_patch")
    model.add_material("elastic", 1e5, .3, density=1.)
    model.add_material("target_elastic", 1e5, .3, density=1., yield_stress=target_yield_stress,
                       hardening_curve=target_hardening_curve)
    source = np.array(((-.5, -.5, 0.), (.5, -.5, 0.), (.5, .5, 0.), (-.5, .5, 0.)))
    target = np.array(((-3., -3., target_z), (0., 4., target_z), (4., -3., target_z)))
    if cut_fraction is not None:
        if not 0 < cut_fraction < 1:
            raise ValueError("development cut fraction must lie in (0,1)")
        left = .5-cut_fraction
        target = np.array(((left, -3., target_z), (left, 3., target_z), (4., -3., target_z)))
    target[:, 0] += target_shift
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    for node, p in enumerate(np.vstack((source, target)) @ rotation.T, 1):
        model.add_node(node, *p)
    model.add_element(1, create_shell_element(1, (1, 2, 3, 4), "elastic", thickness=.05,
                                             reference_normal=tuple(rotation[:, 2])))
    model.add_element(2, create_shell_element(2, (5, 6, 7), "target_elastic", thickness=.05,
                                             formulation="e4-pl-s3",
                                             reference_normal=tuple(-rotation[:, 2]),
                                             **({} if target_options is None else dict(target_options))))
    if fixed_nodal:
        model.add_boundary_condition(BoundaryCondition("fixed_nodal", tuple(range(1, 8)),
            {name: 0. for name in ("ux", "uy", "uz", "rx", "ry", "rz")}))
    system = EnrichedStaticSystem.from_model(model,
        (PairDefinition(1, 2, source_offset, target_offset),), normal_stiffness=1e7,
        quadrature=QuadraturePolicy(1e-10, 1e-8, 1e-5, 512))
    return model, system


if __name__ == "__main__":
    import json
    import numpy as np
    from anyfesolver.general_contact.enriched_static import MixedCriteria, IterationControl
    _, system = build(target_z=-.01)
    result = system.solve(np.zeros(system.offset.size),
                          criteria=MixedCriteria(.04, 1e-6, 1e-8),
                          control=IterationControl(8, 8))
    print(json.dumps(result, sort_keys=True))
