"""Headless virgin native reference modes; development, not qualification."""
from dataclasses import replace
import json
import numpy as np
from anyfesolver import b3_ge, stiffened_shell as ss
from examples.stiffened_shell_native_fibre_bl5 import definition as native_definition


def definition(shell="mixed", family="biaxial", beams="mixed", attachment="bl5", free=False):
    model = native_definition(family, beams if beams != "generalized" else "mixed", shell=shell)
    if beams == "generalized":
        section = model.beams[0].instantiate()[0].section
        members = tuple(b3_ge.define_beam("b3-ge", i+1, beam.instantiate()[0].node_ids,
            beam.instantiate()[0].operator.reference.coordinates,
            beam.instantiate()[0].operator.reference.nodal_triads, section, np.eye(6))
            for i, beam in enumerate(model.beams))
        model = replace(model, beams=members, bl5_material_policy="elastic")
    if attachment == "point":
        model = replace(model, lines=(), bl5_material_policy="elastic",
            points=(ss.PointConnection(1, 101), ss.PointConnection(4, 105)))
    elif attachment != "bl5":
        raise ValueError("point/bl5 attachment required")
    if free:
        model = replace(model, fixed=(), lines=(), bl5_material_policy="elastic",
            points=(ss.PointConnection(1, 101), ss.PointConnection(4, 105)),
            modal_boundary_policy=ss.GE_MIXED_FREE_FREE_MODAL_ONLY_V1)
    return model


def workflow(**kwargs):
    owner = ss.create_analysis(definition=definition(**kwargs))
    try:
        result = owner.modes(ss.GEMixedReferenceModesRequest(count=2))
        flexible = result.flexible if hasattr(result, "flexible") else result
        return dict(status=result.status, frequencies_hz=flexible.frequencies_hz,
            policy=result.policy, mass_policy=flexible.mass_policy,
            failure_reason=result.failure_reason, production_qualified=False)
    finally:
        owner.close()


if __name__ == "__main__":
    print(json.dumps(workflow(), indent=2, allow_nan=False))
