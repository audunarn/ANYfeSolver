"""Experimental finite distributing pose, atomic recovery and binary restart.

This reduced area-mean/director pose is not a pointwise weld. Elastic flat Q4,
generalized B3, monotone translation-force stepping only; no qualified physical
accuracy, yielding, spectral, chart-rebase or controller claim.
"""
from dataclasses import replace
import json
import numpy as np
from anyfesolver import stiffened_shell as ss
from stiffened_shell_ge_distributing_reference import definition as reference_definition


def definition():
    reference=reference_definition()
    return replace(reference,distributing_footprints=(),finite_distributing_footprints=(
        ss.GEFiniteDistributingFootprint(reference.distributing_footprints[0]),))


def main():
    model=definition();owner=ss.create_analysis(definition=model);resumed=ss.create_analysis(definition=model)
    actions=(ss.NodalAction("beam",103,(0.,20.,100.,0.,0.,0.)),)
    try:
        owner.solve(actions,(.25,.5))
        checkpoint=owner.checkpoint();resumed.restore(checkpoint)
        a=owner.solve(actions,(1.,));b=resumed.solve(actions,(1.,))
        np.testing.assert_allclose(a.shell_displacement,b.shell_displacement,rtol=1e-10,atol=1e-12)
        record,=b.distributing_footprint_wrenches
        assert b.free_equilibrium_ratio<=1e-8 and b.constraint_max<=1e-9
        print(json.dumps(dict(status="EXPERIMENTAL_NOT_QUALIFIED",law=record.law,
            model_identity=resumed.preflight()["identity"],accepted_factors=b.accepted_factors,
            equilibrium_ratio=b.free_equilibrium_ratio,constraint_max=b.constraint_max,
            checkpoint_bytes=len(checkpoint),shell_action=record.shell_action.tolist(),
            beam_action=record.beam_action.tolist()),sort_keys=True))
    finally:owner.close();resumed.close()


if __name__=="__main__":main()
