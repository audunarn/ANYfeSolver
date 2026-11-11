"""Experimental fixed physical displacement, restart and physical pad recovery."""
import json
import numpy as np
from anyfesolver import stiffened_shell as ss
from stiffened_shell_ge_distributing_finite import definition


def main():
    model = definition()
    owner = ss.create_analysis(definition=model)
    resumed = ss.create_analysis(definition=model)
    actions = (ss.NodalAction("beam", 103, (0., 20., 100., 0., 0., 0.)),)
    control = ss.DisplacementControl("beam", 103, 2)
    kw = dict(displacement_control=control, max_iterations=20, max_backtracks=12)
    try:
        seed = owner.solve(actions, (.25,))  # Authentic accepted fixed-force seed.
        beam = json.loads(model.beams[0].raw)
        reference = dict(zip(beam["node_ids"], beam["reference"]["coordinates"]))
        node_index = sorted(reference).index(control.node)
        initial = float(seed.beam_positions[node_index, control.component]-
                        reference[control.node][control.component])
        owner.solve(actions, (initial*1.01,), **kw)
        raw = owner.checkpoint()
        resumed.restore(raw)
        a = owner.solve(actions, (initial*1.02,), **kw)
        b = resumed.solve(actions, (initial*1.02,), **kw)
        np.testing.assert_allclose(a.cell_resultants, b.cell_resultants, rtol=1e-10, atol=1e-12)
        record, = b.distributing_footprint_wrenches
        print(json.dumps(dict(status="EXPERIMENTAL_NOT_QUALIFIED",
            target_m=b.target, affine_increment_amplitude=b.amplitude,
            generation=b.generation, law=record.law,
            equilibrium_ratio=b.free_equilibrium_ratio, constraint_max=b.constraint_max,
            shell_action=record.shell_action.tolist(), beam_action=record.beam_action.tolist())))
    finally:
        owner.close(); resumed.close()


if __name__ == "__main__":
    main()
