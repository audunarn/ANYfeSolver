"""Experimental fixed axial translation control, native BL5 history and restart."""
import json
from anyfesolver import stiffened_shell as ss
from examples.stiffened_shell_native_fibre_bl5_material import definition

ACTIONS = (ss.NodalAction('beam', 105, (500., 0., 0., 0., 0., 0.)),)
CONTROL = ss.DisplacementControl('beam', 105, 0)
OPTIONS = dict(max_iterations=24, max_backtracks=12)
TARGETS = (1e-6, 1e-5, 5e-6)


def workflow(family='biaxial', shell_material=True, on_accepted=None):
    model = definition(family=family, shell_material=shell_material)
    owner = ss.create_analysis(definition=model)
    fresh = ss.create_analysis(definition=model)
    try:
        owner.solve(ACTIONS, (.001,), **OPTIONS)
        loaded = owner.solve(ACTIONS, TARGETS[:2], displacement_control=CONTROL,
                             on_accepted=on_accepted, **OPTIONS)
        raw = owner.checkpoint()
        fresh.restore(raw)
        assert fresh.checkpoint() == raw
        final = owner.solve(ACTIONS, TARGETS[2:], displacement_control=CONTROL,
                            on_accepted=on_accepted, **OPTIONS)
        resumed = fresh.solve(ACTIONS, TARGETS[2:], displacement_control=CONTROL, **OPTIONS)
        assert owner.checkpoint() == fresh.checkpoint()
        return dict(status=final.status, scope='experimental fixed axial translation control',
                    family=family, target=final.target, amplitude=final.amplitude,
                    generation=final.generation, resumed_generation=resumed.generation,
                    history=final.history, retained_dofs=owner.preflight()['retained_dofs'],
                    shell_alpha_max=max((max(row) for row in final.shell_layer_alpha), default=0.),
                    fibre_alpha_max=max((max(row) for row in final.beam_station_accumulated), default=0.),
                    free_equilibrium_ratio=final.free_equilibrium_ratio,
                    constraint_max=final.constraint_max,
                    fresh_restore_bytes_equal=True, continued_bytes_equal=True,
                    production_qualified=False)
    finally:
        owner.close()
        fresh.close()


if __name__ == '__main__':
    print(json.dumps(workflow(), indent=2, allow_nan=False))
