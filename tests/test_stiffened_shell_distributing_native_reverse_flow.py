"""Actual biaxial fibre reverse flow; not engineering/capacity qualification."""
import json
from math import fsum

import numpy as np

from anyfesolver import stiffened_shell as ss
from test_stiffened_shell_distributing_native_material import definition, checks


HISTORY = (.25, .5, .75, 1., .5, 0., -.25, -.5, -.75, -1., -.5, 0., .5, 1.)


def test_actual_native_biaxial_reverse_flow_and_restart(tmp_path, record_property):
    # Same existing small-force mixed fixture, not an increased load or a new
    # connection. Its single low-yield fibre is a constitutive test control,
    # not an engineering steel or representative scale workload.
    model = definition(mixed=True, yielding="biaxial", biaxial=True)
    direct = ss.create_analysis(definition=model)
    resumed = ss.create_analysis(definition=model)
    first_signs, reverse_keys, steps, snapshots = {}, set(), [], {}
    negative_snapshot_reverse_keys = set()
    previous = []
    def observe(result):
        checks(direct, result, previous[-1] if previous else None)
        flow, stress = {}, {}
        for station in result.beam_fibre_material:
            for fibre in station.fibres:
                key = (station.element_id, station.cell, station.station, fibre.fibre_id)
                dz = fsum(fibre.plastic_increment)
                flow[str(key)] = dz
                stress[str(key)] = fsum(fibre.stress)
                if dz != 0.:
                    if result.load_factor > 0. and len(steps) < 4:
                        first_signs.setdefault(key, np.sign(dz))
                    elif result.load_factor < 0. and key in first_signs and np.sign(dz) != first_signs[key]:
                        reverse_keys.add(key)
                        if result.load_factor == -1.:
                            negative_snapshot_reverse_keys.add(key)
        raw = direct.checkpoint()
        name = f"accepted-{result.generation:02d}.checkpoint"
        (tmp_path/name).write_bytes(raw)
        if result.load_factor == -1.:
            snapshots["negative"] = raw
        steps.append(dict(factor=result.load_factor, generation=result.generation,
            equilibrium=result.free_equilibrium_ratio, compatibility=result.constraint_max,
            plastic_increments=flow, fibre_stress_pa=stress,
            dissipation_j=sum(fsum(row.incremental_dissipation_j) for row in result.beam_fibre_energy),
            checkpoint=name))
        (tmp_path/'history.json').write_text(json.dumps(steps,indent=2)+'\n')
        previous.append(result)
    try:
        try:
            final = direct.solve(model.cyclic_reference_actions, HISTORY,
                max_iterations=24, max_backtracks=16, on_accepted=observe)
        except ss.FiniteIncrementFailure as error:
            (tmp_path/'failed-prefix.checkpoint').write_bytes(direct.checkpoint())
            (tmp_path/'failure.json').write_text(json.dumps(dict(
                error=str(error), accepted_factors=list(direct.recover().accepted_factors)),indent=2)+'\n')
            raise
        assert first_signs and reverse_keys and negative_snapshot_reverse_keys
        assert final.accepted_factors == HISTORY and len(steps) == len(HISTORY)
        negative = snapshots["negative"]
        restored = resumed.restore(negative)
        assert restored.load_factor == -1. and resumed.checkpoint() == negative
        continued = resumed.solve(model.cyclic_reference_actions, HISTORY[10:],
            max_iterations=24, max_backtracks=16)
        assert resumed.checkpoint() == direct.checkpoint()
        for field in ("shell_displacement", "cell_resultants", "support_reactions"):
            np.testing.assert_array_equal(getattr(continued,field),getattr(final,field))
        assert continued.beam_fibre_energy == final.beam_fibre_energy
        assert [r.fibres for r in continued.beam_fibre_material] == [r.fibres for r in final.beam_fibre_material]
        record_property("native_reverse_flow", json.dumps(dict(
            factors=HISTORY, actual_reverse_flow_keys=sorted(reverse_keys),
            negative_snapshot_reverse_flow_keys=sorted(negative_snapshot_reverse_keys),
            initial_flow_keys=len(first_signs), history_path=str(tmp_path/'history.json'),
            equilibrium_max=max(step['equilibrium'] for step in steps),
            compatibility_max=max(step['compatibility'] for step in steps),
            cumulative_beam_dissipation_j=sum(step['dissipation_j'] for step in steps),
            restart_checkpoint_bytes=len(negative), exact_restart=True,
            scope='small-force constitutive/lifecycle development only; not capacity or engineering qualification')))
    finally:
        direct.close(); resumed.close()
