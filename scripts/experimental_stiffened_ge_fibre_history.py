"""Experimental Q4/native fibre history through the public point-panel owner.

One physical fibre yields while the other section fibres remain elastic.
This is a state/recovery/restart demonstrator, not G3 acceptance or a claim
that strong full-section yielding or pure-fibre S3 convergence is resolved.
"""
import argparse
from dataclasses import replace
import json

import anyfesolver
from anyfesolver import b3_ge
from anyfesolver.stiffened_shell import NodalAction
from experimental_stiffened_ge_panel import make_definition


def run(mixed=False):
    base = make_definition('mixed' if mixed else 'physical_fibre')
    beams = []
    for definition in base.beams:
        element, inertia = definition.instantiate()
        if type(element.section) is not b3_ge.PhysicalFibreSection:
            beams.append(definition)
            continue
        section = b3_ge.PhysicalFibreSection(tuple(
            replace(fibre, curve=b3_ge.FlowCurve((0.,),(1e5,),1e9))
            if i == 0 else fibre for i,fibre in enumerate(element.section.fibres)),
            element.section.background_factor)
        ref = element.operator.reference
        beams.append(b3_ge.define_beam('b3-ge',element.element_id,element.node_ids,
            ref.coordinates,ref.nodal_triads,section,inertia,
            quadrature=element.operator.order))
    definition = replace(base,beams=tuple(beams))
    owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    resumed = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    try:
        actions = (NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
        first = owner.solve(actions,targets=(.5,1.),max_iterations=18)
        saved = owner.checkpoint()
        restored = resumed.restore(saved)
        assert resumed.checkpoint() == saved
        assert restored.beam_station_accumulated == first.beam_station_accumulated
        unloaded = resumed.solve(actions,targets=(.5,),max_iterations=18)
        native = first.beam_fibre_material
        return {'status':first.status,'production_qualified':first.production_qualified,
            'capabilities':owner.capabilities,'accepted_factors':unloaded.accepted_factors,
            'maximum_accumulated_fibre_strain':max(
                value for row in first.beam_station_accumulated for value in row),
            'maximum_last_fibre_plastic_increment':max(abs(sum(f.plastic_increment))
                for record in native for f in record.fibres),
            'native_recovery_stations':len(native),'checkpoint_bytes':len(saved),
            'restart_bytes_equal':True,'equilibrium_ratio':unloaded.free_equilibrium_ratio,
            'constraint_max':unloaded.constraint_max}
    finally:
        owner.close()
        resumed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixed',action='store_true')
    print(json.dumps(run(parser.parse_args().mixed),sort_keys=True,allow_nan=False))
