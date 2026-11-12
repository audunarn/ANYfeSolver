"""Experimental two-stiffener Q4 J2/fibre panel under fixed targets.

Uses the existing coupled-history example's deliberately low-strength laws.
This exercises lifecycle and control, not engineering capacity qualification.
"""
from pathlib import Path
from math import fsum
import argparse,json
from anyfesolver import stiffened_shell as ss
from stiffened_shell_coupled_plastic_history import definition,OPTIONS


def workflow(output):
    output=Path(output);output.mkdir(exist_ok=False)
    model=definition()
    analysis=ss.create_analysis(definition=model)
    resumed=ss.create_analysis(definition=model)
    try:
        loads=tuple(ss.NodalAction('beam',node,(0.,0.,1500.,0.,0.,0.)) for node in (105,205))
        seed=analysis.solve(loads,(.25,.5,.75,1.),**OPTIONS)
        assert max(max(row) for row in seed.shell_layer_alpha)>0.
        assert max(max(row) for row in seed.beam_station_accumulated)>0.
        # First stiffener ends at node105, reference z=0.1m in this definition.
        initial=float(seed.beam_positions[4,2]-.1)
        targets=(initial*.99,initial*1.01)
        direction=tuple(ss.NodalAction('beam',node,(0.,0.,150.,0.,0.,0.)) for node in (105,205))
        options=dict(OPTIONS,displacement_control=ss.DisplacementControl('beam',105,2))
        prefix=analysis.solve(direction,(targets[0],),**options)
        raw=analysis.checkpoint();(output/'prefix.checkpoint').write_bytes(raw)
        restored=resumed.restore(raw)
        assert resumed.checkpoint()==raw and restored.shell_layer_alpha==prefix.shell_layer_alpha
        final=analysis.solve(direction,(targets[1],),**options)
        again=resumed.solve(direction,(targets[1],),**options)
        assert analysis.checkpoint()==resumed.checkpoint() and final.history==again.history
        assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
        assert any(max(after)>max(before) for before,after in zip(prefix.shell_layer_alpha,final.shell_layer_alpha))
        assert any(fsum(row.incremental_dissipation_j)>0. for row in final.beam_fibre_energy)
        result=dict(status=final.status,production_qualified=False,history=final.history,
            generation=final.generation,equilibrium=final.free_equilibrium_ratio,gap=final.constraint_max,
            exact_restart=True,shell_alpha_max=max(max(row) for row in final.shell_layer_alpha),
            fibre_accumulated_max=max(max(row) for row in final.beam_station_accumulated),
            fibre_last_increment_dissipation_j=[fsum(row.incremental_dissipation_j) for row in final.beam_fibre_energy])
        (output/'completed.checkpoint').write_bytes(analysis.checkpoint())
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return result
    finally:analysis.close();resumed.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    print(json.dumps(workflow(parser.parse_args().output)))
