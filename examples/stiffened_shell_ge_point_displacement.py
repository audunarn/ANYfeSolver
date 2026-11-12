"""Experimental native mixed point panel under fixed displacement targets.

Uses the public definition in the adjacent zero-unload example. The low-yield
fibre tests state lifecycle; it is not an engineering capacity material law.
"""
from pathlib import Path
from math import fsum
import argparse
import json
from anyfesolver import stiffened_shell as ss
from stiffened_shell_ge_point_zero_unload import definition


def workflow(output):
    output=Path(output);output.mkdir(exist_ok=False)
    model=definition()
    control=ss.create_analysis(definition=model)
    restarted=ss.create_analysis(definition=model)
    try:
        actions=(ss.NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
        direction=(ss.NodalAction('beam',105,(0.,0.,100.,0.,0.,0.)),)
        options=dict(max_iterations=20,max_backtracks=12)
        seed=control.solve(actions,(.5,1.),**options)
        assert any(fsum(row.incremental_dissipation_j)>0. for row in seed.beam_fibre_energy)
        # This definition places the final beam-axis reference node at z=0.1 m.
        initial=float(seed.beam_positions[-1,2]-.1)
        targets=(initial*.99,initial*1.01)
        options['displacement_control']=ss.DisplacementControl('beam',105,2)
        prefix=control.solve(direction,(targets[0],),**options)
        raw=control.checkpoint();(output/'prefix.checkpoint').write_bytes(raw)
        restored=restarted.restore(raw)
        assert restarted.checkpoint()==raw and restored.history==prefix.history
        final=control.solve(direction,(targets[1],),**options)
        again=restarted.solve(direction,(targets[1],),**options)
        assert control.checkpoint()==restarted.checkpoint() and final.history==again.history
        assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
        assert final.beam_fibre_material and final.beam_generalized_material
        result=dict(status=final.status,production_qualified=False,history=final.history,
            generation=final.generation,equilibrium=final.free_equilibrium_ratio,
            gap=final.constraint_max,exact_restart=True,reference_axis_z_m=.1,
            control={'domain':'beam','node':105,'component':2},
            fibre_last_increment_dissipation_j=[fsum(row.incremental_dissipation_j) for row in final.beam_fibre_energy])
        (output/'completed.checkpoint').write_bytes(control.checkpoint())
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return result
    finally:
        control.close();restarted.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    print(json.dumps(workflow(parser.parse_args().output)))
