"""Experimental simultaneous shell/beam yielding and fixed-predecessor restart.

Centered axial loading isolates state ownership in the reduced finite pad.
The generalized beam law is illustrative. This is not physical-connection,
capacity or convergence qualification; a 1% continuation remains unresolved.
"""
from dataclasses import replace
import json
import numpy as np
from anymaterial import DNVC208MaterialCurve
from anyfesolver import b3_ge, stiffened_shell as ss
from stiffened_shell_ge_distributing_finite import definition as elastic_definition


def definition():
    base=elastic_definition()
    element,inertia=base.beams[0].instantiate()
    coordinates=element.operator.reference.coordinates.copy();coordinates[:,2]=0.
    section=b3_ge.EllipsoidalGeneralizedSection(
        element.section.elastic,element.section.metric,1e6,1e11)
    beam=b3_ge.define_beam('b3-ge',element.element_id,element.node_ids,coordinates,
        element.operator.reference.nodal_triads,section,inertia,
        quadrature=element.operator.order,arithmetic_policy=element.operator.arithmetic_policy)
    shell=replace(base.shell,j2_curve=DNVC208MaterialCurve(200e6,220e6,230e6,.005,.01,1e9,.2))
    return replace(base,beams=(beam,),shell=shell,
        finite_distributing_material_policy=ss.GE_FINITE_DISTRIBUTING_COUPLED_J2_MATERIAL_V1)


def main():
    model=definition();owner=ss.create_analysis(definition=model)
    resumed=b3_ge.create_stiffened_shell_analysis(formulation='b3-ge',definition=model)
    loads=(ss.NodalAction('beam',103,(16e6,0.,0.,0.,0.,0.)),)
    try:
        yielded=owner.solve(loads,(.25,.5,.75,1.),max_iterations=24,max_backtracks=16)
        assert np.max(yielded.shell_layer_alpha)>0. and np.max(yielded.beam_station_accumulated)>0.
        checkpoint=owner.checkpoint();resumed.restore(checkpoint)
        a=owner.solve(loads,(1.0001,),max_iterations=24,max_backtracks=16)
        b=resumed.solve(loads,(1.0001,),max_iterations=24,max_backtracks=16)
        assert owner.checkpoint()==resumed.checkpoint()
        np.testing.assert_array_equal(a.shell_displacement,b.shell_displacement)
        assert b.free_equilibrium_ratio<=1e-8 and b.constraint_max<=1e-9
        print(json.dumps(dict(status='EXPERIMENTAL_NOT_QUALIFIED',
            material_policy=model.finite_distributing_material_policy,
            accepted_factors=b.accepted_factors,equilibrium=b.free_equilibrium_ratio,
            gap=b.constraint_max,shell_alpha=float(np.max(b.shell_layer_alpha)),
            beam_accumulated=float(np.max(b.beam_station_accumulated)),checkpoint_bytes=len(checkpoint))))
    finally:owner.close();resumed.close()


if __name__=='__main__':main()
