"""Development-only S3 mortar pressure/Coulomb press, slide, reverse and open.

Run with an installed ANYfeSolver package. Fixed reference material quadrature,
elastic shells and explicit regularization; this is not C5 qualification.
"""
import json

import numpy as np

from anyfesolver import BoundaryCondition, create_shell_element, solve_static_nonlinear
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact import (
    ContactDefinition, ContactInteraction, ContactSurface, ContactSolverSettings,
    ContactPathControl, ContactPathProgram,
)
from anyfesolver.general_contact.s3_friction_point import S3SkinMaterialPoint
from anyfesolver.general_contact.s3_friction_pair import S3FrictionPointPair
from anyfesolver.general_contact.s3_friction_static import S3MaterialFrictionSettings


def run_example(*, material_anchor=False):
    model = FEModel("S3 pressure-coupled friction development")
    model.add_material("elastic", 1e5, .3, density=1)
    xyz = ((.2,.2,.065),(.6,.2,.065),(.2,.6,.065),
           (0,0,0),(1,0,0),(0,1,0),(1,1,0))
    for node, position in enumerate(xyz,1):
        model.add_node(node,*position)
    for element,nodes in ((1,(1,2,3)),(2,(4,5,6)),(3,(5,7,6))):
        model.add_element(element,create_shell_element(element,nodes,"elastic",thickness=.05,
            formulation="e4-pl-s3-v2d",reference_normal=(0,0,1),reference_surface_offset=.01))
    if material_anchor:
        model.add_boundary_condition(BoundaryCondition("target_plane",(4,5,6,7),
            {d:0 for d in ("uz","rx","ry","rz")}))
        model.add_boundary_condition(BoundaryCondition("target_origin",(4,),{"ux":0,"uy":0}))
        model.add_boundary_condition(BoundaryCondition("target_orientation",(5,),{"uy":0}))
    else:
        model.add_boundary_condition(BoundaryCondition("target",(4,5,6,7),
            {d:0 for d in ("ux","uy","uz","rx","ry","rz")}))
    model.add_boundary_condition(BoundaryCondition("guide",(1,2,3),
        {d:0 for d in ("uy","rx","ry","rz")}))
    contact = ContactDefinition((ContactSurface("source",(1,),"bottom"),
        ContactSurface("target",(2,3),"top")),(),
        (ContactInteraction("normal","source","target",1000.,.02,normal_law="projected_penalty"),))
    settings = ContactSolverSettings(pair_motion="general",normal_discretization="primal_mortar",
        general_s3_difference_step=2e-5,general_s3_differentiation="jet2_s3_dev")
    quadrature = []
    for bary in ((2/3,1/6,1/6),(1/6,2/3,1/6),(1/6,1/6,2/3)):
        xy = np.asarray(bary) @ np.asarray(xyz[:3])[:,:2]
        quadrature.append(S3FrictionPointPair(
            S3SkinMaterialPoint.from_model(model,1,1,"bottom",bary),
            S3SkinMaterialPoint.from_model(model,2,2,"top",(1-xy.sum(),*xy)),
            reference_weight_m2=.08/3,friction_coefficient=.3,
            tangential_stiffness_Pa_per_m=1000.,maximum_active_columns=42))
    friction = S3MaterialFrictionSettings(tuple(quadrature),
        (2e-7,)*model.mesh.dof_manager.total_dofs,maximum_candidate_evaluations=100)
    if material_anchor:
        from anyfesolver.general_contact.s3_sliding_friction import S3SlidingFrictionSettings
        friction = S3SlidingFrictionSettings(.3,1000.,
            (2e-7,)*model.mesh.dof_manager.total_dofs,maximum_candidate_evaluations=100,
            source_quadrature_refinement=3,activation_history="linear_constant_normal_touch_dev")
    sliding = (.39,.385) if material_anchor else (.008,.004)
    path = ContactPathProgram((ContactPathControl.node(1,"ux"),ContactPathControl.node(1,"uz")),
        ((0,0),(0,-.02),(.0002,-.02),(sliding[0],-.02),(sliding[1],-.02),
         (sliding[1],0),(sliding[1],-.02)),1)
    result = solve_static_nonlinear(model,contact=contact,contact_settings=settings,
        contact_path_program=path,contact_friction=friction,max_iterations=18,tolerance=1e-8,
        emit_restart_checkpoint=True)
    if result.status != "completed":
        raise RuntimeError(result.info)
    rows = result.info["force_displacement_history"]
    print(json.dumps(dict(status=result.status,profile=rows[0]["friction"]["profile"],
        phases=[{b:r["friction"]["branch"].count(b) for b in sorted(set(r["friction"]["branch"]))} for r in rows],
        control_reactions_n=[r["reaction_generalized"] for r in rows],
        dissipation_j=sum(r["friction"]["dissipation_increment_j"] for r in rows))))
    return result


if __name__ == "__main__":
    run_example()
