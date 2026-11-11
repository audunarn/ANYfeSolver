"""Owned S3 layer predictor in the unchanged native constitutive station order."""
from inspect import signature
import numpy as np
from .plasticity import plane_stress_elastic_matrix, plane_stress_return_map, _jit_flow_stress
from ._stiffened_shell_q4_material_event import Q4MaterialTrial, _origin_values, _boundary_amount
from .e4_pl_s3_v2d_element import qualified_s3_lobatto_layers


def capture(model, element, displacement, differential, origin, candidate, layers):
    material=model.get_material(element.material_name);curve=material.hardening_curve
    geometry=element._geometry(model.mesh)
    baseline=element._elastic_baseline_constitutive(material)
    operators=element._operators(geometry,baseline)
    membrane,curvature,_=element._effective_station_operators(operators)
    transform=np.asarray(geometry['local_from_external'],dtype=float)
    z,_=qualified_s3_lobatto_layers(layers,element.thickness)
    # Native candidate plastic/layer arrays retain INTERNAL station order.
    jacobian=[(bm+float(level)*bb)@transform@differential
              for bm,bb in zip(membrane,curvature) for level in z]
    flow=_jit_flow_stress(np.asarray(origin['alpha'],dtype=float),*(float(getattr(curve,n)) for n in
        ('sigma_prop','sigma_yield','sigma_yield_2','eps_p_y1','eps_p_y2','K','n','_power_offset')))
    return Q4MaterialTrial(origin,_origin_values(origin),tuple(map(float,displacement)),
        tuple(tuple(map(float,row)) for row in candidate['layer_strain']),
        tuple(tuple(tuple(map(float,row)) for row in layer) for layer in jacobian),
        tuple(tuple(map(float,row)) for row in plane_stress_elastic_matrix(material.elastic_modulus,material.poisson_ratio)),
        tuple(map(float,flow)),float(signature(plane_stress_return_map).parameters['tolerance'].default))


def boundary_amount(trial,direction,*,origin):
    return _boundary_amount(trial,direction,origin=origin,coordinate_count=18)
