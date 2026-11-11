"""Owned fixed-origin Q4 numerical-switch proposal, never acceptance/history."""
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from inspect import signature
from math import isfinite, nextafter, ulp
import numpy as np
from .plasticity import plane_stress_elastic_matrix, plane_stress_return_map, _jit_flow_stress, lobatto_layers

@dataclass(frozen=True)
class Q4MaterialTrial:
    origin: object
    origin_values: tuple
    displacement: tuple
    strain: tuple
    strain_differential: tuple
    elastic_matrix: tuple
    flow: tuple
    tolerance: float

def _origin_values(origin):
    def owned(value):
        if isinstance(value,(tuple,list,np.ndarray)):return tuple(owned(x) for x in value)
        return float(value)
    offsets=tuple((key,owned(origin[key])) for key in ('initial_membrane_stress','initial_bending_stress',
        'initial_membrane_prestrain','initial_curvature_prestrain','initial_generalized_prestrain','initial_generalized_resultant') if key in origin)
    return (tuple(tuple(float(x) for x in row) for row in origin['plastic_strain']),tuple(map(float,origin['alpha'])),offsets)

def capture(model, element, displacement, local, differential, origin, candidate, layers):
    material=model.get_material(element.material_name);curve=material.hardening_curve
    if curve is None:return None
    cache=element._nonlinear_geometry(model.mesh);u=cache['T0']@local; jacobian=[]
    z,_=lobatto_layers(layers,element.thickness)
    for gp in cache['gp']:
        theta=gp['Gw']@u
        nonlinear=np.vstack((theta[0]*gp['Gw'][0],theta[1]*gp['Gw'][1],theta[0]*gp['Gw'][1]+theta[1]*gp['Gw'][0]))
        for level in z:jacobian.append((gp['B_m']+nonlinear+level*gp['B_b'])@cache['T0']@differential)
    flow=_jit_flow_stress(np.asarray(origin['alpha'],dtype=float),*(float(getattr(curve,n)) for n in
        ('sigma_prop','sigma_yield','sigma_yield_2','eps_p_y1','eps_p_y2','K','n','_power_offset')))
    return Q4MaterialTrial(origin,_origin_values(origin),tuple(map(float,displacement)),
        tuple(tuple(map(float,row)) for row in candidate['layer_strain']),
        tuple(tuple(tuple(map(float,row)) for row in layer) for layer in jacobian),
        tuple(tuple(map(float,row)) for row in plane_stress_elastic_matrix(material.elastic_modulus,material.poisson_ratio)),
        tuple(map(float,flow)),float(signature(plane_stress_return_map).parameters['tolerance'].default))

def boundary_amount(trial,direction,*,origin):
    return _boundary_amount(trial,direction,origin=origin,coordinate_count=24)

def _boundary_amount(trial,direction,*,origin,coordinate_count):
    """Local predictor only: pose remainder and kernel/flow/dot rounding are
    not certified by the outward affine-update margin. Actual merit decides.
    """
    if type(trial) is not Q4MaterialTrial or trial.origin is not origin or trial.origin_values!=_origin_values(origin):
        raise ValueError('current origin-bound Q4 material trial required')
    values=tuple(direction)
    if type(coordinate_count) is not int or coordinate_count not in (18,24):
        raise ValueError('admitted shell coordinate dimension required')
    if len(values)!=coordinate_count or any(isinstance(x,(bool,np.bool_)) or not isfinite(float(x)) for x in values):
        raise ValueError('finite complete shell coordinate corrections required')
    if (len(trial.displacement)!=coordinate_count or len(trial.strain)!=len(trial.flow) or
        len(trial.strain)!=len(trial.origin_values[0]) or len(trial.strain)!=len(trial.origin_values[1]) or
        any(len(r)!=3 for r in trial.origin_values[0]) or
        len(trial.strain_differential)!=len(trial.strain) or len(trial.elastic_matrix)!=3 or
        any(len(r)!=3 for r in trial.elastic_matrix) or any(len(r)!=3 for r in trial.strain) or
        any(len(layer)!=3 or any(len(r)!=coordinate_count for r in layer) for layer in trial.strain_differential)):
        raise ValueError('complete Q4 material trial shapes required')
    scalars=(*trial.displacement,*trial.flow,trial.tolerance,*[x for r in trial.strain for x in r],
        *[x for r in trial.elastic_matrix for x in r],*[x for layer in trial.strain_differential for r in layer for x in r],
        *[x for r in trial.origin_values[0] for x in r],*trial.origin_values[1])
    if any(isinstance(x,(bool,np.bool_)) or not isfinite(float(x)) for x in scalars) or trial.tolerance<=0 or any(x<=0 for x in trial.flow):
        raise ValueError('finite positive Q4 material trial data required')
    with localcontext() as context:
        context.prec=80;dec=lambda x:D.from_float(float(x));dp=tuple(map(dec,values))
        matrix=tuple(tuple(map(dec,r)) for r in trial.elastic_matrix)
        def mv(m,x):return tuple(sum((a*b for a,b in zip(r,x)),D(0)) for r in m)
        def phi(x):return (x[0]+x[1])**2/12+(x[0]-x[1])**2/4+x[2]**2
        first=None
        for strain,jacobian,ep,flow in zip(trial.strain,trial.strain_differential,trial.origin_values[0],trial.flow):
            jac=tuple(tuple(map(dec,r)) for r in jacobian)
            stress=mv(matrix,tuple(dec(a)-dec(b) for a,b in zip(strain,ep)));change=mv(matrix,mv(jac,dp));sy=dec(flow)
            a=phi(change);b=(stress[0]+stress[1])*(change[0]+change[1])/6+(stress[0]-stress[1])*(change[0]-change[1])/2+2*stress[2]*change[2]
            c=phi(stress)-sy*sy/3-dec(trial.tolerance)*max(sy*sy,D(1));roots=[]
            if a:
                discriminant=b*b-4*a*c
                if discriminant<=0:continue  # tangent contact does not switch branch
                q=-(b+(discriminant.sqrt() if b>=0 else -discriminant.sqrt()))/2
                roots=(q/a,c/q) if q else (D(0),)
            elif b:roots=(-c/b,)
            for root in roots:
                if not 0<root<1:continue
                # _advance rounds amount*d and q+product. Uniform [0,1]
                # half-ULP bounds propagate through the local strain map.
                error=[]
                for p,d in zip(trial.displacement,values):
                    upper=nextafter(abs(p)+abs(float(d)),float('inf'))
                    if not isfinite(upper):return None
                    error.append((dec(ulp(float(d)))+dec(ulp(upper)))/2)
                strain_error=tuple(sum((abs(j)*e for j,e in zip(row,error)),D(0))+dec(ulp(s))/2+
                    dec(ulp(float(dec(s)-dec(p))))/2 for row,s,p in zip(jac,strain,ep))
                stress_error=mv(tuple(tuple(abs(x) for x in row) for row in matrix),strain_error)
                at=tuple(x+root*y for x,y in zip(stress,change));t0,t1,t2=stress_error;s=t0+t1
                # Bound represented local dot/phi arithmetic separately from
                # the unbounded nonlinear chart remainder. gamma_n is the
                # standard n round-to-nearest operation accumulation bound.
                unit=D(2)**-53;gamma5=5*unit/(1-5*unit);gamma10=10*unit/(1-10*unit)
                affine=tuple(dec(x)-dec(y)+root*v for x,y,v in zip(strain,ep,mv(jac,dp)))
                stress_error=tuple(e+gamma5*sum((abs(a*b) for a,b in zip(row,affine)),D(0)) for row,e in zip(matrix,stress_error))
                t0,t1,t2=stress_error;s=t0+t1
                uncertainty=abs(at[0]+at[1])*s/6+abs(at[0]-at[1])*s/2+2*abs(at[2])*t2+s*s/3+t2*t2
                uncertainty+=gamma10*(phi(at)+sy*sy/3+dec(trial.tolerance)*max(sy*sy,D(1)))
                slope=abs(2*a*root+b)
                if not slope:continue
                proposed=nextafter(float(root+uncertainty/slope),float('inf'))
                if isfinite(proposed) and 0<proposed<1 and (first is None or proposed<first):first=proposed
        return first
