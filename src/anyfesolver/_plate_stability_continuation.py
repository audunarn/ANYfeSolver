"""Narrow opt-in energy trust-region continuation for virgin Q4 edge shear.

No physical regularization: the Hessian shift solves only the trust-region
subproblem. Acceptance uses the original residual and strictly positive curvature.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import eigh


def admit(model):
    from .fe_core import FEModel, FEMesh, Material, Node
    from .boundary import BoundaryCondition
    from .e4_pl_element import QualifiedE4PLShellElement, _require_exact_q4_runtime_authority
    if type(model) is not FEModel or type(model.mesh) is not FEMesh:
        raise ValueError('plate_energy requires builtin FEModel/FEMesh')
    if model.constraint_equations or len(model.boundary_conditions)!=2:
        raise ValueError('plate_energy requires simple clamp and driven edge only')
    if model.mesh.element_activity is not None or any(type(n) is not Node for n in model.mesh.nodes.values()):
        raise ValueError('plate_energy requires builtin nodes without activity/deletion')
    xyz={n:tuple(map(float,(node.x,node.y,node.z))) for n,node in model.mesh.nodes.items()}
    if not xyz: raise ValueError('plate_energy requires a plate mesh')
    xmin=min(x[0] for x in xyz.values());xmax=max(x[0] for x in xyz.values())
    left={n for n,x in xyz.items() if x[0]==xmin};right={n for n,x in xyz.items() if x[0]==xmax}
    names={'ux','uy','uz','rx','ry','rz'};clamp=False;drive=False
    bcs=[]
    for bc in model.boundary_conditions:
        if type(bc) is not BoundaryCondition: raise ValueError('plate_energy requires exact BoundaryCondition')
        values=dict(bc.dof_constraints);nodes=set(bc.node_ids)
        if nodes==left and set(values)==names and all(v==0 for v in values.values()):clamp=True
        elif nodes==right and set(values)=={'uy'} and np.isfinite(values['uy']) and values['uy']!=0:drive=True
        else: raise ValueError('plate_energy requires left clamp/right uniform uy')
        bcs.append((id(bc),tuple(bc.node_ids),tuple(values.items())))
    if not clamp or not drive or left&right: raise ValueError('plate_energy edge topology is unsupported')
    free=6*len(xyz)-6*len(left)-len(right)
    if free<=0 or free>512: raise ValueError('plate_energy admits at most 512 free coordinates')
    elements=[];lengths=[]
    for eid,e in model.mesh.elements.items():
        if type(e) is not QualifiedE4PLShellElement or e.shell_section is not None:
            raise ValueError('plate_energy requires scalar-section qualified Q4 elements')
        _require_exact_q4_runtime_authority(e, context='plate_energy polynomial material admission')
        mat=model.get_material(e.material_name)
        if type(mat) is not Material: raise ValueError('plate_energy requires builtin isotropic material')
        if mat.yield_stress!=0 or mat.hardening_curve is not None:
            raise ValueError('plate_energy requires virgin ideal elastic material')
        coords=np.array([xyz[n] for n in e.node_ids])
        lengths.extend(np.linalg.norm(coords-np.roll(coords,1,axis=0),axis=1).tolist())
        elements.append((eid,id(e),tuple(e.node_ids),e.thickness,e.drilling_stabilization,id(mat),mat.elastic_modulus,mat.poisson_ratio,mat.yield_stress,id(mat.hardening_curve)))
    if not lengths or min(lengths)<=0: raise ValueError('plate_energy requires nondegenerate elements')
    return (tuple(xyz.items()),tuple(bcs),tuple(elements)),min(lengths)


def segment_work(model,old,new,checkpoint,num_layers=5):
    """Exact polynomial local potential difference, without subtracting totals."""
    from .corotational import _corotational_cache,_corotational_deformation_state
    # The admitted virgin elastic local force is cubic along this affine ray;
    # two Gauss points integrate its work exactly (including the linear correction).
    refs=_corotational_cache(model);points,weights=np.polynomial.legendre.leggauss(2)
    terms=[]
    for eid,e in model.mesh.elements.items():
        checkpoint();dofs=e.get_dof_mapping(model.mesh)
        a,_,_=_corotational_deformation_state(refs[eid],e,old[dofs],stable_shell_rotation=True)
        b,_,_=_corotational_deformation_state(refs[eid],e,new[dofs],stable_shell_rotation=True);delta=b-a
        for x,w in zip(points,weights):
            checkpoint()
            f,_,_=e.compute_nonlinear_response(model.mesh,model.get_material(e.material_name),a+.5*(x+1)*delta,None,num_layers,False)
            checkpoint();terms.append(.5*w*float(delta@f))
    value=math.fsum(terms);screen=64*np.finfo(float).eps*math.fsum(abs(v) for v in terms)
    if not np.isfinite(value+screen): raise ValueError('nonfinite plate energy work')
    return value,screen


def trust_steps(values,vectors,gradient,radius):
    """Spectral trust-region solution, including a repeated-eigenvalue hard case."""
    g=vectors.T@gradient;low=float(values[0]);eps=np.finfo(float).eps
    if low>0:
        z=-g/values
        if np.linalg.norm(z)<=radius:return [vectors@z]
    shift=max(0.,-low);denom=values+shift
    singular=denom==0
    if np.any(singular) and np.all(g[singular]==0):
        z=np.zeros_like(g);z[~singular]=-g[~singular]/denom[~singular]
        if np.linalg.norm(z)<=radius and low<0:
            amplitude=math.sqrt(max(radius*radius-float(z@z),0.))
            j=int(np.flatnonzero(singular)[0]);direction=vectors[:,j]
            sign=1. if direction[np.argmax(abs(direction))]>=0 else -1.
            delta=np.zeros_like(z);delta[j]=sign*amplitude
            return [vectors@(z+delta),vectors@(z-delta)]
    # Solve in excess shift tau so tiny low-mode gradients are not lost
    # by adding a sub-ulp increment to a large -lambda_min.
    left=float(np.linalg.norm(g[singular]))/radius if np.any(singular) else 0.
    right=max(left,float(np.linalg.norm(g))/radius,np.finfo(float).tiny)
    def norm_at(lam):
        with np.errstate(divide='ignore',invalid='ignore'):
            z=np.divide(g,denom+lam)
        return np.linalg.norm(z)
    for _ in range(80):
        if norm_at(right)<=radius:break
        right=2*right
    else:raise ValueError('trust-region secular bracket failed')
    for _ in range(80):
        mid=(math.sqrt(left)*math.sqrt(right)
             if left>0 and right>2*left else .5*(left+right))
        if norm_at(mid)>radius:left=mid
        else:right=mid
    step=vectors@(-g/(denom+right))
    if not np.all(np.isfinite(step)):raise ValueError('nonfinite trust-region step')
    return [step]


def solve_increment(q_start,evaluate,work,checkpoint,discard,scale,h,tolerance,reference,max_iterations,diagnostics,small_step_work=None):
    q=q_start.copy();radius=.1*h;current=evaluate(q)
    # evaluate returns original residual, reduced Hessian, and original payload.
    for iteration in range(1,max_iterations+1):
        checkpoint();residual,k,payload=current;norm=float(np.linalg.norm(residual))
        matrix=k.toarray() if hasattr(k,'toarray') else np.asarray(k)
        scaled=(matrix*scale[:,None])*scale[None,:]
        symmetry=np.linalg.norm(scaled-scaled.T)/max(np.linalg.norm(scaled),1.)
        if not np.isfinite(symmetry) or symmetry>1e-12:
            return False,q_start,None,norm,iteration,'stability_nonsymmetric_tangent'
        checkpoint();values,vectors=eigh(.5*(scaled+scaled.T));checkpoint()
        diagnostics['eigensolves']+=1
        threshold=100*np.finfo(float).eps*len(q)*max(np.linalg.norm(scaled,np.inf),1.)
        diagnostics['last_minimum_curvature']=float(values[0]);diagnostics['last_curvature_threshold']=float(threshold)
        if norm<=tolerance*reference and values[0]>threshold:
            diagnostics['accepted_curvatures'].append({'minimum':float(values[0]),'threshold':float(threshold),'residual':norm})
            return True,q,payload,norm,iteration,None
        if norm<=tolerance*reference and abs(values[0])<=threshold:
            return False,q_start,None,norm,iteration,'stability_curvature_indeterminate'
        gradient=-residual*scale
        proposals=trust_steps(values,vectors,gradient,radius)
        best=None
        for step in proposals:
            predicted=-float(gradient@step+.5*step@scaled@step)
            if not np.isfinite(predicted) or predicted<=0:continue
            candidate=q+scale*step;checkpoint()
            tiny_newton = (values[0]>threshold and np.linalg.norm(step)<radius
                and np.linalg.norm(step)<=math.sqrt(np.finfo(float).eps)*max(h,float(np.linalg.norm(q/scale))))
            if tiny_newton and small_step_work is not None:
                difference,error=small_step_work(q,candidate)
                diagnostics['small_step_work_uses']=diagnostics.get('small_step_work_uses',0)+1
            else:
                difference,error=work(q,candidate)
            checkpoint();diagnostics['energy_proposals']+=1
            if difference+error>=0:continue
            ratio=-difference/predicted
            if best is None or difference<best[0]:best=(difference,ratio,candidate,step)
        if best is None or best[1]<.1:
            radius*=.25;diagnostics['rejected_proposals']+=1;discard()
        else:
            if best[1]>.75 and np.linalg.norm(best[3])>.9*radius:radius=min(2*radius,h)
            q=best[2];discard();current=evaluate(q)
        if radius<=np.finfo(float).eps*h:
            return False,q_start,None,norm,iteration,'stability_energy_resolution'
    return False,q_start,None,float(np.linalg.norm(current[0])),max_iterations,'stability_maximum_iterations'

def _seal_helper_authority():
    namespace=globals()
    names=('admit','segment_work','trust_steps','solve_increment','eigh','np','math')
    original=tuple(namespace[name] for name in names)
    codes=tuple(getattr(value,'__code__',None) for value in original)
    def require():
        if (namespace.get('require_authority') is not require
                or any(namespace.get(name) is not value for name,value in zip(names,original))
                or any(getattr(value,'__code__',None) is not code for value,code in zip(original,codes))):
            raise ValueError('plate_energy helper authority changed')
    return require


require_authority=_seal_helper_authority()
