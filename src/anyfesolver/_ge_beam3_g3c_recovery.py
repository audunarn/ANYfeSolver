"""Private source-native scalar recovery diagnostics; not qualification."""
from dataclasses import dataclass
import numpy as np

from ._ge_beam3_g3c_local_beam import owned

POLICY = 'GE_BEAM3_G3C_SCALAR_LOCAL_RECOVERY_V1'
ORDER = ('eps_x', 'gamma_xy', 'gamma_xz', 'kappa_x', 'kappa_y', 'kappa_z')
RESULTANT_ORDER = ('N', 'Vy', 'Vz', 'T', 'My', 'Mz')

class PhysicalRecoveryBlocked(ValueError):
    """The unchanged operator cannot support the claimed physical recovery."""
    code = 'BLOCKED_G3C_B2_PHYSICAL_RECOVERY_SHEAR_CLAMP'

@dataclass(frozen=True)
class StationDiagnostics:
    stations: np.ndarray
    weights: np.ndarray
    strains: np.ndarray
    resultants: np.ndarray
    strain_differential: np.ndarray
    reference_frame: np.ndarray
    current_frame: np.ndarray
    energy: float
    definition_sha256: str
    policy: str = POLICY
    strain_order: tuple = ORDER
    resultant_order: tuple = RESULTANT_ORDER

def recover(definition, kin, transform, element, material, seal):
    """Recover the scalar potential, never the legacy linear stress envelope.

    Called only with a fresh, already evaluated local element. Differential
    columns refer to local deformation in reference-global coordinates, not
    global spatial nodal wrenches. The caller applies the complete pose map.
    """
    ref=np.array(definition['coordinates']); n=len(ref)
    L=float(np.linalg.norm(ref[-1]-ref[0])); section=definition['section']
    E=definition['E']; G=E/(2*(1+definition['nu']))
    S=owned([E*section['area'],G*section['area']*section['shear_factor_y'],
        G*section['area']*section['shear_factor_z'],G*section['J'],
        E*section['Iy'],E*section['Iz']])
    if np.any(S<=0): raise ValueError('nonpositive computed section rigidity')
    points=np.array([-np.sqrt(3/5),0.,np.sqrt(3/5)])
    weights=np.array([5/9,8/9,5/9])*L/2
    local=transform@kin.deformation; rows=[]; jacobians=[]
    if n==2:
        if min(S[1],S[2])*L**2<1e-12:
            raise PhysicalRecoveryBlocked(PhysicalRecoveryBlocked.code)
        # The fresh actual scalar evaluation owns this cache. Do not invent
        # an effective physical section to absorb its historical shear floor.
        K=element._nl_cache['K_noax']; f0=K@local
        slope=(local[6:9]-local[:3])/L
        eps=slope[0]+.5*(slope[1]**2+slope[2]**2)
        deps=np.zeros(12)
        deps[:3]=-np.array([1.,slope[1],slope[2]])/L; deps[6:9]=-deps[:3]
        indices=[9,4,10,5,11]
        q=np.r_[S[0]*eps,f0[indices]]
        dq=np.vstack((S[0]*deps,K[indices]))
        energy=.5*float(local@K@local)+.5*S[0]*L*eps**2
        for xi in points:
            t=(xi+1)/2; H=np.zeros((6,6)); H[0,0]=H[3,1]=1
            H[1,4:6]=-1/L; H[2,2:4]=1/L
            H[4,2:4]=[t-1,t]; H[5,4:6]=[t-1,t]
            rows.append((H@q)/S)
            jacobians.append(((H@dq)/S[:,None])@transform)
    elif n==3:
        values=local.reshape(3,6)
        for xi in points:
            shape=np.array([xi*(xi-1)/2,1-xi**2,xi*(xi+1)/2])
            derivative=np.array([xi-.5,-2*xi,xi+.5])*2/L
            grad=derivative@values; interp=shape@values
            rows.append([grad[0]+.5*(grad[1]**2+grad[2]**2),
                         grad[1]-interp[5],grad[2]+interp[4],*grad[3:]])
            B=np.zeros((6,18))
            for i,(Ni,Di) in enumerate(zip(shape,derivative)):
                b=6*i
                B[0,b:b+3]=Di*np.array([1.,grad[1],grad[2]])
                B[1,b+1]=Di; B[1,b+5]=-Ni
                B[2,b+2]=Di; B[2,b+4]=Ni
                B[3,b+3]=B[4,b+4]=B[5,b+5]=Di
            jacobians.append(B@transform)
        energy=.5*float(np.einsum('i,ij,j,ij->',weights,rows,S,rows))
    else: raise ValueError('unregistered scalar recovery family')
    strain=owned(rows); resultant=owned(strain*S); differential=owned(jacobians)
    integrated=.5*float(np.einsum('i,ij,ij->',weights,strain,resultant))
    if not np.isfinite(energy) or not np.isfinite(integrated) or energy<0 or abs(integrated-energy)>1e-11*max(1.,abs(energy),abs(integrated)):
        raise ValueError('station and actual local potential disagree')
    frame=transform[:3,:3].T
    return StationDiagnostics(owned(points),owned(weights),strain,resultant,
        differential,owned(frame),owned(kin.frame@frame),float(energy),seal)
