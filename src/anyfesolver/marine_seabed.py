"""Stateless compliant frictionless seabed law, not W6 qualification.

Normal force is unilateral. Reference projected area controls stiffness/damping;
the surface gap includes the declared line radius. No friction/history is owned.
"""
from dataclasses import dataclass
import math
import numpy as np

API_VERSION = 'marine_seabed_contact_v1'
CONTINUOUS_API_VERSION = 'marine_seabed_contact_v2'
__all__ = ['API_VERSION', 'CONTINUOUS_API_VERSION', 'SeabedContactError', 'SeabedContact',
           'SeabedContactResponse', 'seabed_contact_force']


class SeabedContactError(ValueError):
    pass


def _number(value,name,*,positive=False,nonnegative=False):
    try:
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating))
                or not math.isfinite(value) or (positive and value<=0.) or (nonnegative and value<0.)):
            raise ValueError()
        return float(value)
    except (TypeError,ValueError,OverflowError) as error:
        raise SeabedContactError('invalid '+name) from error


@dataclass(frozen=True)
class SeabedContact:
    elevation: float
    normal_stiffness: float  # Pa/m
    normal_damping: float    # Pa*s/m
    activation_depth_m: float | None = None  # explicit v2 constitutive parameter

    def __post_init__(self):
        object.__setattr__(self,'elevation',_number(self.elevation,'seabed elevation'))
        object.__setattr__(self,'normal_stiffness',_number(self.normal_stiffness,'normal stiffness',positive=True))
        object.__setattr__(self,'normal_damping',_number(self.normal_damping,'normal damping',nonnegative=True))
        if self.activation_depth_m is not None:
            object.__setattr__(self,'activation_depth_m',_number(self.activation_depth_m,'activation depth',positive=True))

    def to_dict(self):
        data = dict(api_version=API_VERSION,elevation=self.elevation,
                    normal_stiffness=self.normal_stiffness,normal_damping=self.normal_damping)
        if self.activation_depth_m is not None:
            data.update(api_version=CONTINUOUS_API_VERSION,activation_depth_m=self.activation_depth_m)
        return data

    @classmethod
    def from_dict(cls,data):
        fields={'api_version','elevation','normal_stiffness','normal_damping'}
        continuous = isinstance(data,dict) and data.get('api_version')==CONTINUOUS_API_VERSION
        if continuous:fields.add('activation_depth_m')
        if not isinstance(data,dict) or set(data)!=fields or data['api_version'] not in (API_VERSION,CONTINUOUS_API_VERSION):
            raise SeabedContactError('exact named seabed contact schema required')
        depth = _number(data['activation_depth_m'],'activation depth',positive=True) if continuous else None
        return cls(*(data[k] for k in ('elevation','normal_stiffness','normal_damping')),activation_depth_m=depth)


@dataclass(frozen=True)
class SeabedContactResponse:
    force: np.ndarray
    elastic_force: np.ndarray
    damping_force: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    gaps: np.ndarray
    active: np.ndarray
    potential_energy: np.ndarray
    dissipation: np.ndarray


def seabed_contact_force(contact,diameter,reference_length,positions,velocities):
    """Fresh endpoint arrays and K=-dF/dp, C=-dF/dv on the selected branch.

    reference_length is this lump's reference material length (half a segment
    in the assembler). At gap=0 or raw normal=0 the inactive derivative is used.
    No differentiability through either branch boundary is claimed.
    Opt-in v2 ramps damping by min(penetration/activation_depth_m,1), retaining
    elastic storage and clipped outgoing loss. The depth is physical input,
    never derived from a time step. At the ramp corner use the saturated side.
    Legacy v1 retains its strict, discontinuous damping activation.
    """
    if type(contact) is not SeabedContact:raise SeabedContactError('public SeabedContact required')
    diameter=_number(diameter,'contact diameter',positive=True)
    reference_length=_number(reference_length,'contact reference length',positive=True)
    arrays=[]
    for value in (positions,velocities):
        try:
            raw=np.asarray(value)
            if any(isinstance(x,(bool,np.bool_)) for x in np.asarray(value,dtype=object).flat):raise ValueError()
            if raw.ndim!=2 or raw.shape[1]!=3 or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():raise ValueError()
            arrays.append(np.asarray(raw,dtype=float))
        except (TypeError,ValueError,OverflowError) as error:
            raise SeabedContactError('finite N by 3 endpoint arrays required') from error
    p,v=arrays
    if p.shape!=v.shape:raise SeabedContactError('endpoint position/velocity shapes must agree')
    with np.errstate(over='ignore',invalid='ignore'):
        area=diameter*reference_length;k=area*contact.normal_stiffness;c=area*contact.normal_damping
        gaps=p[:,2]-contact.elevation-.5*diameter;penetration=np.maximum(-gaps,0.)
        touching=gaps<0.;elastic=k*penetration
        if contact.activation_depth_m is None:
            effective_c=c
            effective_k=k
        else:
            depth=contact.activation_depth_m
            effective_c=c*np.minimum(penetration/depth,1.)
            effective_k=k-np.where(penetration<depth,(c/depth)*v[:,2],0.)
        raw=elastic-effective_c*v[:,2]
        active=touching&(raw>0.);normal=np.where(active,raw,0.)
        force=np.zeros_like(p);force[:,2]=normal
        elastic_force=np.zeros_like(p);elastic_force[:,2]=elastic
        damping_force=force-elastic_force
        K=np.zeros((len(p),3,3));C=K.copy()
        K[:,2,2]=np.where(active,effective_k,0.);C[:,2,2]=np.where(active,effective_c,0.)
        energy=.5*k*penetration**2
        dissipation=-damping_force[:,2]*v[:,2]
    if not all(np.isfinite(x).all() for x in (area,k,c,force,elastic_force,damping_force,K,C,gaps,energy,dissipation)):
        raise SeabedContactError('nonfinite seabed force/tangent/energy')
    return SeabedContactResponse(force,elastic_force,damping_force,K,C,gaps,active,energy,dissipation)
