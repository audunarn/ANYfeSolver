"""Private registered-context physical recovery, pending numerical qualification.

No public routing, graph commit/restart, material history or general-affine admission.
The unchanged qualified kernel remains the authority for the trial potential.
"""
from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
from threading import Lock
import numpy as np
from ._ge_beam3_g3c_local_shell import (owned, array, rotations, canonical,
    family_objects, tensor)
from ._ge_beam3_g3c_affine_q4_increment_chart import deformation, _exp_terms, NUMERICS_ID as CHART_NUMERICS_ID
from ._ge_beam3_g3c_affine_q4_registry import construction

POLICY = 'GE_BEAM3_G3C_AFFINE_Q4_PHYSICAL_FACADE_V1'
RECOVERY_ID = 'GE_BEAM3_Q4_AFFINE_CHART_PHYSICAL_RECOVERY_V1'
REPRESENTATION_ID = 'GE_BEAM3_Q4_AFFINE_STATION_RECOVERY_64_V1'
OPERATOR_ID = 'E4_PL_QUALIFIED_Q4_HYBRID_V2'
STATION_ASSOCIATION_ID = 'GE_BEAM3_Q4_NATURAL_COORDINATE_BIJECTION_V1'

def _expected_descriptor(c):
    """Complete authority reconstruction, never a digest of caller-supplied facts."""
    return canonical(dict(policy=POLICY,recovery_id=RECOVERY_ID,representation_id=REPRESENTATION_ID,
        operator_id=OPERATOR_ID,chart_numerics_id=CHART_NUMERICS_ID,
        station_association_id=STATION_ASSOCIATION_ID,
        construction_id=c.construction_id,recipe_sha256=c.recipe_sha256,
        family='Q4',node_ids=c.node_ids,coordinates=c.coordinates,normal=c.normal,
        material_direction=c.material_direction,director_polarity=c.director_polarity,
        E=100.,nu=.25,thickness=.1,layers=3,origin='OWNED_VIRGIN_ONLY'))

class AffineRecoveryCancelled(RuntimeError):
    pass

def _payload(value):
    """Injective private fingerprint encoding; returned API bytes stay bytes.

    Every container/scalar has a disjoint type tag, so a caller dictionary
    resembling a bytes tag cannot collide with bytes or a dataclass payload.
    This encoding does not change any inherited public JSON/restart serializer.
    """
    if hasattr(type(value), '__dataclass_fields__'):
        return dict(kind='dataclass',name=type(value).__module__+'.'+type(value).__qualname__,
                    fields=[[f.name,_payload(getattr(value,f.name))] for f in fields(value)])
    if type(value) is bytes:
        return dict(kind='bytes',hex=value.hex())
    if type(value) is np.ndarray:
        if value.dtype != np.dtype('float64') or not np.isfinite(value).all():
            raise ValueError('finite binary64 fingerprint array required')
        return dict(kind='ndarray',dtype=value.dtype.str,shape=list(value.shape),
                    hex=np.ascontiguousarray(value).tobytes().hex())
    if isinstance(value,np.generic):
        return dict(kind='numpy_scalar',dtype=value.dtype.str,value=_payload(value.item()))
    if type(value) in (tuple,list):
        return dict(kind=type(value).__name__,items=[_payload(v) for v in value])
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError('string fingerprint mapping keys required')
        return dict(kind='mapping',items=[[k,_payload(value[k])] for k in sorted(value)])
    if value is None: return dict(kind='none')
    if type(value) is float:
        if not math.isfinite(value): raise ValueError('nonfinite fingerprint scalar')
        return dict(kind='float',hex=value.hex())
    if type(value) in (str,int,bool):
        return dict(kind=type(value).__name__,value=value)
    raise ValueError('unsupported private fingerprint value')

def _fingerprint(value):
    return sha256(canonical(_payload(value))).hexdigest()

def _engineering_map(source_frame, target_frame):
    """Engineering membrane tensor components from source to target frame."""
    columns = []
    for v in np.eye(3):
        T = tensor([v],source_frame,.5)[0]
        L = target_frame.T@T@target_frame
        columns.append((L[0,0],L[1,1],2*L[0,1]))
    return owned(np.array(columns).T)

def _station_bijection(source_natural,target_natural):
    """Exact target-to-source point join; quadrature index is not an identity."""
    original=array(source_natural,(4,2)); target=array(target_natural,(4,2))
    source_keys=tuple(tuple(row) for row in original)
    target_keys=tuple(tuple(row) for row in target)
    if len(set(source_keys))!=4 or len(set(target_keys))!=4:
        raise ValueError('duplicate quadrature point identity')
    if set(source_keys)!=set(target_keys):
        raise ValueError('mixed and nonlinear quadrature points do not coincide')
    order=tuple(source_keys.index(point) for point in target_keys)
    if len(set(order))!=4 or not np.array_equal(original[list(order)],target):
        raise ValueError('nonbijective quadrature association')
    return order

def _nonlinear_terms(gw, d, engineering):
    p = gw@d
    n0 = np.array((.5*p[0]**2,.5*p[1]**2,p[0]*p[1]))
    first0 = np.array((p[0]*gw[0],p[1]*gw[1],p[0]*gw[1]+p[1]*gw[0]))
    second0 = np.array((np.outer(gw[0],gw[0]),np.outer(gw[1],gw[1]),
                        np.outer(gw[0],gw[1])+np.outer(gw[1],gw[0])))
    n = np.zeros(8); first = np.zeros((8,24)); second = np.zeros((8,24,24))
    n[:3] = engineering@n0; first[:3] = engineering@first0
    second[:3] = np.einsum('ab,bij->aij',engineering,second0)
    return owned(n), owned(first), owned(second)

def _station_saddle(C, weight):
    I = np.eye(8); z = np.zeros((8,8))
    return owned(weight*np.block([[C,-I],[-I,z]])), owned(np.block([[z,-I],[-I,-C]])/weight)

def _schur_terms(J, C, s, Fsecond, weight):
    return owned(weight*J.T@C@J), owned(weight*np.einsum('a,aij->ij',s,Fsecond))

def _pullback_station(M,n,Dn,Hn,C,weight,kin):
    B = M+Dn; e = M@kin.deformation+n; s = C@e
    J = B@kin.differential
    Fsecond = np.einsum('ip,aij,jq->apq',kin.differential,Hn,kin.differential)
    Fsecond += np.einsum('ai,ipq->apq',B,kin.second)
    material, geometric = _schur_terms(J,C,s,Fsecond,weight)
    return tuple(owned(v) for v in (e,s,J,Fsecond,weight*J.T@s,material+geometric))

@dataclass(frozen=True)
class Prepared:
    frame: np.ndarray
    physical_frame: np.ndarray
    physical_map: np.ndarray
    constitutive: np.ndarray
    stationary: np.ndarray
    coupling: np.ndarray
    solution: np.ndarray
    mixed_maps: np.ndarray
    natural: np.ndarray
    weights: np.ndarray
    interpolation: np.ndarray
    reference_positions: np.ndarray
    centre_frame: np.ndarray
    engineering: np.ndarray
    membrane_maps: np.ndarray
    transverse_maps: np.ndarray
    source_weights: np.ndarray
    linear_physical: np.ndarray
    operator_id: str
    definition_sha256: str
    source_order: tuple
    nonlinear_source_natural: np.ndarray
    nonlinear_source_positions: np.ndarray

def _prepare_reference(element,mesh,material,coordinates,*,definition_sha256):
    from . import e4_pl_element as source
    mixed = element._recover_planar_mixed_fields(mesh,np.zeros(24),material,source._GAUSS)
    solution = source._solve_stationary_system(mixed['stationary_matrix'],mixed['stationary_coupling'])[0]
    c = source._coefficients(mixed['local_nodes']); frame = mixed['frame']
    transform = source._global_transform(frame).T
    maps = np.array([-source._source_fields(c,r,s)[1]@solution[14:]@transform for r,s in source._GAUSS])
    geometry = element._nonlinear_geometry(mesh)
    source_natural=owned(element.gauss_points)
    target_natural=owned(source._GAUSS)
    source_order=_station_bijection(source_natural,target_natural)
    if len(geometry['gp'])!=4:
        raise ValueError('four identified nonlinear station records required')
    ordered_gp=tuple(geometry['gp'][i] for i in source_order)
    physical_frame, membrane, curvature, shear, _ = element._physical_director_context(frame)
    P = np.zeros((8,8)); P[:3,:3]=membrane; P[3:6,3:6]=curvature; P[6:,6:]=shear
    interpolation = np.array([element.compute_shape_functions(float(r),float(s))[0] for r,s in source._GAUSS])
    source_interpolation=owned([element.compute_shape_functions(float(r),float(s))[0] for r,s in source_natural])
    source_positions=owned(source_interpolation@coordinates)
    if (not np.array_equal(source_interpolation[list(source_order)],interpolation)
            or not np.array_equal(source_positions[list(source_order)],interpolation@coordinates)):
        raise ValueError('associated quadrature shape functions or physical positions differ')
    weights = owned(mixed['jacobian_determinants'])
    source_weights = owned([gp['detw'] for gp in ordered_gp])
    if weights.shape != (4,) or np.any(weights <= 0) or np.any(source_weights <= 0):
        raise ValueError('positive four-station measure required')
    components = element.compute_stiffness_components(mesh,material)
    # Both independently exposed source descriptions are retained, not silently averaged.
    return Prepared(owned(frame),owned(physical_frame),owned(P),owned(mixed['constitutive']),
        owned(mixed['stationary_matrix']),owned(mixed['stationary_coupling']),owned(solution),
        owned(maps),owned(source._GAUSS),weights,owned(interpolation),owned(interpolation@coordinates),
        owned(geometry['R0']),_engineering_map(geometry['R0'],frame),
        owned([gp['B_m']@geometry['T0'] for gp in ordered_gp]),
        owned([gp['Gw']@geometry['T0'] for gp in ordered_gp]),source_weights,
        owned(components['physical']),str(element.formulation_id),definition_sha256,
        source_order,source_natural,source_positions)

@dataclass(frozen=True)
class Station:
    natural: np.ndarray
    weight: float
    source_weight: float
    reference_position: np.ndarray
    current_position: np.ndarray
    numbered_frame: np.ndarray
    physical_frame: np.ndarray
    constitutive: np.ndarray
    mixed_map: np.ndarray
    nonlinear: np.ndarray
    nonlinear_first: np.ndarray
    nonlinear_second: np.ndarray
    strain: np.ndarray
    resultant: np.ndarray
    physical_strain: np.ndarray
    physical_resultant: np.ndarray
    reference_strain_tensors: np.ndarray
    current_strain_tensors: np.ndarray
    reference_resultant_tensors: np.ndarray
    current_resultant_tensors: np.ndarray
    reference_shear_strain: np.ndarray
    current_shear_strain: np.ndarray
    reference_shear_resultant: np.ndarray
    current_shear_resultant: np.ndarray
    chart_first: np.ndarray
    chart_second: np.ndarray
    internal_block: np.ndarray
    internal_inverse: np.ndarray
    internal_residual: np.ndarray
    old_mixed_resultant: np.ndarray
    source_index: int
    source_natural: np.ndarray
    source_reference_position: np.ndarray

def _station_fields(prepared,kin):
    from . import e4_pl_element as source
    stations = []; forces = []; tangents = []
    C = prepared.constitutive; R = kin.frame
    params = -prepared.solution@source._global_transform(prepared.frame).T@kin.deformation
    # The old equilibrium resultant is a separately labelled diagnostic only.
    local = (kin.reference_positions-kin.reference_positions.mean(axis=0))@prepared.frame
    coeff = source._coefficients(local)
    for i,(r,s0) in enumerate(prepared.natural):
        w = float(prepared.weights[i]); M = prepared.mixed_maps[i]
        n,Dn,Hn = _nonlinear_terms(prepared.transverse_maps[i],kin.deformation,prepared.engineering)
        e,s,J,Fsecond,f,H = _pullback_station(M,n,Dn,Hn,C,w,kin)
        block,inverse = _station_saddle(C,w)
        residual = owned(w*np.concatenate((C@e-s,M@kin.deformation+n-e)))
        pe = prepared.physical_map@e; ps = prepared.physical_map@s; F = prepared.physical_frame
        et = tensor([pe[:3],pe[3:6]],F,.5); st = tensor([ps[:3],ps[3:6]],F)
        ev = owned(F[:,:2]@pe[6:]); sv = owned(F[:,:2]@ps[6:])
        stations.append(Station(owned((r,s0)),w,float(prepared.source_weights[i]),
            prepared.reference_positions[i],owned(prepared.interpolation[i]@kin.current_positions),
            prepared.frame,F,C,M,n,Dn,Hn,e,s,owned(pe),owned(ps),et,owned(R@et@R.T),
            st,owned(R@st@R.T),ev,owned(R@ev),sv,owned(R@sv),J,Fsecond,block,inverse,residual,
            owned(source._source_fields(coeff,float(r),float(s0))[0]@params[:14]),
            prepared.source_order[i],prepared.nonlinear_source_natural[prepared.source_order[i]],
            prepared.nonlinear_source_positions[prepared.source_order[i]]))
        forces.append(f); tangents.append(H)
    return tuple(stations),owned(sum(forces)),owned(sum(tangents))

@dataclass(frozen=True)
class NumericalChannel:
    name: str
    energy: float
    local_force: np.ndarray
    local_tangent: np.ndarray
    chart_force: np.ndarray
    chart_hessian: np.ndarray

def _spatial_connection(u,g,H):
    P = np.eye(24); terms = []
    for i,row in enumerate(u.reshape(4,6)):
        _,J,dJ = _exp_terms(row[3:]); sl = slice(6*i+3,6*i+6)
        P[sl,sl] = J; terms.append((sl,dJ))
    r = np.linalg.solve(P.T,g); correction = np.zeros((24,24))
    for sl,dJ in terms: correction[sl,sl] = np.einsum('ijk,i->jk',dJ,r[sl])
    return owned(r),owned(np.linalg.solve(P.T,H-correction))

@dataclass(frozen=True)
class AffineQ4Trial:
    kinematics: object
    stations: tuple
    source_stationary_matrix: np.ndarray
    source_coupling: np.ndarray
    source_solution: np.ndarray
    source_linear_physical: np.ndarray
    local_force: np.ndarray
    local_tangent: np.ndarray
    physical_energy: float
    physical_chart_force: np.ndarray
    physical_chart_hessian: np.ndarray
    source_physical_energy: float
    source_physical_chart_force: np.ndarray
    source_physical_chart_hessian: np.ndarray
    chart_force: np.ndarray
    chart_hessian: np.ndarray
    spatial_force: np.ndarray
    spatial_row_chart_tangent: np.ndarray
    physical_spatial_force: np.ndarray
    physical_spatial_row_chart_tangent: np.ndarray
    numerical_channels: tuple
    internal_block64: np.ndarray
    internal_inverse64: np.ndarray
    internal_coupling64: np.ndarray
    direct_chart_hessian: np.ndarray
    schur_chart_hessian: np.ndarray
    internal_residual64: np.ndarray
    energy: float
    candidate: bytes
    definition_sha256: str
    descriptor_bytes: bytes
    operator_id: str
    recovery_id: str = RECOVERY_ID
    representation_id: str = REPRESENTATION_ID
    state_committed: bool = False
    production_qualified: bool = False
    full_g3c_qualified: bool = False

class AffineQ4PhysicalRecovery:
    __slots__ = ('_body','_seal','_lock','_prepared','_prepared_seal')
    def __setattr__(self,name,value):
        if hasattr(self,name): raise AttributeError('write-once registered definition')
        object.__setattr__(self,name,value)
    def __delattr__(self,name): raise AttributeError('registered definition cannot be deleted')

    def __init__(self,construction_id,*,node_ids=None,coordinates=None,normal=None,
                 material_direction=None,director_polarity=None):
        c = construction(construction_id)
        if node_ids is not None and (type(node_ids) is not tuple or node_ids != c.node_ids
                                    or any(type(n) is not int for n in node_ids)):
            raise ValueError('node identity differs from registered recipe')
        for given,expected,shape in ((coordinates,c.coordinates,(4,3)),(normal,c.normal,(3,)),
                                     (material_direction,c.material_direction,(3,))):
            if given is not None and array(given,shape).tobytes() != expected.tobytes():
                raise ValueError('reference bytes differ from registered recipe; no snapping permitted')
        if director_polarity is not None and (type(director_polarity) is not int or director_polarity != c.director_polarity):
            raise ValueError('director policy differs from registered recipe')
        self._body = _expected_descriptor(c)
        self._seal=sha256(self._body).hexdigest(); self._lock=Lock(); self._prepared=None; self._prepared_seal=None

    def descriptor(self):
        if type(self) is not AffineQ4PhysicalRecovery or type(self._body) is not bytes or type(self._seal) is not str or sha256(self._body).hexdigest()!=self._seal:
            raise ValueError('registered recovery definition changed')
        body,seal=self._body,self._seal
        description=json.loads(body)
        if type(description) is not dict or type(description.get('construction_id')) is not str:
            raise ValueError('registered descriptor construction identity required')
        expected=_expected_descriptor(construction(description['construction_id']))
        if body!=expected or self._body is not body or self._seal!=seal:
            raise ValueError('descriptor differs from complete registered authority')
        return description

    def evaluate(self,displacement,accepted_rotations,*,cancel_check=None):
        if type(self) is not AffineQ4PhysicalRecovery:
            raise ValueError('exact private facade type required')
        body,seal,lock = self._body,self._seal,self._lock
        expected_prepared,expected_prepared_seal = self._prepared,self._prepared_seal
        if not lock.acquire(blocking=False): raise RuntimeError('reentry/concurrent recovery forbidden')
        def guard():
            if self._body is not body or self._seal != seal or self._lock is not lock or sha256(body).hexdigest()!=seal:
                raise ValueError('definition changed during observation')
            if self._prepared is not expected_prepared or self._prepared_seal != expected_prepared_seal:
                raise ValueError('reference cache identity changed during observation')
        def cancellation():
            guard()
            if cancel_check is not None:
                if not callable(cancel_check): raise ValueError('cancellation callback required')
                stop=cancel_check(); guard()
                if type(stop) is not bool: raise ValueError('exact cancellation bool required')
                if stop: raise AffineRecoveryCancelled('private recovery cancelled without publication')
        try:
            guard()
            if expected_prepared is None:
                if expected_prepared_seal is not None:
                    raise ValueError('unpublished reference cache has a seal')
            elif type(expected_prepared) is not Prepared or _fingerprint(expected_prepared)!=expected_prepared_seal:
                raise ValueError('immutable reference operator cache changed')
            guard(); description=self.descriptor(); guard()
            if canonical(description)!=body: raise ValueError('observed descriptor differs from immutable definition')
            c=construction(description['construction_id']); guard()
            expected_body=_expected_descriptor(c); guard()
            if body!=expected_body:
                raise ValueError('complete descriptor differs from registered authority')
            if expected_prepared is not None and expected_prepared.definition_sha256!=seal:
                raise ValueError('reference cache belongs to a different registered definition')
            u=array(displacement,(24,)); guard(); qa=rotations(accepted_rotations,4); guard(); cancellation()
            kin=deformation(c.coordinates,u,qa); guard()
            prepared=expected_prepared; prepared_seal=expected_prepared_seal
            guard()
            model,element,material,origin=family_objects(description); guard()
            if prepared is None:
                prepared=_prepare_reference(element,model.mesh,material,c.coordinates,definition_sha256=seal); guard()
                if prepared.operator_id!=OPERATOR_ID:
                    raise ValueError('source operator differs from registered authority')
                if prepared.definition_sha256!=seal:
                    raise ValueError('fresh reference cache has incorrect definition association')
                prepared_seal=_fingerprint(prepared); guard()
                # The only permitted transition has no caller observation between
                # validating the old identity and installing the new baseline.
                object.__setattr__(self,'_prepared',prepared); object.__setattr__(self,'_prepared_seal',prepared_seal)
                expected_prepared,expected_prepared_seal=prepared,prepared_seal
                guard()
            before=canonical(origin)
            f,k,candidate=element.compute_nonlinear_response(model.mesh,material,kin.deformation,origin,3,True); guard()
            if not before: raise ValueError('missing virgin origin')
            f=owned(f); k=owned(k); components=element.compute_stiffness_components(model.mesh,material); guard()
            stations,gphysical,Hphysical=_station_fields(prepared,kin)
            D,S,d=kin.differential,kin.second,kin.deformation
            channels=[]; numerical_f=np.zeros(24); numerical_k=np.zeros((24,24))
            for name,key in (('NUMERICAL_PL','pl'),('NUMERICAL_HOURGLASS','hourglass')):
                nk=owned(components[key]); nf=owned(nk@d); ng=owned(D.T@nf)
                nH=owned(D.T@nk@D+np.einsum('i,ijk->jk',nf,S))
                channels.append(NumericalChannel(name,float(.5*d@nf),nf,nk,ng,nH))
                numerical_f+=nf; numerical_k+=nk
            fp=f-numerical_f; kp=k-numerical_k
            sourceg=owned(D.T@fp); sourceH=owned(D.T@kp@D+np.einsum('i,ijk->jk',fp,S))
            g=owned(D.T@f); H=owned(D.T@k@D+np.einsum('i,ijk->jk',f,S))
            physical_energy=float(sum(.5*s.weight*s.strain@s.resultant for s in stations))
            source_energy=float(.5*d@components['physical']@d)
            A=100./(1-.25**2)*.1*np.array(((1,.25,0),(.25,1,0),(0,0,(1-.25)/2)))
            # Keep the unchanged source's actual accumulation order. Physical
            # station pairing above follows the distinct mixed-field order.
            for i in sorted(range(4),key=prepared.source_order.__getitem__):
                bm,gw,w=prepared.membrane_maps[i],prepared.transverse_maps[i],prepared.source_weights[i]
                linear=bm@d; p=gw@d; finite=linear+np.array((.5*p[0]**2,.5*p[1]**2,p[0]*p[1]))
                source_energy+=float(.5*w*(finite@A@finite-linear@A@linear))
            block=np.zeros((64,64)); inverse=np.zeros((64,64)); coupling=np.zeros((64,24)); direct=np.zeros((24,24))
            for i,s in enumerate(stations):
                sl=slice(16*i,16*i+16); block[sl,sl]=s.internal_block; inverse[sl,sl]=s.internal_inverse
                coupling[16*i+8:16*i+16]=s.weight*s.chart_first
                direct+=s.weight*np.einsum('a,aij->ij',s.resultant,s.chart_second)
            schur=owned(direct-coupling.T@inverse@coupling)
            spatial,spatialH=_spatial_connection(u,g,H)
            physical_spatial,physical_spatialH=_spatial_connection(u,gphysical,Hphysical)
            cancellation(); guard()
            if self._prepared is not prepared or self._prepared_seal!=prepared_seal or _fingerprint(prepared)!=prepared_seal:
                raise ValueError('reference operators changed during evaluation')
            result=AffineQ4Trial(kin,stations,prepared.stationary,prepared.coupling,prepared.solution,
                prepared.linear_physical,f,k,physical_energy,gphysical,Hphysical,source_energy,sourceg,sourceH,
                g,H,spatial,spatialH,physical_spatial,physical_spatialH,tuple(channels),owned(block),owned(inverse),
                owned(coupling),owned(direct),schur,owned(np.concatenate([s.internal_residual for s in stations])),
                source_energy+sum(c.energy for c in channels),canonical(candidate),seal,body,prepared.operator_id)
            # Validate every returned scalar/contraction, including energies; this is
            # serialization validation only, never an acceptance predicate.
            _fingerprint(result); guard()
            return result
        finally:
            lock.release()
