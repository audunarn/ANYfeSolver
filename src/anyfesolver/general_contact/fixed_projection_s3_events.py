"""Affine pressure events for one explicitly constrained, deforming S3 target.

Uniform source translation along its reference normal fixes projection and source
measure. Target nodal normal motion may be nonuniform: its projected interpolation
still stays fixed. On this represented subspace, shared P1 gaps are affine even
though the full general contact work has moving-normal/projection derivatives.
This sidecar never replaces that work or supplies general moving-overlap events.
"""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from scipy import sparse

from .s3_overlap_derivatives import s3_overlap_derivatives
from .s3_patch_static import GeneralS3PatchManager

_EPS = 256*np.finfo(float).eps


def _calculated(operation):
    """Finite operands alone do not establish finite matrix-derived quantities."""
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            result=np.asarray(operation(),dtype=float)
        if not np.all(np.isfinite(result)):
            raise ValueError
    except (FloatingPointError,OverflowError,ValueError) as exc:
        raise ValueError('fixed-projection S3 derived geometry is not finite') from exc
    return result


def _owned(value):
    result=_calculated(lambda:value)
    return np.frombuffer(result.tobytes(),dtype=result.dtype).reshape(result.shape)


def _parallel(vector,normal):
    return all(vector[i]*normal[j]==vector[j]*normal[i]
               for i,j in ((0,1),(0,2),(1,2)))


@dataclass(frozen=True)
class FixedProjectionS3PressureEvents:
    support: np.ndarray
    reference_gap: np.ndarray
    reduced_gradient: np.ndarray
    features: tuple[tuple[str,int],...]
    normal: np.ndarray
    normal_exact: tuple[Fraction,Fraction,Fraction]
    source_dofs: np.ndarray
    target_dofs: np.ndarray
    full_size: int

    def require_motion(self,value):
        raw=np.asarray(value)
        if (raw.dtype.kind not in 'fiu' or raw.shape!=(self.full_size,)
                or not np.all(np.isfinite(raw))):
            raise ValueError('fixed-projection S3 motion requires a finite full real vector')
        raw=_calculated(lambda:raw)
        source=raw[self.source_dofs];target=raw[self.target_dofs]
        # Epsilon membership is insufficient: tiny tangential motion combined
        # with large normal deformation can produce a quadratic pressure gap.
        # Check exact represented floats against the original rational normal.
        for local in (source,target):
            for vector in local[:,:3]:
                rational=tuple(Fraction(float(value)) for value in vector)
                if not _parallel(rational,self.normal_exact):
                    raise ValueError('fixed-projection S3 requires exact normal nodal translations')
            if np.any(local[:,3:]!=0.):
                raise ValueError('fixed-projection S3 requires zero shell rotations')
        if not np.all(source[:,:3]==source[0,:3]):
            raise ValueError('fixed-projection S3 requires uniform source normal translation')

    def gaps(self,reduced_position):
        position=np.asarray(reduced_position)
        if (position.dtype.kind not in 'fiu' or position.shape!=(self.reduced_gradient.shape[1],)
                or not np.all(np.isfinite(position))):
            raise ValueError('fixed-projection S3 pressure position is invalid')
        position=_calculated(lambda:position)
        return _calculated(lambda:self.reference_gap+self.reduced_gradient@position)

    def require_candidate(self,full_position,reduced_position,result):
        """Check actual motion and independent same-pose geometry agreement."""
        self.require_motion(full_position)
        expected=self.gaps(reduced_position)
        actual=np.asarray(result.normalized_weighted_gap_m)
        support=np.asarray(result.reference_support_m2)
        position=_calculated(lambda:reduced_position)
        scales=_calculated(lambda:np.abs(self.reference_gap)+np.abs(self.reduced_gradient)@np.abs(position))
        if (result.source_pressure_node_ids!=tuple(node for _,node in self.features)
                or actual.shape!=expected.shape or support.shape!=self.support.shape
                or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(support))
                or not np.all(np.isfinite(self.support)) or np.any(self.support<=0.)
                or np.any(np.abs(support-self.support)>16*_EPS*self.support)
                or np.any(np.abs(actual-expected)>16*_EPS*np.maximum(scales,np.finfo(float).tiny))):
            raise ValueError('fixed-projection S3 pressure geometry agreement is unresolved')

    def skin_path(self,manager,start,end,v0,v1):
        """Return represented physical Hermite data only after exact path admission.

        DOF membership alone is insufficient: adding offsets/reference positions
        or pulling velocities through a float work map can change the represented
        physical coefficients. Check these actual endpoints/rates before using
        affine pressure-event assumptions. Unsupported rounding fails explicitly.
        """
        for value in (start,end,v0,v1):self.require_motion(value)
        first=tuple(_calculated(lambda:value) for value in manager._positions(start))
        last=tuple(_calculated(lambda:value) for value in manager._positions(end))
        count=len(first[0])
        rates=[]
        for value in (v0,v1):
            made=_calculated(lambda:manager.map@value).reshape(-1,3)
            rates.append((made[:count],made[count:]))
        # All source physical points must lie on one plane with the original
        # normal; full source motion must be one common normal translation.
        rational_first=[tuple(Fraction(float(x)) for x in row) for row in first[0]]
        origin=rational_first[0]
        for point in rational_first:
            if sum((x-y)*n for x,y,n in zip(point,origin,self.normal_exact))!=0:
                raise ValueError('fixed-projection S3 represented source skin is not the original normal plane')
        for before,after in zip(first,last):
            shifts=[tuple(Fraction(float(y))-Fraction(float(x)) for x,y in zip(a,b))
                    for a,b in zip(before,after)]
            if not all(_parallel(shift,self.normal_exact) for shift in shifts):
                raise ValueError('fixed-projection S3 represented skin path loses exact normal motion')
            if before is first[0] and any(shift!=shifts[0] for shift in shifts):
                raise ValueError('fixed-projection S3 represented source skin loses common translation')
        for source_rate,target_rate in rates:
            for vectors in (source_rate,target_rate):
                if not all(_parallel(tuple(Fraction(float(x)) for x in vector),self.normal_exact)
                           for vector in vectors):
                    raise ValueError('fixed-projection S3 represented skin rate loses exact normal motion')
            if not np.all(source_rate==source_rate[0]):
                raise ValueError('fixed-projection S3 represented source skin rates are nonuniform')
        return first[0],last[0],first[1],last[1],rates[0][0],rates[1][0],rates[0][1],rates[1][1]


def prepare_fixed_projection_s3_events(model,manager,T,*,prescribed=None):
    """Prepare fixed supports and exact Jet2 pullback on the admitted generators.

    Each generator must belong to the exact represented fixed-projection subspace;
    approximate direction equality does not prove affine gaps. Actual path vectors
    are checked again. No positive support is removed or production work replaced.
"""
    if (type(manager) is not GeneralS3PatchManager or not manager.projected_work
            or manager.augmented or manager.law.geometry.differentiation!='jet2_s3'
            or manager.law.pressure_basis is not None
            or len(manager.law.target_face_indices)!=1
            or any(len(face)!=3 for face in (*manager.law.source_face_indices,*manager.law.target_face_indices))):
        raise ValueError('fixed-projection S3 events require Jet2 projected penalty and one S3 target')
    if len(manager.law.source_node_ids)>manager.settings.maximum_patch_normal_dofs:
        raise MemoryError('fixed-projection S3 pressure cap exceeded; no positive support discarded')
    matrix=T.tocsc(copy=True) if sparse.issparse(T) else np.asarray(T)
    if sparse.issparse(matrix):
        matrix.sum_duplicates();matrix.eliminate_zeros()
        values=matrix.data
    else:values=matrix
    if (values.dtype.kind not in 'fiu' or matrix.ndim!=2 or matrix.shape[0]!=manager.n_dof
            or not matrix.shape[1] or not np.all(np.isfinite(values)) or not np.any(values!=0)):
        raise ValueError('fixed-projection S3 requires a finite nonempty real constraint subspace')
    source,target=manager.reference
    source_faces=manager.law.source_face_indices
    target_face=manager.law.target_face_indices[0]
    first=source[source_faces[0]]
    rational=tuple(tuple(Fraction(float(x)) for x in row) for row in first)
    ea=tuple(x-y for x,y in zip(rational[1],rational[0]))
    eb=tuple(x-y for x,y in zip(rational[2],rational[0]))
    normal_exact=tuple(ea[i]*eb[j]-ea[j]*eb[i] for i,j in ((1,2),(2,0),(0,1)))
    for point in source:
        relative=tuple(Fraction(float(x))-y for x,y in zip(point,rational[0]))
        if sum(x*y for x,y in zip(relative,normal_exact))!=0:
            raise ValueError('fixed-projection S3 requires an exact common source plane')
    for ids in source_faces:
        face=tuple(tuple(Fraction(float(x)) for x in row) for row in source[ids])
        fa=tuple(x-y for x,y in zip(face[1],face[0]));fb=tuple(x-y for x,y in zip(face[2],face[0]))
        face_normal=tuple(fa[i]*fb[j]-fa[j]*fb[i] for i,j in ((1,2),(2,0),(0,1)))
        if sum(x*y for x,y in zip(face_normal,normal_exact))<=0:
            raise ValueError('fixed-projection S3 source face polarity is invalid')
    normal=_calculated(lambda:np.cross(first[1]-first[0],first[2]-first[0]))
    length=float(_calculated(lambda:np.linalg.norm(normal)))
    if length<=0:
        raise ValueError('fixed-projection S3 source normal is invalid')
    normal/=length
    def dofs(nodes):
        result=np.asarray([model.mesh.dof_manager.get_node_dofs(node)[:6] for node in nodes],dtype=np.intp)
        return np.frombuffer(result.tobytes(),dtype=result.dtype).reshape(result.shape)
    source_dofs,target_dofs=dofs(manager.law.source_node_ids),dofs(manager.law.target_node_ids)
    count=len(source);support=np.zeros(count);weighted=np.zeros(count)
    integral_gradient=np.zeros((count,manager.law.n_coordinates))
    def mapped():
        product=manager.map@matrix
        return product.toarray() if sparse.issparse(product) else product
    physical_modes=_calculated(mapped)
    # Bounds and force mapping follow the actual original physical skins.
    geometry_scale=max(1.,float(np.max(np.abs(source))),float(np.max(np.abs(target))))
    if np.ptp(source@normal)>_EPS*geometry_scale or np.ptp(target@normal)>_EPS*geometry_scale:
        raise ValueError('fixed-projection S3 reference skins must be parallel planes')
    target_positions=target[target_face]
    target_normal=np.cross(target_positions[1]-target_positions[0],target_positions[2]-target_positions[0])
    if target_normal@normal<=0:
        raise ValueError('fixed-projection S3 reference target polarity is invalid')
    edge=np.column_stack((target_positions[1]-target_positions[0],target_positions[2]-target_positions[0]))
    edge_inverse=np.linalg.pinv(edge)
    for ids in source_faces:
        face=source[ids]
        for point in face:
            bary=edge_inverse@(point-target_positions[0])
            if np.min(np.r_[1-bary.sum(),bary])<=_EPS:
                raise ValueError('fixed-projection S3 source requires strict projected target coverage')
        derivative=s3_overlap_derivatives(face,face,target_positions)
        support[ids]+=derivative.reference_support
        weighted[ids]+=derivative.reference_weighted_gap
        coordinates=np.r_[np.concatenate([3*int(i)+np.arange(3) for i in ids]),
            np.concatenate([3*(len(source)+int(i))+np.arange(3) for i in target_face])]
        integral_gradient[np.ix_(ids,coordinates)]+=derivative.weighted_gap_gradient
        local_modes=physical_modes[coordinates]
        variation=_calculated(lambda:derivative.support_gradient@local_modes)
        budget=_calculated(lambda:16*_EPS*derivative.reference_support[:,None]*np.maximum(1.,np.max(np.abs(local_modes),axis=0)))
        if np.any(np.abs(variation)>budget):
            raise ValueError('fixed-projection S3 support variation is unresolved')
    if not np.all(np.isfinite(support)) or np.any(support<=0):
        raise ValueError('fixed-projection S3 requires positive supported pressure modes')
    base=np.zeros(manager.n_dof) if prescribed is None else np.asarray(prescribed)
    contract=FixedProjectionS3PressureEvents(_owned(support),_owned(weighted/support),
        _owned(_calculated(lambda:(integral_gradient@physical_modes)/support[:,None])),
        tuple((manager.definition.interactions[0].name,int(node)) for node in manager.law.source_node_ids),
        _owned(normal),normal_exact,source_dofs,target_dofs,manager.n_dof)
    for j in range(matrix.shape[1]):
        contract.require_motion(matrix[:,j].toarray().reshape(-1) if sparse.issparse(matrix) else matrix[:,j])
    contract.require_motion(base)
    shift=_calculated(lambda:manager.map@base).reshape(-1)
    contract=FixedProjectionS3PressureEvents(contract.support,
        _owned(_calculated(lambda:contract.reference_gap+integral_gradient@shift/support)),contract.reduced_gradient,
        contract.features,contract.normal,normal_exact,source_dofs,target_dofs,manager.n_dof)
    return contract
