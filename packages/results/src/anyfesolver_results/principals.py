"""Principal display axes from the canonical symmetric stress tensor kernel."""
from dataclasses import dataclass
import math
import numpy as np
from .formulas import principal_stresses


@dataclass(frozen=True)
class PrincipalAxes:
    """Descending values and rank-major unoriented axes; ambiguous axes are NaN.

    ``unique`` reports numerical distinguishability, not physical certainty.
    Degeneracy uses 64 float64 eps times the largest eigenvalue magnitude.
    """
    values: np.ndarray
    axes: np.ndarray
    unique: np.ndarray


def principal_axes(tensors, *, basis=None, cancel_event=None, working_bytes=67108864):
    """Derive axes without changing canonical principal-value admission/results.

    Basis columns, when supplied, are the local axes in global coordinates.
    Axis signs are deterministic for display; principal axes are bidirectional.
    Cancellation raises InterruptedError and returns no partial publication.
    """
    def cancelled():
        if cancel_event is not None:
            if not callable(getattr(cancel_event,'is_set',None)):raise ValueError('cancel_event must provide is_set()')
            if cancel_event.is_set():raise InterruptedError('principal axes cancelled')
    cancelled()
    if type(working_bytes) is not int or working_bytes<4096:raise ValueError('working_bytes must be an integer of at least 4096')
    def shape(value):
        if isinstance(value,np.ndarray):
            dimensions=np.ndarray.shape.__get__(value)
            if type(dimensions) is not tuple or any(type(dimension) is not int or dimension<0 for dimension in dimensions):
                raise ValueError('array shape dimensions must be plain nonnegative integers')
            return dimensions
        if isinstance(value,(list,tuple)):
            if not value:return (0,)
            first=shape(value[0])
            if any(shape(item)!=first for item in value[1:]):raise ValueError('principal tensors must form a rectangular array')
            return (len(value),)+first
        if np.isscalar(value):return ()
        raise ValueError('principal tensors require an ndarray or nested sequences')
    dimensions=shape(tensors)
    if len(dimensions)<2 or dimensions[-2:]!=(3,3):raise ValueError('principal axes require 3 by 3 tensors')
    count=math.prod(dimensions[:-2])
    if count*512+4096>working_bytes:raise MemoryError('principal axes exceed working byte cap')
    if basis is not None:
        basis_dimensions=shape(basis)
        if (len(basis_dimensions)>len(dimensions)
                or any(source!=1 and source!=target
                       for source,target in zip(reversed(basis_dimensions),reversed(dimensions)))):
            raise ValueError('basis must broadcast to the tensor shape')
    tensor=np.asarray(tensors,dtype=np.float64)
    values=np.empty(tensor.shape[:-1],dtype=np.float64)
    rotation=None
    if basis is not None:
        rotation=np.asarray(basis,dtype=np.float64)
        try:rotation=np.broadcast_to(rotation,tensor.shape)
        except ValueError as error:raise ValueError('basis must broadcast to the tensor shape') from error
        if (not np.isfinite(rotation).all()
                or not np.allclose(np.swapaxes(rotation,-1,-2)@rotation,np.eye(3),rtol=1e-12,atol=1e-12)
                or not np.allclose(np.linalg.det(rotation),1.,rtol=1e-12,atol=1e-12)):
            raise ValueError('basis must be a finite proper orthonormal rotation')
    axes=np.empty(tensor.shape,dtype=np.float64)
    flat=tensor.reshape((-1,3,3));output=axes.reshape((-1,3,3))
    for first in range(0,count,4096):
        cancelled();block=flat[first:first+4096]
        # Canonical value admission/semantics per independent tensor; chunking
        # bounds cancellation latency without changing the existing kernel.
        values.reshape((-1,3))[first:first+len(block)]=principal_stresses(block)
        symmetric=.5*(block+np.swapaxes(block,-1,-2))
        try:_,vectors=np.linalg.eigh(symmetric)
        except np.linalg.LinAlgError as error:raise ValueError('principal axes eigensystem failed') from error
        output[first:first+len(block)]=np.swapaxes(vectors[...,::-1],-1,-2)
    if rotation is not None:axes=axes@np.swapaxes(rotation,-1,-2)
    if not np.isfinite(axes).all():raise ValueError('principal axes must be finite')
    indices=np.argmax(np.abs(axes),axis=-1)
    signs=np.take_along_axis(axes,indices[...,None],axis=-1)[...,0]
    axes*=np.where(signs<0,-1.,1.)[...,None]
    scale=np.max(np.abs(values),axis=-1,keepdims=True)
    tolerance=64*np.finfo(np.float64).eps*scale
    gaps=np.abs(values[...,None,:]-values[...,:,None])
    unique=np.all((gaps>tolerance[...,None])|np.eye(3,dtype=bool),axis=-1)
    axes=np.where(unique[...,None],axes,np.nan)
    cancelled()
    for array in (values,axes,unique):array.setflags(write=False)
    return PrincipalAxes(values,axes,unique)
