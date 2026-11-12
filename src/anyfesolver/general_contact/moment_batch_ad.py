"""Point-lane second derivatives for the dedicated nonlinear moment path.

The leading axis contains INDEPENDENT quadrature points. All37 original
geometry derivatives remain dense; there are no cross-point derivatives.
No protected Jet2 implementation or legacy dispatch is modified.
"""
from __future__ import annotations
import numpy as np
from anyfesolver._ge_beam3_mixed_ad import Jet2

class PointJet:
    __slots__=('value','gradient','hessian')
    __array_priority__=1000
    def __init__(self,value,gradient,hessian):
        self.value=np.asarray(value,dtype=float)
        self.gradient=np.asarray(gradient,dtype=float)
        self.hessian=np.asarray(hessian,dtype=float)

    @classmethod
    def constant(cls,value,count,size):
        value=np.broadcast_to(np.asarray(value,dtype=float),(count,))
        return cls(value,np.broadcast_to(np.zeros(size),(count,size)),
                   np.broadcast_to(np.zeros((size,size)),(count,size,size)))

    @classmethod
    def broadcast(cls,jet,count):
        if not isinstance(jet,Jet2):raise TypeError('original Jet2 required')
        return cls(np.broadcast_to(jet.value,(count,)),
                   np.broadcast_to(jet.gradient,(count,jet.gradient.size)),
                   np.broadcast_to(jet.hessian,(count,*jet.hessian.shape)))

    @classmethod
    def stack(cls,jets):
        return cls([x.value for x in jets],np.stack([x.gradient for x in jets]),
                   np.stack([x.hessian for x in jets]))

    def coerce(self,other):
        if isinstance(other,PointJet):return other
        if isinstance(other,Jet2):return PointJet.broadcast(other,self.value.size)
        return PointJet.constant(other,self.value.size,self.gradient.shape[1])

    def __add__(self,other):
        y=self.coerce(other)
        return PointJet(self.value+y.value,self.gradient+y.gradient,self.hessian+y.hessian)
    __radd__=__add__
    def __neg__(self):return PointJet(-self.value,-self.gradient,-self.hessian)
    def __sub__(self,other):return self+(-self.coerce(other))
    def __rsub__(self,other):return self.coerce(other)-self
    def __mul__(self,other):
        # Keep the ORIGINAL coercion (and its observable conversion/errors)
        # exactly once. Only its known constant derivative work is contracted.
        y=self.coerce(other)
        if (not isinstance(other,(PointJet,Jet2)) and _numeric_layout(self)
                and np.all(np.isfinite(self.value))):
            return _numeric_parts_product(self,y.value,y.gradient[:,0],y.hessian[:,0,0])
        return PointJet(self.value*y.value,
            self.gradient*y.value[:,None]+y.gradient*self.value[:,None],
            self.hessian*y.value[:,None,None]+y.hessian*self.value[:,None,None]
            +self.gradient[:,:,None]*y.gradient[:,None,:]
            +y.gradient[:,:,None]*self.gradient[:,None,:])
    __rmul__=__mul__
    def reciprocal(self):
        if np.any(self.value==0.):raise ZeroDivisionError('jet reciprocal is singular')
        inverse=1./self.value
        return unary(self,inverse,-inverse**2,2.*inverse**3)
    def __truediv__(self,other):
        y=self.coerce(other)
        if (not isinstance(other,(PointJet,Jet2)) and _numeric_layout(self)
                and np.all(np.isfinite(self.value))):
            if np.any(y.value==0.):raise ZeroDivisionError('jet reciprocal is singular')
            inverse=1./y.value
            # Identical coefficient operations/order to reciprocal(), once.
            first,second=-inverse**2,2.*inverse**3
            if np.all(np.isfinite(first)) and np.all(np.isfinite(second)):
                gradient_zero=first*y.gradient[:,0]
                hessian_zero=first*y.hessian[:,0,0]+second*(y.gradient[:,0]*y.gradient[:,0])
                return _numeric_parts_product(self,inverse,gradient_zero,hessian_zero)
            # Preserve original nonfinite coefficient/warning behavior without
            # recomputing reciprocal powers or repeating caller coercion.
            return self*unary(y,inverse,first,second)
        return self*y.reciprocal()
    def __rtruediv__(self,other):return self.coerce(other)*self.reciprocal()

    def sum_to_jet(self):
        # Original rule starts at a zeroJet and left-folds the25points.
        value=0.;gradient=np.zeros(self.gradient.shape[1]);hessian=np.zeros(self.hessian.shape[1:])
        for lane in range(self.value.size):
            value=float(value+float(self.value[lane]))
            gradient=gradient+self.gradient[lane]
            hessian=hessian+self.hessian[lane]
        return Jet2(value,gradient,hessian)

def _numeric_layout(jet):
    """Only ordinary dense binary64 point jets enter the contracted path."""
    return (type(jet)is PointJet and type(jet.value)is np.ndarray
        and type(jet.gradient)is np.ndarray and type(jet.hessian)is np.ndarray
        and jet.value.dtype==np.float64 and jet.gradient.dtype==np.float64
        and jet.hessian.dtype==np.float64 and jet.value.ndim==1
        and jet.gradient.ndim==2 and jet.gradient.shape[0]==jet.value.size
        and jet.gradient.shape[1]>0
        and jet.hessian.shape==(jet.value.size,jet.gradient.shape[1],jet.gradient.shape[1]))

def _numeric_parts_product(jet,value,gradient_zero,hessian_zero):
    """Same four product terms/order, with coordinate-constant zeros broadcast.

    NEVER drop zero products: Inf/NaN derivatives and signed zero still
    propagate. Avoid only point-lane dense constant-H/outer temporaries.
    All returned derivative buffers remain independently owned.
    """
    result=jet.value*value
    gradient=jet.gradient*value[:,None]+gradient_zero[:,None]*jet.value[:,None]
    hessian=jet.hessian*value[:,None,None]+(hessian_zero*jet.value)[:,None,None]
    hessian=hessian+(jet.gradient*gradient_zero[:,None])[:,:,None]
    hessian=hessian+(gradient_zero[:,None]*jet.gradient)[:,None,:]
    return PointJet(result,gradient,hessian)

def unary(x,value,first,second):
    return PointJet(value,first[:,None]*x.gradient,
        first[:,None,None]*x.hessian+second[:,None,None]*(x.gradient[:,:,None]*x.gradient[:,None,:]))

def sqrt(x):
    if np.any(x.value<=0.):raise ValueError('jet square root requires a positive value')
    root=np.sqrt(x.value)
    return unary(x,root,.5/root,-.25/root**3)

def choose(mask,yes,no):
    return PointJet(np.where(mask,yes.value,no.value),
        np.where(mask[:,None],yes.gradient,no.gradient),
        np.where(mask[:,None,None],yes.hessian,no.hessian))
