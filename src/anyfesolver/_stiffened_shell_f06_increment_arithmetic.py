"""Opt-in binary64 expansion arithmetic for durable native state and work.

Two components retain roundoff information; no decimal/extended-precision
backend, projection, altered mechanical equation or empirical stabilization.
"""
import math
import numpy as np
POLICY='GE_BEAM3_DURABLE_COMPENSATED_NATIVE_STATE_V1_DEV'
MAGIC=b'ANYF06DD1\0'


def two_sum(a,b):
    h=a+b;v=h-a
    return h,(a-(h-v))+(b-v)

def two_product(a,b):
    h=a*b;split=134217729.
    ca=split*a;cb=split*b
    ah=ca-(ca-a);al=a-ah;bh=cb-(cb-b);bl=b-bh
    return h,((ah*bh-h)+ah*bl+al*bh)+al*bl

class Pair:
    __array_priority__=1000
    def __init__(self,high,low=0.):
        self.high,self.low=two_sum(np.asarray(high,dtype=float),np.asarray(low,dtype=float))
        if not np.isfinite(self.high).all() or not np.isfinite(self.low).all():
            raise ValueError('finite representable compensated native values required')
    @staticmethod
    def make(value):return value if isinstance(value,Pair) else Pair(value)
    def __add__(self,other):
        b=self.make(other);h,l=two_sum(self.high,b.high)
        return Pair(h,l+self.low+b.low)
    __radd__=__add__
    def __neg__(self):return Pair(-self.high,-self.low)
    def __sub__(self,other):return self+-self.make(other)
    def __rsub__(self,other):return self.make(other)+-self
    def __mul__(self,other):
        b=self.make(other);h,l=two_product(self.high,b.high)
        return Pair(h,l+self.high*b.low+self.low*b.high+self.low*b.low)
    __rmul__=__mul__
    def reciprocal(self):
        if np.any(self.high==0.):raise ValueError('singular compensated native reciprocal')
        q=Pair(1./self.high)
        return q+(1.-self*q)*(1./self.high)
    def __truediv__(self,other):return self*self.make(other).reciprocal()
    def __rtruediv__(self,other):return self.make(other)*self.reciprocal()
    def __getitem__(self,index):return Pair(self.high[index],self.low[index])
    @property
    def T(self):return Pair(self.high.T,self.low.T)
    def rounded(self):return self.high+self.low
    def scalar(self):return float(self.high)+float(self.low)

def outer(a,b):return Pair(a.high[:,None],a.low[:,None])*Pair(b.high[None,:],b.low[None,:])

class Jet:
    def __init__(self,value,gradient,hessian):
        self.value=Pair.make(value);self.gradient=Pair.make(gradient);self.hessian=Pair.make(hessian)
    @classmethod
    def constant(cls,value,size):return cls(value,Pair(np.zeros(size)),Pair(np.zeros((size,size))))
    @classmethod
    def variable(cls,value,index,size):
        g=np.zeros(size);g[index]=1.
        return cls(value,Pair(g),Pair(np.zeros((size,size))))
    def make(self,value):return value if isinstance(value,Jet) else Jet.constant(value,self.gradient.high.size)
    def __add__(self,other):
        b=self.make(other);return Jet(self.value+b.value,self.gradient+b.gradient,self.hessian+b.hessian)
    __radd__=__add__
    def __neg__(self):return Jet(-self.value,-self.gradient,-self.hessian)
    def __sub__(self,other):return self+-self.make(other)
    def __rsub__(self,other):return self.make(other)+-self
    def __mul__(self,other):
        b=self.make(other)
        return Jet(self.value*b.value,self.gradient*b.value+self.value*b.gradient,
            self.hessian*b.value+self.value*b.hessian+outer(self.gradient,b.gradient)+outer(b.gradient,self.gradient))
    __rmul__=__mul__
    def __truediv__(self,other):
        if isinstance(other,Jet):
            r=other.value.reciprocal();return self*unary(other,r,-r*r,2*r*r*r)
        return self*Pair.make(other).reciprocal()

def unary(j,v,d,dd):return Jet(v,j.gradient*d,j.hessian*d+outer(j.gradient,j.gradient)*dd)
def transpose(a):return [list(row) for row in zip(*a)]
def mm(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def mv(a,b):return [sum(x*y for x,y in zip(row,b)) for row in a]
def const(a,low,size):
    return [[Jet.constant(Pair(a[i,j],low[i,j]),size) for j in range(3)] for i in range(3)]
def exp(w):
    n=w[0].gradient.high.size;z=Jet.constant(0.,n);t=sum((v*v for v in w),z)
    if math.sqrt(max(0.,t.value.scalar()))>=.9*math.pi:raise ValueError('native increment requires cutback')
    # Entire-function Rodrigues coefficients, evaluated from their defining
    # series. 32 terms cover the unchanged native chart radius.
    a=Jet.constant(1.,n);b=Jet.constant(.5,n);ta=a;tb=b
    stop=2 if t.value.scalar()==0. and not np.any(t.gradient.high) and not np.any(t.gradient.low) else 33
    for k in range(1,stop):
        ta=-ta*t/((2*k)*(2*k+1));tb=-tb*t/((2*k+1)*(2*k+2));a=a+ta;b=b+tb
    S=[[z,-w[2],w[1]],[w[2],z,-w[0]],[-w[1],w[0],z]];SS=mm(S,S)
    return [[Jet.constant(float(i==j),n)+a*S[i][j]+b*SS[i][j] for j in range(3)] for i in range(3)]
def greater_scalar(value,bound):
    """Compare authoritative normalized scalar parts without rounding the low."""
    return bool(value.high>bound or (value.high==bound and value.low>0.))

def pair_sqrt(value):
    value=Pair.make(value)
    if not greater_scalar(value,0.):raise ValueError('positive compensated square root required')
    root=Pair(float(np.sqrt(value.high)))
    for _ in range(2):root=root+(value-root*root)/(2.*root)
    return root

def pair_sin_cos(value):
    """Bounded principal-angle entire series in binary64 expansions."""
    value=Pair.make(value);t=value*value
    sine=value;cosine=Pair(1.);a=value;b=Pair(1.)
    for k in range(1,33):
        a=-a*t/((2*k)*(2*k+1));b=-b*t/((2*k-1)*(2*k))
        sine=sine+a;cosine=cosine+b
    return sine,cosine

def full_log_coefficients(cosine):
    """Same principal acos(c)/sqrt(1-c*c) and its first two derivatives.

    Acos is seeded at the high component, then corrected using the defining
    cosine equation in the same binary64 expansion arithmetic. The recurrences
    (1-c*c) f' = c*f-1 and (1-c*c) f'' = f+3*c*f' avoid introducing a new law.
    """
    from ._ge_beam3_mixed_ad import RotationDomainError,math_cos_limit
    c=Pair.make(cosine)
    if not greater_scalar(c,math_cos_limit()) or greater_scalar(c,1.) or (c.high==1. and c.low==0.):
        raise RotationDomainError('relative rotation must remain below 0.9*pi; refine or cut back')
    t=1.-c*c;sine=pair_sqrt(t);angle=Pair(float(np.arccos(c.high)))
    for _ in range(2):
        sn,cs=pair_sin_cos(angle);angle=angle+(cs-c)/sn
    value=angle/sine;first=(c*value-1.)/t;second=(value+3.*c*first)/t
    return value,first,second

def log(a):
    n=a[0][0].gradient.high.size;c=(a[0][0]+a[1][1]+a[2][2]-1.)/2.
    if greater_scalar(c.value,1.-1e-7):
        d=1.-c;f=1.+d/3.+2.*d*d/15.+2.*d*d*d/35.
    else:
        f=unary(c,*full_log_coefficients(c.value))
    return [f*(a[2][1]-a[1][2])/2.,f*(a[0][2]-a[2][0])/2.,f*(a[1][0]-a[0][1])/2.]

def updated_rotation(high,low,angle):
    w=[Jet.constant(float(v),1) for v in angle];e=exp(w);made=mm(e,const(high,low,1))
    return (np.array([[v.value.high for v in row] for row in made]),
            np.array([[v.value.low for v in row] for row in made]))

def add_state(high,low,delta):
    p=Pair(high,low)+Pair(delta)
    return p.high.copy(),p.low.copy()


def retained_evaluation(reference,x,low,q,q_low,u,u_low,p,p_low,material):
    size=24;variables=[Jet.variable(0.,i,size) for i in range(size)]
    made_q=[mm(exp(variables[6*n+3:6*n+6]),const(q[n],q_low[n],size)) for n in range(3)]
    z=[];ell=[]
    for cell,(left,right) in enumerate(((0,1),(1,2))):
        made_u=mm(exp(variables[18+3*cell:21+3*cell]),const(u[cell],u_low[cell],size))
        chord=[Jet.constant(Pair(x[right,i],low[right,i])-Pair(x[left,i],low[left,i]),size)
            +variables[6*right+i]-variables[6*left+i] for i in range(3)]
        local=mv(transpose(made_u),chord)
        z.extend(v-Pair(reference.coordinates[right,i])+Pair(reference.coordinates[left,i]) for i,v in enumerate(local))
        for node,sign in ((left,-1),(right,1)):
            frame=mm(made_u,const(reference.nodal_triads[node],np.zeros((3,3)),size))
            ell.extend(sign*v for v in log(mm(transpose(frame),made_q[node])))
    values=z+ell;forces=Pair(p,p_low);gradient=Pair(np.zeros(24));geometric=Pair(np.zeros((24,24)))
    potential=Pair(0.)
    for v,ph,pl in zip(values,forces.high,forces.low):
        f=Pair(ph,pl);gradient=gradient+v.gradient*f;geometric=geometric+v.hessian*f;potential=potential+v.value*f
    gh,gl=material['gradient'];compatibility=[v.value-Pair(h,l) for v,h,l in zip(values,gh,gl)]
    jh=np.array([v.gradient.high for v in values]);jl=np.array([v.gradient.low for v in values])
    ch=np.array([r[0] for r in material['hessian']]);cl=np.array([r[1] for r in material['hessian']])
    high=np.block([[geometric.high,jh.T],[jh,-ch]])
    lowh=np.block([[geometric.low,jl.T],[jl,-cl]])
    residual=np.r_[gradient.rounded(),[v.scalar() for v in compatibility]]
    potential=potential-Pair(material['potential'][0][0],material['potential'][1][0])
    return potential.scalar(),residual,high,lowh,np.array([v.value.scalar() for v in values])


def validate_pairs(high,low,label):
    p=Pair(high,low)
    if not np.array_equal(p.high,high) or not np.array_equal(p.low,low):
        raise ValueError('normalized compensated '+label+' required')
    return p

def validate_frames(high,low,label):
    validate_pairs(high,low,label)
    for h,l in zip(np.asarray(high).reshape(-1,3,3),np.asarray(low).reshape(-1,3,3)):
        m=[[Pair(h[i,j],l[i,j]) for j in range(3)] for i in range(3)]
        product=mm(transpose(m),m)
        error=np.array([[v.scalar()-float(i==j) for j,v in enumerate(row)] for i,row in enumerate(product)])
        physical=h+l
        if np.linalg.norm(error)>1e-11 or abs(np.linalg.det(physical)-1.)>1e-11:
            raise ValueError('proper physical compensated '+label+' required')

def pose_evaluation(x,low,frames,frames_low,multipliers,offset,relative):
    size=18;variables=[Jet.variable(0.,i,size) for i in range(12)]
    variables+=[Jet.variable(float(v),12+i,size) for i,v in enumerate(multipliers)]
    made=[mm(exp(variables[6*n+3:6*n+6]),const(frames[n],frames_low[n],size)) for n in range(2)]
    distance=[Jet.constant(Pair(x[1,i],low[1,i])-Pair(x[0,i],low[0,i]),size)+variables[6+i]-variables[i] for i in range(3)]
    ct=[v-float(e) for v,e in zip(mv(transpose(made[0]),distance),offset)]
    rel=mm(const(relative.T,np.zeros((3,3)),size),mm(transpose(made[0]),made[1]))
    constraints=ct+log(rel);potential=sum((v*variables[12+i] for i,v in enumerate(constraints)),Jet.constant(0.,size))
    return (potential.value.scalar(),np.array([v.value.scalar() for v in constraints]),
        np.array([v.gradient.rounded()[:12] for v in constraints]),potential.gradient.rounded(),potential.hessian.rounded())
