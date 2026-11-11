"""Private six-resultant associated ellipsoid section, not a fibre/J2 adapter."""
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from hashlib import sha256
from time import monotonic
import numpy as np
from ._ge_beam3_fibre_section import _owned,_number,canonical
from ._ge_beam3_station_resultant_cell import dec,pair,mv,dot,solve,ident,cholesky
from ._ge_beam3_effective_deadline import effective_deadline_seconds

POLICY='GE_BEAM3_SIX_RESULTANT_ASSOCIATED_ELLIPSOID_HARDENING_V1'
MEASURE='NATIVE_GENERALIZED_STRAIN_AND_REFERENCE_SECTION_RESULTANT_WORK'
RECOVERY='RESULTANT_LEVEL_ONLY_NO_FIBRE_STRESSES_SUPPLIED'


def _pair(x):
    high,low=pair([x]);return high[0],low[0]


def _parts(values,shape):
    a,b=pair(values)
    return _owned(np.array(a).reshape(shape),shape),_owned(np.array(b).reshape(shape),shape)


def _unpair(value):
    if type(value) is not tuple or len(value)!=2:raise ValueError('exact paired section scalar')
    x=dec(value[0])+dec(value[1])
    if _pair(x)!=value:raise ValueError('normalized paired section scalar')
    return x


@dataclass(frozen=True)
class GeneralizedHistory:
    section_identity: str
    plastic: tuple
    accumulated: tuple


@dataclass(frozen=True)
class GeneralizedResponse:
    origin: GeneralizedHistory
    history: GeneralizedHistory
    strain: np.ndarray
    strain_low: np.ndarray
    resultants: np.ndarray
    resultants_low: np.ndarray
    tangent: np.ndarray
    tangent_low: np.ndarray
    compliance: np.ndarray
    compliance_low: np.ndarray
    dual_potential: tuple
    incremental_potential: tuple
    stored_energy: tuple
    dissipation: tuple
    plastic_increment: tuple
    branch: str
    derivative_kind: str
    section_identity: str
    recovery_status: str=RECOVERY
    production_qualified: bool=False


class EllipsoidalGeneralizedSection:
    """C and M are explicit SPD work-coordinate tensors; H and Y are positive.

    q(s)=sqrt(s.T M s). Flow is dz=dlambda*M*s/q and dp=dlambda.
    The complete complementary incremental potential is
    .5*s.T*C^-1*s + z0.s + max(q-Y-H*p0,0)^2/(2H).
    This is a declared resultant law, never an inferred physical fibre model.
    """
    __slots__=('elastic','metric','yield_force','hardening','identity','_c','_m','_s','_compiled','_sealed')

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):raise AttributeError('generalized section is immutable')
        object.__setattr__(self,name,value)

    def __init__(self,elastic,metric,yield_force,hardening):
        self.elastic=_owned(elastic,(6,6));self.metric=_owned(metric,(6,6))
        self.yield_force=_number(yield_force);self.hardening=_number(hardening)
        if self.yield_force<=0 or self.hardening<=0:raise ValueError('positive yield and hardening required')
        if not np.array_equal(self.elastic,self.elastic.T) or not np.array_equal(self.metric,self.metric.T):
            raise ValueError('exact symmetric generalized tensors required')
        with localcontext() as ctx:
            ctx.prec=80
            self._c=tuple(tuple(D.from_float(float(v)) for v in row) for row in self.elastic)
            self._m=tuple(tuple(D.from_float(float(v)) for v in row) for row in self.metric)
            cholesky(self._c);cholesky(self._m)
            self._s=tuple(map(tuple,solve([list(row) for row in self._c],ident(6),lambda:None)))
        self._compiled=(self._c,self._m,self._s)
        self.identity=sha256(canonical(self.descriptor())).hexdigest();self._sealed=True

    def descriptor(self):
        return dict(policy=POLICY,measure=MEASURE,elastic=self.elastic,metric=self.metric,
            yield_force=self.yield_force,hardening=self.hardening,recovery=RECOVERY)

    def guard(self):
        if sha256(canonical(self.descriptor())).hexdigest()!=self.identity:raise ValueError('generalized section authority changed')
        if any(a is not b for a,b in zip((self._c,self._m,self._s),self._compiled)):
            raise ValueError('generalized section compiled authority changed')

    def virgin(self):
        self.guard();return GeneralizedHistory(self.identity,((0.,0.),)*6,(0.,0.))

    def _origin(self,origin):
        if type(origin) is not GeneralizedHistory or origin.section_identity!=self.identity or type(origin.plastic) is not tuple or len(origin.plastic)!=6:
            raise ValueError('owned six-component section history required')
        z=list(map(_unpair,origin.plastic));p=_unpair(origin.accumulated)
        if p<0:raise ValueError('nonnegative accumulated section increment')
        # This point law permits an explicitly supplied initial plastic offset.
        # Reachability from virgin state belongs to its authenticated history chain.
        return z,p

    def _dual(self,s,z0,p0):
        h=dec(self.hardening);y=dec(self.yield_force);ms=mv(self._m,s);q=dot(s,ms).sqrt();radius=y+h*p0
        lam=max(D(0),(q-radius)/h);z=z0[:];comp=[list(row) for row in self._s]
        if lam:
            n=[v/q for v in ms];z=[a+lam*b for a,b in zip(z0,n)]
            for i in range(6):
                for j in range(6):comp[i][j]+=n[i]*n[j]/h+lam*(self._m[i][j]-n[i]*n[j])/q
        elastic=mv(self._s,s);e=[a+b for a,b in zip(elastic,z)]
        p=p0+lam;ue=dot(s,elastic)/2
        dual=ue+dot(z0,s)+max(D(0),q-radius)**2/(2*h)
        potential=ue+h*(p*p-p0*p0)/2+y*lam
        branch='PLASTIC' if lam else ('YIELD_BOUNDARY' if q==radius else 'ELASTIC')
        return e,comp,z,p,lam,dual,potential,ue+h*p*p/2,y*lam,branch

    def response(self,values,*,origin=None,control='strain'):
        self.guard()
        if control not in ('strain','resultant'):raise ValueError('explicit strain or resultant section control')
        data=_owned(values,(6,));origin=self.virgin() if origin is None else origin
        captured=canonical(origin);started=monotonic()
        def check():
            if monotonic()-started>effective_deadline_seconds(15.):raise RuntimeError('generalized section response deadline')
            self.guard()
        with localcontext() as ctx:
            ctx.prec=80;z0,p0=self._origin(origin);v=list(map(lambda x:D.from_float(float(x)),data))
            if control=='resultant':s=v
            else:
                target=[a-b for a,b in zip(v,z0)];s=mv(self._c,target)
                radius=dec(self.yield_force)+dec(self.hardening)*p0;q=dot(s,mv(self._m,s)).sqrt()
                if q>radius:
                    left=D(0);right=1/dec(self.hardening)
                    for iteration in range(256):
                        check();tau=(left+right)/2
                        matrix=[[self._s[i][j]+tau*self._m[i][j] for j in range(6)] for i in range(6)]
                        s=[row[0] for row in solve(matrix,[[x] for x in target],check)]
                        q=dot(s,mv(self._m,s)).sqrt();g=q*(1-dec(self.hardening)*tau)-radius
                        if abs(g)<=D('1e-60')*max(D(1),radius):break
                        if g>0:left=tau
                        else:right=tau
                    else:raise RuntimeError('generalized section return-map iteration limit')
            e,comp,z,p,lam,dual,potential,stored,dissipation,branch=self._dual(s,z0,p0)
            if control=='strain' and max(abs(a-b) for a,b in zip(e,v))>D('1e-50')*max(D(1),*(abs(x) for x in v)):
                raise ValueError('generalized section primal return mismatch')
            cholesky(comp);tangent=solve(comp,ident(6),check)
            ep=_parts(e,(6,));sp=_parts(s,(6,))
            cp=_parts([x for row in comp for x in row],(6,6));tp=_parts([x for row in tangent for x in row],(6,6))
            result=GeneralizedResponse(origin,GeneralizedHistory(self.identity,tuple(map(_pair,z)),_pair(p)),*ep,*sp,*tp,*cp,
                _pair(dual),_pair(potential),_pair(stored),_pair(dissipation),_pair(lam),branch,
                'SEMISMOOTH_ELASTIC_SELECTION' if branch=='YIELD_BOUNDARY' else 'CLASSICAL_SMOOTH_BRANCH',self.identity)
        check()
        if canonical(origin)!=captured:raise ValueError('generalized origin changed during response')
        return result


@dataclass(frozen=True)
class SectionProposal:
    token: object
    response: GeneralizedResponse


class GeneralizedStation:
    """Local transaction building block; global beam integration is separate."""
    __slots__=('_section','_history','_epoch','_pending','_accepted')

    def __init__(self,section):
        if type(section) is not EllipsoidalGeneralizedSection:raise ValueError('exact generalized section')
        self._section=section;self._history=section.virgin();self._epoch=0;self._pending=None;self._accepted=None

    @property
    def section(self):return self._section

    @property
    def history(self):return self._history

    @property
    def epoch(self):return self._epoch

    def trial(self,strain):
        self._pending=None
        data=_owned(strain,(6,));response=self.section.response(data,origin=self.history)
        proposal=SectionProposal(object(),response);self._pending=(proposal,data,self.history)
        return proposal

    def discard(self,proposal):
        if self._pending is None or self._pending[0] is not proposal:raise ValueError('foreign or stale section proposal')
        self._pending=None

    def commit(self,proposal):
        if self._pending is None or self._pending[0] is not proposal:raise ValueError('foreign or stale section proposal')
        _,data,origin=self._pending
        self._pending=None
        if canonical(origin)!=canonical(self.history):raise ValueError('section origin changed before commit')
        response=self.section.response(data,origin=origin)
        if canonical(proposal.response)!=canonical(response):raise ValueError('section proposal response mutated')
        self._history=response.history;self._epoch+=1;self._accepted=(data,origin,response,self.epoch)
        return self.history

    def replay(self):
        if self._accepted is None:raise ValueError('no accepted generalized section response')
        data,origin,response,epoch=self._accepted
        if self.epoch!=epoch or canonical(self.history)!=canonical(response.history):raise ValueError('accepted section state changed')
        made=self.section.response(data,origin=origin)
        if canonical(made)!=canonical(response):raise ValueError('accepted generalized response replay mismatch')
        return made
