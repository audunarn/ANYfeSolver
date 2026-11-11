"""Private constrained complementary cell for six-resultant plastic interaction.

Minimize sum(w W*(n_j,m_j(p))) subject to sum(w V_j.T n_j)=p_force.
Station forces are solved together. No prescribed plastic direction or
elastic-tangent substitution; origin is fixed throughout each response.
"""
from dataclasses import dataclass
from decimal import Decimal as D,localcontext
from hashlib import sha256
from time import monotonic
from ._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection,GeneralizedHistory,_pair
from ._ge_beam3_fibre_section import canonical
from ._ge_beam3_station_resultant_cell import zero,ident,tr,mul,mv,dot,dec,pair,solve,cholesky,frozen
from ._ge_beam3_cell_identity import CapturedCellIdentity
from ._ge_beam3_effective_deadline import effective_deadline_seconds

POLICY='GE_BEAM3_CONSTRAINED_ELLIPSOID_CELL_CONJUGATE_V1'


@dataclass(frozen=True)
class GeneralizedCellHistory:
    cell_identity: str
    stations: tuple


class GeneralizedCellConjugate:
    __slots__=('section','stations','constraint','identity','size','_capture','_compiled','_identity_binding','_sealed')

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):raise AttributeError('generalized cell is immutable')
        object.__setattr__(self,name,value)

    def __init__(self,section,stations):
        if type(section) is not EllipsoidalGeneralizedSection:raise ValueError('explicit generalized ellipsoid section')
        if type(stations) not in (list,tuple) or not 4<=len(stations)<=32:raise ValueError('bounded complete station inventory')
        section.guard();self.section=section;self.size=3*len(stations)
        rows=[];capture=[]
        with localcontext() as ctx:
            ctx.prec=80;b=zero(6,self.size)
            for index,row in enumerate(stations):
                if type(row) is not dict or set(row)!={'cell','t','weight','v'}:raise ValueError('exact station schema')
                cell=row['cell'];t=dec(row['t']);w=dec(row['weight'])
                if type(cell) is not int or cell not in (0,1) or w<=0 or not 0<t<1:raise ValueError('positive interior cell station')
                v=[[dec(x) for x in r] for r in row['v']]
                if len(v)!=3 or any(len(r)!=3 for r in v):raise ValueError('three by three station map')
                n=zero(3,12)
                for i in range(3):
                    n[i][6*cell+i]=1-t;n[i][6*cell+3+i]=t
                    for j in range(3):b[3*cell+i][3*index+j]=w*v[j][i]
                rows.append((cell,t,w,frozen(v),frozen(n)))
                capture.append(dict(cell=cell,t=float(t),weight=float(w),v=[[float(x) for x in r] for r in v]))
            if any(len({row['t'] for row in capture if row['cell']==cell})<2 for cell in (0,1)):
                raise ValueError('two distinct stations in both cells required')
            cholesky(mul(b,tr(b)))
            self.stations=tuple(rows);self.constraint=frozen(b)
        self._capture=canonical(capture);self._compiled=(self.stations,self.constraint)
        self.identity=sha256(canonical(dict(policy=POLICY,section=section.identity,stations=capture))).hexdigest()
        self._identity_binding=CapturedCellIdentity(self)
        self._sealed=True

    def guard(self):
        self.section.guard()
        if self.stations is not self._compiled[0] or self.constraint is not self._compiled[1]:
            raise ValueError('compiled generalized cell changed')
        self._identity_binding.require(self)

    def virgin(self):
        self.guard();return GeneralizedCellHistory(self.identity,tuple(self.section.virgin() for _ in self.stations))

    def _origins(self,origin):
        if type(origin) is not GeneralizedCellHistory or origin.cell_identity!=self.identity or type(origin.stations) is not tuple or len(origin.stations)!=len(self.stations):
            raise ValueError('complete cell-bound generalized history')
        return tuple(self.section._origin(row) for row in origin.stations)

    def _evaluate(self,x,p,origins,check,elastic=False):
        # x = independent station forces followed by the six constraint multipliers.
        gradient=[D(0)]*(self.size+12);h=zero(self.size+12,self.size+12)
        value=D(0);fields=[]
        for j,((cell,t,w,v,n),(z0,p0)) in enumerate(zip(self.stations,origins)):
            check();s=x[3*j:3*j+3]+mv(n,p[6:])
            e,c,z,acc,lam,dual,primal,stored,dissipation,branch=self.section._dual(s,z0,p0)
            if elastic:
                c=[list(row) for row in self.section._s];e=[a+b for a,b in zip(mv(c,s),z0)]
            value+=w*dual
            indices=list(range(3*j,3*j+3))+list(range(self.size+6*cell,self.size+6*cell+6))
            mapping=zero(6,9)
            for i in range(3):
                mapping[i][i]=D(1);mapping[3+i][3:]=list(n[i][6*cell:6*cell+6])
            g=mv(tr(mapping),e);block=mul(mul(tr(mapping),c),mapping)
            for i,a in enumerate(indices):
                gradient[a]+=w*g[i]
                for k,b in enumerate(indices):h[a][b]+=w*block[i][k]
            fields.append((e,s,z,acc,lam,dual,primal,stored,dissipation,branch))
        residual=[a-b for a,b in zip(gradient[:self.size],mv(tr(self.constraint),x[self.size:]))]
        residual += [a-b for a,b in zip(mv(self.constraint,x[:self.size]),p[:6])]
        hn=[row[:self.size] for row in h[:self.size]]
        saddle=[row+[-v for v in col] for row,col in zip(hn,tr(self.constraint))]
        saddle += [list(row)+[D(0)]*6 for row in self.constraint]
        return value,gradient,h,residual,saddle,fields

    def response(self,resultants,origin=None,*,check=None,branch_direction=None,resultants_low=None):
        started=monotonic()
        def guard():
            if check is not None:check()
            if monotonic()-started>effective_deadline_seconds(60.):raise TimeoutError('generalized cell deadline')
            self.guard()
        guard();origin=self.virgin() if origin is None else origin;captured=canonical(origin)
        with localcontext() as ctx:
            ctx.prec=80;p=[dec(v) for v in resultants]
            if len(p)!=18:raise ValueError("18 retained resultants required")
            if resultants_low is not None:
                low=[dec(v) for v in resultants_low]
                if len(low)!=18 or any(not v.is_finite() for v in low):raise ValueError('finite retained resultant corrections required')
                p=[h+l for h,l in zip(p,low)]
            if len(p)!=18:raise ValueError('18 retained resultants required')
            direction=None if branch_direction is None else tuple(dec(v) for v in branch_direction)
            if direction is not None and (len(direction)!=18 or any(not v.is_finite() for v in direction)):
                raise ValueError('finite 18-component retained branch direction required')
            origins=self._origins(origin);x=[D(0)]*(self.size+6)
            # Affine elastic predictor at the actual plastic origin only.
            _,_,_,r,a,_=self._evaluate(x,p,origins,guard,elastic=True)
            x=[row[0] for row in solve(a,[[-v] for v in r],guard)]
            tolerance=D('1e-28')*max(D(1),*(abs(v) for v in p));evaluations=0;cuts=0
            for iteration in range(49):
                guard();value,g,h,r,a,fields=self._evaluate(x,p,origins,guard);evaluations+=1
                if max(abs(v) for v in r)<=tolerance:break
                if iteration==48:raise ValueError('generalized cell Newton limit')
                step=[row[0] for row in solve(a,[[-v] for v in r],guard)]
                old=dot(r,r)
                for cut in range(32):
                    guard();alpha=D(2)**(-cut);trial=[v+alpha*d for v,d in zip(x,step)]
                    rt=self._evaluate(trial,p,origins,guard)[3];evaluations+=1;cuts+=1
                    if dot(rt,rt)<old:x=trial;break
                else:raise ValueError('generalized cell line search limit')
            rhs=[([D(0)]*6+[-v for v in row[self.size:]]) for row in h[:self.size]]
            rhs += [row+[D(0)]*12 for row in ident(6)]
            dx=solve(a,rhs,guard)
            gradient=x[self.size:]+g[self.size:]
            hessian=dx[self.size:]
            lower=mul([row[:self.size] for row in h[self.size:]],dx[:self.size])
            for i,row in enumerate(lower):
                for j in range(12):row[6+j]+=h[self.size+i][self.size+j]
            hessian+=lower
            cholesky(hessian)
            history=GeneralizedCellHistory(self.identity,tuple(GeneralizedHistory(self.section.identity,tuple(map(_pair,f[2])),_pair(f[3])) for f in fields))
            self._origins(history)
            stations=[dict(strain=pair(f[0]),resultants=pair(f[1]),dual_potential=pair([f[5]]),incremental_potential=pair([f[6]]),
                stored_energy=pair([f[7]]),dissipation=pair([f[8]]),plastic_increment=pair([f[4]]),branch=f[9],
                recovery_policy='RESULTANT_LEVEL_ONLY_NO_FIBRE_STRESSES_SUPPLIED') for f in fields]
            material=sum((row[2]*f[6] for row,f in zip(self.stations,fields)),D(0))
            result=dict(policy=POLICY,cell_identity=self.identity,origin=origin,history=history,potential=pair([value]),
                material_potential=pair([material]),gradient=pair(gradient),hessian=[pair(row) for row in hessian],
                stations=stations,iterations=iteration,evaluations=evaluations,line_trials=cuts,residual=str(max(abs(v) for v in r)),
                derivative_kind='SEMISMOOTH_ELASTIC_SELECTION' if any(f[9]=='YIELD_BOUNDARY' for f in fields) else 'CLASSICAL_SMOOTH_BRANCH',
                production_qualified=False)
            if direction is not None:
                # The constrained station forces depend implicitly on all retained
                # resultants. Use the same saddle sensitivity as the cell Hessian.
                force_direction=mv(dx[:self.size],direction)
                indicators=[]
                for j,(station,f,old) in enumerate(zip(self.stations,fields,origins)):
                    ds=force_direction[3*j:3*j+3]+mv(station[4],direction[6:])
                    ms=mv(self.section._m,f[1]);q=dot(f[1],ms).sqrt()
                    radius=dec(self.section.yield_force)+dec(self.section.hardening)*old[1]
                    margin=q-radius
                    indicators.append(dict(station=j,branch=f[9],margin=pair([margin]),
                        margin_derivative=None if not q else pair([dot(ms,ds)/q]),
                        resultant_direction=pair(ds)))
                result['branch_direction']=tuple(indicators)
        guard()
        if canonical(origin)!=captured:raise ValueError('generalized cell origin changed')
        return result
