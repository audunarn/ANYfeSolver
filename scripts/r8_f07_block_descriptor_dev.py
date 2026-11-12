"""Nearby-pencil block proof and current-public installed bridge, private only."""
from pathlib import Path
import argparse,ast,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import r8_f07_public_descriptor_dev as prior
from scripts.r8_f07_public_descriptor_dev import (
    Unresolved,need,sha,write,rank_screen,zero_cluster,canonical,balance,runtime,
    homogeneous_valid)

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'.development/r8-f07-block-descriptor-20261009'
DOC=ROOT/'docs/agent_plans/development/r8-f07-block-descriptor-20261009'
OLD=ROOT/'.development/r8-f07-public-descriptor-20261009/source1/current-public-mixed-point'
need(Path(prior.__file__).resolve()==ROOT/'scripts/r8_f07_public_descriptor_dev.py','prior helper path mismatch')
PRIOR_SHA=sha(prior.__file__)


STRUCTURAL=DOC/'structural-v1'
STRUCTURAL_Z=(126,127,128,132,133,134)
FINITE=DOC/'finite-supplement-v1'


def finite_fraction(value):
    """Exact binary64 operand; reject unsupported arithmetic, never hide subnormals."""
    import math
    from fractions import Fraction
    need(sys.float_info.radix==2 and sys.float_info.mant_dig==53 and
         sys.float_info.rounds==1,'unsupported IEEE binary64 environment')
    value=float(value)
    need(math.isfinite(value) and (value==0 or abs(value)>=sys.float_info.min),
         'nonfinite/subnormal numerical operand')
    return Fraction.from_float(value)


def finite_upper(value):
    """Least binary64 upper rounding of a nonnegative exact rational."""
    import math
    from fractions import Fraction
    need(value>=0,'negative bound')
    try:rounded=float(value)
    except OverflowError:raise Unresolved('certificate bound overflow')
    need(math.isfinite(rounded),'certificate bound overflow')
    if Fraction.from_float(rounded)<value:rounded=math.nextafter(rounded,math.inf)
    need(math.isfinite(rounded) and (rounded==0 or rounded>=sys.float_info.min),
         'certificate bound overflow/underflow')
    return Fraction.from_float(rounded)


def finite_sqrt(value):
    import math
    from fractions import Fraction
    # sqrt is only a starting approximation; exact square comparison certifies it.
    rounded=math.sqrt(float(finite_upper(value)))
    while Fraction.from_float(rounded)**2<value:rounded=math.nextafter(rounded,math.inf)
    return finite_upper(Fraction.from_float(rounded))


def finite_int_matrix(matrix):
    """Common power-of-two scale; no floating dot-product certificate."""
    exponents=[-(v.denominator.bit_length()-1) for row in matrix for v in row if v]
    exponent=min(exponents,default=0)
    need(all(v.denominator&(v.denominator-1)==0 for row in matrix for v in row),
         'non-dyadic certificate operand')
    return [[v.numerator << (-(v.denominator.bit_length()-1)-exponent)
             if v else 0 for v in row] for row in matrix],exponent


def finite_exact_product(left,right):
    """Exact represented products, O(n^3) integer work, never an inverse."""
    from fractions import Fraction
    need(bool(left) and bool(right) and bool(right[0]) and
         all(len(row)==len(right) for row in left) and
         all(len(row)==len(right[0]) for row in right),'exact product dimension mismatch')
    a,ea=finite_int_matrix(left);b,eb=finite_int_matrix(right)
    columns=list(zip(*b));scale=Fraction(2)**(ea+eb)
    return [[sum(x*y for x,y in zip(row,column))*scale for column in columns] for row in a]


def finite_matrix_bound(matrix):
    """||X||2 <= sqrt(||X||1 ||X||inf), exact sums then outward rounding."""
    one=max((sum(abs(v) for v in column) for column in zip(*matrix)),default=0)
    infinity=max((sum(abs(v) for v in row) for row in matrix),default=0)
    return finite_sqrt(one*infinity)


def finite_vector_bound(vector):
    return finite_sqrt(sum(v*v for v in vector))


def finite_triangular_bound(matrix,lower):
    """Comparison recurrence bounds max row sum of triangular inverse."""
    n=len(matrix);bounds=[0]*n
    for i in (range(n) if lower else range(n-1,-1,-1)):
        need(matrix[i][i]!=0,'singular triangular certificate factor')
        js=range(i) if lower else range(i+1,n)
        bounds[i]=finite_upper((1+sum(abs(matrix[i][j])*bounds[j] for j in js))/abs(matrix[i][i]))
    return max(bounds,default=0)


def finite_factor_certificate(jacobian,packed,pivots):
    """Certify raw factor application; permutation is part of exact defect."""
    n=len(jacobian);need(n>0 and len(pivots)==n and len(packed)==n and
        all(len(row)==n for row in jacobian+packed),'factor dimensions')
    order=list(range(n))
    for i,pivot in enumerate(pivots):
        pivot=int(pivot);need(i<=pivot<n,'invalid LU pivot permutation')
        order[i],order[pivot]=order[pivot],order[i]
    lower=[[packed[i][j] if j<i else int(j==i) for j in range(n)] for i in range(n)]
    upper=[[packed[i][j] if j>=i else 0 for j in range(n)] for i in range(n)]
    product=finite_exact_product(lower,upper)
    defect=[[jacobian[order[i]][j]-product[i][j] for j in range(n)] for i in range(n)]
    kt=lambda t,l:finite_triangular_bound(t,l)
    infinity=finite_upper(kt(upper,False)*kt(lower,True))
    one=finite_upper(kt(list(map(list,zip(*lower))),False)*kt(list(map(list,zip(*upper))),True))
    inverse=finite_sqrt(one*infinity);error=finite_matrix_bound(defect)
    neumann=finite_upper(inverse*error)
    need(neumann<1,'unresolved factor reconstruction/formation defect')
    return finite_upper(inverse/(1-neumann)),dict(defect_bound=float(error),
        application_bound=float(inverse),neumann_product=float(neumann),permutation=order)


def finite_certificate(A,B,theta,vector,out=None):
    """Unique REAL normalized pair for this exact represented pencil only.

    No all-unstructured gamma-ball guarantee; old robustness FAIL is untouched.
    J=[A-theta B,-Bv;v.T,0], v.T e=0. F+J(e,d)-(d B e,0)=0.
    All raw descriptor/algebraic roles survive. Ordinary LU solves only; exact
    dyadic certificates cover formation, factor, permutation and solve errors.
    """
    import numpy as np
    from scipy.linalg import lu_factor,lu_solve
    from fractions import Fraction
    n=len(vector);need(0<n<=256 and np.shape(A)==np.shape(B)==(n,n),'bounded finite certificate dimensions')
    need(np.isrealobj(A) and np.isrealobj(B) and np.isrealobj(vector),'real certificate operands required')
    a=[[finite_fraction(v) for v in row] for row in A]
    b=[[finite_fraction(v) for v in row] for row in B]
    v=[finite_fraction(x) for x in vector];t=finite_fraction(theta)
    need(t>0 and any(v),'positive real nonzero normalized candidate required')
    bv=[row[0] for row in finite_exact_product(b,[[x] for x in v])]
    tangent=[[a[i][j]-t*b[i][j] for j in range(n)] for i in range(n)]
    residual=[row[0] for row in finite_exact_product(tangent,[[x] for x in v])]+[Fraction(0)]
    jacobian=[tangent[i]+[-bv[i]] for i in range(n)]+[v+[Fraction(0)]]
    try:represented=np.array([[float(x) for x in row] for row in jacobian],dtype=float)
    except OverflowError:raise Unresolved('Jacobian formation overflow')
    for row in represented:
        for x in row:finite_fraction(x)
    packed,pivots=lu_factor(represented,check_finite=True)
    factor=[[finite_fraction(x) for x in row] for row in packed]
    inverse,diagnostics=finite_factor_certificate(jacobian,factor,pivots)
    try:rhs=np.array([-float(x) for x in residual])
    except OverflowError:raise Unresolved('residual formation overflow')
    dual=np.zeros(n+1);dual[-1]=1.
    for x in rhs:finite_fraction(x)
    primal=lu_solve((packed,pivots),rhs,check_finite=True)
    adjoint=lu_solve((packed,pivots),dual,trans=1,check_finite=True)
    if out is not None:
        np.savez(out/'factor-application.npz',represented_jacobian=represented,packed=packed,
                 pivots=pivots,primal=primal,adjoint=adjoint,theta=theta,vector=vector)
    s=[finite_fraction(x) for x in primal];y=[finite_fraction(x) for x in adjoint]
    js=[row[0] for row in finite_exact_product(jacobian,[[x] for x in s])]
    jt=[row[0] for row in finite_exact_product(list(map(list,zip(*jacobian))),[[x] for x in y])]
    primal_error=finite_vector_bound([x+r for x,r in zip(js,residual)])
    dual_error=finite_vector_bound([x-int(i==n) for i,x in enumerate(jt)])
    bnorm=finite_matrix_bound(b)
    eta_v=finite_upper(finite_vector_bound(s[:n])+inverse*primal_error)
    c_lambda=finite_upper(finite_vector_bound(y)+inverse*dual_error)
    eta_lambda=finite_upper(abs(s[-1])+c_lambda*primal_error)
    yb=finite_exact_product([y[:n]],b)[0]
    lv=finite_upper(inverse*bnorm)
    ll=finite_upper(finite_vector_bound(yb)+inverse*dual_error*bnorm)
    if not any(residual):radius_v=radius_lambda=Fraction(0)
    else:
        need(eta_v>0 and eta_lambda>0,'degenerate component radius unresolved')
        radius_v=finite_upper(2*eta_v);radius_lambda=finite_upper(2*eta_lambda)
        if out is not None:
            write(out/'contraction-admission.json',dict(
                scope='supplementary represented-pencil only; trial bounds, not certified radii',
                inverse_bound=str(inverse),primal_residual_bound=str(primal_error),
                dual_residual_bound=str(dual_error),eta_vector=str(eta_v),eta_lambda=str(eta_lambda),
                vector_action_bound=str(lv),lambda_action_bound=str(ll),
                vector_radius=str(radius_v),scalar_radius=str(radius_lambda),
                vector_image=str(eta_v+lv*radius_v*radius_lambda),
                scalar_image=str(eta_lambda+ll*radius_v*radius_lambda),
                lipschitz_upper=str(max(2*lv*radius_lambda,2*ll*radius_v))))
        need(eta_v+lv*radius_v*radius_lambda<=radius_v and
             eta_lambda+ll*radius_v*radius_lambda<=radius_lambda and
             max(2*lv*radius_lambda,2*ll*radius_v)<1,'finite contraction unresolved')
    need(t-radius_lambda>0,'certified eigenvalue interval reaches zero')
    result=dict(status='CERTIFIED_REPRESENTED_PAIR',theta=float(t),radius=float(radius_lambda),
        vector_radius=float(radius_v),eta_vector=float(eta_v),eta_lambda=float(eta_lambda),
        inverse_bound=float(inverse),vector_action_bound=float(lv),lambda_action_bound=float(ll),
        primal_residual_bound=float(primal_error),dual_residual_bound=float(dual_error),
        normalization='v0.T e=0; unchanged raw v0; real uniqueness in joint vector/scalar ball',
        guarantee='actual represented pencil only; NOT all unstructured perturbations or completeness',
        old_robustness_status='UNCHANGED',qualification=False,**diagnostics)
    result['contraction_exact']=dict(vector_image=str(eta_v+lv*radius_v*radius_lambda),
        scalar_image=str(eta_lambda+ll*radius_v*radius_lambda),
        vector_radius=str(radius_v),scalar_radius=str(radius_lambda),
        lipschitz_upper=str(max(2*lv*radius_lambda,2*ll*radius_v)))
    # Exact rational endpoints, not inward-rounded floating intervals.
    result['interval_exact']=[str(t-radius_lambda),str(t+radius_lambda)]
    return result


def finite_association(results):
    from fractions import Fraction
    intervals=[list(map(Fraction,r['interval_exact'])) for r in results]
    for result,(lo,hi) in zip(results,intervals):
        center=Fraction.from_float(result['theta']);radius=Fraction.from_float(result['radius'])
        need(result['status']=='CERTIFIED_REPRESENTED_PAIR' and lo==center-radius and hi==center+radius,
             'incorrect finite interval association')
    for i,(lo,hi) in enumerate(intervals):
        for other_lo,other_hi in intervals[:i]:
            need(hi<other_lo or other_hi<lo,'overlapping/repeated finite association unresolved')
    return 'disjoint certified real intervals; no all-root completeness claim'


def run_finite_saved(out):
    """Four historical refusals only; no new roots or old screen modification."""
    import numpy as np
    data=dict(np.load(OLD/'pencil.npz'));roots=dict(np.load(OLD/'homogeneous-qz.npz'))
    historical=STRUCTURAL/'attempt-source1/structural-saved-current-public/finite-classification.npz'
    classification=dict(np.load(historical))
    positions=np.flatnonzero(~(classification['near']|classification['nonzero']))
    indices=classification['indices'][positions];need(indices.tolist()==[1,2,4,5],'current four-refusal inventory changed')
    results=[]
    for position,index in zip(positions,indices):
        theta=classification['theta'][position];vector=roots['right'][:,index]
        need(theta.imag==0 and not np.any(vector.imag!=0),'saved candidate is not exactly real')
        mode_out=out/('mode-'+str(int(index)));mode_out.mkdir()
        result=finite_certificate(data['As'],data['Bs'],float(theta.real),vector.real,mode_out)
        result.update(saved_index=int(index),historical_delta=float(classification['delta'][position]))
        results.append(result);write(out/'selected-pairs.json',results)
    association=finite_association(results)
    answer=dict(status='SUPPLEMENT_ONLY',selected_count=4,eventual_positive_obligation=55,
        old_status='FAIL: finite nearzero separation unresolved',association=association,
        physical_gates='unchanged, not reexecuted or supplied by this supplement',results=results,qualification=False)
    write(out/'result.json',answer);return answer


FINITE_CONTROLS=('simple','approximate','nonsymmetric','indefinite-algebraic','close-disjoint',
    'repeated','nearzero','overlap','wrong-association','permutation','wrong-factor','wrong-residual',
    'rounding','subnormal','overflow','triangular-robustness-counterexample')


def run_finite_control(name,out):
    """Analytic fixtures only: no eig/QZ and no production model."""
    import numpy as np
    from fractions import Fraction
    from unittest.mock import patch
    expected={'repeated':'singular triangular','nearzero':'reaches zero',
        'overlap':'association unresolved','wrong-association':'incorrect finite interval',
        'permutation':'factor reconstruction',
        'wrong-factor':'singular triangular','wrong-residual':'contraction unresolved',
        'subnormal':'subnormal','overflow':'overflow'}
    A=np.diag((1.,3.));B=np.eye(2);v=np.array((1.,0.));theta=1.
    try:
        if name=='rounding':
            bound=finite_upper(Fraction(1,3));root=finite_sqrt(Fraction(2))
            need(bound>=Fraction(1,3) and root*root>=2,'outward rounding failed')
            proof=dict(upper=str(bound),sqrt_upper=str(root))
        elif name=='subnormal':finite_fraction(float.fromhex('0x0.0000000000001p-1022'))
        elif name=='overflow':finite_upper(Fraction(2)**1024)
        elif name in ('permutation','wrong-factor'):
            jacobian=[[Fraction(0),Fraction(1)],[Fraction(2),Fraction(3)]]
            packed=[[Fraction(2),Fraction(3)],[Fraction(0),Fraction(1)]]
            if name=='wrong-factor':packed[1][1]=Fraction(0)
            # Valid swapped factor is independently checked before corrupted pivot.
            if name=='permutation':
                _,good=finite_factor_certificate(jacobian,packed,[1,1])
                need(good['defect_bound']==0,'correct pivot factor not exact')
            finite_factor_certificate(jacobian,packed,[0,1])
        else:
            if name=='approximate':theta=1.+2**-30;v=np.array((1.,2**-35))
            elif name=='nonsymmetric':A[0,1]=2.
            elif name=='indefinite-algebraic':
                A=np.array(((2.,0.,0.),(0.,4.,1.),(0.,3.,-2.)))
                B=np.diag((1.,1.,0.));v=np.array((1.,0.,0.));theta=2.
            elif name=='repeated':A[1,1]=1.
            elif name=='close-disjoint':A[1,1]=1.+2**-20
            elif name=='nearzero':A[0,0]=2**-40;theta=2**-39;v=np.array((1.,2**-60))
            elif name=='triangular-robustness-counterexample':A[0,1]=2**20
            if name=='wrong-residual':
                theta=2.
                # Corrupt both ordinary solves; exact original residuals must refuse.
                with patch('scipy.linalg.lu_solve',return_value=np.zeros(3)):
                    proof=finite_certificate(A,B,theta,v)
            else:proof=finite_certificate(A,B,theta,v)
            if name=='overlap':finite_association([proof,proof])
            if name=='wrong-association':
                proof['interval_exact']=['2','2'];finite_association([proof])
            if name=='close-disjoint':
                other=finite_certificate(A,B,A[1,1],np.array((0.,1.)))
                finite_association([proof,other])
            if name=='approximate':
                lo,hi=map(Fraction,proof['interval_exact'])
                need(lo<=1<=hi and proof['radius']>0,'analytic approximate root not enclosed')
            if name=='triangular-robustness-counterexample':
                # det(A+Delta)=2-K*(2/K)=0 exactly; original pair residual zero.
                K=Fraction(2)**20;perturbation=Fraction(2)/K
                need(2-K*perturbation==0 and proof['radius']==0,'robustness counterexample lost')
                proof.update(old_robustness_must_remain_failed=True,
                    zero_root_perturbation_exact=str(perturbation))
                moved=A.copy();moved[1,0]=float(perturbation)
                refused=False
                try:finite_certificate(moved,B,theta,v)
                except Unresolved:refused=True
                need(refused,'old exact certificate incorrectly inherited by perturbed pencil')
        need(name not in expected,'negative finite control admitted')
        write(out/'result.json',dict(status='PASS',proof=proof,qualification=False))
    except Unresolved as exc:
        need(name in expected and expected[name] in str(exc),'unintended finite refusal: '+str(exc))
        write(out/'result.json',dict(status='PASS',expected_refusal=str(exc),qualification=False))


def structural_law(record):
    """Admitted fixture only; this is not a general POINT attachment theorem."""
    need(record.get('point_pairs')==[[1,101],[4,105]],'structural point inventory changed')
    need(record.get('offsets')==[[0.,0.,.0675],[0.,0.,.0675]],'structural offset is not purely normal')
    identity=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    need(record.get('frames')==[identity]*4,'structural point frame changed')
    need(record.get('prestress')==0.,'structural prestress changed')
    need(record.get('selectors')==list(STRUCTURAL_Z),'structural selector roles changed')


def saved_structural_roles(data,physical,blocks,out):
    import numpy as np
    need(blocks['dimensions']==dict(primal=126,constraints=12,pencil=138),'structural fixture dimensions changed')
    nodal=[48+6*i+j for i in range(5) for j in (3,4,5)]
    spin=[6*i+5 for i in range(8)]
    retained=list(range(84,102))+list(range(108,126))
    cells=list(range(78,84))+list(range(102,108))
    need(blocks['native_nodal_rotations']==nodal and blocks['shell_normal_spin']==spin and
         blocks['resultants']==retained and blocks['cells']==cells and
         blocks['multipliers']==list(range(126,138)) and
         blocks['declared_zero']==sorted(nodal+spin+retained),'structural mass role inventory changed')
    # ct=Q_s.T(rb-rs)-e. At the admitted identity frames its derivative is
    # [-I,skew(e),I,0]; cr has [-I,I] rotational blocks. No pose evaluation.
    expected=np.zeros((12,126));identity=np.eye(3);height=.0675
    skew=np.array(((0.,-height,0.),(height,0.,0.),(0.,0.,0.)))
    for rows,shell,beam in ((0,0,48),(6,18,72)):
        expected[rows:rows+3,shell:shell+3]=-identity
        expected[rows:rows+3,shell+3:shell+6]=skew
        expected[rows:rows+3,beam:beam+3]=identity
        expected[rows+3:rows+6,shell+3:shell+6]=-identity
        expected[rows+3:rows+6,beam+3:beam+6]=identity
    A,B=data['A'],data['B'];M,C=physical['M'],physical['C']
    need(M.shape==(126,126) and C.shape==(12,126) and np.array_equal(C,expected),
         'saved point law differs from exact source-derived fixture')
    need(A.shape==B.shape==(138,138) and np.array_equal(A[:126,:126],physical['K']) and
         np.array_equal(A[126:,:126],C) and np.array_equal(A[:126,126:],C.T) and
         not np.any(A[126:,126:]!=0),'saved original pencil/physical law mismatch')
    need(np.array_equal(B[:126,:126],M) and not np.any(B[126:,:]!=0) and not np.any(B[:,126:]!=0),
         'saved original physical mass mismatch')
    algebraic=sorted(blocks['declared_zero']+blocks['multipliers'])
    record=dict(point_pairs=[[1,101],[4,105]],offsets=[[0.,0.,height]]*2,
                frames=[identity.tolist()]*4,prestress=0.,selectors=list(STRUCTURAL_Z))
    structural_law(record)
    need(np.all(np.isfinite(data['D'])) and np.all(data['D']!=0),'invalid diagonal balance')
    write(out/'structural-role-contract.json',dict(law=record,algebraic=algebraic,
          scope='exact admitted saved POINT fixture; source-bound represented checks; no general pose/mass law'))
    return dict(selectors=list(STRUCTURAL_Z),algebraic=algebraic)


def block_infinity(A,B,out,structural=None):
    import numpy as np
    from scipy.linalg import svd,cholesky,solve,svdvals
    norm=lambda x:float(np.linalg.norm(x,2)) if x.size else 0.
    gamma=lambda d:64*np.finfo(float).eps*d
    N=len(A);need(N<=256,'bounded pencil dimension exceeded')
    active=np.any(B!=0,axis=0)|np.any(B!=0,axis=1)
    i=np.flatnonzero(active);a=np.flatnonzero(~active);m=len(i);na=len(a)
    if structural is not None:
        expected=structural['algebraic'];z=structural['selectors']
        need(len(set(expected))==len(expected) and sorted(expected)==a.tolist(),'structural mass-nullspace inventory mismatch')
        need(len(set(z))==len(z) and len(z)>0 and set(z).issubset(expected),'invalid structural selector partition')
        need(not np.any(B[:,z]!=0) and not np.any(B[z,:]!=0),'structural selectors have mass')
    M=B[np.ix_(i,i)];Aa=A[np.ix_(a,a)]
    Aia=A[np.ix_(i,a)];Aai=A[np.ix_(a,i)]
    np.savez(out/'original-blocks.npz',A=A,B=B,M=M,Aaa=Aa,Aia=Aia,Aai=Aai,inertial=i,algebraic=a)
    need(m>0 and np.isfinite(A).all() and np.isfinite(B).all(),'invalid inertial pencil')
    H=np.tril(M)+np.tril(M,-1).conj().T
    try:L=cholesky(H,lower=True)
    except np.linalg.LinAlgError as exc:
        raise Unresolved('nonpositive physical inertia') from exc
    sm=svdvals(M);sl=svdvals(L)
    eMH=norm(M-H);eHL=norm(H-L@L.conj().T)
    deltaM=eMH+eHL+gamma(m)*(norm(M)+norm(abs(L)@abs(L.conj().T)))
    rho=deltaM/float(sm[-1]) if sm[-1]>0 else float('inf')
    np.savez(out/'inertia-proof.npz',M=M,Hlower=H,L=L,singular_M=sm,singular_L=sl,eMH=eMH,eHL=eHL,deltaM=deltaM,rho=rho)
    need(sl[-1]**2>=1000*deltaM and rho<1,'unresolved physical inertia positivity/conditioning')
    if structural is not None:
        z=np.asarray(structural['selectors'],dtype=int)
        at=np.asarray([a.tolist().index(int(v)) for v in z],dtype=int)
        remaining=np.asarray([v for v in range(na) if v not in set(at.tolist())],dtype=int)
        # Slices prove both exact null identities. No inferred/global basis E.
        need(not np.any(Aa[:,at]!=0) and not np.any(Aa[at,:]!=0),'structural algebraic selectors are not exact both-sided null')
        Haa=Aa[np.ix_(remaining,remaining)]
        U,s,Vh=svd(Haa,full_matrices=True);tau=gamma(na)*norm(Aa)
        r=len(remaining);k=len(at)
        np.savez(out/'algebraic-rank.npz',H=Haa,U=U,singular=s,Vh=Vh,tau=tau,rank=r,
                 selector_indices=z,complement_indices=a[remaining])
        need(r>0 and len(s)==r and s[-1]>0 and s[-1]>=1000*tau,'unresolved structural complement rank gap')
        nearby=(U*s)@Vh;E=nearby-Haa
        formation=norm(abs(U)@(s[:,None]*abs(Vh)))
        deltaa=norm(E)+gamma(na)*(norm(Aa)+formation)
        separation=float(s[-1]);ratio=deltaa/(separation-deltaa) if separation>deltaa else float('inf')
        np.savez(out/'nearby-algebraic-perturbation.npz',H=Haa,Ehh=E,Hnear=nearby,delta_a=deltaa,
                 complement_resolution_ratio=ratio,theta=0.,null_defects=(0.,0.),selector_indices=z)
        need(ratio<1,'unresolved structural complement perturbation')
        deltaa=float(deltaa);theta=0.;defects=(0.,0.)
    elif na:
        U,s,Vh=svd(Aa,full_matrices=True);tau=gamma(na)*(float(s[0]) if len(s) else 0.)
        r=int(np.count_nonzero(s>tau));k=na-r
        np.savez(out/'algebraic-rank.npz',Aaa=Aa,U=U,singular=s,Vh=Vh,tau=tau,rank=r)
        need(r==0 or s[r-1]>=1000*tau,'unresolved algebraic rank gap')
        Wa=U[:,r:];Va=Vh[r:,:].conj().T
        Ur=U[:,:r];Vhr=Vh[:r,:]
        nearby=(Ur*s[:r])@Vhr;E=nearby-Aa
        tail=float(s[r]) if r<len(s) else 0.
        formation=norm(abs(Ur)@(s[:r,None]*abs(Vhr)))
        deltaa=max(tail,norm(E))+gamma(na)*(norm(Aa)+formation)
        defects=(norm(Aa@Va),norm(Wa.conj().T@Aa))
        separation=float(s[r-1]) if r else float('inf')
        theta=deltaa/(separation-deltaa) if k and r and separation>deltaa else (0. if not r or not k else float('inf'))
        np.savez(out/'nearby-algebraic-perturbation.npz',Eaa=E,Anear=nearby,Wa=Wa,Va=Va,delta_a=deltaa,null_defects=defects,theta=theta)
        need(max(defects)<=deltaa,'approximate null defect exceeds nearby bound')
        need(theta<1,'unresolved nullspace orientation')
    else:
        r=k=0;Wa=Va=np.zeros((0,0));deltaa=theta=tau=0.;defects=(0.,0.)
    if k:
        if structural is not None:
            F=A[np.ix_(i,z)];G=A[np.ix_(z,i)]
            # Exact extraction has no orientation uncertainty. Retain the
            # old conservative arithmetic allowance, not just the SVD term.
            deltaF=gamma(N)*norm(abs(F));deltaG=gamma(N)*norm(abs(G))
        else:
            F=Aia@Va;G=Wa.conj().T@Aai
            deltaF=gamma(N)*norm(abs(Aia)@abs(Va))+2*theta*norm(Aia)
            deltaG=gamma(N)*norm(abs(Wa.conj().T)@abs(Aai))+2*theta*norm(Aai)
        X=solve(M,F,assume_a='gen');S=G@X
        residual=M@X-F;eta=norm(residual)
        eR=gamma(m)*norm(abs(M)@abs(X)+abs(F))
        deltaX=(eta+eR+deltaF+deltaM*norm(X))/(float(sm[-1])*(1-rho))
        deltaS=deltaG*norm(X)+(norm(G)+deltaG)*deltaX+gamma(N)*norm(abs(G)@abs(X))
        ss=svdvals(S);screen=max(deltaS,gamma(k)*norm(S))
        np.savez(out/'null-coupling-proof.npz',F=F,G=G,X=X,S=S,solve_residual=residual,singular_S=ss,
            deltaF=deltaF,deltaG=deltaG,eta=eta,eR=eR,deltaX=deltaX,deltaS=deltaS,screen=screen)
        need(eta<=eR,'physical inertia solve residual exceeds screen')
        need(ss[-1]>0 and ss[-1]>=1000*screen,'singular or unresolved algebraic null coupling')
    else:ss=np.empty(0);deltaS=screen=0.
    result=dict(scope='nearby-pencil binary64 resolution; NOT exact stored infinity multiplicity',
        representation='exact structural role selectors' if structural is not None else 'SVD inferred null orientation',
        inertial=m,algebraic=na,algebraic_rank=r,nullity=k,simple_infinity=r,size2_infinity=k,
        nearby_infinite=na+k,nearby_finite=m-k,delta_a=deltaa,theta=theta,
        deltaM=deltaM,rho=rho,deltaS=deltaS,coupling_sigma_min=float(ss[-1]) if k else None,
        coupling_screen=screen)
    write(out/'block-proof.json',result);return result


def classify_saved(data,roots,physical,out,structural=None):
    """Reuse saved balanced matrices and homogeneous roots, never new QZ."""
    import numpy as np
    from scipy.linalg import cholesky,svdvals
    A=data['A'];B=data['B'];As=data['As'];Bs=data['Bs'];D=data['D'];frequency=float(data['frequency'])
    alpha=roots['alpha'];beta=roots['beta'];left=roots['left'];right=roots['right'];N=len(A)
    proof=block_infinity(As,Bs,out,structural=structural)
    homogeneous_valid(np.vstack((alpha,beta)))
    p=abs(beta)/np.hypot(abs(alpha),abs(beta));order=np.argsort(p)
    inf=order[:proof['nearby_infinite']];finite=order[proof['nearby_infinite']:]
    screen=64*np.finfo(float).eps*N
    np.savez(out/'raw-root-inventory.npz',alpha=alpha,beta=beta,left=left,right=right,projective=p,infinite_indices=inf,finite_indices=finite,screen=screen)
    need((not len(inf) or np.max(p[inf])<=screen) and np.min(p[finite])>=1000*screen,'homogeneous nearby-cluster separation unresolved')
    allres=[]
    normAs=float(np.linalg.norm(As,2));normBs=float(np.linalg.norm(Bs,2))
    for j in range(N):
        v=right[:,j];res=beta[j]*(As@v)-alpha[j]*(Bs@v)
        den=(abs(beta[j])*normAs+abs(alpha[j])*normBs)*np.linalg.norm(v)
        err=float(np.linalg.norm(res)/den) if den else float(np.linalg.norm(res))
        allres.append(err)
    np.savez(out/'homogeneous-backward.npz',errors=allres,normAs=normAs,normBs=normBs)
    need(max(allres)<=1e-8,'original homogeneous backward equation failed')
    rank,V0,W0,delta0=zero_cluster(As,Bs,out)
    need(N-rank==6,'full-pencil zero nullity is not exactly six')
    R=physical['canonical']/D[:,None]
    primal=len(physical['M']);K=physical.get('K',A);C=physical['C'];canonical_errors=[]
    for v in physical['canonical'].T:
        q=v[:primal];kr=K@q;kb=abs(K)@abs(q);den=np.max(kb*D[:primal])
        eq=float(np.max(abs(kr)*D[:primal])/den) if den else float(np.max(abs(kr)))
        gap=C@q;gb=abs(C)@abs(q)
        cg=float(np.max(abs(gap)*D[primal:])/np.max(gb*D[primal:])) if len(C) and np.max(gb)>0 else 0.
        canonical_errors.append(dict(equation=eq,constraint=cg))
    write(out/'canonical-physical-guards.json',canonical_errors)
    need(max(p['equation'] for p in canonical_errors)<=1e-8 and max(p['constraint'] for p in canonical_errors)<=1e-9,'canonical physical rigid field failed')
    L=cholesky(R.conj().T@Bs@R,lower=True);W=R@np.linalg.solve(L.conj().T,np.eye(6))
    L=cholesky(V0.conj().T@Bs@V0,lower=True);Z=V0@np.linalg.solve(L.conj().T,np.eye(6))
    overlap=svdvals(W.conj().T@Bs@Z)
    need(np.max(abs(overlap-1))<=1e-7,'zero nullspace is not physical rigid space')
    theta=alpha[finite]/beta[finite];near=abs(theta)<=delta0
    nonzero=abs(theta)>=1000*delta0;delta=[]
    for j,t in zip(finite,theta):
        if abs(t)<=delta0:delta.append(delta0);continue
        pair=abs(left[:,j].conj()@Bs@right[:,j]);need(pair>0,'unresolved nonzero pair sensitivity')
        delta.append(screen*(np.linalg.norm(As,2)+abs(t)*np.linalg.norm(Bs,2))*np.linalg.norm(left[:,j])*np.linalg.norm(right[:,j])/pair)
    delta=np.asarray(delta);nonzero&=abs(theta)>=1000*delta
    np.savez(out/'finite-classification.npz',indices=finite,theta=theta,delta0=delta0,delta=delta,near=near,nonzero=nonzero)
    need(np.isfinite(delta).all() and np.all(near|nonzero),'finite nearzero separation unresolved')
    need(np.count_nonzero(near)==6,'finite nearzero count is not exactly six')
    need(np.all(abs(theta.imag)<=np.where(near,delta,1e-8*abs(theta.real))),'nonreal finite root')
    need(np.all(theta.real[nonzero]>0),'negative resolved finite root')
    Q=right[:,finite[near]];L=cholesky(Q.conj().T@Bs@Q,lower=True);Q=Q@np.linalg.solve(L.conj().T,np.eye(6))
    qzoverlap=svdvals(W.conj().T@Bs@Q)
    need(np.max(abs(qzoverlap-1))<=1e-7,'QZ nearby-zero subspace is not physical rigid space')
    positive=finite[nonzero][np.argsort(theta.real[nonzero])]
    M=physical['M'];C=physical['C'];n=len(M);pairs=[]
    for j in finite:
        if j in finite[near]:continue
        value=float((frequency*alpha[j]/beta[j]).real);x=D*right[:,j]
        need(np.linalg.norm(x.imag)<=1e-8*np.linalg.norm(x.real),'complex physical finite vector')
        x=x.real;q=x[:n];kinetic=float(q@M@q);need(kinetic>0,'nonpositive physical modal mass')
        x=x/np.sqrt(kinetic);q=x[:n]
        res=A@x-value*(B@x);bound=abs(A)@abs(x)+abs(value)*(abs(B)@abs(x));gap=C@q;gb=abs(C)@abs(q)
        eq=float(np.max(abs(res)*D)/np.max(bound*D))
        constraint=float(np.max(abs(gap)*D[n:])/np.max(gb*D[n:])) if len(C) else 0.
        np.savez(out/f'finite-pair-{j}.npz',x=x,eigenvalue=value,residual=res,bound=bound,gap=gap,gap_bound=gb)
        pairs.append(dict(index=int(j),value=value,equation=eq,constraint=constraint))
        write(out/'finite-pair-metrics.json',pairs)
        need(eq<=1e-8 and constraint<=1e-9,'original finite physical residual failed')
    first=[next(p['value'] for p in pairs if p['index']==int(j)) for j in positive[:2]]
    answer=dict(status='PASS',proof=proof,finite_count=len(finite),nearzero_count=int(np.count_nonzero(near)),
        first_eigenvalues=first,rigid_overlap=overlap.tolist(),qz_rigid_overlap=qzoverlap.tolist(),
        max_homogeneous_error=max(allres),max_equation=max(p['equation'] for p in pairs),max_constraint=max(p['constraint'] for p in pairs))
    write(out/'result.json',answer);return answer


def run_saved(out):
    import numpy as np
    data=dict(np.load(OLD/'pencil.npz'));roots=dict(np.load(OLD/'homogeneous-qz.npz'))
    physical=dict(np.load(OLD/'original-physical.npz'))
    np.savez(out/'bound-saved-inputs.npz',**data)
    return classify_saved(data,roots,physical,out)


def run_structural_saved(out):
    """Same saved roots/pencil, exact fixture role branch, never new assembly/QZ."""
    import numpy as np
    data=dict(np.load(OLD/'pencil.npz'));roots=dict(np.load(OLD/'homogeneous-qz.npz'))
    physical=dict(np.load(OLD/'original-physical.npz'))
    blocks=json.loads((OLD/'blocks.json').read_text())
    roles=saved_structural_roles(data,physical,blocks,out)
    return classify_saved(data,roots,physical,out,structural=roles)


STRUCTURAL_CONTROLS=('baseline','right-algebraic','left-algebraic','selector-mass',
    'complement-mass','complement-rank','complement-gap','shifted-selector',
    'negative-inertia','tiny-inertia','uncoupled','tangential-offset','rotated-frame','prestress')


def run_structural_control(name,out):
    import numpy as np
    # Independent raw nonsymmetric/indefinite 2 inertial +2 h +1 z example.
    A=np.array(((2.,0.,3.,1.,1.),(0.,4.,2.,-2.,0.),
                (4.,-1.,-5.,1.,0.),(1.,3.,0.,2.,0.),(2.,0.,0.,0.,0.)))
    B=np.diag((1.,2.,0.,0.,0.));roles=dict(algebraic=[2,3,4],selectors=[4])
    expected={'right-algebraic':'not exact both-sided null','left-algebraic':'not exact both-sided null',
              'selector-mass':'mass-nullspace inventory','complement-mass':'mass-nullspace inventory',
              'complement-rank':'complement rank gap','complement-gap':'complement rank gap',
              'shifted-selector':'not exact both-sided null','negative-inertia':'nonpositive physical inertia',
              'tiny-inertia':'unresolved physical inertia','uncoupled':'algebraic null coupling',
              'tangential-offset':'offset is not purely normal','rotated-frame':'frame changed','prestress':'prestress changed'}
    if name=='right-algebraic':A[2,4]=1e-30
    elif name=='left-algebraic':A[4,2]=-1e-30
    elif name=='selector-mass':B[4,4]=1e-30
    elif name=='complement-mass':B[2,2]=1e-30
    elif name=='complement-rank':A[3,3]=0.
    elif name=='complement-gap':A[3,3]=1e-15
    elif name=='shifted-selector':roles['selectors']=[3]
    elif name=='negative-inertia':B[0,0]=-1.
    elif name=='tiny-inertia':B[1,1]=1e-20
    elif name=='uncoupled':A[4,:2]=0.
    np.savez(out/'control-input.npz',A=A,B=B,selectors=roles['selectors'],algebraic=roles['algebraic'])
    try:
        if name in ('tangential-offset','rotated-frame','prestress'):
            identity=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
            law=dict(point_pairs=[[1,101],[4,105]],offsets=[[0.,0.,.0675]]*2,
                     frames=[identity]*4,prestress=0.,selectors=list(STRUCTURAL_Z))
            if name=='tangential-offset':law['offsets']=[[1e-30,0.,.0675],[0.,0.,.0675]]
            elif name=='rotated-frame':law['frames']=[[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]]+[identity]*3
            else:law['prestress']=1e-30
            write(out/'control-law.json',law);structural_law(law)
            raise AssertionError('unexpected fixture law admitted')
        proof=block_infinity(A,B,out,structural=roles)
    except Unresolved as exc:
        need(name in expected and expected[name] in str(exc),'unintended structural refusal: '+str(exc))
        write(out/'result.json',dict(status='PASS',expected_refusal=str(exc),qualification=False));return
    need(name=='baseline','negative structural control admitted')
    need(proof['theta']==0. and proof['algebraic_rank']==2 and proof['nullity']==1 and
         proof['nearby_infinite']==4 and proof['nearby_finite']==1,'structural conditional theorem partition wrong')
    write(out/'result.json',dict(status='PASS',proof=proof,qualification=False,
          scope='structural block resolution only; no eig/QZ/physical model or finite-mode acceptance'))


def structural_binding():
    """Stdlib IO/AST only: fresh candidate, never reuse a frozen registration."""
    paths=('scripts/r8_f07_block_descriptor_dev.py','tests/test_r8_f07_block_descriptor_dev.py',
           'scripts/r8_f07_public_descriptor_dev.py','tests/conftest.py','examples/stiffened_shell_ge_mixed_reference_modes.py',
           'examples/stiffened_shell_native_fibre_bl5.py','src/anyfesolver/_stiffened_shell_ge_experimental.py',
           'src/anyfesolver/_ge_beam3_pose_joint.py')
    for path in paths:ast.parse((ROOT/path).read_text(),filename=path)
    return dict(sources={path:sha(ROOT/path) for path in paths},
        inputs={str(OLD/name):sha(OLD/name) for name in
                ('pencil.npz','homogeneous-qz.npz','original-physical.npz','public-status.json','blocks.json')},
        controls=list(STRUCTURAL_CONTROLS),saved_case='structural-saved-current-public',
        control_count=len(STRUCTURAL_CONTROLS)+1,qualification=False,
        scope='exact admitted saved structural branch; no new assembly/eig/QZ/model/reference/installed job')


def structural_prepare(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    need(not path.exists(),'preserve versioned structural preparation')
    write(path,structural_binding())


def finite_binding():
    binding=structural_binding()
    classification=STRUCTURAL/'attempt-source1/structural-saved-current-public/finite-classification.npz'
    binding['inputs'][str(classification)]=sha(classification)
    binding.update(controls=list(FINITE_CONTROLS),control_count=len(FINITE_CONTROLS)+1,
        saved_case='finite-supplement-saved-current-public',
        scope='supplementary represented-pencil finite certificate; old robustness FAIL unchanged; no new roots/model')
    return binding


def finite_prepare(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    need(not path.exists(),'preserve versioned finite preparation');write(path,finite_binding())


def structural_child(out,authority_path,finite=False):
    import traceback
    out=Path(out).resolve();authority_path=Path(authority_path).resolve()
    authority=json.loads(authority_path.read_text());launch=json.loads((out/'launch.json').read_text())
    home=FINITE if finite else STRUCTURAL;binding=finite_binding if finite else structural_binding
    supervised='F07_FINITE_SUPERVISED' if finite else 'F07_STRUCTURAL_SUPERVISED'
    ledger=json.loads((home/'ledger.json').read_text())
    need(authority['status']=='AUTHORIZED' and authority['sole_slot_released'] is True and
         launch['authority_sha256']==sha(authority_path) and launch['binding']==binding() and
         ledger['status']=='OPEN' and ledger['current_attempt']==out.name and
         os.environ.get(supervised)==sha(authority_path),'fresh matching candidate authority required')
    import psutil
    psutil.Process().cpu_affinity([0])
    selected=ROOT/'src';sys.path[:0]=[str(selected),str(ROOT)]
    os.environ.update(F07_BLOCK_OUTPUT=str(out),F07_BLOCK_SUPERVISED=sha(authority_path),F07_BLOCK_STAGE='finite' if finite else 'structural')
    runtime(out,'runtime-before',selected)
    import pytest
    tests=[str(ROOT/'tests/test_r8_f07_block_descriptor_dev.py')+'::'+name for name in
           (('test_finite_supplement_controls','test_finite_supplement_saved_current_public') if finite else
            ('test_structural_role_controls','test_structural_saved_current_public'))]
    expected_count=len(FINITE_CONTROLS if finite else STRUCTURAL_CONTROLS)+1
    class Counts:
        collected=passed=failed=skipped=0
        def pytest_collection_finish(self,session):
            self.collected=len(session.items)
            need(self.collected==expected_count,'focused candidate collection count changed')
        def pytest_runtest_logreport(self,report):
            if report.failed:self.failed+=1
            elif report.when=='call' and report.passed:self.passed+=1
            elif report.skipped:self.skipped+=1
    counts=Counts()
    code=None
    try:
        code=int(pytest.main(tests+(['-x'] if finite else [])+['-o','pythonpath=','-q','-p','no:cacheprovider',
                    '--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'pytest-tmp')],plugins=[counts]))
        if code==0 and (counts.passed!=expected_count or counts.failed or counts.skipped):code=1
        return code
    except BaseException as exc:
        write(out/'child-failure.json',dict(exception=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        settings=runtime(out,'runtime-after',selected)
        need(not settings['imports'],'saved-only structural assessment imported a product model')
        write(out/'child-status.json',dict(exit_code=code,scope='finite supplement saved-only' if finite else 'structural saved-only',qualification=False,
            collected=counts.collected,passed=counts.passed,failed_reports=counts.failed,skipped=counts.skipped))


def structural_supervise(authority_path,attempt,finite=False):
    """Ordinary fresh cumulative allowance; root supplies all numerical limits."""
    import psutil,subprocess
    authority_path=Path(authority_path).resolve();authority=json.loads(authority_path.read_text())
    home=FINITE if finite else STRUCTURAL
    binding=finite_binding() if finite else structural_binding();lp=home/'ledger.json'
    ledger=json.loads(lp.read_text()) if lp.exists() else None
    need(ledger is None or ledger['status']=='OPEN','CLOSED structural authority cannot be reused')
    need(authority['status']=='AUTHORIZED' and authority['sole_slot_released'] is True,'fresh structural numerical release required')
    if ledger is None:
        need(authority['binding']==binding,'initial structural source/input binding mismatch')
        ledger=dict(status='OPEN',authority_sha256=sha(authority_path),used_s=0.,current_attempt=None,attempts=[])
    need(ledger['authority_sha256']==sha(authority_path) and ledger['current_attempt'] is None,'idle same cumulative structural authority required')
    need(authority['reserve_s']>=60. and authority['memory_cap_bytes']<=2147483648,'structural reserve/memory boundary changed')
    cap=min(authority['cap_s'],authority['budget_s']-authority['reserve_s']-ledger['used_s'])
    need(cap>0 and 0<authority['cap_s']<=45.,'structural process/reserve exhausted')
    need(attempt.isascii() and attempt.replace('-','').isalnum(),'simple unique structural attempt required')
    out=home/('attempt-'+attempt);need(not out.exists(),'preserve all candidate attempts');out.mkdir(parents=True)
    snapshots=out/'source-snapshots';snapshots.mkdir()
    for rel in binding['sources']:(snapshots/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='1',NUMBA_DISABLE_JIT='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        PYTEST_ADDOPTS='',PYTHONPATH=str(ROOT),F07_STRUCTURAL_SUPERVISED=sha(authority_path))
    if finite:env['F07_FINITE_SUPERVISED']=sha(authority_path)
    command=[sys.executable,str(Path(__file__).resolve()),'--finite-child' if finite else '--structural-child',str(out), '--authority',str(authority_path)]
    launch=dict(status='launching',command=command,binding=binding,authority_sha256=sha(authority_path),
        environment={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS',
            'NUMBA_DISABLE_JIT','PYTHONOPTIMIZE','PYTHONPATH')},cap_s=cap,peak_tree_rss_bytes=0)
    ledger['current_attempt']=out.name;write(out/'launch.json',launch);write(lp,ledger)
    start=time.perf_counter_ns();proc=None;code=None;failure=None
    try:
        with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            process=psutil.Process(proc.pid);process.cpu_affinity([0])
            while proc.poll() is None:
                try:members=[process]+process.children(recursive=True)
                except psutil.NoSuchProcess:members=[]
                rss=0
                for member in members:
                    try:rss+=member.memory_info().rss
                    except psutil.NoSuchProcess:pass
                launch['peak_tree_rss_bytes']=max(launch['peak_tree_rss_bytes'],rss)
                if rss>authority['memory_cap_bytes']:failure='MEMORY_LIMIT'
                elif (time.perf_counter_ns()-start)/1e9>=cap:failure='TIMEOUT'
                if failure:
                    for member in reversed(members):
                        try:member.kill()
                        except psutil.NoSuchProcess:pass
                    break
                time.sleep(.01)
            code=proc.wait()
    except BaseException as exc:
        failure='LAUNCH_OR_SUPERVISION_FAILURE';launch['exception']=repr(exc)
        if proc is not None and proc.poll() is None:
            try:members=[psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True)
            except psutil.NoSuchProcess:members=[]
            for member in reversed(members):
                try:member.kill()
                except psutil.NoSuchProcess:pass
            proc.wait()
    finally:
        wall=(time.perf_counter_ns()-start)/1e9;ledger['used_s']+=wall
        launch.update(status='PASS' if code==0 and failure is None else 'FAIL',exit_code=code,failure=failure,
            wall_s=wall,cumulative_wall_s=ledger['used_s'],remaining_s=authority['budget_s']-ledger['used_s'])
        ledger['attempts'].append(dict(attempt=attempt,status=launch['status'],wall_s=wall))
        ledger['current_attempt']=None
        for name in ('stdout.log','stderr.log','runtime-before.json','runtime-after.json','junit.xml','child-status.json'):
            if (out/name).exists():launch[name+'_sha256']=sha(out/name)
        write(out/'receipt.json',launch);write(lp,ledger)
    print(json.dumps({k:launch[k] for k in ('status','wall_s','cumulative_wall_s','remaining_s')}))
    return int(launch['status']!='PASS')


def structural_close(authority_path,finite=False):
    lp=(FINITE if finite else STRUCTURAL)/'ledger.json';ledger=json.loads(lp.read_text());authority=json.loads(Path(authority_path).read_text())
    need(ledger['status']=='OPEN' and ledger['current_attempt'] is None and ledger['authority_sha256']==sha(authority_path),
         'idle same OPEN structural ledger required')
    ledger.update(status='CLOSED',unused_discarded_s=authority['budget_s']-ledger['used_s']);write(lp,ledger)


def run_control(name,out):
    import numpy as np
    from scipy.linalg import block_diag,eig
    if name=='nearby-not-exact':
        from fractions import Fraction
        epsilon=1e-15;A=np.array([[0.,1.,0.],[1.,epsilon,0.],[0.,0.,1.]]);B=np.diag([1.,0.,0.])
        np.savez(out/'control-input.npz',A=A,B=B,epsilon=epsilon)
        proof=block_infinity(A,B,out);raw=eig(A,B,left=False,right=False,homogeneous_eigvals=True)
        np.savez(out/'raw-control-qz.npz',alpha=raw[0],beta=raw[1]);homogeneous_valid(raw)
        # det(A-lambda B)=-1-epsilon*lambda exactly for this binary input.
        exact_epsilon=Fraction.from_float(epsilon);exact_root=-1/exact_epsilon
        need(exact_epsilon!=0 and proof['nearby_infinite']==3,'nearby/stored scope discriminator wrong')
        write(out/'result.json',dict(status='PASS',exact_stored_finite=1,exact_stored_infinite=2,
            exact_determinant_coefficients=['-1',str(-exact_epsilon)],stored_huge_finite=float(exact_root),
            raw_qz_exact_beta_zero=int(np.count_nonzero(raw[1]==0)),nearby_infinite=proof['nearby_infinite'],
            scope='exact degree1 determinant vs nearby degree0; rawQZ not exact arithmetic authority'));return
    if name=='nonsymmetric':
        smallA=np.array([[2.,0.,1.],[0.,1.,1.],[1.,0.,0.]]);smallB=np.diag([1.,0.,0.])
    else:smallA=np.array([[0.,1.],[1.,0.]]);smallB=np.diag([1.,0.])
    A=block_diag(np.diag([0.]*6+[1.,2.]),smallA);B=block_diag(np.eye(8),smallB)
    if name=='rank-ambiguity':
        A=block_diag(A,np.ones((1,1)));B=block_diag(B,np.zeros((1,1)));A[-2,-2]=1e-12
    R=np.zeros((len(A),6));R[:6]=np.eye(6)
    if name=='seventh':A[6,6]=0.
    elif name=='ambiguous-zero':A[6,6]=1e-13
    elif name=='negative-inertia':B[0,0]=-1.
    elif name=='tiny-inertia':B[0,0]=1e-15
    elif name=='singular-coupling':A[-1,:]=0.;A[:,-1]=0.
    elif name=='indeterminate':A[-1,:]=0.;A[:,-1]=0.;B[-1,:]=0.;B[:,-1]=0.
    elif name=='defective-zero':
        A[:2,:2]=np.array([[0.,1.],[0.,0.]])
    np.savez(out/'control-input.npz',A=A,B=B,R=R)
    expected=dict(seventh='zero nullity is not exactly six',
        **{'ambiguous-zero':'zero nullity is not exactly six','negative-inertia':'nonpositive physical inertia',
        'tiny-inertia':'unresolved physical inertia','rank-ambiguity':'unresolved algebraic rank gap',
        'singular-coupling':'algebraic null coupling','indeterminate':'algebraic null coupling',
        'defective-zero':'zero cluster cross-overlap'})
    try:
        if name=='indeterminate':
            raw=eig(A,B,left=False,right=False,homogeneous_eigvals=True)
            np.savez(out/'raw-control-qz.npz',alpha=raw[0],beta=raw[1])
            homogeneous_valid(raw)
            raise AssertionError('indeterminate pair admitted')
        if name in ('negative-inertia','tiny-inertia','rank-ambiguity','singular-coupling'):
            block_infinity(A,B,out);raise AssertionError('negative block control admitted')
        As,Bs,D,freq=balance(A,B,out)
        raw,left,right=eig(As,Bs,left=True,right=True,homogeneous_eigvals=True)
        result=classify_saved(dict(A=A,B=B,As=As,Bs=Bs,D=D,frequency=freq),
            dict(alpha=raw[0],beta=raw[1],left=left,right=right),
            dict(canonical=R,M=B,C=np.zeros((0,len(A)))),out)
    except Unresolved as exc:
        if name=='indeterminate':message='indeterminate homogeneous 0/0 pair'
        else:message=expected.get(name)
        need(message is not None and message in str(exc),'control refused at unintended gate: '+str(exc))
        write(out/'result.json',dict(status='PASS',expected_refusal=str(exc)));return
    need(name in ('baseline','nonsymmetric'),'negative control unexpectedly admitted')
    need(result['finite_count']==8 and result['proof']['nearby_infinite']==len(A)-8,'block theorem control counts wrong')


def run_installed(out):
    import numpy as np
    from anyfesolver import stiffened_shell as ss
    from examples.stiffened_shell_ge_mixed_reference_modes import definition
    saved=dict(np.load(OLD/'pencil.npz'));physical=dict(np.load(OLD/'original-physical.npz'))
    proof=json.loads((HERE/'source1/saved-current-public/result.json').read_text())
    owner=ss.create_analysis(definition=definition('mixed','biaxial','mixed','point',free=True))
    try:
        initial=(owner._identity,owner._generation,owner._K.copy(),owner._C.copy())
        result=owner.modes(ss.GEMixedReferenceModesRequest(count=2))
        public=result.flexible
        write(out/'public-status-before-proof.json',dict(status=result.status,failure_reason=result.failure_reason,
            model_identity=result.model_identity,flexible_model_identity=public.model_identity,
            mass_identity=public.mass_identity,policy=result.policy))
        metrics=dict(status=result.status,failure_reason=result.failure_reason,policy=result.policy,
            model_identity=result.model_identity,flexible_model_identity=public.model_identity,mass_identity=public.mass_identity,
            values=list(public.eigenvalues_rad2_s2),frequencies=list(public.frequencies_hz),
            equation=list(public.equation_residuals),physical_equation=list(public.physical_equation_residuals),
            constraint=list(public.constraint_residuals),rigid_equation=list(result.rigid_equation_residuals),
            rigid_constraint=list(result.rigid_constraint_residuals),rigid_flexible_overlap=result.rigid_flexible_mass_overlap)
        write(out/'public-metrics-before-proof.json',metrics)
        np.savez(out/'public-coordinates-before-proof.npz',coordinates=np.asarray(public.full_coordinates),
            multipliers=np.asarray(public.multiplier_modes),rigid=np.asarray(result.rigid_coordinates),
            rigid_multipliers=np.asarray(result.rigid_multipliers),inverse_eigenvalues=np.asarray(public.inverse_eigenvalues))
        need(result.status=='completed','installed public sparse caller failed')
        need(result.model_identity==public.model_identity==owner._identity,'free/flexible owner identity mismatch')
        mass,zero,massid=owner._mixed_reference_modal_mass
        A=owner._A.toarray();K=owner._K.toarray();C=owner._C.toarray();M=mass.toarray()
        B=np.zeros_like(A);B[:len(M),:len(M)]=M
        np.savez(out/'installed-original-matrices.npz',A=A,B=B,K=K,C=C,M=M,canonical=canonical(owner))
        parity={name:bool(np.array_equal(actual,expected)) for name,actual,expected in
            (('A',A,saved['A']),('B',B,saved['B']),('K',K,physical['K']),('M',M,physical['M']),('C',C,physical['C']))}
        write(out/'matrix-parity.json',parity);need(all(parity.values()),'installed/source full physical pencil differs')
        old=json.loads((OLD/'public-status.json').read_text())
        need(owner._identity==old['identity'],'installed fixture identity changed')
        blocks=json.loads((OLD/'blocks.json').read_text())
        need(list(zero)==blocks['declared_zero'] and massid==blocks['mass_identity'],'physical mass identity/zero channels changed')
        relative=abs(np.asarray(metrics['values'])-proof['first_eigenvalues'])/abs(np.asarray(metrics['values']))
        metrics.update(relative_pair_error=relative.tolist(),source_proof_sha256=sha(HERE/'source1/saved-current-public/result.json'),
            nearby_infinite=proof['proof']['nearby_infinite'],finite_count=proof['finite_count'],nearzero_count=proof['nearzero_count'])
        write(out/'comparison-before-guards.json',metrics)
        need(np.max(relative)<=1e-8,'installed first flexible pair mismatch')
        need(max(metrics['equation']+metrics['physical_equation'])<=1e-8 and max(metrics['constraint'])<=1e-9,'installed public physical residual failed')
        need(max(metrics['rigid_equation'])<=1e-8 and max(metrics['rigid_constraint'])<=1e-9 and metrics['rigid_flexible_overlap']<=1e-7,'installed public rigid guard failed')
        Q=np.column_stack(result.rigid_coordinates);R=physical['canonical'][:len(M)]
        gram=R.T@M@R;L=np.linalg.cholesky(gram);R=R@np.linalg.solve(L.T,np.eye(6))
        overlap=np.linalg.svd(R.T@M@Q,compute_uv=False)
        np.savez(out/'installed-rigid-overlap.npz',overlap=overlap,canonical=R,public=Q)
        need(np.max(abs(overlap-1))<=1e-7,'installed rigid subspace mismatch')
        need(owner._identity==initial[0] and owner._generation==initial[1]==0 and owner._accepted is owner._finite_driver is None and not owner._writing and (owner._K!=initial[2]).nnz==(owner._C!=initial[3]).nnz==0,'installed virgin owner mutated')
        metrics['status']='PASS';write(out/'result.json',metrics);return metrics
    finally:owner.close()


CONTROLS=('baseline','nonsymmetric','seventh','ambiguous-zero','negative-inertia',
    'tiny-inertia','rank-ambiguity','singular-coupling','indeterminate','defective-zero','nearby-not-exact')


def prepare():
    start=time.perf_counter_ns();HERE.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    need(not (DOC/'registration.json').exists(),'registration exists')
    paths=('scripts/r8_f07_block_descriptor_dev.py','tests/test_r8_f07_block_descriptor_dev.py',
        'scripts/r8_f07_public_descriptor_dev.py','examples/stiffened_shell_ge_mixed_reference_modes.py',
        'examples/stiffened_shell_native_fibre_bl5.py','tests/test_stiffened_shell_physical_s3_mass.py')
    candidates={rel:sha(ROOT/rel) for rel in paths}
    for rel in paths:(DOC/('frozen-'+Path(rel).name)).write_bytes((ROOT/rel).read_bytes())
    inputs={str(OLD/name):sha(OLD/name) for name in ('pencil.npz','homogeneous-qz.npz','original-physical.npz','public-status.json','blocks.json')}
    source={p.relative_to(ROOT/'src').as_posix():sha(p) for p in (ROOT/'src/anyfesolver').rglob('*.py')}
    artifact=ROOT/'.development/r8-f13-dependency-repair-20261009/artifact-v2'
    need(all(sha(artifact/'site'/rel)==h for rel,h in source.items()),'source/site mismatch')
    manifest=json.loads((artifact/'artifact-binding.json').read_text());wheel=artifact/'dist/anysolver-0.4.7-py3-none-any.whl'
    need(sha(wheel)==manifest['wheel_sha256'],'wheel changed')
    (DOC/'bound-artifact-manifest.json').write_bytes((artifact/'artifact-binding.json').read_bytes())
    (DOC/'frozen-TASK.md').write_bytes((DOC/'TASK.md').read_bytes())
    env=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMBA_DISABLE_JIT='1',PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',NUMEXPR_NUM_THREADS='1')
    reg=dict(status='CANDIDATE_HOLD',root_go=False,python=str(Path(sys.executable).resolve()),python_sha256=sha(sys.executable),
        candidates=candidates,inputs=inputs,source=source,contract_sha256=sha(DOC/'TASK.md'),
        wheel=str(wheel),wheel_sha256=sha(wheel),source_site=str(ROOT/'src'),installed_site=str(artifact/'site'),
        environment=env,budget_s=180.,reserve_s=60.,cap_s=45.,memory_cap_bytes=2147483648,
        cases=dict(source=list(CONTROLS)+['saved-current-public'],installed=['installed-current-public',
            's3-barycentric-thickness-work','s3-covariance-rigid-work']+['s3-physical-refusal-'+str(i) for i in range(6)]),
        commands={stage:[str(Path(sys.executable).resolve()),str(Path(__file__).resolve()),'--child',stage,'--site',str(ROOT/'src' if stage=='source' else artifact/'site'),'--out',str(HERE/(stage+'1'))] for stage in ('source','installed')})
    write(DOC/'registration.json',reg);write(DOC/'prepare-io.json',dict(wall_s=(time.perf_counter_ns()-start)/1e9,scope='stdlib hashes/copies only'))


def freeze_guard(reg,stage,selected):
    need(reg['status']=='FROZEN_BY_ROOT' and reg['root_go'] is True,'root freezeGO missing')
    need(stage in ('source','installed') and selected==Path(reg[stage+'_site']).resolve(),'stage/site mismatch')
    need(sha(sys.executable)==reg['python_sha256'] and str(Path(sys.executable).resolve())==reg['python'],'Python identity changed')
    need(sha(DOC/'TASK.md')==reg['contract_sha256'],'contract changed')
    for rel,h in reg['candidates'].items():need(sha(ROOT/rel)==h,'candidate changed: '+rel)
    need(PRIOR_SHA==reg['candidates']['scripts/r8_f07_public_descriptor_dev.py'],'loaded prior helper SHA changed')
    for path,h in reg['inputs'].items():need(sha(path)==h,'saved input changed: '+path)
    for rel,h in reg['source'].items():need(sha(ROOT/'src'/rel)==sha(selected/rel)==h,'source/site changed: '+rel)
    need(sha(reg['wheel'])==reg['wheel_sha256'],'bound wheel changed')


def child(stage,site,out):
    import traceback
    reg=json.loads((DOC/'registration.json').read_text());selected=Path(site).resolve();out=Path(out).resolve()
    freeze_guard(reg,stage,selected)
    need(os.environ.get('F07_BLOCK_SUPERVISED')==sha(DOC/'registration.json'),'supervisor-only marker missing')
    need(str(out)==reg['commands'][stage][-1],'unregistered output')
    ledger=json.loads((HERE/'ledger.json').read_text());launch=json.loads((out/'launch.json').read_text())
    need(ledger['status']=='OPEN' and launch['status']=='launching' and launch['registration_sha256']==sha(DOC/'registration.json'),'inactive supervised launch')
    if stage=='installed':need(sha(HERE/'source1/saved-current-public/result.json')==launch['source_result_sha256'],'source proof changed')
    sys.path[:0]=[str(selected),str(ROOT)];os.environ['F07_BLOCK_OUTPUT']=str(out);os.environ['F07_BLOCK_STAGE']=stage
    import psutil
    psutil.Process().cpu_affinity([0]);runtime(out,'runtime-before',selected)
    import pytest
    test=str(ROOT/'tests/test_r8_f07_block_descriptor_dev.py')
    tests=[test+'::test_installed_public_bridge',str(ROOT/'tests/test_stiffened_shell_physical_s3_mass.py')] if stage=='installed' else [test+'::test_independent_block_controls',test+'::test_saved_current_public']
    code=None
    try:
        code=int(pytest.main(tests+['-o','pythonpath=','-q','-x','-p','no:cacheprovider','--junitxml',str(out/'junit.xml'),'--basetemp',str(out/'pytest-tmp')]))
        return code
    except BaseException as exc:
        write(out/'child-failure.json',dict(exception=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        runtime(out,'runtime-after',selected);write(out/'child-status.json',dict(exit_code=code,stage=stage))


def supervise(stage):
    import psutil,subprocess
    reg=json.loads((DOC/'registration.json').read_text());selected=Path(reg[stage+'_site']).resolve();freeze_guard(reg,stage,selected)
    lp=HERE/'ledger.json';ledger=json.loads(lp.read_text()) if lp.exists() else dict(status='OPEN',budget_s=180.,reserve_s=60.,used_s=0.,attempts=[])
    need(ledger['status']=='OPEN','closed ledger')
    if stage=='source':need(not ledger['attempts'],'source already attempted')
    else:need(len(ledger['attempts'])==1 and ledger['attempts'][0]['stage']=='source' and ledger['attempts'][0]['status']=='PASS','source gate not passed once')
    cap=min(reg['cap_s'],reg['budget_s']-reg['reserve_s']-ledger['used_s']);need(cap>0,'budget/reserve exhausted')
    out=HERE/(stage+'1');out.mkdir();snap=out/'source-snapshots';snap.mkdir()
    for rel in reg['candidates']:(snap/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    for name in ('registration.json','TASK.md'):(snap/name).write_bytes((DOC/name).read_bytes())
    inputs=out/'bound-inputs';inputs.mkdir()
    for path in reg['inputs']:(inputs/Path(path).name).write_bytes(Path(path).read_bytes())
    command=reg['commands'][stage];env=os.environ.copy();env.update(reg['environment']);env['F07_BLOCK_SUPERVISED']=sha(DOC/'registration.json')
    receipt=dict(status='launching',stage=stage,command=command,registration_sha256=sha(DOC/'registration.json'),
        environment=reg['environment'],selected_site=str(selected),cap_s=cap,prior_used_s=ledger['used_s'],peak_tree_rss_bytes=0,
        source_result_sha256=sha(HERE/'source1/saved-current-public/result.json') if stage=='installed' else None)
    write(out/'launch.json',receipt);write(lp,ledger)
    start=time.perf_counter_ns();proc=None;code=None;failure=None;error=None
    try:
        with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
            psutil.Process().cpu_affinity([0]);proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            process=psutil.Process(proc.pid);process.cpu_affinity([0])
            while proc.poll() is None:
                try:members=[process]+process.children(recursive=True)
                except psutil.NoSuchProcess:members=[]
                rss=0
                for member in members:
                    try:rss+=member.memory_info().rss
                    except psutil.NoSuchProcess:pass
                receipt['peak_tree_rss_bytes']=max(receipt['peak_tree_rss_bytes'],rss)
                if rss>reg['memory_cap_bytes']:failure='MEMORY_LIMIT'
                elif (time.perf_counter_ns()-start)/1e9>=cap:failure='TIMEOUT'
                if failure:
                    for member in reversed(members):
                        try:member.kill()
                        except psutil.NoSuchProcess:pass
                    break
                time.sleep(.01)
            code=proc.wait()
    except BaseException as exc:
        error=exc;failure='LAUNCH_OR_SUPERVISION_FAILURE';receipt['exception']=repr(exc)
        if proc is not None and proc.poll() is None:
            try:members=[psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True)
            except psutil.NoSuchProcess:members=[]
            for member in reversed(members):
                try:member.kill()
                except psutil.NoSuchProcess:pass
            code=proc.wait()
    finally:
        wall=(time.perf_counter_ns()-start)/1e9;ledger['used_s']+=wall
        receipt.update(status='PASS' if code==0 and failure is None else 'FAIL',exit_code=code,failure=failure,
            wall_s=wall,cumulative_wall_s=ledger['used_s'],remaining_s=reg['budget_s']-ledger['used_s'])
        ledger['attempts'].append(dict(stage=stage,status=receipt['status'],wall_s=wall))
        if stage=='installed' or receipt['status']=='FAIL':ledger.update(status='CLOSED',unused_discarded_s=reg['budget_s']-ledger['used_s'])
        write(out/'receipt.json',receipt);write(lp,ledger)
    print(json.dumps({k:receipt[k] for k in ('status','wall_s','cumulative_wall_s','remaining_s')}))
    if error:raise error
    return int(receipt['status']!='PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',choices=('source','installed'))
    p.add_argument('--child',choices=('source','installed'));p.add_argument('--site');p.add_argument('--out')
    p.add_argument('--structural-prepare');p.add_argument('--structural-run');p.add_argument('--structural-child')
    p.add_argument('--structural-close',action='store_true');p.add_argument('--authority')
    p.add_argument('--finite-prepare');p.add_argument('--finite-run');p.add_argument('--finite-child')
    p.add_argument('--finite-close',action='store_true');a=p.parse_args()
    if a.finite_prepare:finite_prepare(a.finite_prepare)
    elif a.finite_child:raise SystemExit(structural_child(a.finite_child,a.authority,finite=True))
    elif a.finite_run:raise SystemExit(structural_supervise(a.authority,a.finite_run,finite=True))
    elif a.finite_close:structural_close(a.authority,finite=True)
    elif a.structural_prepare:structural_prepare(a.structural_prepare)
    elif a.structural_child:raise SystemExit(structural_child(a.structural_child,a.authority))
    elif a.structural_run:raise SystemExit(structural_supervise(a.authority,a.structural_run))
    elif a.structural_close:structural_close(a.authority)
    elif a.prepare:prepare()
    else:raise SystemExit(child(a.child,a.site,a.out) if a.child else supervise(a.run))
