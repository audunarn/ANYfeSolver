"""Increment-resolved evaluation of the SAME private Procrustes Q4 chart.

Authority: GE_BEAM3_Q4_AFFINE_CANCELLATION_SAFE_CHART_ADDENDUM.md.
Original Davenport eigenstationarity and analytic normalized eigen derivatives;
no independent-checker import, fit replacement, finite differences or fallback.
"""
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
import numpy as np
from ._ge_beam3_mixed_ad import (Jet2, matmul, matvec, transpose, skew, identity,
    constant_matrix, unary, math_cos_limit)
from ._ge_beam3_g3c_so3_numerics import _exp_coefficients, _log_factor, _polynomial, LOG
from ._ge_beam3_g3c_local_shell import Kinematics, owned, array, rotations, canonical
from ._ge_beam3_g3c_affine_q4_chart import _exp_terms

NUMERICS_ID = 'GE_BEAM3_Q4_AFFINE_INCREMENT_RESOLVED_CHART_NUMERICS_V1'
PRECISION = 80
MAX_ITERATIONS = 16
RESIDUAL_LIMIT = Decimal('1e-60')
ORTHOGONALITY_LIMIT = Decimal('1e-70')

class ChartEvaluationError(ValueError):
    pass

class ChartNonconvergence(ChartEvaluationError):
    pass

class ChartBranchError(ChartEvaluationError):
    pass

def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)),Decimal(0))

def _norm(values):
    return _dot(values,values).sqrt()

def _solve(matrix,rhs):
    """Deterministic largest-absolute pivot, lowest current row on ties."""
    n=len(matrix); a=[list(row)+[rhs[i]] for i,row in enumerate(matrix)]
    for col in range(n):
        pivot=max(range(col,n),key=lambda i:abs(a[i][col]))
        if not a[pivot][col]: raise ChartEvaluationError('zero Decimal chart pivot')
        a[col],a[pivot]=a[pivot],a[col]
        for row in range(col+1,n):
            multiplier=a[row][col]/a[col][col]
            for j in range(col+1,n+1): a[row][j]-=multiplier*a[col][j]
            a[row][col]=Decimal(0)
    x=[Decimal(0)]*n
    for i in range(n-1,-1,-1):
        x[i]=(a[i][n]-_dot(a[i][i+1:n],x[i+1:]))/a[i][i]
    return x

def _davenport(A):
    trace=sum((A[i][i] for i in range(3)),Decimal(0))
    z=[A[2][1]-A[1][2],A[0][2]-A[2][0],A[1][0]-A[0][1]]
    return [[trace]+z]+[[z[i]]+[A[i][j]+A[j][i]-(trace if i==j else 0)
                              for j in range(3)] for i in range(3)]

def _decimal_covariance(reference,translations):
    """Return centered exact-input X/u, normalized Davenport K and all24 K_i."""
    D=Decimal.from_float; n=len(reference)
    X=[[D(float(v)) for v in row] for row in reference]
    U=[[D(float(v)) for v in row] for row in translations]
    xc=[[row[j]-sum((r[j] for r in X),Decimal(0))/n for j in range(3)] for row in X]
    uc=[[row[j]-sum((r[j] for r in U),Decimal(0))/n for j in range(3)] for row in U]
    beta=sum((_dot(row,row) for row in xc),Decimal(0))
    if not beta.is_finite() or beta<=0: raise ChartEvaluationError('positive reference covariance scale required')
    baseline=[[Decimal(0)]*3 for _ in range(3)]
    for i in range(3):
        for j in range(i,3):
            baseline[i][j]=baseline[j][i]=sum((row[i]*row[j] for row in xc),Decimal(0))
    delta=[[sum((u[i]*x[j] for u,x in zip(uc,xc)),Decimal(0)) for j in range(3)] for i in range(3)]
    A=[[(baseline[i][j]+delta[i][j])/beta for j in range(3)] for i in range(3)]
    derivatives=[]
    for node in range(n):
        for component in range(6):
            ai=[[Decimal(0)]*3 for _ in range(3)]
            if component<3: ai[component]=[v/beta for v in xc[node]]
            derivatives.append(_davenport(ai))
    return xc,uc,_davenport(A),derivatives

def _orthogonal_seed(seed_vectors):
    basis=[]
    for col in range(4):
        v=[Decimal.from_float(float(seed_vectors[i,col])) for i in range(4)]
        for _ in range(2):
            for earlier in basis:
                projection=_dot(v,earlier)
                v=[a-projection*b for a,b in zip(v,earlier)]
        length=_norm(v)
        if not Decimal('.5')<=length<=Decimal(2):
            raise ChartBranchError('defective eigensolver basis')
        basis.append([v0/length for v0 in v])
    error=max(abs(_dot(a,b)-int(i==j)) for i,a in enumerate(basis) for j,b in enumerate(basis))
    if error>ORTHOGONALITY_LIMIT: raise ChartBranchError('Decimal seed orthogonality failed')
    return basis

def _certify_branch(K,q,lam,basis):
    normK=_norm([v for row in K for v in row])
    if normK<=0: raise ChartBranchError('zero Davenport scale')
    budget=RESIDUAL_LIMIT*normK
    T=[[_dot(a,[_dot(row,b) for row in K]) for b in basis] for a in basis]
    intervals=[]
    for i,row in enumerate(T):
        radius=sum((abs(v) for j,v in enumerate(row) if j!=i),Decimal(0))+budget
        intervals.append((row[i]-radius,row[i]+radius))
    others=max(v[1] for v in intervals[:3])
    if intervals[3][0]<=others: raise ChartBranchError('dominant Gershgorin interval is not isolated')
    residual=[_dot(row,q)-lam*q[i] for i,row in enumerate(K)]
    qnorm=_norm(q)
    if qnorm==0 or abs(_dot(q,q)-1)>RESIDUAL_LIMIT:
        raise ChartBranchError('refined quaternion normalization failed')
    rho=_norm(residual)/qnorm+budget
    if lam-rho<=others or lam+rho<intervals[3][0] or lam-rho>intervals[3][1]:
        raise ChartBranchError('refined root is not certified dominant')
    return dict(eigen_residual=str(_norm(residual)/normK),normalization_residual=str(abs(_dot(q,q)-1)),
                interval_low=str(intervals[3][0]),interval_high=str(intervals[3][1]),radius=str(rho))

def _refine_eigenpair(K,seed_values,seed_vectors):
    basis=_orthogonal_seed(seed_vectors); q=list(basis[-1])
    lam=Decimal.from_float(float(seed_values[-1])); normK=_norm([v for row in K for v in row])
    if normK<=0: raise ChartEvaluationError('positive Davenport norm required')
    for iteration in range(MAX_ITERATIONS):
        residual=[_dot(row,q)-lam*q[i] for i,row in enumerate(K)]
        normalization=(_dot(q,q)-1)/2
        if _norm(residual)/normK<=RESIDUAL_LIMIT and abs(2*normalization)<=RESIDUAL_LIMIT:
            certificate=_certify_branch(K,q,lam,basis)
            certificate['iterations']=iteration+1
            return q,lam,certificate
        if iteration+1==MAX_ITERATIONS: break
        jacobian=[[K[i][j]-(lam if i==j else 0) for j in range(4)]+[-q[i]] for i in range(4)]
        jacobian.append(q+[Decimal(0)])
        step=_solve(jacobian,[-v for v in residual]+[-normalization])
        q=[q[i]+step[i] for i in range(4)]; lam+=step[4]
    raise ChartNonconvergence('fixed16-step Decimal Davenport refinement did not converge')

def _verify_eigen_derivatives(q,lam,K,ki,dq,ddq,first_lambda,second_lambda):
    """Actual eigenpair differential: all coordinates and ordered pairs.

    Scales are sums of physical term norms, never an absolute unit floor.
    Zero scale requires zero residual. No derivative correction is performed.
    """
    arrays=(q,K,ki,dq,ddq,first_lambda,second_lambda)
    if not np.isfinite(lam) or not all(np.isfinite(a).all() for a in arrays):
        raise ChartEvaluationError('nonfinite eigen-derivative witness')
    n=len(ki); norm=np.linalg.norm; nq=norm(q); nk=norm(K)
    metrics=dict(first_normalization=0.,second_normalization=0.,
                 first_stationarity=0.,second_stationarity=0.,second_symmetry=0.)
    def record(name,residual,scale):
        magnitude=float(norm(residual))
        error=magnitude/float(scale) if scale else (0. if magnitude==0. else float('inf'))
        if not np.isfinite(error) or error>1e-11:
            raise ChartEvaluationError('eigen-derivative identity failed: '+name)
        metrics[name]=max(metrics[name],error)
    operator=lam*np.eye(4)-K
    for i in range(n):
        qi=dq[:,i]; ni=norm(qi); ki_scale=norm(ki[i])+abs(first_lambda[i])
        ai=ki[i]-first_lambda[i]*np.eye(4)
        record('first_normalization',np.dot(q,qi),nq*ni)
        record('first_stationarity',operator@qi-ai@q,
               (abs(lam)+nk)*ni+ki_scale*nq)
        for j in range(n):
            qj=dq[:,j]; qij=ddq[:,i,j]; nj=norm(qj); nij=norm(qij)
            aj=ki[j]-first_lambda[j]*np.eye(4)
            record('second_normalization',np.dot(q,qij)+np.dot(qi,qj),nq*nij+ni*nj)
            residual=operator@qij-ai@qj-aj@qi+second_lambda[i,j]*q
            scale=((abs(lam)+nk)*nij+ki_scale*nj
                   +(norm(ki[j])+abs(first_lambda[j]))*ni+abs(second_lambda[i,j])*nq)
            record('second_stationarity',residual,scale)
            record('second_symmetry',qij-ddq[:,j,i],nij+norm(ddq[:,j,i]))
    return dict(derivative_coordinates=n,derivative_pairs=n*n,
                derivative_residuals=metrics)


def _rotation_jets_from_increments(reference,translations):
    """Return R Jets, (R-I) Jets, centered X/u arrays and certificate bytes."""
    with localcontext() as context:
        context.prec=PRECISION; context.rounding=ROUND_HALF_EVEN
        xc,uc,K,ki=_decimal_covariance(reference,translations)
        try:
            eigenvalues,vectors=np.linalg.eigh(np.array(K,dtype=np.float64))
        except np.linalg.LinAlgError as exc: raise ChartEvaluationError('Davenport seed eigensolver failed') from exc
        gap=eigenvalues[-1]-eigenvalues[-2]
        if not gap>1e-11*max(np.finfo(float).tiny,np.max(np.abs(eigenvalues))):
            raise ChartBranchError('nonunique proper shell fit; refine or cut back')
        q,lam,certificate=_refine_eigenpair(K,eigenvalues,vectors)
        bordered=np.array([[float((lam if i==j else 0)-K[i][j]) for j in range(4)]+[float(q[i])]
                           for i in range(4)]+[[float(v) for v in q]+[0.]],dtype=np.float64)
        qvalue=owned([float(v) for v in q]); derivative=owned(ki)
        matrix=owned(K); eigenvalue=float(lam)
        centered_x=owned(xc); centered_u=owned(uc)
    first_lambda=np.einsum('a,iab,b->i',qvalue,derivative,qvalue)
    reduced=derivative-first_lambda[:,None,None]*np.eye(4)
    first_rhs=np.vstack((np.einsum('iab,b->ai',reduced,qvalue),np.zeros(len(derivative))))
    try:
        dq=np.linalg.solve(bordered,first_rhs)[:4]
        second_lambda=2*np.einsum('aj,iab,b->ij',dq,derivative,qvalue)
        term=np.einsum('iab,bj->aij',reduced,dq)
        second_rhs=term+term.transpose(0,2,1)-qvalue[:,None,None]*second_lambda[None,:,:]
        rhs=np.concatenate((second_rhs,(-dq.T@dq)[None,:,:]),axis=0)
        ddq=np.linalg.solve(bordered,rhs.reshape(5,-1))[:4].reshape(4,len(derivative),len(derivative))
    except np.linalg.LinAlgError as exc: raise ChartEvaluationError('analytic eigen-derivative solve failed') from exc
    certificate.update(_verify_eigen_derivatives(qvalue,eigenvalue,matrix,derivative,
                       dq,ddq,first_lambda,second_lambda))
    quaternion=[Jet2(qvalue[i],owned(dq[i]),owned(ddq[i])) for i in range(4)]
    cross=skew(quaternion[1:]); square=matmul(cross,cross)
    delta=[[2*quaternion[0]*cross[i][j]+2*square[i][j] for j in range(3)] for i in range(3)]
    unit=identity(len(derivative))
    rotation=[[unit[i][j]+delta[i][j] for j in range(3)] for i in range(3)]
    return rotation,delta,centered_x,centered_u,canonical(certificate)

def _exp_delta(vector):
    argument=sum((v*v for v in vector),Jet2.constant(0.,vector[0].gradient.size))
    a,b=_exp_coefficients(argument); cross=skew(vector); square=matmul(cross,cross)
    return [[a*cross[i][j]+b*square[i][j] for j in range(3)] for i in range(3)]

def _log_delta(delta_matrix):
    delta=-(delta_matrix[0][0]+delta_matrix[1][1]+delta_matrix[2][2])/2
    cosine=1-delta.value
    if not np.isfinite(cosine) or not math_cos_limit()<cosine:
        raise ChartBranchError('relative rotation must remain below0.9pi')
    if delta.value<=.5:
        factor=unary(delta,*_polynomial(LOG,delta.value))
    else:
        factor=_log_factor(1-delta)
    return [factor*v for v in ((delta_matrix[2][1]-delta_matrix[1][2])/2,
        (delta_matrix[0][2]-delta_matrix[2][0])/2,(delta_matrix[1][0]-delta_matrix[0][1])/2)]

def _translation_deformation(rotation,delta_rotation,centered_reference,centered_increment):
    return [a+b for a,b in zip(matvec(transpose(rotation),centered_increment),
                               matvec(transpose(delta_rotation),centered_reference))]

def deformation(reference,displacement,accepted_rotations):
    n=len(reference)
    if n not in (3,4): raise ValueError('registered shell topology required')
    ref=array(reference,(n,3)); u=array(displacement,(6*n,)); qa=rotations(accepted_rotations,n)
    if np.any(np.linalg.norm(u.reshape(n,6)[:,3:],axis=1)>=.9*np.pi):
        raise ValueError('trial shell increment requires cutback')
    rotation,deltaR,xc,uc,_=_rotation_jets_from_increments(ref,u.reshape(n,6)[:,:3])
    size=6*n; values=[]; trial_rotations=[]; unit=identity(size)
    for i in range(n):
        translation=[]
        for j in range(3):
            gradient=np.zeros(size); gradient[j::6]=-1/n; gradient[6*i+j]+=1
            translation.append(Jet2(uc[i,j],gradient,np.zeros((size,size))))
        values.extend(_translation_deformation(rotation,deltaR,
            [Jet2.constant(v,size) for v in xc[i]],translation))
        eta=[Jet2.variable(u[6*i+3+j],6*i+3+j,size) for j in range(3)]
        e=_exp_delta(eta); da=constant_matrix(qa[i]-np.eye(3),size); product=matmul(e,da)
        dtrial=[[da[a][b]+e[a][b]+product[a][b] for b in range(3)] for a in range(3)]
        drt=transpose(deltaR); cross=matmul(drt,dtrial)
        relative=[[drt[a][b]+dtrial[a][b]+cross[a][b] for b in range(3)] for a in range(3)]
        values.extend(_log_delta(relative))
        trial_rotations.append([[unit[a][b]+dtrial[a][b] for b in range(3)] for a in range(3)])
    return Kinematics(owned([v.value for v in values]),owned([v.gradient for v in values]),
        owned([v.hessian for v in values]),owned([[v.value for v in row] for row in rotation]),
        owned([[[v.value for v in row] for row in q] for q in trial_rotations]),ref,
        owned(ref+u.reshape(n,6)[:,:3]))
