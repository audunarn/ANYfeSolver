"""Nonclassifying elementary SO(3) evaluation diagnosis; no source patching."""
from decimal import Decimal, localcontext
from math import factorial
import json

import numpy as np
from anyfesolver import _ge_beam3_mixed_ad as ad


def oracle(kind, value, terms):
    """Independent convergent scalar series, differentiated term by term."""
    with localcontext() as context:
        context.prec = 90
        x = Decimal.from_float(value)
        coefficient = Decimal(1)
        result = [Decimal(0)] * 3
        for n in range(terms):
            if kind == 'log':
                if n:
                    coefficient *= Decimal(n) / Decimal(2*n+1)
                argument = 1-x
            else:
                coefficient = Decimal((-1)**n) / Decimal(factorial(2*n+(1 if kind == 'sinc' else 2)))
                argument = x
            for derivative in range(3):
                if n >= derivative:
                    sign = (-1)**derivative if kind == 'log' else 1
                    result[derivative] += sign*coefficient*Decimal(factorial(n)//factorial(n-derivative))*argument**(n-derivative)
        return tuple(float(v) for v in result)


def emit(body):
    print('G3C_ELEMENTARY_DIAGNOSTIC '+json.dumps(body,sort_keys=True,allow_nan=False),flush=True)


def test_elementary_rotation_diagnostic():
    # Fixed grid spans both historical series switches and the failed small
    # chart. Decimal 40/60-term agreement checks the independent reference.
    for delta in (1e-12,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2):
        value = 1.0-delta
        reference = oracle('log',value,60)
        assert reference == oracle('log',value,40)
        actual = ad._log_factor(ad.Jet2.variable(value,0,1))
        output = (actual.value,actual.gradient[0],actual.hessian[0,0])
        emit(dict(kind='NONCLASSIFYING_LOG_FACTOR',cosine=value,reference=reference,
                  actual=output,absolute_errors=[abs(a-b) for a,b in zip(output,reference)]))
    for value in (1e-12,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2):
        actual = ad._exp_coefficients(ad.Jet2.variable(value,0,1))
        for kind,jet in zip(('sinc','cosc'),actual):
            reference = oracle(kind,value,60)
            assert reference == oracle(kind,value,40)
            output = (jet.value,jet.gradient[0],jet.hessian[0,0])
            emit(dict(kind='NONCLASSIFYING_EXP_COEFFICIENT',coefficient=kind,theta_squared=value,
                      reference=reference,actual=output,absolute_errors=[abs(a-b) for a,b in zip(output,reference)]))
    # Exact local identity Log(Exp(v))=v: no finite-difference oracle and no
    # alternate mechanics dispatched. Report value/gradient/Hessian separately.
    for magnitude in (1e-5,1e-4,1e-3,.003,.01,.1):
        vector = np.array([1.,-.7,.4])*magnitude
        jets = [ad.Jet2.variable(v,i,3) for i,v in enumerate(vector)]
        result = ad.so3_log(ad.so3_exp(jets))
        emit(dict(kind='NONCLASSIFYING_LOG_EXP_IDENTITY',magnitude=magnitude,
                  value_error=float(np.linalg.norm(np.array([j.value for j in result])-vector)),
                  gradient_error=float(np.linalg.norm(np.array([j.gradient for j in result])-np.eye(3))),
                  hessian_error=float(np.linalg.norm(np.array([j.hessian for j in result])))))
