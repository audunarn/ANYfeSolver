import numpy as np
from anyfesolver._ge_beam3_p5.arrays import _array
from anyfesolver._native_reference_modal import _owned
from anyfesolver._ge_beam3_p5.algebra import skew
from anyfesolver._ge_beam3_g3c_stable.chart import exp_chart_terms


def spatial_jacobian(conservative_residual, conservative_hessian):
    """Derivative of Rc-G, where G is fixed in spatial residual components.

    Rc is the conservative material-minus-dead-line-work residual, NOT Rc-G.
    Hc is its Exp-chart Hessian. D G=0 for this reference spatial load policy.
    """
    residual = _array(conservative_residual, (42,), 'conservative residual')
    jacobian = _array(conservative_hessian, (42, 42), 'conservative Hessian')
    for start in (3, 9, 15, 18, 21):
        jacobian[start:start+3, start:start+3] -= .5*skew(residual[start:start+3])
    return _owned(jacobian)


def chart_pullback(force, jacobian, increments):
    """Pull back a general spatial condensed Jacobian, without symmetrizing."""
    force = _array(force, (18,), 'condensed spatial force')
    jacobian = _array(jacobian, (18, 18), 'condensed spatial Jacobian')
    increments = _array(increments, (3, 3), 'nodal rotation chart increments')
    chart = np.eye(18)
    extra = np.zeros((18, 18))
    for node, vector in enumerate(increments):
        start = 6*node+3
        a, da = exp_chart_terms(vector)
        chart[start:start+3, start:start+3] = a
        for k in range(3):
            extra[start:start+3, start+k] = da[:, :, k].T@force[start:start+3]
    result = chart.T@force
    tangent = chart.T@jacobian@chart+extra
    if not np.isfinite(result).all() or not np.isfinite(tangent).all():
        raise ValueError('nonfinite distributed couple chart response')
    return _owned(result), _owned(tangent), _owned(chart)
