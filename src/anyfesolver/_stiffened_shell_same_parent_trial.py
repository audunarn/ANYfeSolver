"""Single-use full trial handoff, restricted to elastic Q4/generalized points.

No tangent lag, constitutive cache, cross-increment reuse or checkpoint state.
Unsupported inputs take the ordinary complete evaluation route.
"""
from dataclasses import fields
from hashlib import sha256
from decimal import getcontext
from types import FunctionType
import sys
import numpy as np
from scipy.sparse import csc_matrix

from .e4_pl_element import QualifiedE4PLShellElement, _validate_prepared_q4_correction_inputs
from ._ge_beam3_retained_generalized import RetainedGeneralizedOperator
from ._ge_beam3_generalized_cell import GeneralizedCellHistory
from ._ge_beam3_generalized_ellipsoid_section import GeneralizedHistory
from ._ge_beam3_pose_joint import RigidPoseJoint
from ._ge_beam3_mixed_ad import Jet2
from ._stiffened_shell_point_recovery import PointTransferWrench
from ._stiffened_shell_q4_algorithmic_pullback import POLICY


class _UnsupportedSnapshot(Exception):
    pass


def _digest(value):
    """Byte binding of exact owned data; never invoke arbitrary serializers."""
    digest = sha256()
    def write(value):
        kind = type(value)
        if kind in (str, bytes, int, float, bool, type(None)):
            raw = value if kind is bytes else repr(value).encode('utf-8')
            digest.update(kind.__name__.encode('ascii'))
            digest.update(str(len(raw)).encode('ascii')+b':'+raw)
        elif kind is np.ndarray and value.dtype.kind in 'bifu':
            write((id(value), value.dtype.str, value.shape))
            digest.update(value.tobytes())
        elif kind is csc_matrix:
            write((id(value), value.shape, value.data, value.indices, value.indptr))
        elif kind in (list, tuple):
            digest.update(kind.__name__.encode('ascii'))
            write(len(value))
            for item in value:
                write(item)
        elif kind is dict:
            if any(type(key) not in (str, int) for key in value):
                raise _UnsupportedSnapshot()
            write(tuple((key, value[key]) for key in sorted(value, key=repr)))
        elif kind in (GeneralizedCellHistory, GeneralizedHistory, PointTransferWrench):
            write((kind.__name__, id(value)))
            for field in fields(kind):
                write(getattr(value, field.name))
        else:
            raise _UnsupportedSnapshot()
    write(value)
    return digest.digest()


def _eligible(driver):
    owner = driver.owner
    definition = owner._definition
    return (definition.q4_pullback_policy == POLICY and
        not definition.shell.s3_elements and definition.shell.j2_curve is None and
        not definition.lines and not definition.footprints and
        not any(mode in ('call', 'log', 'print') for mode in np.geterr().values()) and
        np.geterr()['under'] != 'warn' and
        driver._external is None and driver._programme is None and
        driver._adaptive_state is None and
        not getattr(owner, '_curved_reference', False) and
        all(type(element) is QualifiedE4PLShellElement and
            driver._q4_corrections.get(eid) is not None
            for eid, element, _slots, _origin in owner._shell_parts) and
        all(type(element.operator) is RetainedGeneralizedOperator
            for element, _slots, _origin in owner._beam_parts) and
        all(type(joint) is RigidPoseJoint for joint in owner._connections) and
        all(station.accumulated == (0., 0.)
            for history in driver._beam_origins for station in history.stations))


_DISPATCH_MODULES = (
    '_stiffened_shell_ge_finite_experimental', '_stiffened_shell_q4_algorithmic_pullback',
    '_stiffened_shell_algorithmic_pullback', '_ge_beam3_variational_shell',
    '_ge_beam3_mixed_ad', '_ge_beam3_retained_generalized', '_ge_beam3_generalized_cell',
    '_ge_beam3_generalized_ellipsoid_section', '_ge_beam3_station_resultant_cell',
    '_ge_beam3_pose_joint', '_ge_beam3_p5.algebra', '_ge_beam3_p5.compensated_coordinates',
    '_ge_beam3_p5.compensated', '_ge_beam3_p5.arrays',
    '_ge_beam3_generalized_static_boundary',
)


def _callable_binding(value):
    function = value.__func__ if type(value) in (staticmethod, classmethod) else value
    if type(function) is not FunctionType:
        return (id(value), None, None, None, None)
    code = function.__code__
    defaults = function.__defaults__
    keywords = function.__kwdefaults__
    # Keep the function/code alive; detect replaced bodies and defaults too.
    return (id(value), function if code is not None else None, code,
            id(defaults), None if keywords is None else
            tuple((key, id(item)) for key, item in keywords.items()))


def _dispatch_binding():
    return tuple((name, tuple((key, _callable_binding(value))
        for key, value in vars(sys.modules['anyfesolver.'+name]).items()
        if not key.startswith('__'))) for name in _DISPATCH_MODULES)


def _identity(driver, trial, pattern):
    owner = driver.owner
    decimal = getcontext()
    arithmetic = (decimal.prec, decimal.rounding, decimal.Emin, decimal.Emax,
        decimal.clamp, tuple((kind.__name__, value) for kind, value in decimal.traps.items()))
    operators = [element.operator for element, _slots, _origin in owner._beam_parts]
    classes = {type(operator) for operator in operators}
    classes.update(type(operator.cell) for operator in operators)
    classes.update(type(operator.section) for operator in operators)
    classes.add(RigidPoseJoint)
    classes.add(Jet2)
    classes.add(type(driver))
    # Rebinding a local operator implementation invalidates the handoff too.
    methods = tuple((cls, tuple((name, _callable_binding(value)) for name, value in cls.__dict__.items()))
                    for cls in sorted(classes, key=lambda cls: cls.__name__))
    return (id(driver), id(trial), id(driver._state), owner._generation,
        id(owner._definition), id(driver._shell_origins), id(driver._beam_origins),
        id(pattern), id(owner._shell_parts), id(owner._beam_parts),
        tuple((id(element), id(element.operator)) for element, _slots, _origin in owner._beam_parts),
        tuple((eid, id(element), id(driver._q4_corrections.get(eid)))
              for eid, element, _slots, _origin in owner._shell_parts),
        tuple(id(joint) for joint in owner._connections), methods, _dispatch_binding(),
        tuple(np.geterr().items()), id(np.geterrcall()), arithmetic)


def _inputs(driver, trial, factor, pattern):
    owner = driver.owner
    return (float(factor), pattern, driver._load_factor, driver._accepted_factors,
        driver._state.q, driver._state.frames, driver._state.cells,
        driver._state.resultants, driver._state.multipliers, driver._state.shell_anchors,
        driver._shell_predecessor_q, driver._shell_predecessor_anchors,
        trial.q, trial.frames, trial.cells,
        trial.resultants, trial.multipliers, trial.shell_anchors,
        driver._shell_origins, driver._beam_origins,
        driver._reference_positions, driver._reference_frames,
        owner._shell_xyz, owner._shell_count, owner._K.shape, driver._beam_nodal,
        owner._free, owner._coordinate_scale, owner._constraint_scale,
        owner._beam_ids, owner._beam_index, owner._shell_ids, owner._shell_index,
        tuple((point.shell_node, point.beam_node) for point in owner._point_incidence),
        tuple(slots for _eid, _element, slots, _origin in owner._shell_parts),
        tuple(slots for _element, slots, _origin in owner._beam_parts),
        owner._joint_slots)


def _contents(driver, trial, factor, pattern, evaluation):
    return (_inputs(driver, trial, factor, pattern), evaluation,
        driver._point_trial_recovery[1], driver._line_trial_recovery[1],
        driver._generalized_trial_recovery[2])


class FixedTrialHandoff:
    @classmethod
    def begin(cls, driver, trial, factor, pattern):
        if type(factor) not in (int, float) or not _eligible(driver):
            return None
        try:
            return (_identity(driver, trial, pattern),
                    _digest(_inputs(driver, trial, factor, pattern)))
        except _UnsupportedSnapshot:
            return None

    @classmethod
    def capture(cls, driver, trial, factor, pattern, evaluation, *, before):
        if (before is None or type(factor) not in (int, float) or
                not _eligible(driver) or any(station.accumulated != (0., 0.)
                for history in evaluation[5] for station in history.stations)):
            return None
        try:
            if before != (_identity(driver, trial, pattern),
                          _digest(_inputs(driver, trial, factor, pattern))):
                return None
            contents = _digest(_contents(driver, trial, factor, pattern, evaluation))
        except _UnsupportedSnapshot:
            return None
        made = cls()
        made._evaluation = evaluation
        made._identity = _identity(driver, trial, pattern)
        made._contents = contents
        made._recovery = (driver._point_trial_recovery, driver._line_trial_recovery,
                          driver._generalized_trial_recovery)
        return made

    def take(self, driver, trial, factor, pattern):
        evaluation, self._evaluation = self._evaluation, None
        if evaluation is None:
            return None
        if (type(factor) not in (int, float) or not _eligible(driver) or
                self._identity != _identity(driver, trial, pattern)):
            return None
        recovery = (driver._point_trial_recovery, driver._line_trial_recovery,
                    driver._generalized_trial_recovery)
        if any(old is not new for old, new in zip(self._recovery, recovery)):
            return None
        try:
            if self._contents != _digest(_contents(driver, trial, factor, pattern, evaluation)):
                return None
        except _UnsupportedSnapshot:
            return None
        owner = driver.owner
        owner._require_open()
        for eid, element, _slots, _origin in owner._shell_parts:
            material = owner._shell_model.get_material(element.material_name)
            _validate_prepared_q4_correction_inputs(element, owner._shell_model.mesh,
                                                  material, driver._q4_corrections[eid])
        for element, _slots, _origin in owner._beam_parts:
            element.operator.guard()
        for joint in owner._connections:
            joint.guard()
        # Guard observations may themselves expose a changed dependency.
        if (self._identity != _identity(driver, trial, pattern) or
                self._contents != _digest(_contents(driver, trial, factor, pattern, evaluation))):
            return None
        return evaluation
