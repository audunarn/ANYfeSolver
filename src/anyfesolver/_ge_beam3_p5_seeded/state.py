# V5 successor of the preserved V4 load-aware candidate. Mechanical expressions
# remain unchanged; authoritative operator seeding is bound by the V5 core.
"""Opt-in load parameter bound to an existing native trial transaction.

No global/thread-local load factor and no change to the generic state store.
Accepted parameter authority lives in every element's replay-validated state.
An assembly must be explicitly scoped; a missing parameter never means zero.
"""

from contextlib import contextmanager
import math
import threading

from anyfesolver.nonlinear_state import NonlinearStateStore, StateTransactionError


LOAD_POLICY = 'SPATIAL_DEAD_FORCE_PER_REFERENCE_ARCLENGTH_V1'


def parameter_value(value):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('finite exact float load parameter required')
    return value


class LoadStateStore(NonlinearStateStore):
    """Standalone scalar candidate only; exact token/parameter ownership."""

    def __init__(self):
        super().__init__()
        self._load_request = None
        self._load_trial = None

    @classmethod
    def from_shell_layouts(cls, layouts, committed_states=None):
        if layouts:
            raise StateTransactionError('load-aware candidate forbids shell batching')
        store = super().from_shell_layouts(layouts,committed_states)
        store.committed_load_parameter
        return store

    @contextmanager
    def load_parameter_scope(self, parameter):
        parameter = parameter_value(parameter)
        with self._lock:
            if self._load_trial is not None and self._load_trial[2] is not threading.current_thread():
                raise StateTransactionError('active load trial belongs to another thread')
            if self._load_request is not None:
                raise StateTransactionError('nested load parameter scope is forbidden')
            before = self._active_token
            # Hold the Thread object, not its recyclable operating-system ID.
            self._load_request = (parameter, threading.current_thread())
            try:
                yield
            except BaseException:
                if self.has_active_trial and self._active_token is not before:
                    self.discard_trial(self.active_trial_token())
                raise
            finally:
                self._load_request = None

    @property
    def committed_load_parameter(self):
        with self._lock:
            if not self._fallback_committed or self._batches:
                raise StateTransactionError('explicit scalar native load states required')
            try:
                values = [parameter_value(state['material_state']['load_parameter'])
                          for state in self._fallback_committed.values()]
            except (KeyError, TypeError, ValueError) as error:
                raise StateTransactionError('committed load parameter authority missing') from error
            if any(value != values[0] for value in values):
                raise StateTransactionError('committed element load parameters disagree')
            return values[0]

    def begin_trial(self, *, full_displacement=None, full_coordinates=None):
        with self._lock:
            if self._load_request is None or self._load_request[1] is not threading.current_thread():
                raise StateTransactionError('explicit owned load parameter scope required')
            self.committed_load_parameter
            token = super().begin_trial(full_displacement=full_displacement,full_coordinates=full_coordinates)
            self._load_trial = (token, *self._load_request)
            return token

    def native_load_parameter(self, token, element_id):
        with self._lock:
            self._require_active(token)
            item = self._load_trial
            if item is None or item[0] is not token or item[2] is not threading.current_thread():
                raise StateTransactionError('load parameter belongs to another trial or thread')
            if self._load_request is not None and self._load_request != item[1:]:
                raise StateTransactionError('active load parameter cannot be rebound')
            if type(element_id) is not int or element_id not in (self._native_element_bindings or {}):
                raise StateTransactionError('load parameter needs a bound native element')
            return item[1]

    def _validate_native_material_candidate(self, element_id, state, full):
        expected = self.native_load_parameter(self.active_trial_token(),element_id)
        try:
            actual = parameter_value(state['material_state']['load_parameter'])
        except (KeyError, TypeError, ValueError) as error:
            raise StateTransactionError('candidate load parameter missing or malformed') from error
        if actual != expected:
            raise StateTransactionError('candidate load parameter disagrees with its trial')
        super()._validate_native_material_candidate(element_id,state,full)

    def commit(self, token, *, accepted_full_displacement=None, accepted_full_coordinates=None):
        with self._lock:
            for element_id in self._fallback_committed:
                self.native_load_parameter(token,element_id)
            result = super().commit(token,accepted_full_displacement=accepted_full_displacement,
                accepted_full_coordinates=accepted_full_coordinates)
            self._load_trial = None
            return result

    def discard_trial(self, token):
        with self._lock:
            self._require_active(token)
            if self._load_trial is not None and self._load_trial[2] is not threading.current_thread():
                raise StateTransactionError('active load trial belongs to another thread')
            super().discard_trial(token)
            self._load_trial = None
