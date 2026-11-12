"""Explicit elastic native-fibre BL5 admission; no material equation changes."""
from ._ge_beam3_retained_fibre import RetainedFibreOperator


class BL5NativeElasticDomainError(ValueError):
    """An element/station lies outside the selected elastic native BL5 domain."""
    def __init__(self, element_id, station, cause):
        self.element_id, self.station, self.cause = element_id, station, cause
        super().__init__(f"elastic native BL5 element={element_id} station={station}: {cause}")


def require_elastic(element, response, origin):
    # Original histories and candidate histories must remain virgin elastic;
    # accepting an unloaded plastic predecessor would extend this policy.
    fibre = type(element.operator) is RetainedFibreOperator
    for family, history in (("predecessor", origin), ("candidate", response.history)):
        for index, station in enumerate(history.stations):
            values = (tuple(value for row in station.rows for value in row) if fibre else
                      tuple(station.accumulated)+tuple(value for row in station.plastic for value in row))
            if any(value != 0. for value in values):
                raise BL5NativeElasticDomainError(element.element_id, index, family+" advanced plastic history")
    # Reuse the unchanged native every-fibre inventory/branch/ULP-margin
    # witness and generalized witness, before the trial can be published.
    from ._stiffened_shell_ge_point_buckling import _elastic_fibre, _elastic_beam, _ElasticBranchDomainError
    try:
        (_elastic_fibre if fibre else _elastic_beam)(element.operator, response, origin)
    except _ElasticBranchDomainError as exc:
        raise BL5NativeElasticDomainError(element.element_id, None, str(exc)) from exc
