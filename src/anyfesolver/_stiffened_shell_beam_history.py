"""Model-bound native beam histories for the experimental mixed owner.

Offsets and identities come only from compiled beam cells. Both halves of
every compensated history value are preserved; no constitutive update occurs
while packing or reconstructing a history.
"""
from decimal import localcontext

import numpy as np

from ._ge_beam3_fibre_cell import CellHistory
from ._ge_beam3_fibre_section import FibreHistory, PhysicalFibreSection
from ._ge_beam3_generalized_cell import GeneralizedCellHistory
from ._ge_beam3_generalized_ellipsoid_section import GeneralizedHistory, EllipsoidalGeneralizedSection


def compensated_kind(parts, *, coupled=False):
    """Exact compiled member families; the owner identity binds their order/laws."""
    from ._stiffened_shell_f06_increment_arithmetic import POLICY
    families = []
    for element, _, _ in parts:
        if getattr(element.operator, 'arithmetic_policy', None) != POLICY:
            raise ValueError('uniform compensated native material policy required')
        if type(element.section) is PhysicalFibreSection:
            families.append('fibre')
        elif type(element.section) is EllipsoidalGeneralizedSection:
            families.append('generalized')
        else:
            raise ValueError('explicit physical-fibre or generalized material family required')
    if not families:
        raise ValueError('complete compensated native member inventory required')
    family = families[0] if len(set(families)) == 1 else 'mixed'
    return 'finite_compensated_'+family+('_coupled' if coupled else '_material')


def shapes(parts, prefix):
    generalized = sum(len(e.operator.cell.stations) for e, _, _ in parts
                      if type(e.section) is not PhysicalFibreSection)
    fibres = sum(len(e.operator.cell.stations)*len(e.section.fibres)
                 for e, _, _ in parts if type(e.section) is PhysicalFibreSection)
    return ((prefix+"_plastic", (generalized, 6, 2)),
            (prefix+"_accumulated", (generalized, 2)),
            (prefix+"_fibres", (fibres, 4)))


def arrays(parts, histories, prefix):
    if type(histories) is not tuple or len(histories) != len(parts):
        raise ValueError("complete native beam histories required")
    result = {name: np.empty(shape) for name, shape in shapes(parts, prefix)}
    gi = fi = 0
    with localcontext() as context:
        context.prec = 80
        for (element, _, _), history in zip(parts, histories):
            element.operator.cell._origins(history)
            for station in history.stations:
                if type(element.section) is PhysicalFibreSection:
                    count = len(element.section.fibres)
                    result[prefix+"_fibres"][fi:fi+count] = station.rows
                    fi += count
                else:
                    result[prefix+"_plastic"][gi] = station.plastic
                    result[prefix+"_accumulated"][gi] = station.accumulated
                    gi += 1
    return tuple(result.items())


def histories(parts, values, prefix):
    captured = {}
    for name, shape in shapes(parts, prefix):
        value = np.array(values[name], dtype=float, copy=True)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("complete finite model-bound native history required")
        captured[name] = value
    gi = fi = 0
    result = []
    with localcontext() as context:
        context.prec = 80
        for element, _, _ in parts:
            cell = element.operator.cell
            stations = []
            physical = type(element.section) is PhysicalFibreSection
            for _ in cell.stations:
                if physical:
                    count = len(element.section.fibres)
                    rows = tuple(tuple(map(float, row)) for row in
                                 captured[prefix+"_fibres"][fi:fi+count])
                    stations.append(FibreHistory(element.section.identity, rows))
                    fi += count
                else:
                    stations.append(GeneralizedHistory(element.section.identity,
                        tuple(tuple(map(float, row)) for row in
                              captured[prefix+"_plastic"][gi]),
                        tuple(map(float, captured[prefix+"_accumulated"][gi]))))
                    gi += 1
            history = (CellHistory if physical else GeneralizedCellHistory)(
                cell.identity, tuple(stations))
            cell._origins(history)
            result.append(history)
    return tuple(result)


def observables(histories):
    """Generalized accumulated strain, or maximum fibre accumulation per station."""
    return tuple(tuple(
        max((float(row[2]+row[3]) for row in station.rows), default=0.)
        if type(station) is FibreHistory else float(sum(station.accumulated))
        for station in history.stations) for history in histories)
