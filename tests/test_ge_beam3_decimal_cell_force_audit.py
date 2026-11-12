"""Audit preserved cell inputs; do not rerun the failed cell solver."""
import json
from decimal import Decimal as D, localcontext
from pathlib import Path
import numpy as np
import pytest
from docs.reference_cases.ge_beam3_decimal_cell_force_audit import audit
from test_ge_beam3_plastic_cell_conjugate_probe import build
from anyfesolver._ge_beam3_p5_seeded.core import canonical

ARCHIVE = Path('C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease/ge-beam3-complementary-plastic-development-20260907-9325e35bd701/cell-final-failure')


@pytest.mark.parametrize('index,contrast', [(0, 1.), (1, 10000.), (2, 1000000000000.)])
def test_preserved_cell_force_from_source_station_equations(index, contrast, tmp_path):
    root = ARCHIVE/('test_shared_cell_conjugate_mat'+str(index))
    if not root.exists(): pytest.skip('explicit preserved cell evidence required')
    trace = json.loads((root/'optimality-progress.json').read_text())
    with np.load(root/'cell-inputs.npz', allow_pickle=False) as arrays:
        pattern = np.array(trace[-1]['increments']); p = arrays['retained']
        gradient = arrays['c0']@p+arrays['g'].T@(arrays['z0']+pattern)
    # Construct only the bound section/station data. Never call response().
    probe = build(contrast); source = probe.source.section
    stations = []
    for cell in (0, 1):
        for index, t, _, weight, v, _, _ in probe.source._stations[cell]:
            origin = probe.origins[cell*probe.source.order+index]
            stations.append(dict(cell=cell, t=float(t), weight=float(weight), v=v.tolist(),
                origin=[origin.plastic_coordinate, origin.accumulated]))
    data = dict(elastic=source._elastic.tolist(), direction=source._direction.tolist(),
        yield_force=source._yield, hardening=source._hardening, stations=stations,
        retained=p.tolist(), old_gradient=gradient.tolist(), increment_pattern=pattern.tolist())
    with (tmp_path/'source-inputs.json').open('xb') as stream: stream.write(canonical(data))
    low = audit(data, digits=60); high = audit(data, digits=90)
    with localcontext() as context:
        context.prec = 100
        assert max(abs(D(a)-D(b)) for a, b in zip(low['gradient'], high['gradient'])) < D('1e-40')
    with (tmp_path/'audit.json').open('xb') as stream: stream.write(canonical(dict(contrast=contrast, low=low, high=high)))
    print(canonical(dict(contrast=contrast, **{k:v for k,v in high.items() if k not in ('gradient','increments')})).decode(), flush=True)
    assert D(high['source_solution_force_error']) < D('1e-35')
    assert D(high['paired_gradient_force_error']) < D('1e-11')
