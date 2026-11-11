"""Read-only inspection of the saved development comparison; no new solves."""
from hashlib import sha256
from pathlib import Path
import pytest
from docs.reference_cases.ge_beam3_preserved_arch_load_comparison import parse,canonical
from docs.reference_cases import ge_beam3_dead_line_probe as probe
from docs.reference_cases import ge_beam3_dead_line_comparison as compare

ROOT=Path(__file__).resolve().parents[1]
EXTERNAL=Path('C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease/ge-beam3-dead-line-20260907-v1')
REVISION='db0a8da3379e6bfbda774ebda5cd6b15d664da77'


def saved():
    return parse((ROOT/'docs/reference_cases/ge_beam3_dead_line_development_result.json').read_bytes().replace(b'\r\n',b'\n'))


def test_saved_development_counts_and_boundaries():
    result=saved()
    assert result['revision']==REVISION and result['production_qualified'] is False
    assert result['independent_review']=='PENDING' and result['conservative_spectral_authority'] is False
    assert result['status']=='DEVELOPMENT_COMPARISON_NOT_QUALIFICATION'
    assert len(result['artifacts'])==8 and len(result['rows'])==9
    assert sum(row['metrics']['stations'] for row in result['rows'])==168


@pytest.mark.parametrize('target',[.25,.5,1.])
def test_recorded_refinement_and_continuum_profile_agreement(target):
    rows=[row['metrics'] for row in saved()['rows'] if row['target']==target]
    for key in ('tip_displacement_relative_error','strain_energy_norm_relative_error','integrated_energy_relative_error',
                'spatial_force_max_relative_error','spatial_moment_max_relative_error'):
        assert rows[2][key]<rows[1][key]<rows[0][key]
    assert rows[2]['tip_displacement_relative_error']<.006
    assert rows[2]['strain_energy_norm_relative_error']>.05  # Never claim all field errors are below 2%.
    assert max(row['reference_profile_field_max_difference'] for row in rows)<2.1e-9
    for row in rows:
        assert max(row[k] for k in ('native_equilibrium_metric','reaction_force_absolute_error','reaction_moment_absolute_error'))<=1e-11


def test_external_hash_dag_and_complete_recomputation_without_equilibrium_solves(monkeypatch):
    if not EXTERNAL.is_dir(): pytest.skip('External development packet not installed; not qualification')
    def forbidden(*args,**kwargs): raise AssertionError('Inspection must not solve')
    monkeypatch.setattr(compare,'build',forbidden)
    monkeypatch.setattr(compare.reference,'solve',forbidden)
    from anyfesolver import _ge_beam3_fibre_line_program as native
    monkeypatch.setattr(native,'solve_force_program',forbidden)
    result=saved()
    for row in result['artifacts']:
        raw=(EXTERNAL/row['path']).read_bytes()
        assert len(raw)==row['bytes'] and sha256(raw).hexdigest()==row['sha256']
    raw=(EXTERNAL/'comparison.json').read_bytes()
    assert canonical(result)==raw==canonical(probe.summary(EXTERNAL,REVISION))
