"""Committed physical-fibre reporting for the separate native contact owner.

This does not solve a new path or commit a material trial. Incremental variational
potential is explicitly distinct from physical stored energy and dissipation.
"""
from math import fsum,isfinite
import numpy as np
from anyfesolver.control import cancellation_safe_point
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5_seeded.codec import _load,MAX_BYTES

SCHEMA='NATIVE_CONTACT_PHYSICAL_FIBRE_RECOVERY_DEV_V1'


def recover_native_material(owner,checkpoint,expected_sha256,*,include_stations=False,cancellation_token=None):
    if not owner._fibre_mode:
        raise ValueError('physical-fibre material recovery requires its exact native contact family')
    if type(include_stations) is not bool:
        raise ValueError('include_stations must be an explicit boolean')
    context=owner.context
    with context.operation(cancellation_token):
        context.activate_cancellation(cancellation_token)
        cancellation_safe_point(cancellation_token,'native material recovery validation')
        accepted,records=owner._restore(checkpoint,expected_sha256)
        origins=owner.initial.native_histories;increments=[];dissipation_terms=[]
        for index,record in enumerate(records,1):
            cancellation_safe_point(cancellation_token,'native material recovery increment')
            mechanical=context.make(record['mechanical'],decoded=True)
            histories=context.histories(record['histories'],decoded=True)
            recovery=context.recover(mechanical,origins,histories,index)
            elements=[]
            for i,((eid,element),fields) in enumerate(zip(context.elements,recovery)):
                stored=[];dissipated=[];potential=[];stations=[]
                for row,origin in zip(fields['stations'],origins[i].stations):
                    response=element.section.response(row['strain'],origin,strain_low=row['strain_low'],
                        check=context._backend.check)
                    weight=row['measure']
                    stored.append(weight*fsum(response.stored_energy))
                    dissipated.append(weight*fsum(response.dissipation))
                    potential.append(weight*fsum(response.incremental_potential))
                    if include_stations:
                        stations.append(dict(row,stored_energy_per_reference_length_n=fsum(response.stored_energy),
                            plastic_dissipation_per_reference_length_n=fsum(response.dissipation),
                            incremental_material_potential_per_reference_length_n=fsum(response.incremental_potential)))
                values=dict(element_id=eid,section_identity=element.section.identity,
                    stored_material_energy_j=fsum(stored),plastic_dissipation_increment_j=fsum(dissipated),
                    incremental_material_potential_j=fsum(potential))
                if include_stations:values['stations']=stations
                elements.append(values)
            stored=fsum(e['stored_material_energy_j'] for e in elements)
            dissipated=fsum(e['plastic_dissipation_increment_j'] for e in elements)
            dissipation_terms.append(dissipated)
            displacement=np.asarray(record['shell_displacement'],dtype=float)
            shell_stored=float(.5*displacement@owner.shell_stiffness@displacement)
            contact_stored=record['contact']['potential_j']
            row=dict(target=index,load_parameter=owner.program.targets[index-1],elements=elements,
                stored_native_material_energy_j=stored,stored_shell_energy_j=shell_stored,
                contact_regularization_storage_j=contact_stored,
                total_stored_energy_j=fsum((stored,shell_stored,contact_stored)),
                plastic_dissipation_increment_j=dissipated,plastic_dissipation_cumulative_j=fsum(dissipation_terms),
                incremental_material_potential_j=fsum(e['incremental_material_potential_j'] for e in elements))
            if any(not isfinite(v) for v in row.values() if type(v) is float):
                raise ValueError('finite native material recovery required')
            increments.append(row);origins=histories
        result=dict(schema=SCHEMA,profile=owner.profile,definition_identity=owner.identity,
            checkpoint_sha256=expected_sha256,completed_targets=accepted.completed_targets,
            measure='reference_beam_arclength',energy_evaluation='paired_recovered_station_strain',
            independent_engineering_acceptance=False,energy_balance_qualified=False,
            increments=increments,plastic_dissipation_cumulative_j=fsum(dissipation_terms))
        raw=canonical(result)
        if len(raw)>MAX_BYTES:raise MemoryError('native material recovery byte limit')
        cancellation_safe_point(cancellation_token,'native material recovery publication')
        context.guard()
        return _load(raw.decode('ascii'))
