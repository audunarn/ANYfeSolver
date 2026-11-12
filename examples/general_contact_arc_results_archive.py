"""Public affine/preload arc producer and bounded inert data consumer."""
from examples.general_contact_arc_length_affine import analysis as solve
from anyfesolver.general_contact import read_arc_contact_results


def analysis():
    result=solve(2)
    return result,read_arc_contact_results(result.to_contact_archive_bytes())


if __name__=='__main__':
    result,archive=analysis()
    print(result.status,len(result.steps),len(archive.to_bytes()),archive.sha256)
