"""Public DATA-only consumer of a portable native accepted-prefix capsule."""
from pathlib import Path
from anyfesolver.general_contact import read_native_contact_checkpoint_results
from anyfesolver.general_contact.native_checkpoint_results import MAX_BYTES


def read_portable(path):
    """Read an existing capsule; no owner, model, material or restart replay."""
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_BYTES+1)
    return read_native_contact_checkpoint_results(raw)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    result = read_portable(parser.parse_args().checkpoint)
    print(dict(identity=result.identity, completed_targets=result.completed_targets,
               bytes=len(result.to_bytes()), scope='DATA_INTEGRITY_ONLY'))
