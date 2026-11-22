"""Fail when a junit report has no tests, or any skipped, failed or errored test."""
import sys
from xml.etree import ElementTree


def main(path):
    cases = list(ElementTree.parse(path).getroot().iter("testcase"))
    bad = [f"{case.get('classname')}::{case.get('name')}: {child.tag} {child.get('message') or ''}"
           for case in cases for child in case if child.tag in ("skipped", "failure", "error")]
    if not cases:
        bad.append("no tests ran")
    for line in bad:
        print(line)
    print(f"{len(cases)} tests, {len(bad)} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
