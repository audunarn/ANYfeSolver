"""Compatibility aliases for the ANYfileio SESAM record layer."""

from anyfileio.sesam.records import (
    FemRawRecord,
    canonical_record_lines,
    format_numeric,
    parse_record_lines,
    read_raw_records,
    records_to_text,
    strict_int,
)

__all__ = [
    "FemRawRecord",
    "canonical_record_lines",
    "format_numeric",
    "parse_record_lines",
    "read_raw_records",
    "records_to_text",
    "strict_int",
]
