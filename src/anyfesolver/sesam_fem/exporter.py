"""Compatibility facade for SESAM document export owned by ANYfileio."""

from anyfileio.sesam.exporter import (
    SesamFemExportReport,
    export_sesam_fem,
    write_sesam_fem_document,
)

__all__ = ["SesamFemExportReport", "export_sesam_fem", "write_sesam_fem_document"]
