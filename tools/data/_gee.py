"""Earth Engine initialization for the download scripts.

The cloud project id is private: it comes from ``CHLA_GEE_PROJECT`` or from
the YAML that ``CHLA_GEE_RUNTIME_CONFIG`` points at, never from a config,
a script or a command line.
"""

from __future__ import annotations

import os
from pathlib import Path

import ee
import yaml

__all__ = ["initialize"]

DEFAULT_RUNTIME_CONFIG = Path.home() / ".config/chla-prediction/gee_runtime.yaml"


def initialize() -> None:
    """Initialize Earth Engine against the privately configured project."""
    path = Path(os.environ.get("CHLA_GEE_RUNTIME_CONFIG", DEFAULT_RUNTIME_CONFIG))
    project = os.environ.get("CHLA_GEE_PROJECT") or yaml.safe_load(path.read_text())["earth_engine_project"]
    ee.Initialize(project=project)
