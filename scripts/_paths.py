"""Shared paths. Everything resolves under one project root.

The root defaults to the repository folder. Point it elsewhere (for example a scratch disk
holding the large .h5ad files) with the FIB_ATLAS_ROOT environment variable:

    export FIB_ATLAS_ROOT=/path/to/project
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FIB_ATLAS_ROOT", REPO))
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESOURCES = REPO / "resources"

for d in (DATA, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)
