"""Brain-section Aβ/Iba1/CD68 analysis."""

import os
import tempfile
from pathlib import Path


_CACHE = Path(os.environ.get("CELL_ANALYZER_CACHE_DIR", Path(__file__).resolve().parent.parent / "local_artifacts" / "cache" / "application")).expanduser().resolve()
_TEMP = _CACHE / "tmp"
for _directory in (_CACHE, _TEMP, _CACHE / "cjdk", _CACHE / "matplotlib"):
    _directory.mkdir(parents=True, exist_ok=True)
for _name, _path in {
    "CELL_ANALYZER_CACHE_DIR": _CACHE,
    "CJDK_CACHE_DIR": _CACHE / "cjdk",
    "MPLCONFIGDIR": _CACHE / "matplotlib",
    "TEMP": _TEMP,
    "TMP": _TEMP,
    "TMPDIR": _TEMP,
    "XDG_CACHE_HOME": _CACHE,
}.items():
    os.environ[_name] = str(_path)
tempfile.tempdir = str(_TEMP)

from .analysis import analyze_arrays, run_analysis
from .batch import run_batch_analysis
from .config import create_default_config, load_config, save_config

__all__ = [
    "analyze_arrays",
    "create_default_config",
    "load_config",
    "run_analysis",
    "run_batch_analysis",
    "save_config",
]

__version__ = "0.1.0"
