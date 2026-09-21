"""Cell segmentation and per-channel ROI measurements."""

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

from .batch import run_batch_analysis
from .pipeline import run_analysis

__all__ = ["run_analysis", "run_batch_analysis"]
__version__ = "0.1.0"
