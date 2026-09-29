"""Replicate the ORIGINAL run environment's CPU probing WITHOUT spawning wmic.

Background
----------
The original SensoDat results were produced with anaconda python 3.12.7 /
sklearn 1.5.1 / numpy 1.26.4 (recorded in the original's sensodat_run.json
"environment" block).  In that stack, sklearn's HGB fit calls
`_openmp_effective_n_threads()` -> joblib.parallel.cpu_count(
only_physical_cores=True) -> loky `_count_physical_cores()`, which on win32
spawns `wmic CPU Get NumberOfCores /Format:csv`.

Some sandboxed environments blacklist wmic.exe and kill the whole command tree
when it is attempted.  When wmic fails, loky falls back to LOGICAL cores, so
the fit still works but with a different OpenMP thread count than the
original run.  To reproduce the original EXACTLY we pre-seed the physical
core count (14 on this machine, measured via CIM) so loky never attempts
the wmic spawn and the thread count matches the original run.

Apply BEFORE sklearn/joblib performs any fit.  Importing this module is
enough.
"""
import os as _os
PHYSICAL_CORES = int(_os.environ.get("REPROALLOC_PHYSICAL_CORES", "14"))
LOGICAL_CORES = int(_os.environ.get("REPROALLOC_LOGICAL_CORES", "20"))


def apply():
    try:
        from joblib.externals.loky.backend import context as _loky_ctx
        _loky_ctx._count_physical_cores = lambda: (PHYSICAL_CORES, None)
        return True
    except Exception:
        return False


apply()
