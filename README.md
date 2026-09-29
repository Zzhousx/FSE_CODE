# ReproAlloc replication package

This repository provides code, replay inputs, and run-level results for the
reported ReproAlloc comparisons on SensoDat-replay, Amini-TransFuser, and
CANNIER-Shuffle. It includes the shared confirmation rule, external baselines,
internal controls, analysis scripts, and tests.

## Contents

| Path | Contents |
|---|---|
| `src/` | ReproAlloc and SensoDat/Amini replay harnesses |
| `scripts/` | Experiment and analysis commands, including CANNIER replay |
| `tests/` | Algorithm and protocol checks |
| `configs/` | Machine-readable experiment settings |
| `data/` | Amini traces and SensoDat archive/splits; see `data/README.md` |
| `results/` | Run-level records and aggregate tables |
| `logs/` | Decision logs needed by the analysis builders |

Use Python 3.12 and install `requirements.txt`. The pinned SensoDat prior-fit
environment is in `environment.yml`. Run the tests with
`python -m pytest tests -q`. Summary tables can be rebuilt with
`python scripts/build_final_statistics.py` and
`python scripts/cannier/summarize_published_gai.py`.

The CANNIER source archive is available through
`python scripts/cannier/fetch_data.py`; the script verifies its checksum.
Prepared replay pools are included in `results/cannier/pools/`. Large source
databases and execution-event logs are omitted from this GitHub package.
The evaluation uses the eight projects in
`configs/cannier_eval.json` and 12 seeds.

The literature-based GAI comparison methods are implemented in
`scripts/cannier/gai_published_baselines.py`. Their paper sources and the
adaptations needed for this repeated-confirmation task are listed in
`results/published_gai/README.md`.
