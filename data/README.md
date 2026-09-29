# data/

## Shipped

### `amini_transfuser_canonical.csv` (44 KB, 500 rows)

Canonical record of the Amini-TransFuser benchmark: 500 recorded CARLA
0.9.10 / TransFuser executions over 36 routes (251 fail, 249 pass), with the
per-execution outcome and wall-clock duration used by the replay harness
(`src/experiment_amini_finite_trace.py` loads it directly). First-party data
produced by this project; redistributable.

### `sensodat_splits.json` (4.8 MB, 30 entries)

The source-target split definitions used by every SensoDat run: a list of
`{key, train, history, target}` index sets into the SensoDat scenario pool
(10 splits x 3 repetitions layout; split seeds 20261001+i). First-party
derived indices; redistributable. Required together with the npz below for
bit-exact SensoDat reruns.

### `sensodat_fresh.npz` (48.4 MB)

Feature/label archive derived from the public **SensoDat** benchmark
(AmbieGen -> Frenetic transfer setting). It is included in the replication
package and is required to rerun the SensoDat simulations. The evaluated
target pool comprises 12,135 scenarios (4,322 failures) after applying the
source-target split construction in `src/sensodat_final_experiment.py`.

Contents:

| key | shape | dtype | meaning |
|---|---|---|---|
| `X_spec` | (36006, 42) | float32 | scenario specification features |
| `X_config` | (36006, 8) | float32 | generator configuration features |
| `y` | (36006,) | int32 | outcome label (1 = failure) |
| `duration_s` | (36006,) | float64 | recorded execution duration (seconds) |
| `generator` | (36006,) | int32 | generator id (AmbieGen / Frenetic) |
| `scenario_id` | (36006,) | str | scenario identifier |
| `group` / `signature` | (36006,) | str | grouping / dedup keys |
| `severity` | (36006,) | int32 | severity score |
| `sequence` | (36006, 196, 2) | float32 | road sequence geometry |
| `subject` | () | str | subject system tag |

The package includes this derived archive so the SensoDat runs can be
reproduced without a machine-specific data path. The source benchmark is
credited in the paper's bibliography.

