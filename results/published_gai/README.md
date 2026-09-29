# Literature-based GAI comparison methods

The package includes LUCB-G and Murphy Sampling as comparison schedulers.
Their selection rules are reimplemented in Python in
`scripts/cannier/gai_published_baselines.py`; no external source repository
is copied into this package.

- **LUCB-G:** Kano et al., *Good Arm Identification via Bandit Feedback*,
  Machine Learning 2019, Algorithm 1,
  https://doi.org/10.1007/s10994-019-05784-4 .
- **Murphy Sampling:** Kaufmann, Koolen, and Garivier, *Sequential Test for
  the Lowest Mean: From Thompson to Murphy Sampling*, NeurIPS 2018,
  https://proceedings.neurips.cc/paper_files/paper/2018/hash/7c78335a8924215ea5c22fda1aac7b75-Abstract.html .
  The Bernoulli sampling rule was cross-checked against the APGAI authors'
  public implementation:
  https://github.com/clreda/apgai/blob/main/fixed-conf-anytime-code/gai/samplingrules.jl .

For comparison with ReproAlloc, both schedulers use the common anytime
confidence-sequence confirmation rule, remove confirmed arms, respect the
recorded execution-time budget, and stop at finite-trace exhaustion. These
are adaptations to the benchmark objective; the cited papers did not study
this exact repeated-confirmation setting. Neither scheduler normalizes its
selection index by execution cost.

Run-level results are in `amini_runs.csv`, `sensodat_runs.csv`, and
`cannier_test_runs.csv`. The CANNIER projects and budgets are specified
in `configs/cannier_eval.json`.

Run from the repository root:

```sh
python scripts/cannier/run_published_gai.py
python scripts/cannier/run_published_gai_cannier.py
```
