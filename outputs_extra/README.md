# Outputs of the checks added during revision

Written by `tools\run_revision_checks.bat` (see "Checks added during revision" in the main README) after a full run of
`run_pipeline.bat` on the machine of the reported run (Windows 11, Python 3.14.4, the library versions of
`requirements-lock.txt`). They are not part of `outputs/manifest.json`; every number taken from them into the article
goes through `results_extra.tex`.

| File | Script | Article |
|---|---|---|
| `rerun_manifest.json` | manifest of the full rerun; `tools/compare_manifests.py` found every result table identical to the reported run, apart from differences of the order of 1e-16 in a few random-forest scores and thresholds | Section 4.9 |
| `tables/rq3_decision_curve_bands.csv`, `figures/fig13_rq3_net_benefit_bands.*` | `tools/net_benefit_bands.py` | Fig. 10 |
| `tables/rq2_chronology_*.csv` | `tools/rq2_chronology_check.py` | Section 4.6, Table 10 |
| `tables/policy_timing_*.csv` | `tools/policy_timing_check.py` | Sections 4.8 and 5.8 |
| `tables/recal_cap_check_*.csv` | `tools/recal_cap_check.py` | Section 5.7, Table 10 |
| `tables/seed{1..10}_*.csv`, `tables/seed_check_*.csv` | `tools/extra_seeds.py`, `tools/summarise_extra.py` | Table 10, Section 6.3 |
| `tables/sample_*.csv`, `tables/sample_check_*.csv` | `tools/sample_variants.py`, `tools/summarise_extra.py` | Section 3.1, Table 10 |
| `results_extra.tex` | `tools/gen_extra_tex.py` | LaTeX macros of the article |

Notes:

- The bands of Fig. 10, the chronology check and the timing check use only scores saved by the main run and fit no
  model.
- `recal_cap_check_reproduction.csv` shows that the refits reproduce the saved latency-weighted probabilities of both
  learners (differences of the order of 1e-16).
- The seed check refits XGBoost with seeds 1 to 10 and the configurations that the reported run selected in each window.
  Logistic regression does not depend on the seed.
- The sample variants use the configurations of the reported run. `unreviewed0` is a bound, not a label: the synthetic
  reviews also enter the as-of history features of later orders.
- The run log is written to `logs/`, which is not kept in the repository.
