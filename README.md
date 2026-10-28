# Leakage-aware evaluation of early-warning models for customer dissatisfaction

This repository contains the code, the literature-audit data and the outputs of the article

> Dong Bui-Huu, Vinh Pham-An and Khuong Nguyen-An. *Leakage-Aware Evaluation of Early-Warning Models for Customer Dissatisfaction in
> E-Commerce: Performance Inflation, Timeliness Trade-offs, and Label-Latency Bias in Recalibration.* Submitted to
> *Electronic Commerce Research and Applications*, 2026.

The code is archived on Zenodo; all versions are at <https://doi.org/10.5281/zenodo.23661181>. The article uses
version 1.0.2 (<https://doi.org/10.5281/zenodo.23724874>), which adds the Dockerfile, the reference check and the
checks added during revision (`tools/`, `outputs_extra/`). Versions 1.0.1
(<https://doi.org/10.5281/zenodo.23668193>) and 1.0.0 (<https://doi.org/10.5281/zenodo.23661182>) have the same analysis
code and outputs; they differ in the documentation, the coding of one audited study (P10, from 1.0.1) and
`tools/gen_results_tex.py`.

A single command rebuilds the analysis sample from the public Olist data, runs every experiment and sensitivity
analysis, and writes every table and figure of the article and its supplementary material. The outputs of the reported
run are included in `outputs/`. Their hashes, and the hashes of the code that produced them, are recorded in
`outputs/manifest.json`, so a reader can check the reported results without rerunning the pipeline.

## Contents

- [What the pipeline does](#what-the-pipeline-does)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Data](#data)
- [Running the pipeline](#running-the-pipeline)
- [Checking the reported run without rerunning it](#checking-the-reported-run-without-rerunning-it)
- [Checks added during revision](#checks-added-during-revision)
- [Outputs and the article](#outputs-and-the-article)
- [Experimental design and number of model fits](#experimental-design-and-number-of-model-fits)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)
- [License](#license)

## What the pipeline does

The task is to predict, when an Olist marketplace order is handed to the carrier, whether the customer will give it a
one- or two-star review. Models are evaluated as a deployed system would experience them:

- rolling origins every 14 days (30 evaluation windows, June 2017 to July 2018);
- features restricted to the information available at the prediction time (as-of features);
- a label-availability purge, which removes from each training set the orders whose rating was not yet known at the
  origin.

On top of this reference protocol the pipeline answers four research questions.

- **RQ1.** How much do four common evaluation shortcuts inflate reported performance? The shortcuts are random splits,
  no purge, target aggregates computed over the full period, and features recorded after hand-over.
  - A full factorial design is run for logistic regression, XGBoost and random forests.
  - Shapley values attribute the inflation to the shortcuts.
  - Leaky models are scored on admissible features under four serving rules (realised performance, a counterfactual
    serving test).
  - The pipeline checks whether the shortcuts change which learner looks best, with fixed configurations and with
    nested tuning.
  - The analysis is repeated for a second task, late-delivery prediction at payment approval.
- **RQ2.** When should a warning be issued? Six milestones of the order lifecycle are compared by the early catch rate,
  the share of low-rated orders that are flagged before the customer reviews.
- **RQ3.** How does outcome-dependent label latency bias recalibration, and can the bias be removed? Nine recalibration
  strategies are compared, including latency-weighted Platt scaling.
- **RQ4.** Which recalibration and alert policies pay off? Alert policies are evaluated by expected cost, net benefit
  and daily workload.

The stage `rq3` produces the results of both RQ3 and RQ4, so the code and the output files use the prefix `rq3` for both.

A `robustness` stage runs the sensitivity analyses of the article (110 tasks):

- ten further seeds for the learner comparison;
- nested tuning of the full factorial design;
- serving rules;
- finer milestones and other alert budgets;
- recalibration variants and a cost-ratio grid;
- the sample definition and the smoothing of the as-of rates.

## Repository layout

```
.
├── run_pipeline.py              entry point (all stages, or selected stages)
├── run_pipeline.bat             Windows: creates the environment and runs the full pipeline
├── run_pipeline.sh              Linux / macOS equivalent
├── config.yaml                  parameters of the pipeline (sample rules, windows, models, seeds)
├── requirements.txt             minimum library versions
├── requirements-lock.txt        exact library versions of the reported run
├── Dockerfile                   container with the library versions of the reported run (Linux)
├── data/
│   ├── README.md                how to obtain the Olist data
│   └── raw/                     the nine Olist CSV tables (not included)
├── src/leakaware/
│   ├── config.py                configuration loading
│   ├── pipeline.py              stages, parallel orchestration and manifest
│   ├── data/                    raw tables and SHA-256 checks, analysis sample, as-of features, second task
│   ├── evaluation/              rolling-origin and random splits, learners, metrics, block bootstrap
│   ├── experiments/             rq1_inflation, rq2_timing, rq3_recalibration, robustness
│   └── reporting/               figures, paired inference tables, robustness summaries
├── tests/
│   └── test_leakage_guards.py   tests of the sample rules, the as-of history, the purge and the feature sets
├── tools/
│   ├── archive_outputs.py       moves earlier outputs to _archive/ before a full run
│   ├── verify_manifest.py       checks the code and the outputs against outputs/manifest.json
│   ├── gen_results_tex.py       writes the LaTeX number macros of the article from the outputs
│   └── ...                      checks added during revision (see below)
├── docs/
│   ├── literature_audit.csv     coding sheet of the literature audit (Table 1, Supplementary Table S1)
│   ├── literature_audit.md      fields and coding rules of the audit
│   ├── reference_check.csv      Crossref check of the DOIs in the reference list of the article
│   └── supplementary/           learner comparison and library versions of an earlier run (Supplementary Table S3)
└── outputs/                     outputs of the reported run
    ├── tables/                  result tables (CSV)
    ├── figures/                 figures (vector PDF and 600-dpi PNG)
    └── manifest.json            software versions and SHA-256 of the code and of every output
```

A full run also writes `outputs/logs/` (one log per stage), `outputs/data/` (feature files and saved predictions, about
160 MB) and `outputs/robustness/` (raw results of the sensitivity analyses, about 450 MB). These are excluded from the
repository; the manifest still lists them (29 entries), so a full run of your own can be checked against it:

| Not shipped | Entries | Reason |
|---|---|---|
| `outputs/logs/*.log` | 10 | run logs (progress and timings), not results |
| `outputs/data/features_t0.pkl`, `features_t0_csv.zip` | 2 | order-level features derived from the Olist tables (CC BY-NC-SA 4.0), about 125 MB; rebuilt by `--stage sample` in under a minute |
| `outputs/data/*predictions*.csv.gz`, `rq2_scores.csv.gz` | 3 | order-level scores; derived from the Olist tables and rebuilt by a full run |
| `outputs/robustness/features_*.pkl` | 4 | feature files of the smoothing variants, about 98 MB each (above the GitHub file limit) |
| `outputs/robustness/*.csv`, `*.csv.gz` | 10 | raw results of the sensitivity analyses; their summaries are in `outputs/tables/robustness_*.csv` |

## Requirements

- Python 3.12 or later. The reported run used Python 3.14.4 on Windows 11.
- NumPy, pandas, scikit-learn, SciPy, XGBoost, Matplotlib and PyYAML. `requirements-lock.txt` pins the exact versions
  of the reported run; `requirements.txt` gives the minimum versions.
- For the full run, at least 16 CPU cores and 16 GB of RAM. The default run starts about fifteen worker processes with
  two threads each. On smaller machines, use `--sequential` or `--workers N`.

The reported run took 7.2 hours on an Intel Core i9-14900HX (24 cores, 32 GB RAM). Its longest stages are the
robustness analyses and the second-task learner comparison (`task2b`). A smoke test (`--quick`) takes a few minutes.

## Data

The pipeline uses the *Brazilian E-Commerce Public Dataset by Olist*, released by Olist on Kaggle under the
CC BY-NC-SA 4.0 licence (<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>). The data are not
redistributed here.

1. Download the data set from Kaggle and unzip it.
2. Copy the nine CSV files into `data/raw/` (`data/README.md` lists the file names).

Before any analysis, the pipeline compares the SHA-256 of every table with `src/leakaware/data/raw_sha256.json`. It
stops if a file differs from the release used in the article. The 61 MB geolocation table may be stored compressed as
`olist_geolocation_dataset.csv.gz`.

## Running the pipeline

**Windows.** Double-click `run_pipeline.bat`, or run it from a command prompt.

**Linux or macOS.** Run `./run_pipeline.sh`.

On the first run, both scripts create a virtual environment in `.venv/`, install the pinned libraries and run every
stage. They log to `outputs/logs/pipeline.log`.

How the scripts treat earlier outputs:

- Without arguments, the scripts first move the previous contents of `outputs/` to `_archive/outputs_<date>_<time>/`,
  so that every full run starts from an empty folder. If a file cannot be moved (for example, because it is open in
  another program), nothing is moved and the run does not start.
- `--resume` continues an interrupted full run without archiving. Finished robustness tasks are not repeated.
- Any other argument is passed on to `run_pipeline.py`, and the existing outputs are kept.

With an environment already set up, the pipeline can be called directly:

```bash
python run_pipeline.py                     # full run
python run_pipeline.py --quick             # smoke test on a few windows
python run_pipeline.py --stage sample      # analysis sample and feature files only
python run_pipeline.py --stage rq2         # a single stage
python run_pipeline.py --stage rq1b_rf,collect,inference,figures,manifest   # several stages, in this order
python run_pipeline.py --sequential        # analysis stages one after another
python run_pipeline.py --workers 6         # number of robustness worker processes
```

A run with `--quick` or `--stage` writes into `outputs/` and overwrites the shipped files of the reported run.
`git checkout -- outputs` restores them.

The stages run in this order:

| Step | Stages | What they do |
|---|---|---|
| 1 | `sample` | Builds the analysis sample and the as-of feature files. |
| 2 | `rq1`, `rq1_rf`, `rq1b`, `rq1b_rf`, `rq2`, `rq3`, `task2`, `task2b`, `robustness` | The analysis stages, run as parallel processes. Each writes its own log to `outputs/logs/<stage>.log`. |
| 3 | `collect`, `inference`, `figures`, `manifest` | Combine the per-learner RQ1 results, compute the paired inference tables, draw the figures and write the manifest. |

After rerunning a single RQ1 stage, run `collect` and the later stages again. The robustness stage saves each task as
soon as it finishes, so an interrupted run resumes where it stopped. A saved task from a different configuration or
tuning grid is refused rather than mixed with new results.

**Container.** `Dockerfile` installs the pinned libraries on Python 3.14 (Linux). Mount the folder with the nine Olist
CSV files read-only and an empty folder for the outputs; a run overwrites the tables, figures and manifest in the folder
mounted on `/app/outputs`, so never mount the `outputs/` folder of this repository:

```bash
docker build -t leakaware .
docker run --rm -v /path/to/olist-csv:/app/data/raw:ro -v /path/to/empty-folder:/app/outputs leakaware --quick
```

The reported run used Windows. On Linux, as in the container, logistic regression reproduces the reported scores (differences below 1e-8). XGBoost does
not: its row and column subsampling draws random numbers through the C++ standard library, whose distributions differ
between compilers, so the trees differ slightly. In a check on the RQ3 stage the window-level PR-AUC of XGBoost differed
by 0.005 on average and the mean over the 30 windows by at most 0.003.

The leakage-guard tests need the feature file, so build it first with `--stage sample` and then run:

```bash
python -m pytest tests -q
```

They check four things:

- no order in the sample has feedback before hand-over;
- the as-of count of a seller's earlier ratings equals a brute-force recount for 200 random orders;
- no training order kept by the purge has its label after the origin, and no training order falls in the evaluation
  window;
- the feature set of the reference protocol contains no post-hand-over variable and no full-period aggregate.

## Checking the reported run without rerunning it

```bash
python tools/verify_manifest.py
```

The script makes two checks:

- **Code.** It hashes the source files as the pipeline does and compares the result with `code_files` and
  `code_sha256` in `outputs/manifest.json`. For the reported run, `code_sha256` begins with `ad2a62fcd20d`. A match
  means that the code in the checkout is the code that produced the outputs.
- **Outputs.** It compares every shipped output file with its manifest entry. The 96 tables and figures should match;
  the logs, feature files and robustness files listed in the manifest are not shipped and are reported as absent.

The same check runs on every push (`.github/workflows/verify-manifest.yml`).

The numbers printed in the article are written as LaTeX macros by

```bash
python tools/gen_results_tex.py outputs path/to/results.tex path/to/tab_rq3.tex
```

The script reads the result tables, `outputs/data/features_t0.pkl`, `outputs/logs/pipeline.log` (for the run time),
`config.yaml`, the code and `docs/`. The feature file and the log are not shipped, so the script applies to the outputs
of a full run of your own. It also writes `supp_tables.tex` (the tables of the supplementary material) next to
`tab_rq3.tex`.

## Checks added during revision

These scripts import the released package without changing it and write to a separate folder (`EXTRA_DIR`), so the
shipped outputs and the manifest are not touched. They need the outputs of a full run (feature and prediction files).

| Script | What it does | Fits models |
|---|---|---|
| `tools/net_benefit_bands.py OUT EXTRA_DIR` | 95% block-bootstrap bands for the net-benefit curves of Fig. 10, from the saved RQ3 predictions; first checks that the pooled curves equal `rq3_decision_curve.csv` | no |
| `tools/recal_cap_check.py OUT EXTRA_DIR` | weight cap 100, no cap from 4 days of age, latency by predicted transit time (Section 5.7); refits the RQ3 models with the configurations the stage selected and checks that logistic regression reproduces the saved probabilities | yes, 60 refits |
| `tools/gen_extra_tex.py EXTRA_DIR results_extra.tex` | LaTeX macros for the checks in this table | no |
| `tools/audit_agreement.py SECOND_CODER_CSV [OUT_CSV]` | agreement and Cohen's kappa between the two coders of the literature audit | no |
| `tools/extra_seeds.py OUT EXTRA_DIR --seeds ...` | XGBoost seed sensitivity of RQ2 and RQ3, with the selected configurations or (`--nested`) with tuning repeated | yes |
| `tools/sample_variants.py DATA_DIR OUT EXTRA_DIR` | RQ1 to RQ4 under alternative sample conditions (undelivered orders after hand-over; unreviewed orders as satisfied, as a bound), RQ2 to RQ4 with the configurations of the reported run; the unchanged sample is refitted first and logistic regression must reproduce `rq1_window_metrics.csv` | yes |
| `tools/rq2_chronology_check.py OUT EXTRA_DIR` | early catch rates of RQ2 on the orders whose training labels all precede their scoring time, from the saved RQ2 scores; first checks that the full cohort reproduces `rq2_window_metrics.csv` | no |
| `tools/policy_timing_check.py OUT EXTRA_DIR` | the daily top-10% policy as an end-of-day batch (reviews answered before the end of the hand-over day), an online rule with a threshold from the past seven days, and a threshold matched to the same alert volume; first checks that the saved scores reproduce `rq3_daily_policy.csv` | no |
| `tools/summarise_extra.py OUT EXTRA_DIR` | spread over seeds of the RQ2 to RQ4 quantities, and the summary of each sample variant | no |
| `tools/compare_manifests.py OLD NEW` | compares two runs file by file through their manifests (for example the reported run and a rerun); time stamps in PDF figures and compressed files are ignored | no |

On Windows, `tools\run_revision_checks.bat` runs these checks, except the audit agreement, after `run_pipeline.bat`
and writes a log to `outputs_extra\logs`. By default it uses seeds 1 to 10, the extra seeds of the learner comparison
in `config.yaml`; other seeds can be given as arguments. It first compares the new run with the newest archived full
run (`compare_manifests.py --archive _archive`) and keeps the new manifest as `outputs_extra/rerun_manifest.json`.
Result files of an interrupted run are reused only if they cover all evaluation windows.

Logistic regression is deterministic and reproduces the reported scores on any platform. XGBoost refits on another
platform than the reported one (Windows) differ slightly (see the container note above).

## Outputs and the article

Paths below are relative to `outputs/`. Figures are written as `<name>.pdf` and `<name>.png`.

| Article | Output |
|---|---|
| Table 1, literature audit | `docs/literature_audit.csv` (repository root) |
| Section 3.4, analysis sample | `tables/sample_flow.csv`, `tables/sample_notes.json` |
| Fig. 1, reference protocol | `figures/fig02_rolling_origin_protocol` |
| Fig. 2, evaluation context | `figures/fig03_evaluation_context` |
| Fig. 3, inflation | `figures/fig04_rq1_inflation`, `tables/rq1_inflation.csv`, `tables/rq1b_tuned_inflation.csv` |
| Fig. 4, Shapley attribution | `figures/fig05_rq1_shapley`, `tables/rq1_shapley.csv` |
| Fig. 5, realised performance | `figures/fig06_rq1_realised_performance`, `tables/rq1_deployment_summary.csv` |
| Fig. 6, learner comparison | `figures/fig07_rq1_model_ranking`, `tables/rq1_ranking_flip.csv`, `tables/rq1b_tuned_ranking_flip.csv` |
| Table 6, number of model fits | computed by `tools/gen_results_tex.py` from `config.yaml`, the code and the table row counts |
| Table 7, summary of RQ1 (both tasks) | `tables/rq1_*`, `tables/rq1b_*`, `tables/task2_*` |
| Table 8 and Fig. 7, timing of the warning | `figures/fig09_rq2_timing`, `tables/rq2_summary.csv`, `tables/rq2_paired_vs_t0.csv` |
| Fig. 8, label latency | `figures/fig10_rq3_label_latency` |
| Fig. 9, calibration | `figures/fig11_rq3_calibration`, `tables/rq3_window_metrics.csv`, `tables/rq3_summary.csv` |
| Table 9, recalibration strategies | `tables/rq3_summary.csv`, `tables/rq3_paired_strategies.csv`, `tables/supp_rq3_contrasts.csv` |
| Fig. 10, net benefit | `figures/fig13_rq3_net_benefit`, `tables/rq3_decision_curve.csv` |
| Fig. 11, daily workload | `figures/fig14_rq3_daily_workload`, `tables/rq3_daily_policy.csv` |
| Table 10, sensitivity analyses | `tables/robustness_*.csv`, `tables/rq1b_tuned_window_metrics.csv` |
| Paired contrasts with Holm-adjusted p-values | `tables/supp_paired_contrasts.csv`, `tables/supp_rq3_contrasts.csv` |
| Supplementary Table S1, audit coding | `docs/literature_audit.csv` (repository root) |
| Supplementary Table S2, intervals for Table 9 | `tables/rq3_summary.csv` (columns `*_lo`, `*_hi`) |
| Supplementary Table S3, earlier software environment | `docs/supplementary/rq1b_tuned_ranking_flip_xgboost305_linux.csv` (repository root), `tables/rq1b_tuned_ranking_flip.csv` |
| Supplementary Table S4, alternative multiplicity adjustments | `tables/supp_paired_contrasts.csv`, `tables/supp_rq3_contrasts.csv` (computed by `tools/gen_results_tex.py`) |
| Supplementary Fig. S1, sample flow | `figures/fig01_sample_flow` |
| Supplementary Fig. S2, second task | `figures/fig08_rq1_second_task` |
| Supplementary Figs. S3 and S4, ROC-AUC versions of Figs. 3 and 4 | `figures/figS1_rq1_inflation_auc`, `figures/figS2_rq1_shapley_auc` |
| Supplementary Fig. S5, PR-AUC by window | `figures/figS3_rq1_prauc_over_time` |
| Supplementary Fig. S6, selection on test data | `figures/figS4_rq1_selection_shortcuts`, `tables/rq1_selection_summary.csv`, `tables/rq1_threshold_summary.csv` |
| Supplementary Fig. S7, all learner comparisons | `figures/figS5_rq1_ranking_fixed` |
| Supplementary Fig. S8, other alert budgets | `figures/figS6_rq2_alert_budgets`, `tables/robustness_milestone_budgets.csv` |
| Supplementary Fig. S9, cost savings with intervals | `figures/fig12_rq3_cost_savings` |

Figures are drawn at their printed size for the two-column Elsevier layout.

## Experimental design and number of model fits

A temporal window costs one fit, and a random split one fit per fold (five folds). Nested tuning costs one fit per grid
configuration and inner split (three inner folds under a random split), followed by one refit. Recalibration maps are
not counted.

| Experiment | Design | Fits |
|---|---|---|
| RQ1 factorial design | 14 protocols (12 factorial, plus full-period raw and leave-one-out aggregates) x 3 learners, fixed configurations, 30 windows | 960 |
| RQ1 realised performance | 4 leaky pipelines x 30 windows x 3 learners | 360 |
| RQ1 selection shortcuts | 12 candidate configurations x 30 windows | 360 |
| RQ1 nested learner comparison | 3 protocols x 3 learners; grids of 20 (XGBoost), 11 (LR) and 12 (RF) configurations | 2,700 |
| RQ2 milestones | 6 milestones x 30 windows x 2 learners, nested tuning | 5,940 |
| RQ3 and RQ4 recalibration | 30 windows x 2 learners, nested tuning; 9 recalibration strategies | 1,050 |
| Second task | 13 protocols, 4 leaky pipelines and a nested comparison (grids of 36 and 15), 2 learners | 3,960 |
| Sensitivity analyses | 110 tasks of the `robustness` stage | 25,065 |
| Total | | 40,395 |

Statistical inference uses:

- a moving-block bootstrap over the 30 windows, with blocks of three windows, 2,000 replicates for intervals and 20,000
  for p-values;
- Holm adjustment within families of related contrasts (256 contrasts in the main analyses, 178 in the sensitivity
  analyses);
- a Newey-West test with lag two as a check that does not resample;
- 11 seeds for the comparison of logistic regression and XGBoost.

## Reproducibility notes

- **Seeds.** All random components use seed 42 (`config.yaml`); the seed analysis adds the ten seeds listed there.
- **Rerunning on the same machine.** Rerunning the pipeline on the same machine from an empty `outputs/` folder
  reproduced every result table exactly, with one exception. A few random-forest decision thresholds in
  `tables/rq1_threshold_windows.csv` differ by about 1e-16, because the forest sums its trees in two threads. F1,
  precision and recall are unaffected.
- **Other platforms and library versions.** These can change the XGBoost results slightly, through floating-point
  differences in the library. The reported run used the versions in `requirements-lock.txt` on Windows 11.
  `docs/supplementary/` contains the nested learner comparison of an earlier run that combined other library versions
  with narrower tuning grids (Supplementary Table S3 of the article).
- **Fitting inside the training set.** Imputation, scaling and one-hot encoding are fitted inside each training set.
  The category vocabulary of XGBoost is fixed in advance and uses no labels. Hyper-parameters are tuned on training
  data only.

## Citation

If you use this code or the audit data, please cite the article. `CITATION.cff` contains the citation metadata.

## License

The code is released under the MIT License (see `LICENSE`). The Olist data are not part of this repository and remain
under the licence stated by Olist on Kaggle.
