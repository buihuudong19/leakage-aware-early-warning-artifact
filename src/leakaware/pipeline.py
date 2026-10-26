"""Pipeline stages and their orchestration.

Stages
  sample      build the analysis sample and the feature file from the nine raw tables
  rq1         shortcut factorial design and realised performance (XGBoost, LR), selection shortcuts
  rq1_rf      the same for random forests (a separate process, so that it runs in parallel)
  rq1b        learner comparison with nested hyper-parameter tuning (XGBoost, LR)
  rq1b_rf     the same for random forests
  rq2         prediction milestones and the early catch rate (milestones in parallel processes)
  rq3         recalibration and alert policies under label latency
  task2       the RQ1 analyses repeated for late-delivery prediction at payment approval (fixed configurations)
  task2b      the nested learner comparison of the second task
  robustness  sensitivity analyses (seeds, tuned factorial, serving rules, milestones and budgets, recalibration
              variants, sample definition, smoothing)
  collect     combine the per-learner RQ1 results; inflation, Shapley and learner-comparison tables
  inference   Holm-adjusted paired contrasts and the robustness summary tables
  figures     all figures of the article
  manifest    SHA-256 of every output file with the software versions
The stage `all` runs `sample`, then the analysis stages in parallel processes, then `collect`, `inference`, `figures`
and `manifest`.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata

import pandas as pd

from .config import ROOT
from .data.io import load_raw, sha256, load_features, export_features_csv_zip
from .data.sample import build_sample
from .data.features import build_features, BASE_CAT
from .data import late_delivery
from .experiments import rq1_inflation as rq1, rq2_timing as rq2, rq3_recalibration as rq3

ANALYSIS_STAGES = ("robustness", "rq2", "rq1b", "task2b", "rq3", "rq1", "rq1_rf", "rq1b_rf", "task2")
FINAL_STAGES = ("collect", "inference", "figures", "manifest")
ALL_STAGES = ("sample",) + ANALYSIS_STAGES + FINAL_STAGES


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- stages
def stage_sample(cfg, out, args):
    raw, _ = load_raw(cfg["data_dir"])
    S, flow, notes = build_sample(raw, cfg)
    F = build_features(S, raw, cfg)
    flow.to_csv(out / "tables" / "sample_flow.csv", index=False, encoding="utf-8-sig")
    (out / "tables" / "sample_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    F.to_pickle(out / "data" / "features_t0.pkl")
    export_features_csv_zip(F, out / "data" / "features_t0_csv.zip")
    print(flow.to_string(index=False)); print(notes)


def _rq1_data(cfg, out):
    F = load_features(out)
    D = F[F["y"] >= 0].reset_index(drop=True)
    return D, {c: sorted(F[c].unique()) for c in BASE_CAT}


def _learners(cfg, rf):
    """Learners of a main-task RQ1 stage: XGBoost and LR, or RF (run as a separate stage so that it runs in parallel)."""
    ls = cfg.get("learners", ["xgb", "lr"])
    return tuple(m for m in ls if (m == "rf") == rf)


def _rq1_compute(cfg, out, args, rf):
    D, vocab = _rq1_data(cfg, out)
    models = _learners(cfg, rf)
    if not models:
        return
    if args.quick:
        cfg["rolling"]["step_days"] = 56
    extra = [dict(split="temporal", purge=True, agg="global_raw", post=False, shortcuts="Graw"),
             dict(split="temporal", purge=True, agg="global_loo", post=False, shortcuts="Gloo")]
    tag = "rf" if rf else "main"; pdir = out / "data" / "parts"; pdir.mkdir(parents=True, exist_ok=True)
    R, P = rq1.run_factorial(D, vocab, cfg, models=models, extra_conditions=extra)
    R.to_csv(pdir / f"rq1_{tag}_window_metrics.csv", index=False)
    P.to_csv(pdir / f"rq1_{tag}_reference_predictions.csv.gz", index=False, compression="gzip")
    G = rq1.run_deployment_gap(D, vocab, cfg, models=models)
    G.to_csv(pdir / f"rq1_{tag}_deployment_windows.csv", index=False)
    if not rf and not args.quick:
        t = out / "tables"
        Sel = rq1.run_selection(D, vocab, cfg)
        Sel.to_csv(t / "rq1_selection_windows.csv", index=False)
        rq1.selection_summary(Sel, cfg).to_csv(t / "rq1_selection_summary.csv", index=False)


def _rq1b_compute(cfg, out, args, rf):
    D, vocab = _rq1_data(cfg, out)
    models = _learners(cfg, rf)
    if not models:
        return
    if args.quick:
        cfg["rolling"]["step_days"] = 140
    tag = "rf" if rf else "main"; pdir = out / "data" / "parts"; pdir.mkdir(parents=True, exist_ok=True)
    T, Ch = rq1.run_tuned_ranking(D, vocab, cfg, models=models)
    T.to_csv(pdir / f"rq1b_{tag}_window_metrics.csv", index=False)
    Ch.to_csv(pdir / f"rq1b_{tag}_chosen_params.csv", index=False)


def stage_rq1(cfg, out, args):
    _rq1_compute(cfg, out, args, rf=False)


def stage_rq1_rf(cfg, out, args):
    _rq1_compute(cfg, out, args, rf=True)


def stage_rq1b(cfg, out, args):
    _rq1b_compute(cfg, out, args, rf=False)


def stage_rq1b_rf(cfg, out, args):
    _rq1b_compute(cfg, out, args, rf=True)


def stage_collect(cfg, out, args):
    """Combine the per-learner RQ1 results and write the RQ1 tables (summaries, Shapley, learner comparisons)."""
    pdir, t = out / "data" / "parts", out / "tables"

    def cat(name, **kw):
        L = [pd.read_csv(p, **kw) for p in sorted(pdir.glob(f"{name.format(tag='*')}")) if p.is_file()]
        return pd.concat(L, ignore_index=True) if L else None

    R = cat("rq1_{tag}_window_metrics.csv")
    if R is not None:
        R.to_csv(t / "rq1_window_metrics.csv", index=False)
        P = cat("rq1_{tag}_reference_predictions.csv.gz")
        P.to_csv(out / "data" / "rq1_reference_predictions.csv.gz", index=False, compression="gzip")
        Rm = R[~R["shortcuts"].isin(["Graw", "Gloo"])]
        rq1.summarize_inflation(R, cfg).to_csv(t / "rq1_inflation.csv", index=False)
        rq1.shapley_decomposition(Rm, cfg).to_csv(t / "rq1_shapley.csv", index=False)
        rq1.ranking_flips(R, cfg).to_csv(t / "rq1_ranking_flip.csv", index=False)
        T, Ts = rq1.threshold_experiment(P, cfg)
        T.to_csv(t / "rq1_threshold_windows.csv", index=False); Ts.to_csv(t / "rq1_threshold_summary.csv", index=False)
        G = cat("rq1_{tag}_deployment_windows.csv")
        G.to_csv(t / "rq1_deployment_windows.csv", index=False)
        rq1.deployment_summary(G, R, cfg).to_csv(t / "rq1_deployment_summary.csv", index=False)
    T = cat("rq1b_{tag}_window_metrics.csv")
    if T is not None:
        Ch = cat("rq1b_{tag}_chosen_params.csv")
        T.to_csv(t / "rq1b_tuned_window_metrics.csv", index=False)
        Ch.to_csv(t / "rq1b_tuned_chosen_params.csv", index=False)
        rq1.tuned_complexity(Ch).to_csv(t / "rq1b_tuned_complexity.csv", index=False)
        rq1.ranking_flips(T, cfg).to_csv(t / "rq1b_tuned_ranking_flip.csv", index=False)
        rq1.summarize_inflation(T, cfg).to_csv(t / "rq1b_tuned_inflation.csv", index=False)


def _rq2_milestone(cfg, m, quick):
    """One prediction milestone of RQ2, run in its own process."""
    import warnings
    warnings.filterwarnings("ignore")
    out = cfg["out_dir"]
    if quick:
        cfg["rolling"]["step_days"] = 140
    F = load_features(out)
    raw, _ = load_raw(cfg["data_dir"], verify=False)
    return rq2.run_rq2(F, raw, cfg, milestones=[m])


def stage_rq2(cfg, out, args):
    """The six milestones run in parallel processes (three at a time); the results do not depend on the order."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(3, len(rq2.MILESTONES))) as ex:
        res = list(ex.map(_rq2_milestone, [cfg] * len(rq2.MILESTONES), rq2.MILESTONES,
                          [args.quick] * len(rq2.MILESTONES)))
    R, L, Ch, Sc = (pd.concat([r[i] for r in res], ignore_index=True) for i in range(4))
    if args.quick:
        cfg["rolling"]["step_days"] = 140
    t = out / "tables"
    R.to_csv(t / "rq2_window_metrics.csv", index=False)
    Ch.to_csv(t / "rq2_chosen_params.csv", index=False)
    Sc.to_csv(out / "data" / "rq2_scores.csv.gz", index=False, compression="gzip")
    rq2.summarize_rq2(R, L, cfg).to_csv(t / "rq2_summary.csv", index=False)
    rq2.paired_vs_t0(R, cfg).to_csv(t / "rq2_paired_vs_t0.csv", index=False)


def stage_rq3(cfg, out, args):
    F = load_features(out)
    if args.quick:
        cfg["rolling"]["step_days"] = 140
    R, Dd, C, Ch = rq3.run_rq3(F, cfg)
    t = out / "tables"
    R.to_csv(t / "rq3_window_metrics.csv", index=False)
    Ch.to_csv(t / "rq3_chosen_params.csv", index=False)
    Dd.to_csv(t / "rq3_daily_policy.csv", index=False)
    rq3.summarize_rq3(R, cfg).to_csv(t / "rq3_summary.csv", index=False)
    pd.concat([rq3.paired_strategy(R, cfg), rq3.paired_strategy(R, cfg, "platt_ipw_roll", "platt"),
               rq3.paired_strategy(R, cfg, "platt_ipw_nowcast", "platt"), rq3.paired_strategy(R, cfg, "platt_ipw_roll", "raw"),
               rq3.paired_strategy(R, cfg, "platt", "static")]).to_csv(t / "rq3_paired_strategies.csv", index=False)
    pd.concat([rq3.decision_curve(C, m) for m in ("xgb", "lr")]).to_csv(t / "rq3_decision_curve.csv", index=False)
    C.to_csv(out / "data" / "rq3_predictions.csv.gz", index=False, compression="gzip")


def _task2_data(cfg, out, args):
    raw, _ = load_raw(cfg["data_dir"], verify=False)
    D, flow = late_delivery.build_task2(raw, cfg)
    if args.quick:
        cfg["rolling"]["step_days"] = 98
    return D, flow, {c: sorted(D[c].unique()) for c in late_delivery.T2_CAT}


def stage_task2(cfg, out, args):
    """Second task: factorial design and realised performance (fixed configurations)."""
    D, flow, vocab = _task2_data(cfg, out, args)
    t = out / "tables"; fs = late_delivery.t2_feature_set
    flow.to_csv(t / "task2_sample_flow.csv", index=False, encoding="utf-8-sig")
    R, _ = rq1.run_factorial(D, vocab, cfg, extra_conditions=[dict(split="temporal", purge=True, agg="global_raw",
                                                                   post=False, shortcuts="Graw")], fs=fs)
    R.to_csv(t / "task2_window_metrics.csv", index=False)
    Rm = R[R["shortcuts"] != "Graw"]
    rq1.summarize_inflation(R, cfg).to_csv(t / "task2_inflation.csv", index=False)
    rq1.shapley_decomposition(Rm, cfg).to_csv(t / "task2_shapley.csv", index=False)
    rq1.ranking_flips(R, cfg).to_csv(t / "task2_ranking_flip.csv", index=False)
    G = rq1.run_deployment_gap(D, vocab, cfg, fs=fs, cols=late_delivery.T2_COLS)
    G.to_csv(t / "task2_deployment_windows.csv", index=False)
    rq1.deployment_summary(G, R, cfg).to_csv(t / "task2_deployment_summary.csv", index=False)


def stage_task2b(cfg, out, args):
    """Second task: learner comparison with nested tuning (a separate process, so that it runs in parallel)."""
    D, _, vocab = _task2_data(cfg, out, args)
    t = out / "tables"
    T, Ch = rq1.run_tuned_ranking(D, vocab, cfg, fs=late_delivery.t2_feature_set, grid=rq1.TUNE_GRID_TASK2)
    T.to_csv(t / "task2_tuned_window_metrics.csv", index=False)
    Ch.to_csv(t / "task2_tuned_chosen_params.csv", index=False)
    rq1.tuned_complexity(Ch, rq1.TUNE_GRID_TASK2).to_csv(t / "task2_tuned_complexity.csv", index=False)
    rq1.ranking_flips(T, cfg).to_csv(t / "task2_tuned_ranking_flip.csv", index=False)


def stage_robustness(cfg, out, args):
    from .experiments.robustness import run_robustness, GROUPS
    groups = GROUPS if args.groups in (None, "all") else tuple(args.groups.split(","))
    run_robustness(cfg, groups=groups, workers=args.workers, quick=args.quick)


def stage_inference(cfg, out, args):
    from .reporting.inference import paired_contrasts
    from .reporting.robustness_summary import summarize
    paired_contrasts(cfg, out)
    if (out / "robustness").is_dir():
        summarize(cfg, out, quick=args.quick)


def stage_figures(cfg, out, args):
    from .reporting import figures
    figures.make_all(cfg, out)


def stage_manifest(cfg, out, args):
    libs = {}
    for lib in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "matplotlib", "pyyaml"):
        try:
            libs[lib] = metadata.version(lib)
        except Exception:
            libs[lib] = "n/a"
    files = [dict(file=p.relative_to(out).as_posix(), sha256=sha256(p)) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and "parts" not in p.parts]
    # code that produced the outputs: one hash per source file (line endings normalised, so that a checkout with
    # Windows line endings gives the same values) and one hash over all of them
    root = ROOT
    src = sorted([*(root / "src").rglob("*.py"), root / "run_pipeline.py", root / "config.yaml", root / "requirements-lock.txt"])
    code = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in src if p.is_file()}
    code_all = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    man = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(), platform=platform.platform(),
               libs=libs, seed=cfg["seed"], quick=args.quick, code_sha256=code_all, code_files=code, outputs=files)
    (out / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")


STAGES = {"sample": stage_sample, "rq1": stage_rq1, "rq1_rf": stage_rq1_rf, "rq1b": stage_rq1b, "rq1b_rf": stage_rq1b_rf,
          "rq2": stage_rq2, "rq3": stage_rq3, "task2": stage_task2, "task2b": stage_task2b,
          "robustness": stage_robustness,
          "collect": stage_collect, "inference": stage_inference, "figures": stage_figures, "manifest": stage_manifest}


# ----------------------------------------------------------------------------------------------- orchestration
def run_stage(name, cfg, out, args):
    t = time.time()
    log(f"Stage {name}: start")
    STAGES[name](cfg, out, args)
    log(f"Stage {name}: done ({(time.time() - t) / 60:.1f} min)")


def _child_command(stage, args):
    cmd = [sys.executable, "-u", str(ROOT / "run_pipeline.py"), "--stage", stage]
    if args.quick:
        cmd.append("--quick")
    if args.workers:
        cmd += ["--workers", str(args.workers)]
    if args.config:
        cmd += ["--config", str(args.config)]
    if stage == "robustness" and args.groups:
        cmd += ["--groups", args.groups]
    return cmd


def run_parallel(stages, out, args):
    """Run independent analysis stages as separate processes; each writes its own log file in outputs/logs."""
    logs = out / "logs"; logs.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    procs = {}
    for s in stages:
        fh = open(logs / f"{s}.log", "w", encoding="utf-8")
        procs[s] = (subprocess.Popen(_child_command(s, args), stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env), fh)
        log(f"Stage {s}: started (log: outputs/logs/{s}.log)")
    failed, t0 = [], time.time()
    while procs:
        for s, (p, fh) in list(procs.items()):
            if p.poll() is not None:
                fh.close(); del procs[s]
                if p.returncode == 0:
                    log(f"Stage {s}: done ({(time.time() - t0) / 60:.1f} min since start)")
                else:
                    failed.append(s); log(f"Stage {s}: FAILED (exit code {p.returncode}); see outputs/logs/{s}.log")
        time.sleep(5)
    return failed


def main(argv=None):
    import argparse
    import warnings
    from .config import load_config
    warnings.filterwarnings("ignore", message="Unknown solver options")
    ap = argparse.ArgumentParser(description="Leakage-aware evaluation pipeline (Olist).")
    ap.add_argument("--stage", default="all",
                    help="all (default), one stage or a comma-separated list run in order: " + ", ".join(ALL_STAGES))
    ap.add_argument("--config", default=None, help="path to an alternative config.yaml")
    ap.add_argument("--quick", action="store_true", help="smoke test on a few windows")
    ap.add_argument("--workers", type=int, default=0, help="worker processes of the robustness stage (default: CPUs / 4)")
    ap.add_argument("--groups", default=None, help="comma-separated robustness groups (default: all)")
    ap.add_argument("--sequential", action="store_true", help="run the analysis stages one after another")
    ap.add_argument("--skip-robustness", action="store_true", help="omit the robustness stage")
    ap.add_argument("--reuse-sample", action="store_true",
                    help="skip the sample stage when outputs/data/features_t0.pkl already exists")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = cfg["out_dir"]
    for d in ("tables", "figures", "data", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.stage != "all":
        stages = [s.strip() for s in args.stage.split(",") if s.strip()]
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            ap.error(f"unknown stage(s): {', '.join(unknown)}; choose from {', '.join(ALL_STAGES)}")
        for s in stages:
            run_stage(s, load_config(args.config), out, args)
        return
    if not (args.reuse_sample and (out / "data" / "features_t0.pkl").is_file()):
        run_stage("sample", cfg, out, args)
    analysis = [s for s in ANALYSIS_STAGES if not (args.skip_robustness and s == "robustness")
                and not (s.endswith("_rf") and "rf" not in cfg.get("learners", []))]
    if args.sequential:
        failed = []
        for s in analysis:
            try:
                run_stage(s, load_config(args.config), out, args)
            except Exception as e:
                failed.append(s); log(f"Stage {s}: FAILED ({e!r})")
    else:
        failed = run_parallel(analysis, out, args)
    if failed:
        log(f"Stopped before the final stages because these stages failed: {', '.join(failed)}. "
            "Fix the cause and rerun them with --stage <name>, then run --stage collect, inference, figures and manifest.")
        sys.exit(1)
    for s in FINAL_STAGES:
        run_stage(s, load_config(args.config), out, args)
    log(f"Pipeline finished in {(time.time() - t0) / 60:.1f} min")
