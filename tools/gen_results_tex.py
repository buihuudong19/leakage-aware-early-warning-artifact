"""Write the LaTeX number macros of the article from the outputs of a full run.

usage: python tools/gen_results_tex.py OUTPUTS_DIR RESULTS_TEX TAB_RQ3_TEX

RESULTS_TEX receives one \newcommand per number printed in the article; TAB_RQ3_TEX receives the body of the table
of recalibration strategies, and supp_tables.tex (written next to it) the tables of the supplementary material.
Every value is computed from OUTPUTS_DIR/tables/*.csv, OUTPUTS_DIR/data/features_t0.pkl (sample stage),
OUTPUTS_DIR/logs/pipeline.log (run time), config.yaml, the code, and the files in docs/ (literature audit and the
earlier run of docs/supplementary).
"""
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) != 4:
    raise SystemExit(__doc__)
OUT = Path(sys.argv[1]); NEW = Path(sys.argv[2]); TAB = Path(sys.argv[3])
T = OUT / "tables"
r = lambda f: pd.read_csv(T / f, encoding="utf-8-sig")
import os
for _cand in (Path(os.environ.get("LEAKAWARE_REPO", "")), Path(__file__).resolve().parents[1]):
    if (_cand / "src" / "leakaware").is_dir():
        sys.path.insert(0, str(_cand / "src")); break


def num(x, d=3):
    s = f"{abs(x):.{d}f}"
    return f"$-{s}$" if x < 0 and float(s) != 0 else s


def pct(x, d=1):
    return f"{x:.{d}f}\\%"


def ci(lo, hi, d=3, scale=1.0):
    lo, hi = lo * scale, hi * scale
    if lo < 0 or hi < 0:
        sgn = lambda x: f"$-{abs(x):.{d}f}$" if x < 0 else f"{x:.{d}f}"  # keep the sign of a bound that rounds to zero
        return f"{sgn(lo)} to {sgn(hi)}"
    return f"{lo:.{d}f}--{hi:.{d}f}"


def big(n):
    return f"{int(round(n)):,}".replace(",", "{,}")


blocks = []          # (comment, [(name, value, note)])


def block(title):
    blocks.append((title, []))


def m(name, value, note=""):
    blocks[-1][1].append((name, value, note))


# ------------------------------------------------------------------ run time (pipeline.log, UTF-16 when written by PowerShell)
log = OUT / "logs" / "pipeline.log"
if not log.is_file():
    raise SystemExit(f"{log} not found: the run time is read from the log of a full run")
raw = log.read_bytes()
txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
stamps = re.findall(r"^\[(\d\d):(\d\d):(\d\d)\]", txt, re.M)
sec = [int(h) * 3600 + int(mi) * 60 + int(s) for h, mi, s in stamps]
total = 0.0
for a, b in zip(sec, sec[1:]):                              # wall time, allowing one pass over midnight
    total += (b - a) % 86400
stage = {}
for k, v in re.findall(r"Stage (\w+): done \(([\d.]+) min", txt):   # an appended single-stage run never replaces
    stage[k] = max(stage.get(k, 0.0), float(v))                       # the longer full-run entry
fin = re.findall(r"Pipeline finished in ([\d.]+) min", txt)
if fin:
    total = 60 * max(float(f) for f in fin)                           # the full run is the longest run in the log
block("computing time (outputs/logs/pipeline.log of the final full run)")
m("RunTimeMin", f"{total / 60:.0f}")
m("RunTimeHours", f"{total / 3600:.1f}")
m("RunTimeRQtwo", f"{stage.get('rq2', float('nan')):.0f}")
m("RunTimeRobust", f"{stage.get('robustness', float('nan')):.0f}")

# ------------------------------------------------------------------ sample and latency: not from the tables
block("sample (sample_flow.csv, sample_notes.json)")
import json
from leakaware.data.features import feature_set
SF = r("sample_flow.csv").n_orders.tolist(); SN = json.loads((T / "sample_notes.json").read_text(encoding="utf-8-sig"))
for name, v in zip(("NOrdersAll", "NDelivered", "NHandover", "NReviewed"), SF[:4]):
    m(name, big(v))
m("ExclNonNeutralShare", pct(100 * (SF[0] - SN["primary_n"] - SN["neutral_3star"]) / SF[0]), "orders removed by delivery, hand-over, review and feedback-before conditions")
m("NFeedbackBefore", big(SN["excluded_feedback_before_t0"])); m("NFeedbackBeforeLow", big(SN["excluded_feedback_before_t0_low"]))
m("NNeutral", big(SN["neutral_3star"])); m("NAnalysis", big(SN["primary_n"])); m("NPositive", big(SN["primary_pos"]))
m("PrevAll", pct(100 * SN["primary_prevalence"], 2)); m("NMultiReview", big(SN["orders_multi_review"]))
_fn, _fc = feature_set("asof"); m("NFeatures", str(len(_fn) + len(_fc)))
W = r("rq1_window_metrics.csv"); ref = W[(W.model == "xgb") & (W.shortcuts == "none")]
block("rolling-origin design (rq1_window_metrics.csv, reference rows)")
m("NOrigins", str(ref.k.nunique())); m("NEvalOrders", big(ref.n.sum())); m("NEvalPositive", big(ref.pos.sum()))
m("NTrainMin", big(ref.n_train.min())); m("NTrainMax", big(ref.n_train.max()))
block("label latency (features_t0.pkl; days after hand-over t0)")
_F = pd.read_pickle(OUT / "data" / "features_t0.pkl"); _D = _F[_F.y >= 0]
_lag = (_D.label_available_ts - _D.t0).dt.total_seconds() / 86400
_dlag = (_D.label_available_ts - _D.delivered_ts).dt.total_seconds() / 86400
_tr = (_D.delivered_ts - _D.t0).dt.total_seconds() / 86400
m("LatMedAll", f"{_lag.median():.1f}")
for c, L in ((1, "Low"), (0, "High")):
    lc = _lag[_D.y == c]
    m(f"LatMed{L}", f"{lc.median():.1f}"); m(f"LatFourteen{L}", pct(100 * (lc <= 14).mean()))
    m(f"LatTwentyEight{L}", pct(100 * (lc <= 28).mean())); m(f"LatDeliv{L}", f"{_dlag[_D.y == c].median():.2f}")
    m(f"Lat{L}Fast", f"{_lag[(_D.y == c) & (_tr <= 7)].median():.1f}"); m(f"Lat{L}Slow", f"{_lag[(_D.y == c) & (_tr > 7)].median():.1f}")
_u = np.arange(0.5, 28.001, 0.5)
_G = {c: np.array([(_lag[_D.y == c] <= u).mean() for u in _u]) for c in (0, 1)}
m("LatCapAge", f"{_u[np.argmax(_G[1] >= 1 / 20)]:.1f}", "first half-day age with G_1(u) >= 1/20 (full sample)")
_ok = _G[1] < _G[0]; _first = _u[np.argmax(_ok & np.flip(np.cumprod(np.flip(_ok))).astype(bool))]
m("LatCrossFrom", f"{_first:.1f}", "G_1(u) < G_0(u) for every half-day age from this value to 28 days (full sample)")
block("labels not yet available at the origin (features_t0.pkl)")
FEAT = OUT / "data" / "features_t0.pkl"
if FEAT.exists():  # same population and windows as Fig. 3c (figures.fig_context)
    import os
    import yaml
    for cand in (Path(os.environ.get("LEAKAWARE_REPO", "")), Path(__file__).resolve().parents[1]):
        if (cand / "src" / "leakaware").is_dir():
            sys.path.insert(0, str(cand / "src")); REPO = cand; break
    from leakaware.evaluation.splits import rolling_windows
    Fd = pd.read_pickle(FEAT); Fd = Fd[Fd.y >= 0]
    sh = []
    for w in rolling_windows(yaml.safe_load(open(REPO / "config.yaml", encoding="utf-8"))):
        past = Fd.t0 < w["origin"]; rec = past & (Fd.t0 >= w["origin"] - pd.Timedelta(days=28))
        un = past & (Fd.label_available_ts >= w["origin"])
        sh.append((un.sum() / past.sum(), (un & rec).sum() / rec.sum()))
    sh = pd.DataFrame(sh, columns=["all", "recent"])
    m("NPLeakLo", f"{100 * sh.recent.min():.0f}\\%"); m("NPLeakHi", f"{100 * sh.recent.max():.0f}\\%")
    m("NPLeakMed", f"{100 * sh.recent.median():.0f}\\%")
    m("NPLeakAllLo", f"{100 * sh['all'].min():.0f}\\%"); m("NPLeakAllHi", f"{100 * sh['all'].max():.0f}\\%")
    a1 = list(rolling_windows(yaml.safe_load(open(REPO / "config.yaml", encoding="utf-8"))))[0]["origin"]
    m("NTrainFrozen", big(((Fd.t0 < a1 - pd.Timedelta(days=28)) & (Fd.label_available_ts < a1)).sum()),
      "frozen RQ3 model: t0 < o_1 - 28 d, label known at o_1")
    # fallbacks of the latency weights (< 50 matured orders of a class) and of the nowcast (< 30 labelled orders)
    _mat, _now = [], []
    for w in rolling_windows(yaml.safe_load(open(REPO / "config.yaml", encoding="utf-8"))):
        o = w["origin"]; mt = (Fd.t0 < o - pd.Timedelta(days=90)) & (Fd.label_available_ts < o)
        roll = mt & (Fd.t0 >= o - pd.Timedelta(days=90 + 180))
        _mat += [int((mt & (Fd.y == c)).sum()) for c in (0, 1)] + [int((roll & (Fd.y == c)).sum()) for c in (0, 1)]
        _now.append(int(((Fd.t0 >= o - pd.Timedelta(days=14)) & (Fd.t0 < o) & (Fd.label_available_ts < o)).sum()))
    m("NMaturedMin", big(min(_mat)), "fewest matured orders of a class (expanding or rolling) over origins; fallback below 50")
    m("NNowcastMin", big(min(_now)), "fewest labelled orders in the last 14 days over origins; fallback below 30")
else:
    raise SystemExit(f"{FEAT} not found: run `python run_pipeline.py --stage sample` first")

# ------------------------------------------------------------------ RQ1 inflation
I = r("rq1_inflation.csv"); g = lambda mm, sc, met="ap": I[(I.model == mm) & (I.shortcuts == sc) & (I.metric == met)].iloc[0]
L3 = (("xgb", "Xgb"), ("lr", "Lr"), ("rf", "Rf"))
block("reference protocol, fixed configuration (rq1_inflation.csv)")
for mm, L in L3:
    m(f"RefAp{L}", f"{g(mm, 'R+G+P').ref_mean:.3f}"); m(f"RefAuc{L}", f"{g(mm, 'R+G+P', 'auc').ref_mean:.3f}")
m("RefPrev", pct(100 * ref.prevalence.mean()))
block("RQ1 inflation, PR-AUC, fixed configuration: delta (95% CI) and relative gain (rq1_inflation.csv)")
for mm, L in L3:
    a = g(mm, "R+G+P")
    m(f"InfAll{L}", f"{a.delta:.3f}"); m(f"InfAll{L}CI", ci(a.ci_lo, a.ci_hi)); m(f"InfAll{L}Rel", pct(a.rel_pct))
    m(f"AllAp{L}", f"{a.cond_mean:.3f}"); m(f"AllAuc{L}", f"{g(mm, 'R+G+P', 'auc').cond_mean:.3f}")
    for sc in ("P", "R", "G", "Graw", "NP"):
        m(f"Inf{sc}{L}Rel", pct(g(mm, sc).rel_pct))
    m(f"InfNP{L}", f"{g(mm, 'NP').delta:.3f}"); m(f"InfNP{L}CI", ci(g(mm, "NP").ci_lo, g(mm, "NP").ci_hi))
    m(f"Gloo{L}Rel", pct(g(mm, "Gloo").rel_pct)); m(f"Gloo{L}Mean", f"{g(mm, 'Gloo').cond_mean:.4f}")
m("GlooXgbMin", f"{W[(W.model == 'xgb') & (W.shortcuts == 'Gloo')].ap.min():.4f}", "lowest window")
TI = r("rq1b_tuned_inflation.csv"); ti = lambda mm: TI[(TI.model == mm) & (TI.shortcuts == "R+G+P") & (TI.metric == "ap")].iloc[0]
block("RQ1 inflation with nested tuning, R+G+P (rq1b_tuned_inflation.csv): headline values")
for mm, L in L3:
    m(f"TunInfAll{L}", pct(ti(mm).rel_pct)); m(f"TunRefAp{L}", f"{ti(mm).ref_mean:.3f}"); m(f"TunAllAp{L}", f"{ti(mm).cond_mean:.3f}")
tun = [ti(mm) for mm, _ in L3]
m("TunRefApLo", f"{min(t.ref_mean for t in tun):.2f}"); m("TunRefApHi", f"{max(t.ref_mean for t in tun):.2f}")
m("TunAllApLo", f"{min(t.cond_mean for t in tun):.2f}"); m("TunAllApHi", f"{max(t.cond_mean for t in tun):.2f}")
m("TunInfLo", f"{min(t.rel_pct for t in tun):.0f}"); m("TunInfHi", f"{max(t.rel_pct for t in tun):.0f}")

# ------------------------------------------------------------------ Shapley
S = r("rq1_shapley.csv"); sh = lambda mm, f, met="ap": S[(S.model == mm) & (S.factor == f) & (S.metric == met)].iloc[0]
block("RQ1 Shapley attribution, PR-AUC, fixed configuration (rq1_shapley.csv)")
for mm, L in L3:
    m(f"ShapP{L}", f"{sh(mm, 'P').phi:.3f}")
    for f in ("P", "G", "R"):
        m(f"Shap{f}{L}Share", pct(100 * sh(mm, f).share_of_total))
m("ShapNPShareMax", f"{100 * max(sh(mm, 'NP').share_of_total for mm, _ in L3):.0f}\\%")
ga = [100 * sh(mm, "G", "auc").share_of_total for mm, _ in L3]
m("ShapGAucShareLo", f"{min(ga):.0f}\\%"); m("ShapGAucShareHi", f"{max(ga):.0f}\\%")
pa = [100 * sh(mm, "P").share_of_total for mm, _ in L3]
m("ShapPShareLo", f"{min(pa):.0f}"); m("ShapPShareHi", f"{max(pa):.0f}")
SR = r("robustness_shapley_rules.csv"); p_ = SR[SR.shortcut == "P"]
rules = p_[["shapley", "standalone", "last_in", "shapley_RGP", "shapley_NPGP"]].to_numpy().ravel()
block("Shapley sensitivity (robustness_shapley_rules.csv): share of P over rules, settings and learners")
m("ShapPRuleLo", f"{100 * np.nanmin(rules):.0f}"); m("ShapPRuleHi", f"{100 * np.nanmax(rules):.0f}")
for mm, L in (("xgb", "Xgb"), ("lr", "Lr")):
    m(f"ShapPNested{L}Share", pct(100 * p_[(p_.setting == "nested tuning") & (p_.model == mm)].shapley.iloc[0]))

# ------------------------------------------------------------------ realised performance
D = r("rq1_deployment_summary.csv"); dp = lambda mm, v, met="ap": D[(D.model == mm) & (D.variant == v) & (D.metric == met)].iloc[0]
block("RQ1 reported versus realised, PR-AUC, fixed configuration (rq1_deployment_summary.csv)")
x = dp("xgb", "G+P"); m("DepGPXgbRep", f"{x.reported_mean:.3f}"); m("DepGPXgbReal", f"{x.realized_mean:.3f}")
m("DepGPXgb", f"{-x.realized_minus_correct:.3f}"); m("DepGPXgbCI", ci(-x.ci_hi, -x.ci_lo))
m("DepGPXgbRel", f"{-100 * x.realized_minus_correct / x.correct_mean:.0f}\\%")
x = dp("xgb", "P"); m("DepPXgb", f"{-x.realized_minus_correct:.3f}"); m("DepPXgbCI", ci(-x.ci_hi, -x.ci_lo))
m("DepGPXgbPten", f"{-dp('xgb', 'G+P', 'p_at_10').realized_minus_correct:.3f}")
for mm, L in (("lr", "Lr"), ("rf", "Rf")):
    v = D[(D.model == mm) & (D.metric == "ap")].realized_minus_correct.abs()
    m(f"Dep{L}Min", f"{v.min():.3f}"); m(f"Dep{L}Max", f"{v.max():.3f}")
x = dp("xgb", "Graw"); m("DepGrawXgb", f"{-x.realized_minus_correct:.3f}"); m("DepGrawXgbCI", ci(x.ci_lo, x.ci_hi))
SC = r("supp_paired_contrasts.csv"); dep = SC[SC.family == "RQ1 realised performance, PR-AUC"]
m("DepNAll", str(len(dep))); m("DepNBelow", str(int((dep["mean"] < 0).sum())))
m("DepSigAll", str(int(((dep.p_holm < 0.05) & (dep["mean"] < 0)).sum())))
nsig = dep[~((dep.p_holm < 0.05) & (dep["mean"] < 0))]
m("DepNotSig", "; ".join(f"{a} {b}" for a, b in zip(nsig.model, nsig.contrast)), "the exceptions")
SV = r("robustness_serving_rules.csv"); sv = lambda mm, v, rl: SV[(SV.model == mm) & (SV.pipeline == v) & (SV.rule == rl)].iloc[0]
block("serving rules for the missing post-hand-over inputs (robustness_serving_rules.csv), relative realised loss")
for mm, L in (("xgb", "Xgb"), ("lr", "Lr")):
    for v, VL in (("P", "P"), ("G+P", "GP")):
        for rl, RL in (("missing", "Miss"), ("train_median", "Med"), ("promise", "Prom"), ("expected", "Exp")):
            m(f"Serv{VL}{RL}{L}", f"{-sv(mm, v, rl).rel_pct:.1f}\\%")
    m(f"ServPExp{L}P", f"{sv(mm, 'P', 'expected').p_holm:.2f}", "Holm p")
m("ServMax", f"{-SV.rel_pct.min():.0f}\\%"); m("ServMin", f"{-SV.rel_pct.max():.0f}\\%")
gpe = SV[(SV.pipeline == "G+P") & (SV.rule == "expected")].rel_pct
m("ServGPExpLo", f"{-gpe.max():.1f}\\%"); m("ServGPExpHi", f"{-gpe.min():.1f}\\%")

# ------------------------------------------------------------------ ranking
F = r("rq1_ranking_flip.csv"); FB = r("rq1b_tuned_ranking_flip.csv")
fl = lambda sc, a="xgb", b="lr", met="ap", t=F: t[(t.a == a) & (t.b == b) & (t.shortcuts == sc) & (t.metric == met)].iloc[0]
block("RQ1 learner comparison, PR-AUC difference (rq1_ranking_flip.csv, rq1b_tuned_ranking_flip.csv)")
x = fl("none"); m("RankFixRef", num(x["diff"])); m("RankFixRefCI", ci(x.ci_lo, x.ci_hi)); m("RankFixRefWins", f"{100 * x.a_wins_share:.0f}\\%")
x = fl("R"); m("RankFixR", f"{x['diff']:.3f}"); m("RankFixRCI", ci(x.ci_lo, x.ci_hi))
x = fl("R+G+P"); m("RankFixAll", f"{x['diff']:.3f}"); m("RankFixAllCI", ci(x.ci_lo, x.ci_hi)); m("RankFixAllWins", f"{100 * x.a_wins_share:.0f}\\%")
x = fl("none", t=FB); m("RankTunRefRelLr", pct(100 * abs(x["diff"]) / TI[(TI.model == "lr") & (TI.shortcuts == "R+G+P") & (TI.metric == "ap")].iloc[0].ref_mean), "|XGB-LR| relative to tuned LR reference PR-AUC"); m("RankTunRef", num(x["diff"])); m("RankTunRefCI", ci(x.ci_lo, x.ci_hi)); m("RankTunRefWins", f"{100 * x.a_wins_share:.0f}\\%")
x = fl("none", met="p_at_10", t=FB); m("RankTunRefPten", num(x["diff"])); m("RankTunRefPtenCI", ci(x.ci_lo, x.ci_hi))
x = fl("R", t=FB); m("RankTunR", f"{x['diff']:.3f}"); m("RankTunRCI", ci(x.ci_lo, x.ci_hi)); m("RankTunRWins", f"{100 * x.a_wins_share:.0f}\\%")
x = fl("R+G+P", t=FB); m("RankTunAll", f"{x['diff']:.3f}"); m("RankTunAllCI", ci(x.ci_lo, x.ci_hi))
for a, b, L in (("rf", "lr", "RfLr"), ("xgb", "rf", "XgbRf")):
    for sc, s in (("none", "Ref"), ("R", "R"), ("R+G+P", "All")):
        for t, tag in ((F, "Fix"), (FB, "Tun")):
            x = fl(sc, a, b, t=t); m(f"Rank{tag}{s}{L}", num(x["diff"])); m(f"Rank{tag}{s}{L}CI", ci(x.ci_lo, x.ci_hi))
fam = SC[SC.family == "RQ1 learner comparison, nested tuning"]
pr_ = lambda pair, sc: fam[(fam.model == pair) & (fam.contrast == sc)].iloc[0].p_holm
m("RankTunRXgbRfP", f"{pr_('xgb - rf', 'R'):.2f}", "Holm p"); m("RankTunRefXgbRfP", f"{pr_('xgb - rf', 'none'):.2f}")
m("RankTunRefRfLrP", f"{pr_('rf - lr', 'none'):.2f}")
SE = r("robustness_seeds_summary.csv"); se = lambda sc: SE[SE.condition == sc].iloc[0]
block("learner comparison over seeds, nested tuning, XGBoost - LR (robustness_seeds_summary.csv)")
m("SeedN", str(int(se("none").n_seeds)))
for sc, L in (("none", "Ref"), ("R", "R")):
    s_ = se(sc)
    m(f"Seed{L}Mean", num(s_["mean"])); m(f"Seed{L}Min", num(s_["min"])); m(f"Seed{L}Max", num(s_["max"]))
    m(f"Seed{L}CI", ci(s_.ci_lo, s_.ci_hi, 4))
m("SeedRefNSig", str(int(se("none").n_ci_below_0))); m("SeedRNSig", str(int(se("R").n_ci_above_0)))
CP = r("rq1b_tuned_chosen_params.csv"); CP["p"] = CP.params.apply(ast.literal_eval)
par = lambda mm, sc: pd.DataFrame(list(CP[(CP.model == mm) & (CP.shortcuts == sc)].p))
block("selected hyper-parameters, nested tuning (rq1b_tuned_chosen_params.csv)")
xr, lr_, rr = par("xgb", "none"), par("lr", "none"), par("rf", "none")
m("TunDepthShallow", f"{100 * (xr.max_depth <= 3).mean():.0f}\\%", "XGBoost depth <= 3, reference")
m("TunLrStrongReg", f"{100 * (lr_.C <= 0.01).mean():.0f}\\%", "LR C <= 0.01, reference")
rr_d = rr.max_depth.fillna(np.inf)
m("TunRfDeepRef", f"{100 * (rr_d >= 20).mean():.0f}\\%", "RF depth >= 20, reference")
rR = par("rf", "R").max_depth.fillna(np.inf); nR = int((rR >= 20).sum())
m("TunRfDeepR", "all five" if nR == len(rR) == 5 else f"{nR} of {len(rR)}", "RF depth >= 20, random split")
xR, lR = par("xgb", "R"), par("lr", "R")
m("TunRandomXgbDepth", f"{int(xR.max_depth.mode()[0])}"); m("TunRandomXgbDepthN", f"{int((xR.max_depth == xR.max_depth.mode()[0]).sum())} of {len(xR)}")
m("TunRandomLrC", f"{lR.C.mode()[0]:g}"); m("TunRandomLrCN", f"{int((lR.C == lR.C.mode()[0]).sum())} of {len(lR)}")

# ------------------------------------------------------------------ selection on test data
SS = r("rq1_selection_summary.csv"); TS = r("rq1_threshold_summary.csv")
block("RQ1 selection on test data (rq1_selection_summary.csv, rq1_threshold_summary.csv)")
m("SelOracleRel", pct(SS[(SS.metric == "ap") & (SS.shortcut == "oracle_per_window")].rel_pct.iloc[0]))
m("SelBoardRel", pct(SS[(SS.metric == "ap") & (SS.shortcut == "leaderboard_on_test")].rel_pct.iloc[0]))
for mm, L in (("lr", "Lr"), ("xgb", "Xgb"), ("rf", "Rf")):
    m(f"ThrFOne{L}Rel", pct(TS[TS.model == mm].rel_pct.iloc[0]))

# ------------------------------------------------------------------ task 2
block("Task 2: late delivery at payment approval (task2_*.csv)")
from leakaware.data.late_delivery import t2_feature_set
_T2 = r("task2_sample_flow.csv").n_orders.tolist()
m("TwoN", big(_T2[-2])); m("TwoPos", big(_T2[-1])); m("TwoPrev", pct(100 * _T2[-1] / _T2[-2]))
_tn, _tc = t2_feature_set("asof"); m("TwoNFeatures", str(len(_tn) + len(_tc)))
TW = r("task2_window_metrics.csv"); tref = TW[(TW.model == "xgb") & (TW.shortcuts == "none")]
m("TwoEvalN", big(tref.n.sum())); m("TwoEvalPrev", pct(100 * tref.prevalence.mean()))
I2 = r("task2_inflation.csv"); g2 = lambda mm, sc, met="ap": I2[(I2.model == mm) & (I2.shortcuts == sc) & (I2.metric == met)].iloc[0]
S2 = r("task2_shapley.csv")
for mm, L in (("xgb", "Xgb"), ("lr", "Lr")):
    m(f"TwoRefAp{L}", f"{g2(mm, 'R+G+P').ref_mean:.3f}"); m(f"TwoInfAll{L}Rel", f"{g2(mm, 'R+G+P').rel_pct:.0f}\\%")
    m(f"TwoAllAuc{L}", f"{g2(mm, 'R+G+P', 'auc').cond_mean:.3f}")
    m(f"TwoShapP{L}Share", f"{100 * S2[(S2.model == mm) & (S2.factor == 'P') & (S2.metric == 'ap')].share_of_total.iloc[0]:.0f}\\%")
m("TwoInfRXgbRel", f"{g2('xgb', 'R').rel_pct:.0f}\\%"); m("TwoInfNPXgbRel", f"{g2('xgb', 'NP').rel_pct:.0f}\\%")
F2 = r("task2_tuned_ranking_flip.csv")
x = fl("none", t=F2); m("TwoTunRef", num(x["diff"])); m("TwoTunRefCI", ci(x.ci_lo, x.ci_hi))
x = fl("R", t=F2); m("TwoTunR", f"{x['diff']:.3f}"); m("TwoTunRCI", ci(x.ci_lo, x.ci_hi))
D2 = r("task2_deployment_summary.csv"); d2 = D2[D2.metric == "ap"]
m("TwoDepSig", str(int((d2.ci_hi < 0).sum())), "interval below zero, of eight")
for v, L in (("G", "G"), ("Graw", "Graw")):
    m(f"TwoDep{L}Xgb", f"{d2[(d2.model == 'xgb') & (d2.variant == v)].realized_minus_correct.iloc[0]:.3f}")
block("tuned choices at the edge of the grid (rq1b/task2_tuned_complexity.csv, reference protocol)")
def edge(f, mm, param, value):
    E = r(f); e = E[(E.model == mm) & (E.shortcuts == "none") & (E.param == param)]
    e = e[np.isclose(e.value.astype(float), float(value))]
    return f"{100 * (e.share.iloc[0] if len(e) else 0):.0f}\\%"
m("EdgeXgbDepthOne", edge("rq1b_tuned_complexity.csv", "xgb", "max_depth", 1))
m("EdgeRfDepthInf", edge("rq1b_tuned_complexity.csv", "rf", "max_depth", np.inf))
m("EdgeLrCMin", edge("rq1b_tuned_complexity.csv", "lr", "C", 1e-4))
# task-2 grid: XGBoost depth up to 20 plus unlimited (max_depth=0, written as inf in the complexity table); C up to 1000
m("EdgeTwoXgbDepthMax", edge("task2_tuned_complexity.csv", "xgb", "max_depth", np.inf), "unlimited depth")
m("EdgeTwoXgbDepthTwenty", edge("task2_tuned_complexity.csv", "xgb", "max_depth", 20))
m("EdgeTwoXgbDepthTwelve", edge("task2_tuned_complexity.csv", "xgb", "max_depth", 12))
m("EdgeTwoLrCMax", edge("task2_tuned_complexity.csv", "lr", "C", 1000))
m("EdgeTwoLrCHundred", edge("task2_tuned_complexity.csv", "lr", "C", 100))
C2 = r("task2_tuned_chosen_params.csv"); C2["p"] = C2.params.apply(ast.literal_eval)
x2 = pd.DataFrame(list(C2[(C2.model == "xgb") & (C2.shortcuts == "none")].p))
x2R = pd.DataFrame(list(C2[(C2.model == "xgb") & (C2.shortcuts == "R")].p))
xd, xdR = x2.max_depth.replace(0, np.inf), x2R.max_depth.replace(0, np.inf)       # XGBoost max_depth=0: unlimited
m("TwoDepthDeep", f"{100 * (xd >= 5).mean():.0f}\\%", "depth >= 5 or unlimited, reference")
m("TwoDepthTop", f"{100 * (xd == np.inf).mean():.0f}\\%", "unlimited depth, reference")
mR = xdR.mode()[0]
m("TwoDepthR", "unlimited" if mR == np.inf else f"{int(mR)}"); m("TwoDepthRN", f"{int((xdR == mR).sum())} of {len(xdR)}")

# ------------------------------------------------------------------ RQ2
block("RQ2 milestones (rq2_summary.csv, rq2_paired_vs_t0.csv, robustness_milestone_*.csv)")
m("DelivBeforeSeven", pct(100 * (_tr < 7).mean(), 0), "delivery before t0 + 7 days, analysis sample")
Q = r("rq2_summary.csv"); q = lambda mm, ms: Q[(Q.model == mm) & (Q.milestone == ms)].iloc[0]
m("ShiftPurchase", f"{q('lr', 'purchase').train_cutoff_shift_days:.1f}"); m("ShiftApproval", f"{q('lr', 'approval').train_cutoff_shift_days:.1f}")
for mm, L in (("lr", "Lr"), ("xgb", "Xgb")):
    m(f"ECtZero{L}", pct(100 * q(mm, "t0").early_catch)); m(f"ECPurchase{L}", pct(100 * q(mm, "purchase").early_catch))
    m(f"ECDeliv{L}", pct(100 * q(mm, "delivery").early_catch)); m(f"ECSeven{L}", pct(100 * q(mm, "t0+7d").early_catch))
    m(f"AucDeliv{L}", f"{q(mm, 'delivery').auc:.3f}"); m(f"AucTzero{L}", f"{q(mm, 't0').auc:.3f}")
    for ms, ML in (("purchase", "Purchase"), ("t0", "Tzero"), ("t0+7d", "Seven"), ("delivery", "Deliv")):
        m(f"ApOpen{ML}{L}", f"{q(mm, ms).ap:.3f}")
PV = r("rq2_paired_vs_t0.csv"); pv = lambda mm, c: PV[(PV.model == mm) & (PV.comparison == f"t0 - {c}") & (PV.metric == "early_catch")].iloc[0]
for c, L in (("purchase", "Purch"), ("approval", "Appr"), ("t0+7d", "Seven"), ("delivery", "Deliv"), ("t0+3d", "Three")):
    dd = 2 if L == "Three" else 1                    # the hand-over vs three-days contrast is small: two decimals
    m(f"DEC{L}Lr", f"{100 * pv('lr', c)['diff']:.{dd}f}"); m(f"DEC{L}Xgb", f"{100 * pv('xgb', c)['diff']:.{dd}f}")
x = pv("xgb", "t0+3d"); m("DECThreeXgbCI", ci(x.ci_lo, x.ci_hi, 2, 100))
x = pv("lr", "t0+3d"); m("DECThreeLrCI", ci(x.ci_lo, x.ci_hi, 2, 100))
for ms, L in (("t0+3d", "Three"), ("t0+7d", "Seven"), ("delivery", "Deliv")):
    m(f"Lost{L}", f"{100 * (1 - q('lr', ms).share_pos_still_open):.0f}\\%")
# per-milestone table (Table 8): early catch rate, ROC-AUC and share of low-rated orders already reviewed
for ms, ML in (("purchase", "Purch"), ("approval", "Appr"), ("t0", "Tzero"), ("t0+3d", "Three"), ("t0+7d", "Seven"),
               ("delivery", "Deliv")):
    for mm, L in (("lr", "Lr"), ("xgb", "Xgb")):
        m(f"TabEC{ML}{L}", f"{100 * q(mm, ms).early_catch:.1f}"); m(f"TabAuc{ML}{L}", f"{q(mm, ms).auc:.3f}")
    m(f"TabLost{ML}", f"{100 * (1 - q('lr', ms).share_pos_still_open):.1f}")
MC = r("robustness_milestone_contrasts.csv")
near = MC[MC.contrast.isin(["t0 - t0+1d", "t0 - t0+2d", "t0 - t0+3d"])]
m("NearMaxPP", f"{100 * near['mean'].abs().max():.1f}", "largest |t0 - (t0+1..3d)| over learners and budgets, pp")
near23 = MC[MC.contrast.isin(["t0 - t0+2d", "t0 - t0+3d"])]
m("NearNSig", f"{int((near23.p_holm < 0.05).sum())} of {len(near23)}", "t0 vs t0+2d and t0+3d, significant after Holm")
m("OneNSig", f"{int(((MC.contrast == 't0 - t0+1d') & (MC.p_holm < 0.05)).sum())}")
far = MC[MC.contrast.isin(["t0 - purchase", "t0 - approval", "t0 - delivery"])]
m("FarAllSig", "yes" if ((far.p_holm < 0.05) & (far["mean"] > 0)).all() else "no")
sev = MC[MC.contrast == "t0 - t0+7d"]; m("SevenNSig", f"{int((sev.p_holm < 0.05).sum())} of {len(sev)}")

block("plain numbers for the graphical abstract (Figs/ga.tex)")
for ms, ML in (("purchase", "Purchase"), ("approval", "Appr"), ("t0", "Tzero"), ("t0+3d", "Three"), ("t0+7d", "Seven"), ("delivery", "Deliv")):
    m(f"GaEC{ML}Lr", f"{100 * q('lr', ms).early_catch:.1f}")
_B = r("robustness_recal_base_rates.csv").iloc[0]
m("GaPiLab", f"{100 * _B.pi_cal:.1f}"); m("GaPiIpw", f"{100 * _B.pi_ipw:.1f}"); m("GaPiEvent", f"{100 * _B.pi_eventual_28d:.1f}")

# ------------------------------------------------------------------ RQ3
block("RQ3 recalibration (rq3_summary.csv, rq3_paired_strategies.csv, supp_rq3_contrasts.csv, robustness_recal_*.csv)")
B = r("robustness_recal_base_rates.csv").iloc[0]
m("PiObserved", pct(100 * B.pi_eval)); m("PiLabelled", pct(100 * B.pi_cal)); m("PiIPW", pct(100 * B.pi_ipw))
m("PiEventual", pct(100 * B.pi_eventual_28d))
m("PiGapClosed", f"{100 * (B.pi_ipw - B.pi_cal) / (B.pi_eventual_28d - B.pi_cal):.0f}\\%")
R3 = r("rq3_summary.csv"); s3 = lambda mm, s: R3[(R3.model == mm) & (R3.strategy == s)].iloc[0]
for mm, L in (("xgb", "Xgb"), ("lr", "Lr")):
    m(f"CitlPlatt{L}", num(100 * s3(mm, "platt").citl_diff, 1)); m(f"CitlIpw{L}", num(100 * s3(mm, "platt_ipw").citl_diff, 1))
    m(f"InterPlatt{L}", num(s3(mm, "platt").cal_intercept, 2)); m(f"InterIpw{L}", num(s3(mm, "platt_ipw").cal_intercept, 2))
PS = r("rq3_paired_strategies.csv"); ps = lambda mm, c, met: PS[(PS.model == mm) & (PS.comparison == c) & (PS.metric == met)].iloc[0]
ix, il = ps("xgb", "platt_ipw - platt", "cal_intercept"), ps("lr", "platt_ipw - platt", "cal_intercept")
m("InterDiff", f"{-(ix['diff'] + il['diff']) / 2:.2f}")
m("InterDiffCIXgb", ci(-ix.ci_hi, -ix.ci_lo, 2)); m("InterDiffCILr", ci(-il.ci_hi, -il.ci_lo, 2))
for rr_, L in ((5, "Five"), (10, "Ten")):
    x = ps("xgb", "platt - static", f"saving_r{rr_}"); m(f"SavePlattStaticXgb{L}", f"{100 * x['diff']:.1f}")
    m(f"SavePlattStaticXgb{L}CI", ci(x.ci_lo, x.ci_hi, 1, 100))
m("ApPlattXgb", f"{s3('xgb', 'platt').ap:.3f}"); m("ApStaticXgb", f"{s3('xgb', 'static').ap:.3f}")
S3C = r("supp_rq3_contrasts.csv")
for mm, L in (("xgb", "Xgb"), ("lr", "Lr")):
    x = ps(mm, "platt_ipw - platt", "saving_r10"); m(f"SaveIpwTen{L}", f"{100 * x['diff']:.1f}"); m(f"SaveIpwTen{L}CI", ci(x.ci_lo, x.ci_hi, 1, 100))
    h = S3C[(S3C.model == mm) & (S3C.family == "RQ3 platt_ipw - platt") & (S3C.contrast == "saving_r10")].iloc[0]
    m(f"HolmIpwTen{L}", f"{h.p_holm:.3f}")
    h = S3C[(S3C.model == mm) & (S3C.family == "RQ3 platt_ipw - raw") & (S3C.contrast == "saving_r10")].iloc[0]
    m(f"IpwRawTen{L}", f"{100 * h['mean']:+.1f}".replace("-", "$-$")); m(f"IpwRawTen{L}CI", ci(h.ci_lo, h.ci_hi, 1, 100))
    m(f"RawInter{L}", f"{s3(mm, 'raw').cal_intercept:.2f}"); m(f"RawSaveTen{L}", f"{100 * s3(mm, 'raw').saving_r10:.1f}")
    m(f"IpwSaveTen{L}", f"{100 * s3(mm, 'platt_ipw').saving_r10:.1f}"); m(f"StaticSaveTen{L}", f"{100 * s3(mm, 'static').saving_r10:.1f}")
    m(f"RawSaveFive{L}", f"{100 * s3(mm, 'raw').saving_r5:.1f}"); m(f"PlattSaveFive{L}", f"{100 * s3(mm, 'platt').saving_r5:.1f}")
feas = R3[~R3.strategy.isin(["prior_em"])]
for mm, L in (("xgb", "Xgb"), ("lr", "Lr")):
    b = feas[feas.model == mm].sort_values("saving_r5").iloc[-1]
    m(f"BestFive{L}", pct(100 * b.saving_r5)); m(f"BestFive{L}Strategy", b.strategy)
m("SavePlattFiveLr", pct(100 * s3("lr", "platt").saving_r5))
m("SaveOracleFiveXgb", pct(100 * s3("xgb", "platt").saving_oracle_r5)); m("SaveOracleFiveLr", pct(100 * s3("lr", "platt").saving_oracle_r5))
m("EmEceXgb", f"{s3('xgb', 'prior_em').ece:.3f}"); m("EmEceLr", f"{s3('lr', 'prior_em').ece:.3f}")
m("EmSaveTenXgb", f"$-{abs(100 * s3('xgb', 'prior_em').saving_r10):.1f}\\%$")
iso = [s3(mm, "isotonic").cal_slope for mm in ("xgb", "lr")]
m("IsoSlopeLo", f"{min(iso):.2f}"); m("IsoSlopeHi", f"{max(iso):.2f}")
RS = r("robustness_recal_strategies.csv"); rs = lambda mm, s: RS[(RS.model == mm) & (RS.strategy == s)].iloc[0]
RC = r("robustness_recal_contrasts.csv"); rc = lambda mm, c, met: RC[(RC.model == mm) & (RC.contrast == c) & (RC.metric == met)].iloc[0]
block("RQ3 variants (robustness_recal_strategies.csv, robustness_recal_contrasts.csv)")
for mm, L in (("xgb", "Xgb"), ("lr", "Lr")):
    m(f"DropYoungInter{L}", num(rs(mm, "platt_drop_young").cal_intercept, 2))
    m(f"StratInter{L}", num(rs(mm, "platt_ipw_strat").cal_intercept, 2))
    m(f"CapTenInter{L}", num(rs(mm, "platt_ipw_cap10").cal_intercept, 2)); m(f"CapFiftyInter{L}", num(rs(mm, "platt_ipw_cap50").cal_intercept, 2))
    m(f"EmPlattEce{L}", f"{rs(mm, 'platt_em').ece:.3f}"); m(f"EmIpwEce{L}", f"{rs(mm, 'platt_ipw_em').ece:.3f}")
    m(f"EmPlattInter{L}", num(rs(mm, "platt_em").cal_intercept, 2))
    x = rc(mm, "platt_ipw - platt_drop_young", "cal_intercept"); m(f"IpwYoungInter{L}", num(x["mean"], 2)); m(f"IpwYoungInter{L}CI", ci(x.ci_lo, x.ci_hi, 2))
pi_strat = B.pi_ipw_strat; m("PiIPWStrat", pct(100 * pi_strat))
RB = r("robustness_recal_residual_by_stratum.csv")
pt = RB[(RB.stratifier == "promise_tercile")]
m("ResidPromiseIpwMax", f"{100 * pt[pt.strategy == 'platt_ipw']['mean'].abs().max():.1f}")
pp_ = pt[pt.strategy == "platt"]["mean"]; m("ResidPromisePlattLo", num(100 * pp_.min(), 1)); m("ResidPromisePlattHi", num(100 * pp_.max(), 1))
dv = RB[(RB.stratifier == "delivered_within_7d") & (RB.strategy == "platt_ipw")]
m("ResidSlowLo", num(100 * dv[dv.stratum == "no"]["mean"].min(), 1)); m("ResidSlowHi", num(100 * dv[dv.stratum == "no"]["mean"].max(), 1))
m("ResidFastLo", f"{100 * dv[dv.stratum == 'yes']['mean'].min():.1f}"); m("ResidFastHi", f"{100 * dv[dv.stratum == 'yes']['mean'].max():.1f}")
G = r("robustness_cost_ratio_grid.csv")
block("cost-ratio grid, effective ratio e*r (robustness_cost_ratio_grid.csv)")
peak = []
for mm in ("xgb", "lr"):
    gg = G[(G.model == mm) & (G.strategy.isin(["platt", "platt_ipw", "raw"]))]
    best = gg.groupby("r").saving.max(); peak.append(best.idxmax())
m("GridPeakLo", f"{min(peak):g}"); m("GridPeakHi", f"{max(peak):g}")
neg = []
for mm in ("xgb", "lr"):
    gi = G[(G.model == mm) & (G.strategy == "platt_ipw")].set_index("r").saving
    neg.append(gi[(gi.index > 5) & (gi < 0)].index.min())
m("GridIpwNegLo", f"{min(neg):g}"); m("GridIpwNegHi", f"{max(neg):g}")
GC = r("robustness_cost_ratio_contrasts.csv"); gc = GC[(GC.contrast == "platt_ipw - platt") & (GC.ci_lo > 0)]
m("GridIpwGainFrom", f"{gc.r.min():g}")

# ------------------------------------------------------------------ RQ3 workload
block("RQ3 daily workload (rq3_daily_policy.csv), days with >= 30 hand-overs")
P = r("rq3_daily_policy.csv"); P = P[P.n >= 30]
m("NDays", str(P[P.model == "xgb"].day.nunique()))
_DC = r("rq3_decision_curve.csv"); _first_gain = []
for _mm in ("xgb", "lr"):
    _g = _DC[_DC.model == _mm]; _base = _g[_g.strategy == "alert_all"].set_index("threshold").net_benefit.clip(lower=0)
    for _s in ("platt", "ipw", "static"):
        _d = _g[_g.strategy == _s].set_index("threshold").net_benefit - _base
        _first_gain.append(_d[_d > 0.0001].index.min())      # gain above 0.01 points of net benefit (excludes rounding noise)
m("DecisionThreshold", f"{min(_first_gain):.2f}", "smallest threshold at which a model beats the better of all/none by > 0.01 points")
m("DecisionRatio", f"{1 / min(_first_gain) - 1:.0f}", "cost ratio r = 1/t - 1 at that threshold")
_fg = lambda st: min(_DC[_DC.model == mm2].pipe(lambda g: (g[g.strategy == st].set_index("threshold").net_benefit
          - g[g.strategy == "alert_all"].set_index("threshold").net_benefit.clip(lower=0))).pipe(lambda d: d[d > 0.0001].index.min())
          for mm2 in ("xgb", "lr"))
m("DecisionThresholdIpw", f"{_fg('ipw'):.2f}", "first threshold at which the weighted map gains, best learner")
m("DecisionThresholdPlatt", f"{_fg('platt'):.2f}", "first threshold at which rolling Platt gains, best learner")
_pk = {(_mm, _s): (_DC[(_DC.model == _mm) & (_DC.strategy == _s)].set_index("threshold").net_benefit
                   - _DC[(_DC.model == _mm) & (_DC.strategy == "alert_all")].set_index("threshold").net_benefit.clip(lower=0)).idxmax()
       for _mm in ("xgb", "lr") for _s in ("platt", "ipw")}
m("NetBenefitPeak", f"{np.median(list(_pk.values())):.2f}", "threshold of the largest net-benefit gain, rolling and weighted Platt")
for mm, L in (("xgb", "Xgb"), ("lr", "Lr")):
    h = P[P.model == mm]; tr, tp_ = h.thr_alerts / h.n, h.topk_alerts / h.n
    m(f"WlPrecFix{L}", f"{h.thr_tp.sum() / h.thr_alerts.sum():.2f}"); m(f"WlPrecTop{L}", f"{h.topk_tp.sum() / h.topk_alerts.sum():.2f}")
    m(f"WlRecFix{L}", f"{h.thr_tp.sum() / h.pos.sum():.2f}"); m(f"WlRecTop{L}", f"{h.topk_tp.sum() / h.pos.sum():.2f}")
    m(f"WlCvFix{L}", f"{tr.std() / tr.mean():.2f}"); m(f"WlFixMax{L}", pct(100 * tr.max())); m(f"WlShareFix{L}", pct(100 * h.thr_alerts.sum() / h.n.sum()))
h = P[P.model == "xgb"]; tp_ = h.topk_alerts / h.n
m("WlCvTop", f"{tp_.std() / tp_.mean():.2f}"); m("WlShareTop", pct(100 * h.topk_alerts.sum() / h.n.sum()))

# ------------------------------------------------------------------ sensitivity: sample, smoothing, inference
SM = r("robustness_sample_and_smoothing.csv")
block("sensitivity: three-star orders as y = 0, smoothing weight and prior (robustness_sample_and_smoothing.csv)")
ne = SM[SM.analysis == "three-star as y = 0"]
for mm, L in (("xgb", "Xgb"), ("lr", "Lr")):
    m(f"NeutralInf{L}", pct(ne[(ne.model == mm) & (ne.condition == "R+G+P")].rel_pct.iloc[0]))
NS = r("robustness_neutral_shapley.csv")
for mm, L in (("xgb", "Xgb"), ("lr", "Lr")):
    m(f"NeutralShapP{L}", pct(100 * NS[(NS.model == mm) & (NS.shortcut == "P")].shapley.iloc[0]))
nsp = NS[NS.shortcut == "P"][["shapley", "standalone", "last_in", "shapley_RGP", "shapley_NPGP"]].to_numpy().ravel()
m("NeutralShapPRuleLo", f"{100 * np.nanmin(nsp):.0f}"); m("NeutralShapPRuleHi", f"{100 * np.nanmax(nsp):.0f}")
nr = SM[(SM.analysis == "three-star as y = 0, XGBoost - LR, nested tuning") & (SM.condition == "none")].iloc[0]
m("NeutralRankRef", num(nr["mean"])); m("NeutralRankRefCI", ci(nr.ci_lo, nr.ci_hi))
nr = SM[(SM.analysis == "three-star as y = 0, XGBoost - LR, nested tuning") & (SM.condition == "R")].iloc[0]
m("NeutralRankR", f"{nr['mean']:.3f}")
smv = SM[SM.analysis.str.startswith("smoothing") & (SM.condition == "R+G+P")]
for mm, L in (("xgb", "Xgb"), ("lr", "Lr")):
    v = smv[smv.model == mm].rel_pct; m(f"SmoothInf{L}Lo", f"{v.min():.0f}"); m(f"SmoothInf{L}Hi", f"{v.max():.0f}\\%")
block("inference checks (supp_paired_contrasts.csv)")
sig = (SC.ci_lo > 0) | (SC.ci_hi < 0)
flips = set()
for b in (1, 2, 5, 6):
    s2 = (SC[f"ci_lo_block{b}"] > 0) | (SC[f"ci_hi_block{b}"] < 0)
    for _, rw in SC[s2 != sig].iterrows():
        flips.add((rw.family, rw.model, rw.contrast))
flip_rows = SC[[(f, mo, c) in flips for f, mo, c in zip(SC.family, SC.model, SC.contrast)]]
m("NContrasts", str(len(SC))); m("BlockFlipN", str(len(flips)))
m("BlockFlipHolmSig", str(int((flip_rows.p_holm < 0.05).sum())), "of the flipping contrasts, significant after Holm")
_nf = int((flip_rows.p_holm < 0.05).sum()); m("BlockFlipHolmSigWord", {0: "none", 1: "one"}.get(_nf, str(_nf)))
m("HacAgree", str(int(((SC.p < 0.05) == (SC.p_hac < 0.05)).sum())))
block("literature audit, Table 1 (docs/literature_audit.csv, studies P01-P17)")
_REPO_A = next(c for c in (Path(os.environ.get("LEAKAWARE_REPO", "")), Path(__file__).resolve().parents[1])
               if (c / "docs" / "literature_audit.csv").is_file())
_AU = pd.read_csv(_REPO_A / "docs" / "literature_audit.csv")
_AU = _AU[_AU.id.str.startswith("P")]
m("AuditN", str(len(_AU)))
m("AuditRandom", str(int(_AU.split.isin(["random_holdout", "not_stated", "unclear"]).sum())), "random or unstated split")
m("AuditUnstated", str(int(_AU.split.isin(["not_stated", "unclear"]).sum())), "split not described")
m("AuditTemporal", str(int((_AU.split == "temporal").sum())))
m("AuditPost", str(int((_AU.post_outcome_features == "yes").sum())))
m("AuditPredTime", str(int(_AU.prediction_time.astype(str).str.match(r"(stated|after review)").sum())))
m("AuditPurge", str(int((_AU.label_availability_purge == "yes").sum())))
# headline accuracy of the satisfaction studies that report one (review-text and delivery tasks excluded)
_acc = []
for _, _a in _AU[_AU.task.isin(["low_review_binary", "review_score_multiclass", "satisfaction"])].iterrows():
    _mm = re.match(r"(?:max )?accuracy (\d+(?:\.\d+)?)(%?)", str(_a.headline_metrics))
    if _mm:
        _acc.append(float(_mm.group(1)) * (1 if _mm.group(2) else 100))
m("AuditAccMedian", f"{np.median(_acc):.1f}\\%", f"median over {len(_acc)} studies")
m("AuditAccLo", f"{min(_acc):.0f}\\%"); m("AuditAccHi", f"{max(_acc):.0f}\\%")

# ------------------------------------------------------------------ write results.tex
hdr = r"""%% =====================================================================
%%  results.tex -- single source of truth for every number printed in the
%%  manuscript. Cite the macro, never a hard-coded value.
%%
%%  Generated by tools/gen_results_tex.py from the outputs of a full run
%%  (outputs/manifest.json records the software versions and the hashes
%%  of the code and of every output file).
%%  Proportions are reported as percentages; PR-AUC, ROC-AUC and
%%  precision values as decimals with three digits.
%% =====================================================================
"""
lines = [hdr]
seen = set()
# ------------------------------------------------------------------ design and computational budget
# Counts follow the code (rq1_inflation, rq2_timing, rq3_recalibration, robustness.plan) and the row counts of the
# output tables: a temporal window is one fit, a random split is one fit per fold, nested tuning adds one fit per grid
# configuration and inner split, then one refit. Recalibration maps (Platt, isotonic) are not counted as model fits.
import os
import yaml
for cand in (Path(os.environ.get("LEAKAWARE_REPO", "")), Path(__file__).resolve().parents[1]):
    if (cand / "src" / "leakaware").is_dir():
        if str(cand / "src") not in sys.path:
            sys.path.insert(0, str(cand / "src"))
        REPO_B = cand; break
from leakaware.experiments import rq1_inflation as rq1m
from leakaware.experiments import robustness as robm
cfgB = yaml.safe_load(open(REPO_B / "config.yaml", encoding="utf-8"))
NW = int(r("rq1_window_metrics.csv").k.nunique()); NF = int(cfgB["random_split"]["n_folds"]); NIN = 3
G1 = {m: len(v) for m, v in rq1m.TUNE_GRID.items()}; G2 = {m: len(v) for m, v in rq1m.TUNE_GRID_TASK2.items()}
fix = lambda split: NW if split == "temporal" else NF
tun = lambda split, g: NW * (g + 1) if split == "temporal" else NF * (NIN * g + 1)
def factorial_fits(f):
    d = r(f)[["model", "shortcuts", "split"]].drop_duplicates()
    return int(sum(fix(x) for x in d.split))
def tuned_fits(f, grid):
    C = r(f); return int(sum(tun("random" if sc.startswith("R") else "temporal", grid[m]) // (NW if not sc.startswith("R") else NF)
                             for m, sc in zip(C.model, C.shortcuts)))
budget = {
    "rq1_factorial": factorial_fits("rq1_window_metrics.csv"),
    "rq1_realised": len(r("rq1_deployment_windows.csv")),
    "rq1_selection": len(r("rq1_selection_windows.csv")),
    "rq1_tuned": tuned_fits("rq1b_tuned_chosen_params.csv", G1),
    "rq2": int(sum(G1[m] + 1 for m in r("rq2_chosen_params.csv").model)),
    "rq3": int(sum(G1[m] + 2 for m in r("rq3_chosen_params.csv").model)),
    "task2_factorial": factorial_fits("task2_window_metrics.csv"),
    "task2_realised": len(r("task2_deployment_windows.csv")),
    "task2_tuned": tuned_fits("task2_tuned_chosen_params.csv", G2),
}
COND = {c["shortcuts"]: c["split"] for c in rq1m.conditions()}
smooth = [f"mu{m}" for m in cfgB["robustness"]["smoothing_variants"]] + \
         [f"lp{int(round(100 * p_)):02d}" for p_ in cfgB["robustness"]["late_prior_variants"]]
plan = robm.plan(cfgB, robm.GROUPS, smooth)
rob = 0
for t in plan:
    kind = t[0]
    if kind == "fixed":
        rob += fix(COND[t[2]])
    elif kind == "tuned":
        rob += tun(COND[t[2]], G1[t[3]])
    elif kind == "milestone":
        rob += NW * (G1[t[2]] + 1)
    elif kind == "recal":
        rob += NW * (G1[t[1]] + 2)
    elif kind == "serving":
        rob += NW * (1 + 4)                       # leaky model + four auxiliary XGBoost models (expected values)
budget["robustness"] = rob
ntot = sum(budget.values())
block("design and computational budget (code + table row counts)")
m("NWindows", str(NW)); m("NFolds", str(NF)); m("NInnerFolds", str(NIN))
m("NCondMain", str(r("rq1_window_metrics.csv").shortcuts.nunique()), "12 factorial + Graw + Gloo")
m("NCondTwo", str(r("task2_window_metrics.csv").shortcuts.nunique()), "12 factorial + Graw")
m("NGridXgb", str(G1["xgb"])); m("NGridLr", str(G1["lr"])); m("NGridRf", str(G1["rf"]))
m("NGridTwoXgb", str(G2["xgb"])); m("NGridTwoLr", str(G2["lr"]))
m("NSelectionCands", str(len(rq1m.SELECTION_GRID)))
m("NRobustTasks", str(len(plan)))
m("NBootCI", big(cfgB["bootstrap"]["n_boot"])); m("NBootP", big(20000)); m("BlockLen", str(cfgB["bootstrap"]["block_len"]))
m("NHolmContrasts", big(len(r("supp_paired_contrasts.csv")) + len(r("supp_rq3_contrasts.csv"))), "paired contrasts with Holm-adjusted p")
rob_holm = sum(len(r(f_)) for f_ in ("robustness_serving_rules.csv", "robustness_milestone_contrasts.csv",
                                    "robustness_recal_contrasts.csv") if "p_holm" in r(f_).columns)
m("NHolmContrastsRob", big(rob_holm), "Holm-adjusted contrasts in the sensitivity analyses")
DIG = {"1": "One", "2": "Two", "3": "Three"}                # LaTeX command names cannot contain digits
for k_, v_ in budget.items():
    m("NFits" + "".join(w.capitalize() for w in k_.split("_")).translate(str.maketrans({d: w for d, w in DIG.items()})), big(v_))
m("NFitsTaskTwoTotal", big(budget["task2_factorial"] + budget["task2_realised"] + budget["task2_tuned"]))
m("NFitsTotal", big(ntot))
print("budget:", budget, "total", ntot, "robustness tasks", len(plan))


# ------------------------------------------------------------------ inflation by period and the earlier run (Table 10, Section 6.3)
block("inflation by period: first and last 15 windows (rq1b_tuned_window_metrics.csv)")
_tw = r("rq1b_tuned_window_metrics.csv").pivot_table(index=["model", "k"], columns="shortcuts", values="ap").reset_index()
_K = int(_tw.k.max()); _half = _K // 2
_per = {}
for _lab, _sel in (("First", _tw.k <= _half), ("Last", _tw.k > _half)):
    _g = _tw[_sel].groupby("model")[["none", "R+G+P"]].mean()
    _inf = 100 * (_g["R+G+P"] / _g["none"] - 1)
    m(f"InfHalf{_lab}Lo", f"{_inf.min():.0f}"); m(f"InfHalf{_lab}Hi", f"{_inf.max():.0f}")
m("InfHalfN", str(_half), "windows per half")

block("earlier run with XGBoost 3.0.5 (docs/supplementary/rq1b_tuned_ranking_flip_xgboost305_linux.csv)")
_REPO = next(c for c in (Path(os.environ.get("LEAKAWARE_REPO", "")), Path(__file__).resolve().parents[1])
             if (c / "src" / "leakaware").is_dir())
_OLD = pd.read_csv(_REPO / "docs" / "supplementary" / "rq1b_tuned_ranking_flip_xgboost305_linux.csv")
_NEWF = r("rq1b_tuned_ranking_flip.csv"); _NEWF = _NEWF[(_NEWF.a == "xgb") & (_NEWF.b == "lr")]
_o = _OLD[(_OLD.shortcuts == "none") & (_OLD.metric == "ap")].iloc[0]
m("OldRefDiffAp", num(_o["diff"])); m("OldRefDiffApCI", ci(_o.ci_lo, _o.ci_hi))
_o = _OLD[(_OLD.shortcuts == "R") & (_OLD.metric == "ap")].iloc[0]
m("OldRDiffAp", num(_o["diff"])); m("OldRDiffApCI", ci(_o.ci_lo, _o.ci_hi))


for title, items in blocks:
    lines.append(f"\n%% ---- {title}")
    for name, val, note in items:
        if name in seen:
            raise SystemExit(f"duplicate macro {name}")
        seen.add(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}" + (f"  % {note}" if note else ""))
NEW.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ RQ3 table body (Table 9 of the article)
names = [("static", "Frozen model and Platt map"), ("raw", "Refit, raw probabilities"), ("platt", "Refit, rolling Platt"),
         ("isotonic", "Refit, rolling isotonic"), ("prior_recent", "Refit, prior shift to labelled 28-day rate"),
         ("prior_em", "Refit, EM prior shift"), ("platt_ipw", r"Refit, latency-weighted Platt (expanding $\hat G_y$)"),
         ("platt_ipw_roll", r"Refit, latency-weighted Platt (rolling $\hat G_y$)"),
         ("platt_ipw_nowcast", r"Refit, latency-weighted Platt (rolling $\hat G_y$) and nowcast")]


def sv_(x, d=1):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$")


rows = []
for s, lab in names:
    a, b = s3("xgb", s), s3("lr", s)
    cells = [lab, f"{a.ap:.3f} / {b.ap:.3f}", f"{sv_(a.cal_intercept, 2)} / {sv_(b.cal_intercept, 2)}",
             f"{a.ece:.3f} / {b.ece:.3f}", f"{sv_(100 * a.saving_r5)} / {sv_(100 * b.saving_r5)}",
             f"{sv_(100 * a.saving_r10)} / {sv_(100 * b.saving_r10)}"]
    if s == "platt_ipw":                                   # the proposed correction is set in bold
        cells = [f"\\textbf{{{c}}}" for c in cells]
    rows.append(" & ".join(cells) + " \\\\")
a, b = s3("xgb", "platt"), s3("lr", "platt")
rows.append(f"Best threshold in hindsight, rolling Platt scores (bound) & -- & -- & -- & {sv_(100 * a.saving_oracle_r5)} / "
            f"{sv_(100 * b.saving_oracle_r5)} & {sv_(100 * a.saving_oracle_r10)} / {sv_(100 * b.saving_oracle_r10)} \\\\")
TAB.write_text("%% generated by gen_results_tex.py from rq3_summary.csv; body of the RQ3 table, tab:rq3 (used inside tabularx)\n"
               "\\newcommand{\\TabRQthreeRows}{%\n" + "\n".join(rows) + "\n}\n", encoding="utf-8")
print(f"{len(seen)} macros written to {NEW}; RQ3 table body written to {TAB}")
# supplementary tables, written next to tab_rq3.tex
_lab = {"none": "Reference", "R": r"Random split (\textsf{R})", "R+G+P": r"\textsf{R}{+}\textsf{G}{+}\textsf{P}"}
_met = {"ap": "PR-AUC", "auc": "ROC-AUC", "p_at_10": "Precision at 10\\%"}
def _ci4(lo, hi):  # three decimals, or four when a bound would round to zero
    d = 4 if min(abs(lo), abs(hi)) < 0.0005 else 3
    return ci(lo, hi, d)


_srows = []
for _s in ("none", "R", "R+G+P"):
    for _mt in ("ap", "auc", "p_at_10"):
        _a = _NEWF[(_NEWF.shortcuts == _s) & (_NEWF.metric == _mt)].iloc[0]
        _b = _OLD[(_OLD.shortcuts == _s) & (_OLD.metric == _mt)].iloc[0]
        _srows.append(f"{_lab[_s] if _mt == 'ap' else ''} & {_met[_mt]} & {num(_a['diff'])} ({_ci4(_a.ci_lo, _a.ci_hi)}) & "
                      f"{num(_b['diff'])} ({_ci4(_b.ci_lo, _b.ci_hi)}) \\\\")
_crow = []
for s, labx in names:
    a_, b_ = s3("xgb", s), s3("lr", s)
    def _c(x, c, k=1.0, d=2):
        return f"{sv_(k * x[c], d)} [{sv_(k * x[c + '_lo'], d)}, {sv_(k * x[c + '_hi'], d)}]"
    for mm_, x in (("XGBoost", a_), ("LR", b_)):
        _crow.append(f"{labx if mm_ == 'XGBoost' else ''} & {mm_} & {_c(x, 'ap', d=3)} & {_c(x, 'cal_intercept')} & "
                     f"{_c(x, 'ece', d=3)} & {_c(x, 'saving_r5', 100, 1)} & {_c(x, 'saving_r10', 100, 1)} \\\\")
SUPP = TAB.parent / "supp_tables.tex"
SUPP.write_text("%% generated by gen_results_tex.py; tables of the supplementary material\n"
                "\\newcommand{\\TabSuppVersionRows}{%\n" + "\n".join(_srows) + "\n}\n"
                "\\newcommand{\\TabSuppRQthreeRows}{%\n" + "\n".join(_crow) + "\n}\n", encoding="utf-8")
print("supplementary tables written to", SUPP)

# coding sheet of the literature audit (docs/literature_audit.csv), studies P01-P17 behind Table 1
_A = pd.read_csv(_REPO / "docs" / "literature_audit.csv")
_A = _A[_A.id.str.startswith("P")]
_KEYS = {"P01": "leite2024uso", "P02": "gogineni2025end", "P03": "gopinadh2025customer", "P04": "lesmana2025machine",
         "P05": "zaghloul2024predicting", "P06": "utari2026factor", "P07": "wangkiat2023machine", "P08": "orman2026hybrid",
         "P09": "wong2020optimising", "P10": "madane2026value", "P11": "mangla2025machine", "P12": "mahdjoub2024enhancing",
         "P13": "bamba2025naive", "P14": "li2026when", "P15": "hartman2026learning", "P16": "pandey2023predicting",
         "P17": "fouad2023effective"}


def _ev(note):
    e = re.search(r"\[evidence: ([^\]]+)\]", note).group(1)
    return "full text" if e.startswith("full text") else ("metadata" if e.startswith("metadata") else "abstract")


def _ds(note):
    d = re.search(r"\[dataset: ([^\]]+)\]", note).group(1)
    if d.startswith("Olist confirmed"):
        return "named"
    if "probable" in d.lower():
        return "probable"
    if d.startswith("unconfirmed"):
        return "unconfirmed"
    raise SystemExit(f"unclassified dataset note: {d}")


def _pt(x):
    x = str(x)
    return "stated" if x.startswith("stated") else ("after review" if x.startswith("after review") else
                                                   ("unclear" if x.startswith("unclear") else "not stated"))


_SPL = {"random_holdout": "random", "temporal": "time-ordered", "not_stated": "not stated", "unclear": "unclear"}
_TSK = {"low_review_binary": "low rating", "review_score_multiclass": "score classes", "late_delivery": "late delivery",
        "satisfaction": "satisfaction", "delivery_time_regression": "delivery time"}
_arows = []
for _, a in _A.iterrows():
    _t = _TSK.get(a.task, "review-text classes")
    _arows.append(f"{a.id} & \\citet{{{_KEYS[a.id]}}} & {_t} & {_ev(a.note)} & {_ds(a.note)} & {_SPL[a.split]} & "
                  f"{_pt(a.prediction_time)} & {a.post_outcome_features} \\\\")
_cnt = dict(full=sum(_ev(n) == "full text" for n in _A.note), abs=sum(_ev(n) == "abstract" for n in _A.note),
            meta=sum(_ev(n) == "metadata" for n in _A.note), post=int((_A.post_outcome_features == "yes").sum()),
            temporal=int((_A.split == "temporal").sum()), pt=int(sum(_pt(x) != "not stated" and _pt(x) != "unclear"
                                                                   for x in _A.prediction_time)))
with open(SUPP, "a", encoding="utf-8") as _fh:
    _fh.write("\\newcommand{\\TabSuppAuditRows}{%\n" + "\n".join(_arows) + "\n}\n")
    for _k, _v in _cnt.items():
        _fh.write(f"\\newcommand{{\\SuppAudit{_k.capitalize()}}}{{{_v}}}\n")
print("audit counts", _cnt)
