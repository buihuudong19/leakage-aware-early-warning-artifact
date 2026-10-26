"""Figures of the article and the supplementary material (vector PDF and 600-dpi PNG).

Every figure is drawn at its final printed size in the two-column Elsevier layout (one column 3.30 in, full width
6.84 in), so that the font sizes below are the sizes in print (6.5 to 8 pt). Colours follow the Okabe-Ito palette,
which stays distinguishable for readers with colour-vision deficiencies; learners are also told apart by marker shape.
File names follow the figure numbers of the article; figS* files are supplementary.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
import numpy as np
import pandas as pd

from ..data.io import load_features

# ----------------------------------------------------------------------------------------------- style
COL, FULL = 3.30, 6.84                       # column and text width of the cas-dc layout, in inches
INK, INK2, MUTED, GRID, LIGHT = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#e3e3e3", "#f3f3f3"
OI = dict(blue="#0072B2", vermillion="#D55E00", green="#009E73", orange="#E69F00", sky="#56B4E9",
          purple="#CC79A7", yellow="#F0E442")
MODELS = ("xgb", "lr", "rf")
MODEL_COLOR = {"xgb": OI["blue"], "lr": OI["vermillion"], "rf": OI["green"]}
MODEL_MARKER = {"xgb": "o", "lr": "s", "rf": "^"}
MODEL_LABEL = {"xgb": "XGBoost", "lr": "Logistic regression", "rf": "Random forest"}
MODEL_SHORT = {"xgb": "XGBoost", "lr": "LR", "rf": "RF"}
SC_COLOR = {"R": "#332288", "NP": "#88CCEE", "G": "#DDCC77", "P": "#CC6677"}
SC_LABEL = {"none": "Reference", "R": "Random split (R)", "NP": "No purge (NP)", "G": "Full-period aggregates (G)",
            "Graw": "G without shrinkage", "Gloo": "G, leave-one-out", "P": "Post-hand-over features (P)",
            "R+P": "R + P", "G+P": "G + P", "NP+G+P": "NP + G + P", "R+G": "R + G", "NP+G": "NP + G",
            "NP+P": "NP + P", "R+G+P": "All shortcuts (R + G + P)"}
STRAT_LABEL = {"static": "Frozen model + Platt", "raw": "Raw scores", "platt": "Rolling Platt",
               "isotonic": "Rolling isotonic", "prior_recent": "Prior shift to the\nrecent labelled rate",
               "prior_em": "EM prior shift", "platt_ipw": "Latency-weighted Platt",
               "platt_ipw_roll": "Latency-weighted Platt,\nrolling latency model",
               "platt_ipw_nowcast": "Latency-weighted Platt\n+ nowcast prior"}
STRAT_LINE = {k: v.replace("\n", " ") for k, v in STRAT_LABEL.items()}
STRAT_STYLE = {"static": (MUTED, "--"), "platt": ("#332288", "-"), "platt_ipw": ("#117733", "-"),
               "raw": ("#AA4499", ":")}
MS_LABEL = {"purchase": "Purchase", "approval": "Approval", "t0": "$t^0$", "t0+1d": "$t^0$+1 d",
            "t0+2d": "$t^0$+2 d", "t0+3d": "$t^0$+3 d", "t0+7d": "$t^0$+7 d", "delivery": "Delivery"}
MS_ORDER = ["purchase", "approval", "t0", "t0+1d", "t0+2d", "t0+3d", "t0+7d", "delivery"]
_CFG = None


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Inter", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 5,
        "axes.titlecolor": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
        "legend.handletextpad": 0.4, "legend.columnspacing": 1.2, "legend.borderaxespad": 0.2,
        "lines.linewidth": 1.1, "lines.markersize": 3.5, "errorbar.capsize": 0,
        "figure.dpi": 150, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": True, "mathtext.default": "regular"})


def _subplots(*args, **kw):
    """plt.subplots with constrained layout, so that the saved figure has exactly the requested printed size."""
    gs = dict(kw.pop("gridspec_kw", {}))
    gs.pop("wspace", None); gs.pop("hspace", None)
    fig, axes = plt.subplots(*args, layout="constrained", gridspec_kw=gs or None, **kw)
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.015, wspace=0.03, hspace=0.04)
    return fig, axes


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"{name}.{ext}", facecolor="white")
    plt.close(fig)


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.5); ax.set_axisbelow(True)


def _panel(ax, i, title=""):
    ax.set_title(f"({'abcdefgh'[i]}) {title}".rstrip())


def _models(df, col="model"):
    present = set(df[col])
    return [m for m in MODELS if m in present]


def _offsets(n, step=0.24):
    return (np.arange(n) - (n - 1) / 2) * -step


def _dot(ax, x, y, lo, hi, color, marker="o", filled=True, ms=3.6, lw=0.9, horizontal=True, zorder=3):
    x, y, lo, hi = map(np.asarray, (x, y, lo, hi))
    if horizontal:
        ax.hlines(y, lo, hi, color=color, lw=lw, zorder=zorder - 1)
    else:
        ax.vlines(x, lo, hi, color=color, lw=lw, zorder=zorder - 1)
    ax.scatter(x, y, s=ms ** 2, marker=marker, facecolor=color if filled else "white", edgecolor=color, lw=0.8,
               zorder=zorder)


def _model_handles(models, filled=True):
    return [Line2D([], [], ls="", marker=MODEL_MARKER[m], ms=4, mfc=MODEL_COLOR[m] if filled else "white",
                   mec=MODEL_COLOR[m], mew=0.8, label=MODEL_LABEL[m]) for m in models]


def _date_axis(ax):
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))


def _read(out, name, **kw):
    p = out / "tables" / name
    return pd.read_csv(p, **kw) if p.is_file() else None


def _boot(x, cfg):
    from ..evaluation.metrics import moving_block_bootstrap_mean
    bs = cfg["bootstrap"]
    m, lo, hi, _ = moving_block_bootstrap_mean(np.asarray(x, float), bs["n_boot"], bs["block_len"], cfg["seed"])
    return m, lo, hi


def _cfg(out):
    if _CFG is not None:
        return _CFG
    import yaml
    return yaml.safe_load((out.parent / "config.yaml").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------------- Fig. 1 sample flow
def fig_sample_flow(out):
    flow = pd.read_csv(out / "tables" / "sample_flow.csv")
    notes = json.loads((out / "tables" / "sample_notes.json").read_text(encoding="utf-8"))
    main = ["All Olist orders", "Delivered", "Hand-over time $t^0$ recorded", "At least one review",
            "No review answered before $t^0$", "Analysis sample"]
    excl = [None, "Not delivered", "No hand-over time", "No review",
            f"Answered before $t^0$\n({notes['excluded_feedback_before_t0_low']:,} low-rated)", "Three stars (neutral)"]
    fig, ax = _subplots(figsize=(COL, 3.1)); ax.axis("off")
    step, n = 1.0, len(main)
    ax.set_xlim(0, 10); ax.set_ylim(-0.95, n * step)
    for i, (lab, cnt) in enumerate(zip(main, flow["n_orders"])):
        y = (n - i) * step - 0.5
        last = i == n - 1
        h = 0.78 if last else 0.6
        yc = y - (0.09 if last else 0)
        ax.add_patch(FancyBboxPatch((0.1, yc - h / 2), 5.1, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="#e8f1f8" if last else LIGHT, ec=OI["blue"] if last else MUTED,
                                    lw=0.9 if last else 0.5))
        ax.text(2.65, yc + (0.2 if last else 0.1), lab, ha="center", va="center", fontsize=6.4, color=INK)
        ax.text(2.65, yc + (0.0 if last else -0.14), f"n = {cnt:,}", ha="center", va="center", fontsize=6.3,
                color=INK, weight="bold")
        if last:
            ax.text(2.65, yc - 0.2, f"{notes['primary_pos']:,} low-rated ({100 * notes['primary_prevalence']:.1f}%)",
                    ha="center", va="center", fontsize=6.0, color=INK)
        else:
            ax.annotate("", xy=(2.65, y - step + 0.3), xytext=(2.65, y - 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, mutation_scale=6))
        if excl[i]:
            d = int(flow["removed"].iloc[i]); yb = y + step / 2
            ax.annotate("", xy=(5.75, yb), xytext=(2.65, yb),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.5, mutation_scale=6))
            first, *rest = excl[i].split("\n")
            hb = 0.5 if rest else 0.32
            ax.add_patch(FancyBboxPatch((5.75, yb - hb / 2), 4.2, hb, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc="white", ec=GRID, lw=0.5))
            ax.text(7.85, yb, "\n".join([f"{first}: {d:,}"] + rest), ha="center", va="center", fontsize=5.8,
                    color=INK2, linespacing=1.15)
    ax.text(0.1, -0.7, f"Low-rated: one or two stars, compared with four and five stars. The\n"
            f"{notes['orders_multi_review']:,} orders with several reviews take the worst score.", fontsize=5.4,
            color=INK2, va="center", linespacing=1.2)
    _save(fig, out, "fig01_sample_flow")


# ----------------------------------------------------------------------------------------------- Fig. 2 protocol
def fig_design(out):
    fig, ax = _subplots(figsize=(COL, 2.2)); ax.axis("off")
    ax.set_xlim(0, 10.4); ax.set_ylim(-3.25, 3.75)
    blue_l = "#cfe3f1"
    for j, (lab, a) in enumerate([("Origin $o_k$", 6.0), ("Origin $o_{k+1}$", 7.0), ("Origin $o_{k+2}$", 8.0)]):
        y = 3.2 - j * 0.95
        ax.text(0.0, y, lab, va="center", ha="left", fontsize=6.5, color=INK)
        ax.add_patch(Rectangle((2.0, y - 0.25), a - 2.0 - 1.3, 0.5, fc=blue_l, ec="none"))
        ax.add_patch(Rectangle((a - 1.3, y - 0.25), 1.3, 0.5, fc="white", ec=MUTED, lw=0.5, hatch="//////"))
        ax.add_patch(Rectangle((a, y - 0.25), 1.0, 0.5, fc=OI["vermillion"], ec="none"))
        ax.plot([a, a], [y - 0.36, y + 0.36], color=INK, lw=0.9)
    ax.annotate("", xy=(10.3, 0.5), xytext=(1.9, 0.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6,
                                                                       mutation_scale=7))
    ax.text(10.3, 0.12, "hand-over time $t^0$", ha="right", fontsize=6.2, color=INK2)
    leg = [(blue_l, None, "none", "Training: $t^0$ before the origin and label known"),
           ("white", "//////", MUTED, "Purged: label not yet known at the origin"),
           (OI["vermillion"], None, "none", "Evaluation: hand-overs in the 14 days after the origin")]
    for i, (fc, hatch, ec, lab) in enumerate(leg):
        yy = -0.55 - i * 0.62
        ax.add_patch(Rectangle((0.0, yy - 0.18), 0.55, 0.36, fc=fc, ec=ec, lw=0.5, hatch=hatch))
        ax.text(0.75, yy, lab, va="center", fontsize=6.2, color=INK)
    ax.text(0.0, -2.75, "30 origins, 14 days apart (June 2017 to July 2018); all features as of $t^0$.",
            fontsize=6.0, color=INK2, va="center")
    _save(fig, out, "fig02_rolling_origin_protocol")


# ----------------------------------------------------------------------------------------------- Fig. 3 context
def fig_context(out):
    from ..evaluation.splits import rolling_windows
    F = load_features(out)
    D = F[F.y >= 0].copy()
    D["month"] = D["t0"].dt.to_period("M").dt.to_timestamp()
    mon = D.groupby("month").agg(n=("y", "size"), low=("y", "mean"), late=("p_late", "mean")).reset_index()
    mon = mon[mon.n >= 300]
    lag = (D["label_available_ts"] - D["t0"]).dt.total_seconds() / 86400
    share = []
    for w in rolling_windows(_cfg(out)):
        past = D["t0"] < w["origin"]
        recent = past & (D["t0"] >= w["origin"] - pd.Timedelta(days=28))
        unav = past & (D["label_available_ts"] >= w["origin"])
        share.append((w["origin"], unav.sum() / past.sum(), (unav & recent).sum() / max(recent.sum(), 1)))
    share = pd.DataFrame(share, columns=["origin", "all", "recent"])
    fig, axes = _subplots(1, 3, figsize=(FULL, 2.0), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.38))
    ax = axes[0]
    ax.plot(mon.month, 100 * mon.low, color=OI["blue"], marker="o", ms=2.4, label="Low-rated (1–2 stars)")
    ax.plot(mon.month, 100 * mon.late, color=OI["orange"], marker="s", ms=2.2, label="Delivered after the promised date")
    ax.set_ylabel("Share of orders (%)"); _panel(ax, 0, "Monthly rates by month of $t^0$")
    ax.set_ylim(0, max(100 * mon.low.max(), 100 * mon.late.max()) * 1.4); _grid(ax, "y"); _date_axis(ax)
    ax.legend(loc="upper left", ncol=1)
    ax = axes[1]
    ax.hist(lag.clip(upper=40), bins=np.arange(0, 41, 1), color="#9cc3e0", edgecolor="white", linewidth=0.3)
    med = lag.median()
    ax.axvline(med, color=INK, lw=0.7, ls="--")
    ax.text(med + 1, ax.get_ylim()[1] * 0.93, f"median {med:.1f} d", fontsize=6.2, color=INK, va="top")
    ax.set_xticks([0, 10, 20, 30, 40]); ax.set_xticklabels(["0", "10", "20", "30", "≥40"])
    ax.set_xlabel("Days from $t^0$ until the label is known"); ax.set_ylabel("Orders")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
    _panel(ax, 1, "Label latency"); _grid(ax, "y")
    ax = axes[2]
    ax.plot(share.origin, 100 * share.recent, color=OI["purple"], marker="o", ms=2.2, label="$t^0$ in the last 28 days")
    ax.plot(share.origin, 100 * share["all"], color=OI["green"], marker="s", ms=2.0, label="All orders before the origin")
    ax.set_ylabel("Labels not yet known (%)"); _panel(ax, 2, "Labels not yet known")
    ax.set_ylim(0, 70); _grid(ax, "y"); _date_axis(ax); ax.legend(loc="center left", bbox_to_anchor=(0, 0.33))
    _save(fig, out, "fig03_evaluation_context")


# ----------------------------------------------------------------------------------------------- Fig. 4 inflation
INF_ORDER = ["NP", "R", "G", "Graw", "P", "R+P", "G+P", "R+G+P"]


def fig_inflation(out, metric="ap"):
    I = _read(out, "rq1_inflation.csv")
    I = I[(I.metric == metric) & I.shortcuts.isin(INF_ORDER)]
    models = _models(I)
    order = [s for s in INF_ORDER if s in set(I.shortcuts)]
    ys = np.arange(len(order))[::-1].astype(float)
    off = _offsets(len(models), 0.25)
    fig, axes = _subplots(1, 2, figsize=(FULL, 2.55), sharey=True, gridspec_kw=dict(wspace=0.06))
    lab = {"ap": "PR-AUC", "auc": "ROC-AUC"}[metric]
    for p, (ax, kind) in enumerate(zip(axes, ("abs", "rel"))):
        for o, m in zip(off, models):
            g = I[I.model == m].set_index("shortcuts").loc[order]
            k = 1.0 if kind == "abs" else 100 / g.ref_mean
            _dot(ax, k * g.delta, ys + o, k * g.ci_lo, k * g.ci_hi, MODEL_COLOR[m], MODEL_MARKER[m])
        ax.axvline(0, color=INK, lw=0.6)
        ax.axhspan(ys[order.index("P")] + 0.5, ys[0] + 0.5, color=LIGHT, zorder=0, lw=0)
        _grid(ax, "x")
        if kind == "abs":
            _panel(ax, p, f"Absolute gain in {lab}"); ax.set_xlabel(f"{lab}, shortcut minus reference")
        else:
            _panel(ax, p, "Relative gain"); ax.set_xlabel("Gain relative to the reference protocol (%)")
    axes[0].set_yticks(ys); axes[0].set_yticklabels([SC_LABEL[s] for s in order])
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    fig.legend(handles=_model_handles(models), loc="outside upper center", ncol=len(models))
    _save(fig, out, "fig04_rq1_inflation" if metric == "ap" else f"figS1_rq1_inflation_{metric}")


# ----------------------------------------------------------------------------------------------- Fig. 5 Shapley
def fig_shapley(out, metric="ap"):
    S = _read(out, "rq1_shapley.csv"); S = S[S.metric == metric]
    models = _models(S)
    order = ["R", "NP", "G", "P"]
    short = {"R": "R", "NP": "NP", "G": "G", "P": "P"}
    fig, axes = _subplots(1, len(models), figsize=(FULL, 2.15), sharey=True, gridspec_kw=dict(wspace=0.08))
    axes = np.atleast_1d(axes)
    top = 0
    for p, (ax, m) in enumerate(zip(axes, models)):
        g = S[S.model == m].set_index("factor").loc[order]
        v0, v1 = g.v_none.iloc[0], g.v_all.iloc[0]; tot = v1 - v0
        ax.bar(0, v0, color="#c9c9c9", width=0.64)
        ax.text(0, v0 + 0.008, f"{v0:.3f}", ha="center", va="bottom", fontsize=6.2)
        cum = v0
        for i, f in enumerate(order, start=1):
            ph = g.loc[f, "phi"]
            ax.bar(i, ph, bottom=cum, color=SC_COLOR[f], width=0.64, ec="white", lw=0.3)
            ax.plot([i - 0.32 - 0.36, i - 0.32], [cum, cum], color=MUTED, lw=0.4, ls=":")
            ax.vlines(i, cum + g.loc[f, "ci_lo"], cum + g.loc[f, "ci_hi"], color=INK, lw=0.7)
            ytxt = cum + max(ph, g.loc[f, "ci_hi"]) + 0.008
            ax.text(i, ytxt, f"{100 * ph / tot:.0f}%", ha="center", va="bottom", fontsize=6.2,
                    weight="bold" if f == "P" else "normal")
            cum += ph
        ax.bar(5, v1, color=INK2, width=0.64)
        ax.text(5, v1 + 0.008, f"{v1:.3f}", ha="center", va="bottom", fontsize=6.2)
        cum2 = v0
        for f in order:
            top = max(top, cum2 + g.loc[f, "ci_hi"]); cum2 += g.loc[f, "phi"]
        top = max(top, v1)
        ax.set_xticks(range(6)); ax.set_xticklabels(["Ref."] + [short[f] for f in order] + ["All"])
        _panel(ax, p, MODEL_LABEL[m]); _grid(ax, "y")
    axes[0].set_ylim(0, top * 1.12)
    axes[0].set_ylabel(f"Mean {'PR-AUC' if metric == 'ap' else 'ROC-AUC'}")
    hs = [Patch(fc=SC_COLOR[f], label=lab) for f, lab in zip(order, ("Random split (R)", "No purge (NP)",
                                                                    "Full-period aggregates (G)",
                                                                    "Post-hand-over features (P)"))]
    fig.legend(handles=hs, loc="outside upper center", ncol=4, handlelength=1.0)
    _save(fig, out, "fig05_rq1_shapley" if metric == "ap" else f"figS2_rq1_shapley_{metric}")


# ----------------------------------------------------------------------------------------------- Fig. 6 realised
def fig_deployment(out):
    G = _read(out, "rq1_deployment_summary.csv")
    if G is None:
        return
    G = G[G.metric == "ap"]; models = _models(G)
    order = ["G", "Graw", "P", "G+P"]
    lab = {"G": "Full-period aggregates (G)", "Graw": "G without shrinkage", "P": "Post-hand-over features (P)",
           "G+P": "G + P"}
    fig, axes = _subplots(len(models), 1, figsize=(COL, 1.05 + 1.0 * len(models)), sharex=True)
    axes = np.atleast_1d(axes)
    ys = np.arange(len(order))[::-1]
    for p, (ax, m) in enumerate(zip(axes, models)):
        g = G[G.model == m].set_index("variant").loc[order]
        ref = g.correct_mean.iloc[0]
        ax.axvline(ref, color=INK, lw=0.6, ls="--")
        for y, (_, r) in zip(ys, g.iterrows()):
            ax.annotate("", xy=(r.realized_mean, y), xytext=(r.reported_mean, y),
                        arrowprops=dict(arrowstyle="-|>", color="#bdbdbd", lw=0.9, mutation_scale=6,
                                        shrinkA=2.5, shrinkB=2.5), zorder=1)
            ax.scatter(r.reported_mean, y, s=16, facecolor="white", edgecolor=MODEL_COLOR[m], lw=0.8, zorder=3,
                       marker=MODEL_MARKER[m])
            _dot(ax, [r.realized_mean], [y], [ref + r.ci_lo], [ref + r.ci_hi], MODEL_COLOR[m], MODEL_MARKER[m])
        _panel(ax, p, MODEL_LABEL[m]); _grid(ax, "x")
        ax.text(ref, len(order) - 0.45, " reference", fontsize=5.8, color=INK2, va="bottom", ha="left")
        ax.set_yticks(ys); ax.set_yticklabels([lab[v] for v in order]); ax.set_ylim(-0.5, len(order) - 0.2)
    axes[-1].set_xlabel("Mean PR-AUC")
    hs = [Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK2, label="Reported (leaky evaluation)"),
          Line2D([], [], ls="", marker="o", ms=4, mfc=INK2, mec=INK2, label="Realised at $t^0$ (95% CI)")]
    fig.legend(handles=hs, loc="outside upper center", ncol=2)
    _save(fig, out, "fig06_rq1_realised_performance")


# ----------------------------------------------------------------------------------------------- Fig. 7 learner comparison
def _pairs(df):
    if "a" not in df:
        df = df.assign(a="xgb", b="lr")
    order = [("xgb", "lr"), ("rf", "lr"), ("xgb", "rf")]
    present = set(zip(df.a, df.b))
    return df, [p for p in order if p in present]


def fig_tuned_ranking(out):
    A, B = _read(out, "rq1_ranking_flip.csv"), _read(out, "rq1b_tuned_ranking_flip.csv")
    if B is None:
        return
    A, _ = _pairs(A); B, pairs = _pairs(B)
    order = ["none", "R", "R+G+P"]
    ys = np.arange(len(order))[::-1].astype(float)
    fig, axes = _subplots(len(pairs), 1, figsize=(COL, 0.75 + 0.95 * len(pairs)), sharex=True)
    axes = np.atleast_1d(axes)
    for p, (ax, (a, b)) in enumerate(zip(axes, pairs)):
        for o, (T, filled) in zip((0.13, -0.13), ((A, False), (B, True))):
            g = T[(T.metric == "ap") & (T.a == a) & (T.b == b)].set_index("shortcuts").reindex(order)
            for y, (_, r) in zip(ys + o, g.iterrows()):
                col = MODEL_COLOR[a] if r["diff"] > 0 else MODEL_COLOR[b]
                sig = r.ci_lo > 0 or r.ci_hi < 0
                _dot(ax, [r["diff"]], [y], [r.ci_lo], [r.ci_hi], col if sig else MUTED, "o", filled)
        ax.axvline(0, color=INK, lw=0.6); _grid(ax, "x")
        _panel(ax, p, f"{MODEL_SHORT[a]} − {MODEL_SHORT[b]}")
        ax.text(0.02, 0.02, f"{MODEL_SHORT[b]} better", transform=ax.transAxes, fontsize=5.8, color=MODEL_COLOR[b])
        ax.text(0.98, 0.02, f"{MODEL_SHORT[a]} better", transform=ax.transAxes, fontsize=5.8, color=MODEL_COLOR[a],
                ha="right")
        ax.set_yticks(ys); ax.set_yticklabels(["Reference", "Random split (R)", "R + G + P"])
        ax.set_ylim(-0.75, len(order) - 0.45)
    lim = max(max(abs(np.array(ax.get_xlim()))) for ax in axes); axes[0].set_xlim(-lim, lim)
    axes[-1].set_xlabel("Difference in PR-AUC")
    hs = [Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK2, label="Fixed hyper-parameters"),
          Line2D([], [], ls="", marker="o", ms=4, mfc=INK2, mec=INK2, label="Nested tuning"),
          Line2D([], [], ls="", marker="o", ms=4, mfc=MUTED, mec=MUTED, label="95% CI includes 0")]
    fig.legend(handles=hs, loc="outside upper center", ncol=2)
    _save(fig, out, "fig07_rq1_model_ranking")


def fig_ranking(out):
    """Supplementary: learner comparison under every condition with fixed hyper-parameters."""
    A = _read(out, "rq1_ranking_flip.csv")
    if A is None:
        return
    A, pairs = _pairs(A); A = A[A.metric == "ap"]
    order = [s for s in ["none", "NP", "G", "Graw", "Gloo", "NP+G", "P", "NP+P", "G+P", "NP+G+P", "R", "R+G", "R+P",
                         "R+G+P"] if s in set(A.shortcuts)]
    ys = np.arange(len(order))[::-1]
    fig, axes = _subplots(1, len(pairs), figsize=(FULL, 2.9), sharey=True, gridspec_kw=dict(wspace=0.1))
    axes = np.atleast_1d(axes)
    for p, (ax, (a, b)) in enumerate(zip(axes, pairs)):
        g = A[(A.a == a) & (A.b == b)].set_index("shortcuts").reindex(order)
        rest = g.drop(index=[s_ for s_ in ("Gloo",) if s_ in g.index])
        hi_lim = max(abs(rest.ci_lo).max(), abs(rest.ci_hi).max()) * 1.15
        for y, (sc, r) in zip(ys, g.iterrows()):
            col = MODEL_COLOR[a] if r["diff"] > 0 else MODEL_COLOR[b]
            sig = r.ci_lo > 0 or r.ci_hi < 0
            if abs(r["diff"]) > hi_lim:          # leave-one-out aggregates: far off the scale of the other conditions
                ax.annotate(f"{r['diff']:+.2f} {'→' if r['diff'] > 0 else '←'}", (np.sign(r["diff"]) * hi_lim, y),
                            ha="right" if r["diff"] > 0 else "left", va="center", fontsize=5.8, color=col)
                continue
            _dot(ax, [r["diff"]], [y], [r.ci_lo], [r.ci_hi], col if sig else MUTED, "o", sig)
        ax.set_xlim(-hi_lim, hi_lim)
        ax.axvline(0, color=INK, lw=0.6); _grid(ax, "x")
        ax.axhspan(-0.5, ys[order.index("R")] + 0.5, color=LIGHT, zorder=0, lw=0)
        _panel(ax, p, f"{MODEL_SHORT[a]} − {MODEL_SHORT[b]}"); ax.set_xlabel("Difference in PR-AUC")
    axes[0].set_yticks(ys); axes[0].set_yticklabels([SC_LABEL.get(s, s) for s in order])
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    _save(fig, out, "figS5_rq1_ranking_fixed")


# ----------------------------------------------------------------------------------------------- Fig. 8 second task
def fig_task2(out):
    I2 = _read(out, "task2_inflation.csv")
    if I2 is None:
        return
    I = {"main": _read(out, "rq1_inflation.csv"), "task2": I2}
    Dp = {"main": _read(out, "rq1_deployment_summary.csv"), "task2": _read(out, "task2_deployment_summary.csv")}
    order = ["NP", "R", "G", "P", "R+G+P"]
    lab = {"NP": "No purge (NP)", "R": "Random split (R)", "G": "Full-period aggregates (G)",
           "P": "Post-prediction features (P)", "R+G+P": "All shortcuts"}
    models = ["xgb", "lr"]
    fig, axes = _subplots(1, 2, figsize=(FULL, 2.25), gridspec_kw=dict(wspace=0.5, width_ratios=[1.1, 1]))
    ax = axes[0]; ys = np.arange(len(order))[::-1].astype(float)
    offs = {("main", "xgb"): 0.27, ("main", "lr"): 0.09, ("task2", "xgb"): -0.09, ("task2", "lr"): -0.27}
    for task in ("main", "task2"):
        for m in models:
            g = I[task]; g = g[(g.model == m) & (g.metric == "ap")].set_index("shortcuts").reindex(order)
            k = 100 / g.ref_mean
            _dot(ax, k * g.delta, ys + offs[(task, m)], k * g.ci_lo, k * g.ci_hi, MODEL_COLOR[m], MODEL_MARKER[m],
                 task == "main")
    ax.axvline(0, color=INK, lw=0.6); _grid(ax, "x")
    ax.set_yticks(ys); ax.set_yticklabels([lab[s] for s in order]); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Gain in PR-AUC relative to the reference (%)"); _panel(ax, 0, "Inflation")
    ax = axes[1]; vorder = ["G", "P", "G+P"]; vy = np.arange(len(vorder))[::-1].astype(float)
    for task in ("main", "task2"):
        for m in models:
            g = Dp[task]; g = g[(g.model == m) & (g.metric == "ap")].set_index("variant").reindex(vorder)
            k = 100 / g.correct_mean
            _dot(ax, k * g.realized_minus_correct, vy + offs[(task, m)], k * g.ci_lo, k * g.ci_hi, MODEL_COLOR[m],
                 MODEL_MARKER[m], task == "main")
    ax.axvline(0, color=INK, lw=0.6); _grid(ax, "x")
    ax.set_yticks(vy); ax.set_yticklabels(["Full-period aggregates (G)", "Post-prediction features (P)", "G + P"])
    ax.set_ylim(-0.6, len(vorder) - 0.4)
    ax.set_xlabel("Realised minus reference PR-AUC (%)"); _panel(ax, 1, "Realised performance")
    hs = _model_handles(models) + [
        Line2D([], [], ls="", marker="o", ms=4, mfc=INK2, mec=INK2, label="Low rating at hand-over (main task)"),
        Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK2, label="Late delivery at approval (second task)")]
    fig.legend(handles=hs, loc="outside upper center", ncol=4)
    _save(fig, out, "fig08_rq1_second_task")


# ----------------------------------------------------------------------------------------------- Fig. 9 RQ2
def _milestone_windows(out):
    R = _read(out, "rq2_window_metrics.csv")
    if R is None:
        return None
    ext = out / "robustness" / "milestone_metrics.csv"
    if not ext.is_file():
        ext = out / "robustness" / "milestone_metrics_quick.csv"
    if ext.is_file():
        E = pd.read_csv(ext)
        R = pd.concat([R, E[E.milestone.isin(["t0+1d", "t0+2d"])]], ignore_index=True)
    return R


def fig_rq2(out):
    R = _milestone_windows(out)
    if R is None:
        return
    cfg = _cfg(out)
    ms = [m for m in MS_ORDER if m in set(R.milestone)]
    extra = {"t0+1d", "t0+2d"}
    rows = []
    for (m, s), g in R.groupby(["model", "milestone"]):
        r = dict(model=m, milestone=s, open=g.pos_eligible.sum() / g.pos_base.sum())
        for met in ("early_catch", "ap"):
            r[met], r[met + "_lo"], r[met + "_hi"] = _boot(g[met].to_numpy(), cfg)
        rows.append(r)
    S = pd.DataFrame(rows)
    models = _models(S); x = np.arange(len(ms)); off = _offsets(len(models), -0.12)
    fig, axes = _subplots(1, 3, figsize=(FULL, 2.15), gridspec_kw=dict(wspace=0.36, width_ratios=[1.2, 1.2, 0.9]))
    for p, (ax, met, title, ylab, k) in enumerate(((axes[0], "early_catch", "Early catch rate",
                                                    "Low-rated orders reached before review (%)", 100),
                                                   (axes[1], "ap", "PR-AUC on orders not yet reviewed", "PR-AUC", 1))):
        for o, m in zip(off, models):
            g = S[S.model == m].set_index("milestone").reindex(ms)
            ok = g[met].notna().to_numpy()
            ax.plot((x + o)[ok], k * g[met][ok], color=MODEL_COLOR[m], lw=0.8, zorder=2)
            for xi, s in zip(x + o, ms):
                r = g.loc[s]
                if pd.isna(r[met]):
                    continue
                _dot(ax, [xi], [k * r[met]], [k * r[met + "_lo"]], [k * r[met + "_hi"]], MODEL_COLOR[m],
                     MODEL_MARKER[m], s not in extra, horizontal=False)
        ax.axvspan(ms.index("t0") - 0.4, ms.index("t0") + 0.4, color=LIGHT, zorder=0, lw=0)
        ax.set_xticks(x); ax.set_xticklabels([MS_LABEL[s] for s in ms], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylab); _panel(ax, p, title); _grid(ax, "y")
    g = S.groupby("milestone").open.first().reindex(ms).to_frame()
    ax = axes[2]
    lost = 100 * (1 - g.open)
    ax.bar(x, lost, color="#b8b8b8", width=0.62)
    for i, v in enumerate(lost):
        ax.text(i, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([MS_LABEL[s] for s in ms], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Already reviewed (% of low-rated)"); _panel(ax, 2, "Already reviewed")
    ax.set_ylim(0, max(lost) * 1.2 + 2); _grid(ax, "y")
    hs = _model_handles(models)
    if extra & set(ms):
        hs.append(Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK2, label="Milestones added in the robustness analysis"))
    fig.legend(handles=hs, loc="outside upper center", ncol=len(hs))
    _save(fig, out, "fig09_rq2_timing")


def fig_milestone_budgets(out):
    """Supplementary: early catch rate by milestone for alert budgets of 5%, 10% and 20%."""
    E = _read(out, "robustness_milestone_budget_windows.csv")
    if E is None:
        return
    cfg = _cfg(out)
    budgets = sorted(E.budget.unique()); ms = [m for m in MS_ORDER if m in set(E.milestone)]
    models = _models(E); x = np.arange(len(ms)); off = _offsets(len(models), -0.12)
    fig, axes = _subplots(1, len(budgets), figsize=(FULL, 1.95), gridspec_kw=dict(wspace=0.3))
    for p, (ax, b) in enumerate(zip(np.atleast_1d(axes), budgets)):
        for o, m in zip(off, models):
            vals = [_boot(E[(E.model == m) & (E.budget == b) & (E.milestone == s)].sort_values("k").early_catch, cfg)
                    if ((E.model == m) & (E.budget == b) & (E.milestone == s)).any() else (np.nan,) * 3 for s in ms]
            v = np.array(vals) * 100
            ax.plot(x + o, v[:, 0], color=MODEL_COLOR[m], lw=0.8)
            _dot(ax, x + o, v[:, 0], v[:, 1], v[:, 2], MODEL_COLOR[m], MODEL_MARKER[m], horizontal=False)
        ax.axvspan(ms.index("t0") - 0.4, ms.index("t0") + 0.4, color=LIGHT, zorder=0, lw=0)
        ax.set_xticks(x); ax.set_xticklabels([MS_LABEL[s] for s in ms], rotation=45, ha="right", rotation_mode="anchor")
        _panel(ax, p, f"Alert budget {100 * b:.0f}%"); _grid(ax, "y")
    np.atleast_1d(axes)[0].set_ylabel("Early catch rate (%)")
    fig.legend(handles=_model_handles(models), loc="outside upper center", ncol=len(models))
    _save(fig, out, "figS6_rq2_alert_budgets")


# ----------------------------------------------------------------------------------------------- Fig. 10 label latency
def fig_label_latency(out):
    F = load_features(out); D = F[F.y >= 0]
    lag = (D.label_available_ts - D.t0).dt.total_seconds() / 86400
    fig, axes = _subplots(2, 1, figsize=(COL, 3.7), gridspec_kw=dict(hspace=0.62, height_ratios=[1, 1.05]))
    ax = axes[0]; d = np.linspace(0, 40, 161)
    for cls, col, ls, lab in ((1, OI["vermillion"], "-", "Low-rated (1–2 stars)"), (0, OI["blue"], "--", "High-rated (4–5 stars)")):
        L = np.sort(lag[D.y == cls].to_numpy())
        ax.plot(d, 100 * np.searchsorted(L, d, side="right") / len(L), color=col, ls=ls, lw=1.3, label=lab)
    for cls, col in ((1, OI["vermillion"]), (0, OI["blue"])):
        med = float(np.median(lag[D.y == cls]))
        ax.plot([med, med], [0, 50], color=col, lw=0.5, ls=":")
    ax.axhline(50, color=MUTED, lw=0.4, ls=":")
    ax.set_xlabel("Days since hand-over ($t^0$)"); ax.set_ylabel("Labels known (%)")
    _panel(ax, 0, "Low ratings arrive later"); _grid(ax, "y"); ax.legend(loc="lower right")
    ax.set_xlim(0, 40); ax.set_ylim(0, 102)
    R = _read(out, "rq3_window_metrics.csv", parse_dates=["origin"])
    if R is not None:
        g = R[(R.model == "xgb") & (R.strategy == "platt")].sort_values("origin")
        ax = axes[1]
        ax.plot(g.origin, 100 * g.obs, color=INK, lw=1.2, marker="o", ms=2.0, label="Observed in the evaluation window")
        ax.plot(g.origin, 100 * g.pi_recent_labeled, color=OI["purple"], lw=1.0, marker="s", ms=1.9,
                label="Last 28 days, labels received so far")
        col_w = "pi_recent_ipw_roll" if "pi_recent_ipw_roll" in g else "pi_recent_ipw"
        ax.plot(g.origin, 100 * g[col_w], color=OI["green"], lw=1.0, marker="^", ms=2.1,
                label="Last 28 days, latency-weighted")
        ax.set_ylabel("Low-rating rate (%)"); _panel(ax, 1, "What a recalibration window sees")
        _date_axis(ax); _grid(ax, "y")
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=1, handlelength=1.6)
    _save(fig, out, "fig10_rq3_label_latency")


# ----------------------------------------------------------------------------------------------- Fig. 11 calibration
RQ3_ORDER = ["static", "raw", "platt", "isotonic", "prior_recent", "prior_em", "platt_ipw", "platt_ipw_roll",
             "platt_ipw_nowcast"]


def fig_rq3_calibration(out):
    R = _read(out, "rq3_window_metrics.csv", parse_dates=["origin"]); S = _read(out, "rq3_summary.csv")
    if R is None:
        return
    fig, axes = _subplots(1, 2, figsize=(FULL, 2.45), gridspec_kw=dict(wspace=0.75, width_ratios=[1.15, 1]))
    ax = axes[0]; g = R[R.model == "xgb"]
    for s in ["static", "platt", "platt_ipw"]:
        h = g[g.strategy == s].sort_values("origin")
        col, ls = STRAT_STYLE[s]
        ax.plot(h.origin, 100 * h.citl_diff, color=col, ls=ls, lw=1.0, marker="o", ms=1.8, label=STRAT_LINE[s])
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_ylabel("Mean predicted minus observed (pp)"); _panel(ax, 0, "Calibration in the large, XGBoost")
    _date_axis(ax); _grid(ax, "y"); ax.legend(loc="lower left", ncol=1)
    ax = axes[1]
    order = [s for s in RQ3_ORDER if s in set(S.strategy)]
    models = _models(S); off = _offsets(len(models), 0.26)
    LO, HI = -0.2, 0.62
    ys = np.arange(len(order))[::-1].astype(float)
    for o, m in zip(off, models):
        h = S[S.model == m].set_index("strategy").loc[order]
        for y, s in zip(ys + o, order):
            v, lo, hi = h.loc[s, ["cal_intercept", "cal_intercept_lo", "cal_intercept_hi"]]
            if v > HI:
                ax.annotate(f"{v:.2f} →", (HI, y), xytext=(-4, 0), textcoords="offset points", ha="right",
                            va="center", fontsize=5.8, color=MODEL_COLOR[m])
                continue
            _dot(ax, [v], [y], [max(lo, LO)], [min(hi, HI)], MODEL_COLOR[m], MODEL_MARKER[m])
    ax.axvline(0, color=INK, lw=0.6)
    ax.axhspan(ys[order.index("platt_ipw")] + 0.5, -0.5, color=LIGHT, zorder=0, lw=0)
    ax.set_yticks(ys); ax.set_yticklabels([STRAT_LABEL[s] for s in order]); ax.set_ylim(-0.5, len(order) - 0.5)
    ax.set_xlim(LO, HI); ax.set_xlabel("Calibration intercept (> 0: under-prediction)")
    _panel(ax, 1, "Systematic bias by strategy"); _grid(ax, "x")
    ax.legend(handles=_model_handles(models), loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(models))
    _save(fig, out, "fig11_rq3_calibration")


# ----------------------------------------------------------------------------------------------- Fig. 12 expected cost
def fig_rq3_cost(out):
    S = _read(out, "rq3_summary.csv")
    if S is None:
        return
    from ..experiments.rq3_recalibration import RATIOS
    order = [s for s in RQ3_ORDER if s in set(S.strategy) and s != "prior_em"]
    models = _models(S); off = _offsets(len(models), 0.26)
    ys = np.arange(len(order))[::-1].astype(float)
    fig, axes = _subplots(1, len(RATIOS), figsize=(FULL, 2.3), sharey=True, gridspec_kw=dict(wspace=0.1))
    for p, (ax, r) in enumerate(zip(axes, RATIOS)):
        c = f"saving_r{r}"
        for o, m in zip(off, models):
            h = S[S.model == m].set_index("strategy").loc[order]
            _dot(ax, 100 * h[c], ys + o, 100 * h[c + "_lo"], 100 * h[c + "_hi"], MODEL_COLOR[m], MODEL_MARKER[m])
            orc = 100 * S[(S.model == m) & (S.strategy == "platt")][f"saving_oracle_r{r}"].iloc[0]
            ax.axvline(orc, color=MODEL_COLOR[m], lw=0.7, ls=(0, (3, 2)))
        ax.axvline(0, color=INK, lw=0.6)
        ax.axhspan(ys[order.index("platt_ipw")] + 0.5, -0.5, color=LIGHT, zorder=0, lw=0)
        _panel(ax, p, f"Cost ratio r = {r}"); _grid(ax, "x")
        ax.set_xlabel("Cost saving (%)")
    axes[0].set_yticks(ys); axes[0].set_yticklabels([STRAT_LABEL[s] for s in order]); axes[0].set_ylim(-0.5, len(order) - 0.5)
    hs = _model_handles(models) + [Line2D([], [], color=INK2, lw=0.7, ls=(0, (3, 2)),
                                          label="Best threshold in hindsight (rolling Platt)")]
    fig.legend(handles=hs, loc="outside upper center", ncol=len(hs))
    _save(fig, out, "fig12_rq3_cost_savings")


# ----------------------------------------------------------------------------------------------- Fig. 13 net benefit
def fig_rq3_decision_curve(out):
    Dc = _read(out, "rq3_decision_curve.csv")
    if Dc is None:
        return
    lab = {"platt": "Rolling Platt", "ipw": "Latency-weighted Platt", "static": "Frozen model + Platt"}
    key = {"platt": "platt", "ipw": "platt_ipw", "static": "static"}
    models = _models(Dc)
    fig, axes = _subplots(len(models), 1, figsize=(COL, 0.7 + 1.45 * len(models)), sharex=True, sharey=True)
    for p, (ax, m) in enumerate(zip(np.atleast_1d(axes), models)):
        g = Dc[(Dc.model == m) & (Dc.threshold >= 0.04) & (Dc.threshold <= 0.30)]
        base = g[g.strategy == "alert_all"].set_index("threshold").net_benefit.clip(lower=0)
        for s in ["static", "platt", "ipw"]:
            h = g[g.strategy == s].set_index("threshold").net_benefit
            col, ls = STRAT_STYLE[key[s]]
            ax.plot(h.index, 100 * (h - base.reindex(h.index)), color=col, ls=ls, lw=1.1, label=lab[s])
        ax.axhline(0, color=INK, lw=0.6)
        for r_ in (10, 5, 3):
            ax.axvline(1 / (1 + r_), color=MUTED, lw=0.5, ls=":")
            ax.text(1 / (1 + r_), 1.0, f"r = {r_}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=5.8, color=INK2)
        ax.set_xlim(0.04, 0.30)
        ax.set_title(f"({'ab'[p]}) {MODEL_LABEL[m]}", pad=11); _grid(ax, "y")
        ax.set_ylabel("Net benefit gain (points)")
    np.atleast_1d(axes)[-1].set_xlabel("Probability threshold $t = 1/(1+r)$")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2)
    _save(fig, out, "fig13_rq3_net_benefit")


# ----------------------------------------------------------------------------------------------- Fig. 14 workload
def fig_rq3_workload(out):
    Dd = _read(out, "rq3_daily_policy.csv", parse_dates=["day"])
    if Dd is None:
        return
    g = Dd[(Dd.model == "xgb") & (Dd.n >= 30)].copy()
    fig, axes = _subplots(2, 1, figsize=(COL, 3.7), gridspec_kw=dict(height_ratios=[1, 1.05]))
    ax = axes[0]
    ax.plot(g.day, 100 * g.thr_alerts / g.n, color=OI["orange"], lw=0.7, label="Fixed probability threshold")
    ax.plot(g.day, 100 * g.topk_alerts / g.n, color=OI["blue"], lw=1.0, label="Top 10% of each day")
    ax.set_ylabel("Orders alerted (% of the day)"); _panel(ax, 0, "Daily alert volume, XGBoost")
    _date_axis(ax); _grid(ax, "y"); ax.set_ylim(0, None)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=2, handlelength=1.6)
    ax = axes[1]; rows = []
    for m in _models(Dd):
        h = Dd[(Dd.model == m) & (Dd.n >= 30)]
        for lab, a_, tp, col in (("fixed threshold", "thr_alerts", "thr_tp", OI["orange"]),
                                 ("daily top 10%", "topk_alerts", "topk_tp", OI["blue"])):
            rate = 100 * h[a_] / h.n
            rows.append(dict(name=f"{MODEL_SHORT[m]}, {lab}", rate=rate.to_numpy(), col=col,
                             prec=h[tp].sum() / h[a_].sum(), rec=h[tp].sum() / h.pos.sum(), cv=rate.std() / rate.mean()))
    ys = np.arange(len(rows))[::-1]
    bp = ax.boxplot([r["rate"] for r in rows], positions=ys, orientation="horizontal", widths=0.5, patch_artist=True,
                    flierprops=dict(marker=".", ms=2, mec=MUTED, alpha=0.6), whiskerprops=dict(lw=0.6),
                    capprops=dict(lw=0.6), medianprops=dict(color=INK, lw=0.8), boxprops=dict(lw=0.5))
    for patch, r in zip(bp["boxes"], rows):
        patch.set_facecolor(r["col"]); patch.set_alpha(0.8)
    # precision, recall and coefficient of variation in a fixed column to the right of the axes, clear of any outlier
    # precision, recall and coefficient of variation as a second line of each label, clear of any outlier
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['name']}\nP {r['prec']:.2f}  R {r['rec']:.2f}  CV {r['cv']:.2f}" for r in rows],
                       fontsize=5.8, linespacing=1.15)
    ax.set_xlabel("Orders alerted (% of the day)"); _panel(ax, 1, "Distribution over days"); _grid(ax, "x")
    _save(fig, out, "fig14_rq3_daily_workload")


# ----------------------------------------------------------------------------------------------- supplementary
def fig_over_time(out):
    R = _read(out, "rq1_window_metrics.csv", parse_dates=["origin"])
    models = _models(R)
    lines = [("none", INK, "-"), ("R", SC_COLOR["R"], "-"), ("P", SC_COLOR["P"], "-"), ("R+G+P", "#882255", "--")]
    fig, axes = _subplots(len(models), 1, figsize=(COL, 0.7 + 1.45 * len(models)), sharex=True, sharey=True)
    for p, (ax, m) in enumerate(zip(np.atleast_1d(axes), models)):
        g = R[R.model == m]
        ref = g[g.shortcuts == "none"].sort_values("origin")
        ax.fill_between(ref.origin, 0, ref.prevalence, color=LIGHT, lw=0, label="Base rate")
        for sc, col, ls in lines:
            h = g[g.shortcuts == sc].sort_values("origin")
            ax.plot(h.origin, h.ap, color=col, ls=ls, lw=1.0, label=SC_LABEL[sc])
        _panel(ax, p, MODEL_LABEL[m]); _date_axis(ax); _grid(ax, "y")
    np.atleast_1d(axes)[0].set_ylabel("PR-AUC in the window"); np.atleast_1d(axes)[0].set_ylim(0, None)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=5)
    _save(fig, out, "figS3_rq1_prauc_over_time")


def fig_selection(out):
    S, T = _read(out, "rq1_selection_summary.csv"), _read(out, "rq1_threshold_summary.csv")
    fig, axes = _subplots(1, 2, figsize=(FULL, 1.9), gridspec_kw=dict(wspace=0.75, width_ratios=[1.2, 1]))
    if S is not None:
        S = S[S.metric == "ap"]; ax = axes[0]
        labels = ["Best in the previous window (valid)", "Best over all test windows", "Best in each test window"]
        vals = [S.forward_mean.iloc[0], S[S.shortcut == "leaderboard_on_test"].shortcut_mean.iloc[0],
                S[S.shortcut == "oracle_per_window"].shortcut_mean.iloc[0]]
        ys = np.arange(3)[::-1]
        for y, v, c in zip(ys, vals, (INK2, OI["orange"], OI["vermillion"])):
            ax.plot([vals[0], v], [y, y], color=GRID, lw=2, solid_capstyle="round", zorder=1)
            ax.scatter(v, y, s=18, color=c, zorder=3)
            ax.text(v, y + 0.22, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
        ax.set_yticks(ys); ax.set_yticklabels(labels); ax.set_ylim(-0.5, 2.7)
        ax.set_xlim(vals[0] - 0.006, max(vals) + 0.006)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.set_xlabel("Mean PR-AUC of the selected configuration"); _panel(ax, 0, "Model chosen on test data")
        _grid(ax, "x")
    ax = axes[1]; models = _models(T); x = np.arange(len(models))
    for i, m in enumerate(models):
        r = T.set_index("model").loc[m]
        ax.bar(i - 0.18, r.f1_forward, width=0.34, color=MODEL_COLOR[m], alpha=0.45)
        ax.bar(i + 0.18, r.f1_oracle, width=0.34, color=MODEL_COLOR[m])
        ax.text(i + 0.18, r.f1_oracle + 0.006, f"{r.rel_pct:+.0f}%", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL[m] for m in models])
    ax.set_ylabel("Mean F1"); ax.set_ylim(0, T.f1_oracle.max() * 1.3); _grid(ax, "y")
    _panel(ax, 1, "Threshold chosen on test data")
    ax.legend(handles=[Patch(fc=INK2, alpha=0.45, label="From the previous window"),
                       Patch(fc=INK2, label="Tuned on the test window")], loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=2)
    _save(fig, out, "figS4_rq1_selection_shortcuts")


def make_all(cfg, out):
    global _CFG
    _CFG = cfg
    _style()
    fig_sample_flow(out); fig_design(out); fig_context(out)
    for m in ("ap", "auc"):
        fig_inflation(out, m); fig_shapley(out, m)
    fig_deployment(out); fig_tuned_ranking(out); fig_task2(out); fig_rq2(out); fig_label_latency(out)
    fig_rq3_calibration(out); fig_rq3_cost(out); fig_rq3_decision_curve(out); fig_rq3_workload(out)
    fig_over_time(out); fig_selection(out); fig_ranking(out); fig_milestone_budgets(out)
