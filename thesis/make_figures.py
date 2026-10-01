"""Generate every figure in Chapter Four from the result artefacts.

    python thesis/make_figures.py

Writes PNG (200 dpi, for the Markdown draft) and PDF (vector, for LaTeX) into
``thesis/figures/``. Exits non-zero if any input is missing, so a stale figure
set cannot silently survive a deleted CSV.

Design constraints
------------------
* **Greyscale-safe.** Theses are often printed in monochrome, so series are
  separated by lightness, marker and hatch — never by hue alone.
* **Tier colours are consistent across figures**: reference baselines, classical
  methods, machine learning. The reader learns the encoding once.
* No truncated bar axes, no 3-D, no decorative colour.

Provenance
----------
Every figure is derived from a CSV in ``outputs/reports/`` **except** the three
noted below, whose underlying data exists only inside the written record. Those
are transcribed with their source named, and their captions in the chapter say
"reproduced from" rather than implying fresh computation:

* ``LEARNING_CURVE`` and ``FEATURE_ABLATION`` — redo/results/05. Regenerating
  them means re-running experiments on the retired 2020-2023 window.
* ``DAY_AHEAD_LADDER`` — the validate_nwp gate run recorded in RESULTS.md.
* the ceiling curve in ``fig_outlook_ceiling`` — redo/results/10. Computing it
  needs the hourly capacity-factor series, which the monthly CSV does not carry.

The fold-difficulty figure was *not* transcribed: `src.bakeoff --per-fold` was
added so it comes from a fresh artefact like everything else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

REPORTS = Path("outputs/reports")
FIGDIR = Path("thesis/figures")

# --- greyscale-safe palette: separated by lightness, not hue ----------------
C_REF = "#B0B0B0"        # reference baselines — light
C_CLASSICAL = "#6C8EBF"  # classical methods — mid
C_ML = "#1F3864"         # machine learning — dark
C_ACCENT = "#C00000"     # the one thing the reader should look at
C_GRID = "#DDDDDD"

TIER_COLOUR = {"reference": C_REF, "classical": C_CLASSICAL, "ml": C_ML}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

# ---------------------------------------------------------------------------
# Transcribed data — see the module docstring
# ---------------------------------------------------------------------------

# Source: redo/results/05_ceiling_and_information.md, 2020-2023 window, 24 h.
LEARNING_CURVE = pd.DataFrame({
    "train_years": [0.4, 0.8, 1.6, 2.4, 3.2],
    "test_mae": [1.5017, 1.1933, 1.1534, 1.1479, 1.1410],
    "train_test_ratio": [2.36, 1.52, 1.25, 1.19, 1.16],
})

FEATURE_ABLATION = pd.DataFrame({
    "group": ["lag/rolling", "+ calendar", "+ surface", "+ upper-air", "+ spatial"],
    "n_features": [9, 13, 25, 33, 53],
    "test_mae": [1.2564, 1.1760, 1.1615, 1.1444, 1.1410],
})

# Source: RESULTS.md, authoritative validate_nwp gate run, 2024-03 -> 2025-12.
DAY_AHEAD_LADDER = pd.DataFrame({
    "predictor": ["Climatology", "Persistence", "OLS on hub(t)",
                  "XGBoost\nwithout NWP", "XGBoost\nwith NWP"],
    "mae": [1.3490, 1.3097, 1.1349, 1.0513, 0.7295],
    "r2": [-0.0711, -0.0149, 0.2225, 0.3300, 0.6653],
    "tier": ["reference", "reference", "classical", "ml", "ml"],
})

# Tier assignment for the eleven-predictor ladder.
LADDER_TIER = {
    "xgboost": "ml", "lightgbm": "ml", "ridge": "ml",
    "mos_multi": "classical", "mos_single": "classical", "kalman_mos": "classical",
    "analog_ensemble": "classical", "sarimax_nwp": "classical",
    "persistence": "reference", "damped_persistence": "reference",
    "climatology": "reference",
}

PRETTY = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "ridge": "Ridge",
    "mos_multi": "Multi-model MOS", "mos_single": "Single-model MOS",
    "kalman_mos": "Kalman MOS", "analog_ensemble": "Analogue ensemble",
    "sarimax_nwp": "SARIMAX + NWP", "persistence": "Naive persistence",
    "damped_persistence": "Damped persistence", "climatology": "Climatology",
    "climatology_archive": "Climatology (archive)",
    "climatology_window": "Climatology (window)",
    "ensemble_mean": "Raw ensemble mean",
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read(name: str) -> pd.DataFrame:
    path = REPORTS / name
    if not path.exists():
        raise SystemExit(
            f"missing input {path}. Regenerate it before building figures — a "
            f"partial figure set is worse than none, because it looks complete.")
    return pd.read_csv(path)


def save(fig, stem: str, title: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  {stem:28s} {title}")


def bar_labels(ax, bars, values, fmt="{:.4f}", pad=0.004, fontsize=8):
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2,
                fmt.format(v), va="center", ha="left", fontsize=fontsize)


# ---------------------------------------------------------------------------
# 4.1 Predictability ceiling
# ---------------------------------------------------------------------------

def fig_ceiling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))

    lc = LEARNING_CURVE
    ax1.plot(lc.train_years, lc.test_mae, "o-", color=C_ML, lw=1.8, ms=6)
    # Ratios sit below the curve from the second point on, so they cannot
    # collide with the annotation arrow that lands on the final point.
    for i, (x, y, r) in enumerate(zip(lc.train_years, lc.test_mae,
                                      lc.train_test_ratio)):
        ax1.annotate(f"ratio {r:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 10 if i == 0 else -16), ha="center",
                     fontsize=7.5, color="#555555")
    ax1.annotate("final doubling buys 1.1%",
                 xy=(3.2, 1.148), xytext=(1.35, 1.33),
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2),
                 color=C_ACCENT, fontsize=8.5)
    ax1.set_ylim(1.10, 1.56)
    ax1.set_xlabel("Training data (years)")
    ax1.set_ylabel("Test MAE (m s$^{-1}$)")
    ax1.set_title("(a) Learning curve — flat after ~1.6 years")

    ab = FEATURE_ABLATION
    x = np.arange(len(ab))
    bars = ax2.bar(x, ab.test_mae, color=C_CLASSICAL, width=0.62,
                   edgecolor="white")
    bars[-1].set_color(C_ML)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{g}\n({n})" for g, n in zip(ab.group, ab.n_features)],
                        fontsize=8)
    ax2.set_ylim(1.05, 1.30)
    ax2.set_ylabel("Test MAE (m s$^{-1}$)")
    ax2.set_title("(b) Feature ablation — saturated")
    for xi, v in zip(x, ab.test_mae):
        ax2.text(xi, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
    ax2.annotate("last 28 columns buy 0.02",
                 xy=(3.5, 1.152), xytext=(1.4, 1.24),
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2),
                 color=C_ACCENT, fontsize=8.5)

    fig.suptitle("Predictability ceiling: the problem is information-limited, "
                 "not variance-limited", y=1.02, fontsize=11.5)
    save(fig, "fig_4_01_ceiling", "predictability ceiling")


# ---------------------------------------------------------------------------
# 4.2 Day-ahead predictor ladder
# ---------------------------------------------------------------------------

def fig_day_ahead_ladder():
    d = DAY_AHEAD_LADDER
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))
    x = np.arange(len(d))
    colours = [TIER_COLOUR[t] for t in d.tier]

    ax1.bar(x, d.mae, color=colours, width=0.62, edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(d.predictor, fontsize=8)
    ax1.set_ylabel("MAE (m s$^{-1}$)")
    ax1.set_title("(a) Mean absolute error")
    for xi, v in zip(x, d.mae):
        ax1.text(xi, v + 0.02, f"{v:.4f}", ha="center", fontsize=8)
    ax1.annotate("", xy=(4, 0.7295), xytext=(3, 1.0513),
                 arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=1.4))
    ax1.text(3.5, 0.93, "−30.6%", color=C_ACCENT, fontsize=10,
             ha="center", fontweight="bold")
    ax1.set_ylim(0, 1.55)

    ax2.bar(x, d.r2, color=colours, width=0.62, edgecolor="white")
    ax2.axhline(0.326, color=C_ACCENT, ls="--", lw=1.3)
    ax2.text(0.05, 0.345, "autocorrelation ceiling  R$^2$ = 0.326",
             color=C_ACCENT, fontsize=8.5)
    ax2.axhline(0, color="#888888", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(d.predictor, fontsize=8)
    ax2.set_ylabel("Test R$^2$")
    ax2.set_title("(b) R$^2$, against the state-only ceiling")
    for xi, v in zip(x, d.r2):
        ax2.text(xi, v + (0.02 if v >= 0 else -0.05), f"{v:.4f}",
                 ha="center", fontsize=8)

    fig.suptitle("Day-ahead (24 h) wind-speed prediction, 2024-03 → 2025-12, "
                 "n = 15,912 h", y=1.02, fontsize=11.5)
    save(fig, "fig_4_02_day_ahead_ladder", "day-ahead predictor ladder")


# ---------------------------------------------------------------------------
# 4.3 Classical vs ML — eleven predictors
# ---------------------------------------------------------------------------

def fig_classical_vs_ml():
    b = read("bakeoff.csv")
    d = b[b.feature_set == "study"].copy()
    d["tier"] = d.model.map(LADDER_TIER)
    d = d.sort_values("mae_mean", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(d))
    bars = ax.barh(y, d.mae_mean, xerr=d.mae_sd, height=0.66,
                   color=[TIER_COLOUR[t] for t in d.tier], edgecolor="white",
                   error_kw=dict(ecolor="#555555", lw=1, capsize=3))
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[m] for m in d.model], fontsize=9)
    ax.set_xlabel("MAE (m s$^{-1}$) — lower is better; whiskers are sd across folds and seeds")
    ax.set_xlim(0, 2.05)
    # Labels clear the whisker cap, not just the bar end.
    for bar, v, sd in zip(bars, d.mae_mean, d.mae_sd):
        ax.text(v + sd + 0.035, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", ha="left", fontsize=8)

    best = d[d.model == "xgboost"].mae_mean.iloc[0]
    mos = d[d.model == "mos_multi"].mae_mean.iloc[0]
    ax.axvline(best, color=C_ACCENT, ls="--", lw=1.1)
    ax.annotate(f"21-parameter MOS is {100*(mos/best-1):.1f}% behind\n"
                f"gradient boosting (DM p < 0.0001)",
                xy=(mos + 0.13, list(d.model).index("mos_multi")),
                xytext=(1.30, 5.4), fontsize=8.5, color=C_ACCENT,
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2))

    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLOUR[t])
               for t in ("reference", "classical", "ml")]
    ax.legend(handles, ["Reference baseline", "Classical", "Machine learning"],
              loc="upper right", fontsize=9)
    ax.set_title("Eleven predictors on one protocol — 24 h horizon, "
                 "4 rolling-origin folds, 5 seeds")
    save(fig, "fig_4_03_classical_vs_ml", "classical vs ML ladder")


# ---------------------------------------------------------------------------
# 4.4 Where the skill enters
# ---------------------------------------------------------------------------

def fig_skill_decomposition():
    b = read("bakeoff.csv")
    d = b[b.feature_set == "study"].set_index("model").mae_mean
    rungs = ["persistence", "damped_persistence", "mos_single", "mos_multi", "xgboost"]
    vals = [d[m] for m in rungs]
    steps = [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]
    labels = ["Persistence\n→ damped", "Damped\n→ single MOS",
              "Single\n→ multi MOS", "Multi MOS\n→ gradient boosting"]
    kinds = ["reference", "classical", "classical", "ml"]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bottoms = vals[:-1]
    x = np.arange(len(steps))
    for xi, (b0, s, k) in enumerate(zip(bottoms, steps, kinds)):
        ax.bar(xi, s, bottom=b0 - s, width=0.6,
               color=TIER_COLOUR[k], edgecolor="white")
        # Offset left of centre so the connecting line does not strike through.
        ax.text(xi - 0.17, b0 - s / 2, f"−{s:.2f}", ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
    ax.plot(np.arange(len(vals)) - 0.5, vals, "o--", color="#333333",
            lw=1.1, ms=5, zorder=3)
    for xi, v in enumerate(vals):
        ax.text(xi - 0.5, v + 0.02, f"{v:.4f}", ha="center", fontsize=8.5)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("MAE (m s$^{-1}$)")
    ax.set_ylim(0.65, 1.45)
    ax.set_title("Where the skill enters — the two largest steps are both classical")
    ax.annotate("everything machine\nlearning adds", xy=(3.31, 0.822),
                xytext=(2.45, 1.13), fontsize=8.5, color=C_ACCENT, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2))
    save(fig, "fig_4_04_skill_decomposition", "skill decomposition")


# ---------------------------------------------------------------------------
# 4.5 Per-fold difficulty
# ---------------------------------------------------------------------------

def fig_per_fold():
    f = read("bakeoff_folds.csv")
    s = f[(f.feature_set == "study") & (f.seed == 42)]
    piv = s.pivot_table(index="fold", columns="model", values="mae")
    models = ["persistence", "damped_persistence", "mos_multi", "xgboost"]
    periods = ["Dec–Mar", "Mar–Jun", "Jun–Sep", "Sep–Dec"]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(piv))
    w = 0.2
    for i, m in enumerate(models):
        ax.bar(x + (i - 1.5) * w, piv[m], width=w, label=PRETTY[m],
               color=TIER_COLOUR[LADDER_TIER[m]],
               alpha=1.0 if m in ("persistence", "xgboost") else 0.75,
               edgecolor="white",
               hatch="" if i % 2 == 0 else "///")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n{p}" for f, p in zip(piv.index, periods)])
    ax.set_ylabel("MAE (m s$^{-1}$)")
    ax.legend(fontsize=8.5, ncol=2)
    lo, hi = piv.persistence.min(), piv.persistence.max()
    ax.set_title(f"Fold difficulty varies {100*(hi/lo-1):.0f}% across the year — "
                 "why a single split is not enough")
    ax.annotate(f"persistence {lo:.4f} → {hi:.4f}",
                xy=(1 - 1.5 * w, hi), xytext=(1.6, 1.62), fontsize=8.5,
                color=C_ACCENT,
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2))
    ax.set_ylim(0, 1.85)
    save(fig, "fig_4_05_per_fold", "per-fold difficulty")


# ---------------------------------------------------------------------------
# 4.6 Deep models — accuracy vs cost, and seed sensitivity
# ---------------------------------------------------------------------------

def fig_deep_models():
    d = read("bakeoff_deep.csv")
    g = d.groupby("model").agg(mae=("mae", "mean"), sd=("mae", "std"),
                               lo=("mae", "min"), hi=("mae", "max"),
                               mins=("minutes", "mean")).sort_values("mae")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for model, r in g.iterrows():
        colour = C_ML if model == "xgboost" else C_CLASSICAL
        ax1.errorbar(r.mins, r.mae, yerr=[[r.mae - r.lo], [r.hi - r.mae]],
                     fmt="o", ms=8, color=colour, ecolor="#666666",
                     capsize=4, lw=1.2)
        ax1.annotate(model, (r.mins, r.mae), textcoords="offset points",
                     xytext=(9, 4), fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlabel("Training time per fit (minutes, log scale)")
    ax1.set_ylabel("MAE (m s$^{-1}$)")
    ax1.set_title("(a) Accuracy against cost\n(whiskers: full range over 3 seeds)")
    ax1.annotate("~2,000× the cost,\n20% worse",
                 xy=(g.loc["tft", "mins"], g.loc["tft", "mae"]),
                 xytext=(4, 0.93), fontsize=8.5, color=C_ACCENT,
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2))

    order = g.index.tolist()
    y = np.arange(len(order))
    for yi, m in enumerate(order):
        sub = d[d.model == m]
        colour = C_ML if m == "xgboost" else C_CLASSICAL
        ax2.hlines(yi, sub.mae.min(), sub.mae.max(), color=colour, lw=3, alpha=0.5)
        ax2.plot(sub.mae, [yi] * len(sub), "o", color=colour, ms=6)
    ax2.set_yticks(y); ax2.set_yticklabels(order)
    ax2.set_xlabel("MAE (m s$^{-1}$) — one point per seed")
    ax2.set_title("(b) Seed sensitivity\nTFT spans 0.1090; XGBoost 0.0068")
    ax2.invert_yaxis()

    fig.suptitle("Deep architectures against gradient boosting — 24 h horizon, "
                 "single split, 3 seeds, study features", y=1.03, fontsize=11.5)
    save(fig, "fig_4_06_deep_models", "deep models: cost and seed spread")


# ---------------------------------------------------------------------------
# 4.7 Operational vs study
# ---------------------------------------------------------------------------

def fig_operational_vs_study():
    b = read("bakeoff.csv")
    piv = b.pivot(index="model", columns="feature_set", values="mae_mean")
    piv = piv.loc[["xgboost", "lightgbm", "ridge", "mos_multi", "kalman_mos",
                   "analog_ensemble", "mos_single"]]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(piv))
    w = 0.36
    ax.bar(x - w / 2, piv["study"], width=w, label="study (94 features)",
           color=C_CLASSICAL, edgecolor="white")
    ax.bar(x + w / 2, piv["operational"], width=w,
           label="operational (55 features, no reanalysis)",
           color=C_ML, edgecolor="white", hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[m] for m in piv.index], fontsize=8.5, rotation=12)
    ax.set_ylabel("MAE (m s$^{-1}$)")
    ax.set_ylim(0.7, 1.05)
    ax.legend(fontsize=9)

    diff = piv["operational"] - piv["study"]
    for xi, m in enumerate(piv.index):
        if abs(diff[m]) > 1e-6:
            ax.text(xi, max(piv.loc[m]) + 0.012, f"{diff[m]:+.4f}",
                    ha="center", fontsize=8, color=C_ACCENT, fontweight="bold")
    ax.set_title("Deployment rehearsal — removing all 39 latency-blocked features\n"
                 "makes the models marginally better, not worse")
    save(fig, "fig_4_07_operational_vs_study", "operational vs study")


# ---------------------------------------------------------------------------
# 4.8 Skill decay 24 h -> 168 h
# ---------------------------------------------------------------------------

def fig_skill_decay():
    sweep = {}
    for h, name in ((24, "bakeoff.csv"), (48, "bakeoff_h48.csv"), (72, "bakeoff_h72.csv")):
        d = read(name)
        sub = d[d.feature_set == "operational"].set_index("model")
        if sub.empty:
            sub = d.set_index("model")
        sweep[h] = (sub.loc["xgboost", "mae_mean"], sub.loc["climatology", "mae_mean"])

    w = read("weekly_leads.csv")
    wa = w[w.arm == "all"].sort_values("lead_day")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    hs = sorted(sweep)
    ax1.plot(hs, [1 - sweep[h][0] / sweep[h][1] for h in hs], "s-",
             color=C_ML, lw=1.8, ms=7, label="Rolling-origin sweep (4 folds)")
    ax1.plot(wa.horizon_h, wa.skill_vs_clim, "o--", color=C_CLASSICAL, lw=1.6,
             ms=6, label="Per-lead weekly models (single split)")
    ax1.set_xlabel("Forecast horizon (hours)")
    ax1.set_ylabel("Skill vs climatology")
    ax1.set_xticks([24, 48, 72, 96, 120, 144, 168])
    ax1.set_ylim(0, 0.45)
    ax1.legend(fontsize=8.5)
    ax1.set_title("(a) Skill against climatology decays but decelerates")
    ax1.annotate(f"7 days: {wa.skill_vs_clim.iloc[-1]:.3f}",
                 xy=(168, wa.skill_vs_clim.iloc[-1]), textcoords="offset points",
                 xytext=(-70, -22), fontsize=8.5, color=C_ACCENT,
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2))

    ax2.plot(wa.horizon_h, wa.climatology, "^-", color=C_REF, lw=1.6, ms=6,
             label="Climatology")
    ax2.plot(wa.horizon_h, wa.mos_multi, "s--", color=C_CLASSICAL, lw=1.6, ms=6,
             label="Multi-model MOS")
    ax2.plot(wa.horizon_h, wa.xgboost, "o-", color=C_ML, lw=1.8, ms=6,
             label="XGBoost")
    ax2.set_xlabel("Forecast horizon (hours)")
    ax2.set_ylabel("MAE (m s$^{-1}$)")
    ax2.set_xticks([24, 48, 72, 96, 120, 144, 168])
    ax2.legend(fontsize=8.5)
    ax2.set_title("(b) Per-lead error, 24 h → 168 h")
    ax2.annotate("ICON stops at d6:\nday 7 has 2 members",
                 xy=(168, wa.xgboost.iloc[-1]), xytext=(104, 0.72),
                 fontsize=8, color="#555555",
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1))

    fig.suptitle("Forecast skill from one day to one week", y=1.02, fontsize=11.5)
    save(fig, "fig_4_08_skill_decay", "skill decay 24-168 h")


# ---------------------------------------------------------------------------
# 4.9 Weekly energy aggregation
# ---------------------------------------------------------------------------

def fig_energy_aggregation():
    w = read("weekly_leads.csv")
    wa = w[w.arm == "all"].sort_values("lead_day")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(wa))
    ax.bar(x - 0.19, wa.hourly_nmae_pct, width=0.38, label="Hourly nMAE",
           color=C_CLASSICAL, edgecolor="white")
    ax.bar(x + 0.19, wa.weekly_nmae_of_rated, width=0.38,
           label="Weekly-energy nMAE (of rated)", color=C_ML,
           edgecolor="white", hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels([f"d{int(l)}\n{int(h)} h" for l, h in
                        zip(wa.lead_day, wa.horizon_h)], fontsize=8.5)
    ax.set_ylabel("Normalised MAE (% of rated power)")
    ax.legend(fontsize=9)
    for xi, (a, b_) in enumerate(zip(wa.hourly_nmae_pct, wa.weekly_nmae_of_rated)):
        ax.text(xi - 0.19, a + 0.15, f"{a:.2f}", ha="center", fontsize=7.5)
        ax.text(xi + 0.19, b_ + 0.15, f"{b_:.2f}", ha="center", fontsize=7.5)
    ratio = wa.hourly_nmae_pct.iloc[-1] / wa.weekly_nmae_of_rated.iloc[-1]
    ax.set_title("Hourly errors partially cancel when summed over a week "
                 f"(≈{ratio:.1f}× at 168 h)")
    ax.set_ylim(0, 12.5)
    save(fig, "fig_4_09_energy_aggregation", "weekly energy aggregation")


# ---------------------------------------------------------------------------
# 4.10 Quantile calibration
# ---------------------------------------------------------------------------

def fig_calibration():
    w = read("weekly_leads.csv")
    wa = w[w.arm == "all"].sort_values("lead_day")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    ax1.plot(wa.horizon_h, wa.coverage, "o-", color=C_ACCENT, lw=1.8, ms=7,
             label="Observed P10–P90 coverage")
    ax1.axhline(0.80, color=C_ML, ls="--", lw=1.5, label="Nominal 0.80")
    ax1.fill_between(wa.horizon_h, wa.coverage, 0.80, color=C_ACCENT, alpha=0.12)
    ax1.set_xlabel("Forecast horizon (hours)")
    ax1.set_ylabel("Coverage")
    ax1.set_ylim(0.55, 0.9)
    ax1.set_xticks([24, 48, 72, 96, 120, 144, 168])
    ax1.legend(fontsize=8.5, loc="lower right")
    ax1.set_title(f"(a) Intervals are too narrow at every lead\n"
                  f"({wa.coverage.min():.3f}–{wa.coverage.max():.3f} "
                  f"against 0.80)")

    ax2.plot(wa.horizon_h, wa.interval_width, "s-", color=C_ML, lw=1.8, ms=6)
    ax2.set_xlabel("Forecast horizon (hours)")
    ax2.set_ylabel("P10–P90 interval width (m s$^{-1}$)")
    ax2.set_xticks([24, 48, 72, 96, 120, 144, 168])
    ax2.set_title("(b) Width does grow with lead —\nthe spread signal is real")

    fig.suptitle("Probabilistic forecasts: uncalibrated, but not uninformative",
                 y=1.03, fontsize=11.5)
    save(fig, "fig_4_10_calibration", "quantile calibration")


# ---------------------------------------------------------------------------
# 4.11 Monthly exceedance product
# ---------------------------------------------------------------------------

def fig_monthly_exceedance():
    m = read("monthly_climatology.csv").sort_values("month")
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(12)
    w = 0.26
    ax.bar(x - w, m.p50_cf, width=w, label="P50", color=C_ML, edgecolor="white")
    ax.bar(x, m.p75_cf, width=w, label="P75", color=C_CLASSICAL, edgecolor="white")
    ax.bar(x + w, m.p90_cf, width=w, label="P90", color=C_REF,
           edgecolor="white", hatch="///")
    ax.set_xticks(x); ax.set_xticklabels(MONTHS)
    ax.set_ylabel("Capacity factor")
    ax.legend(title="Exceedance level", fontsize=9, title_fontsize=9)
    ratio = m.mean_cf.max() / m.mean_cf.min()
    ax.set_title(f"Monthly resource: exceedance capacity factor "
                 f"(August yields {ratio:.2f}× December)")
    ax.annotate("P90 = exceeded 90% of the time,\ni.e. the conservative figure",
                xy=(7 + w, m.p90_cf.iloc[7]), xytext=(8.4, 0.40),
                fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    save(fig, "fig_4_11_monthly_exceedance", "monthly exceedance product")


# ---------------------------------------------------------------------------
# 4.12 Diurnal heatmap
# ---------------------------------------------------------------------------

def fig_diurnal():
    d = read("monthly_diurnal.csv")
    d = d.set_index(d.columns[0])
    arr = d.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    im = ax.imshow(arr, aspect="auto", cmap="cividis", origin="upper")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_yticks(range(12)); ax.set_yticklabels(list(d.index))
    ax.set_xlabel("Hour of day (UTC = local time in Ghana)")
    ax.set_ylabel("Month")
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Mean capacity factor")

    # Mark the solar generation window against the wind trough. Both labels sit
    # *inside* the axes: outside they collide with the title and the tick row.
    for xpos in (8.5, 15.5):
        ax.axvline(xpos, color="white", ls="--", lw=1.4)
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82)
    ax.text(12, 0.42, "solar generation window", ha="center", fontsize=8.5,
            color="#222222", bbox=box)
    ax.text(19.4, 0.42, "wind peak 16:00–21:00", ha="center", fontsize=8.5,
            color="#222222", bbox=box)
    ax.text(10.6, 10.6, "wind trough", ha="center", fontsize=8.5,
            color="#222222", bbox=box)
    ax.grid(False)
    ax.set_title("Diurnal wind cycle by month — the wind trough sits inside the "
                 "solar peak\n(wind and solar are close to anti-correlated at "
                 "this site)")
    save(fig, "fig_4_12_diurnal_heatmap", "diurnal heatmap")


# ---------------------------------------------------------------------------
# 4.13 Annual energy variability
# ---------------------------------------------------------------------------

def fig_annual_variability():
    a = read("monthly_annual_energy.csv")
    full = a[a.complete].sort_values("year")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(full))
    ax.bar(x, full.energy_mwh, width=0.62, color=C_CLASSICAL, edgecolor="white")
    mean = full.energy_mwh.mean()
    ax.axhline(mean, color=C_ML, ls="--", lw=1.4,
               label=f"Mean {mean:,.0f} MWh")
    lo, hi = full.energy_mwh.min(), full.energy_mwh.max()
    ax.set_xticks(x); ax.set_xticklabels(full.year.astype(int), rotation=0)
    ax.set_ylabel("Annual energy, one E-101/3050 (MWh)")
    ax.set_ylim(0, hi * 1.22)
    ax.legend(fontsize=9)
    for xi, v in zip(x, full.energy_mwh):
        ax.text(xi, v + 90, f"{v:,.0f}", ha="center", fontsize=7.5)
    ax.annotate(f"best year is {100*(hi/lo-1):.0f}% above the worst",
                xy=(float(np.argmax(full.energy_mwh.to_numpy())), hi),
                xytext=(1.2, hi * 1.14), fontsize=9, color=C_ACCENT,
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2))
    ax.set_title(f"Interannual variability, {len(full)} complete years "
                 "(2010–2025)")
    save(fig, "fig_4_13_annual_variability", "annual energy variability")


# ---------------------------------------------------------------------------
# 4.14 Monthly outlook ceiling
# ---------------------------------------------------------------------------

def fig_outlook_ceiling():
    """⚠️ The ceiling curve is TRANSCRIBED from redo/results/10.

    Computing it needs the hourly capacity-factor series, which `monthly_series.csv`
    does not carry (it is one row per month). `mae_clim` and the sample size below
    *are* read from that file, so the caption's supporting figures are derived even
    though the curve itself is reproduced.
    """
    ms = read("monthly_series.csv")
    ms = ms.rename(columns={ms.columns[0]: "period"})
    mae_clim = float((ms["cf"] - ms["clim_loyo"]).abs().mean())

    # Source: redo/results/10_monthly_ceiling.md, "perfect first N days" table.
    days = [0, 3, 5, 7, 10, 14, 21, 30]
    skills = [0.0, 0.090, 0.097, 0.131, 0.215, 0.307, 0.515, 0.945]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(days, [s * 100 for s in skills], "o-", color=C_ML, lw=1.9, ms=7)
    ax.axvline(7, color=C_ACCENT, ls="--", lw=1.4)
    ax.plot([7], [13.1], "o", ms=13, mfc="none", mec=C_ACCENT, mew=2)
    ax.annotate("NWP reaches 7 days:\nceiling 13.1% even with\na perfect forecast",
                xy=(7.4, 13.6), xytext=(10.0, 30), fontsize=9, color=C_ACCENT,
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.3))
    ax.plot([7], [4.4], "s", ms=9, color="#333333")
    ax.annotate("measured: +4.4%\n(CI spans zero)", xy=(7.4, 4.2),
                xytext=(11.5, -6), fontsize=8.5, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1))
    ax.set_xlabel("Days of the month known perfectly")
    ax.set_ylabel("Skill vs leave-one-year-out climatology (%)")
    ax.set_xticks(days)
    ax.set_ylim(-12, 104)
    ax.set_title("Why a monthly forecast cannot work here — the first week is the "
                 "least valuable week to know\n"
                 f"(climatology MAE {mae_clim:.4f} CF, n = {len(ms)} months)")
    save(fig, "fig_4_14_outlook_ceiling", "monthly outlook ceiling")


# ---------------------------------------------------------------------------
# 4.15 Outlook ladder
# ---------------------------------------------------------------------------

def fig_outlook_ladder():
    b = read("outlook_block_summary.csv")
    b = b.rename(columns={b.columns[0]: "arm"}).set_index("arm")
    order = ["climatology_window", "ensemble_mean", "mos_single",
             "mos_multi", "xgboost"]
    d = b.loc[order]

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y = np.arange(len(d))
    skills = d.skill_vs_clim * 100
    colours = [C_REF if s < 0 else C_ML for s in skills]
    bars = ax.barh(y, skills, height=0.62, color=colours, edgecolor="white")
    ax.axvline(0, color="#333333", lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY[m] for m in d.index], fontsize=9)
    ax.set_xlabel("Skill vs archive climatology (%) — 7-day energy blocks, n = 114")
    ax.invert_yaxis()

    for yi, (s, p) in enumerate(zip(skills, d.dm_p)):
        sig = "n.s." if p > 0.05 else (f"p = {p:.1e}" if p < 1e-3 else f"p = {p:.3f}")
        ax.text(s + (2.5 if s >= 0 else -2.5), yi, f"{s:+.1f}%   ({sig})",
                va="center", ha="left" if s >= 0 else "right", fontsize=8.5)
    # Wide enough that neither the negative labels reach the tick column nor the
    # positive ones clip at the frame.
    ax.set_xlim(-92, 78)
    ax.set_title("Re-powered outlook ladder — the forecastable week works\n"
                 "(the raw ensemble mean loses: it is 98% bias, which a 168-hour "
                 "sum preserves)")
    save(fig, "fig_4_15_outlook_ladder", "outlook ladder")


# ---------------------------------------------------------------------------

FIGURES = [
    fig_ceiling, fig_day_ahead_ladder, fig_classical_vs_ml,
    fig_skill_decomposition, fig_per_fold, fig_deep_models,
    fig_operational_vs_study, fig_skill_decay, fig_energy_aggregation,
    fig_calibration, fig_monthly_exceedance, fig_diurnal,
    fig_annual_variability, fig_outlook_ceiling, fig_outlook_ladder,
]


def main() -> None:
    print(f"writing {len(FIGURES)} figures to {FIGDIR}/ ...")
    for fn in FIGURES:
        fn()
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
