"""Assemble the auto-generated markdown report."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from .stage1_windspeed import Stage1Result
from .stage2_powercurve import TurbineSpec
from .utils import Config, ensure_dir


def _git_sha(cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _fmt_metrics_row(row: dict[str, float]) -> str:
    return (
        f"| {row['variant']} | {row['MAE']:.1f} W | {row['RMSE']:.1f} W | "
        f"{row['MAPE']:.2f}% | {row['R2']:.3f} | {row['nMAE_%']:.2f}% | "
        f"{row['nRMSE_%']:.2f}% |"
    )


def write_report(
    config: Config,
    spec: TurbineSpec,
    stage1: Stage1Result,
    feature_ranking: pd.DataFrame,
    variants_summary: dict[str, Any],
    figure_paths: dict[str, Path],
    output_dir: str | Path,
    stage1_comparison: list[tuple[str, Stage1Result]] | None = None,
    stage1_algo_label: str | None = None,
) -> Path:
    reports = ensure_dir(Path(output_dir) / "reports")
    out_path = reports / "report.md"

    sha = _git_sha(Path(config.paths.outputs).resolve().parent)

    lines: list[str] = []
    lines.append(f"# Wind power forecasting report — {config.site.name}")
    lines.append("")
    lines.append(
        "> Reference \"truth\" for power metrics is physics-derived (observed "
        "hub wind speed → manufacturer power curve, density-corrected). Ghana "
        "has no operating utility wind farm publishing SCADA; this validates "
        "the *forecasting* step, not absolute physical yield. GMet station "
        "records or a future met-mast are the path to true validation."
    )
    lines.append("")

    lines.append("## Site")
    lines.append(f"- **Name:** {config.site.name}")
    lines.append(f"- **Coordinates:** {config.site.latitude:.4f}, {config.site.longitude:.4f}")
    lines.append(f"- **Hub height:** {config.site.hub_height_m:.0f} m")
    lines.append(f"- **Data source:** {config.data.source} ({config.data.start} → {config.data.end})")
    lines.append("")

    lines.append("## Turbine")
    lines.append(f"- **Model:** {spec.name}")
    lines.append(f"- **Rotor diameter:** {spec.rotor_diameter_m:.1f} m")
    lines.append(f"- **Nominal power:** {spec.nominal_power_w / 1e6:.2f} MW")
    lines.append(f"- **Cut-in / rated / cut-out:** {spec.cut_in_ms:.1f} / {spec.rated_ms:.1f} / {spec.cut_out_ms:.1f} m/s")
    lines.append("")

    # Optional head-to-head model comparison (from --compare)
    if stage1_comparison and len(stage1_comparison) > 1:
        lines.append("## Stage 1 — model comparison (head-to-head)")
        lines.append("")
        lines.append("| Rank | Model | Test MAE (m/s) | Test RMSE | Test R² | Beats persistence? |")
        lines.append("|---:|:---|---:|---:|---:|:---:|")
        pers_mae = stage1_comparison[0][1].metrics_persistence.mae
        # Sort by ascending test MAE and bold the winner
        ranked = sorted(stage1_comparison, key=lambda t: t[1].metrics_test.mae)
        best_name = ranked[0][0]
        for rank, (name, res) in enumerate(ranked, start=1):
            beat = "yes" if res.metrics_test.mae < pers_mae else "**NO**"
            label = f"**{name}**" if name == best_name else name
            lines.append(
                f"| {rank} | {label} | {res.metrics_test.mae:.3f} | {res.metrics_test.rmse:.3f} | "
                f"{res.metrics_test.r2:.3f} | {beat} |"
            )
        lines.append(f"| — | Persistence baseline | {pers_mae:.3f} | "
                     f"{stage1_comparison[0][1].metrics_persistence.rmse:.3f} | "
                     f"{stage1_comparison[0][1].metrics_persistence.r2:.3f} | — |")
        lines.append("")
        lines.append(f"**Downstream two-stage variants use: {best_name}**")
        lines.append("")

    # In --compare mode the reported Stage 1 is the *winner*, which need not be
    # the algo named in the config.
    algo_label = stage1_algo_label or config.model.stage1_algo

    lines.append("## Stage 1 — hub wind speed")
    lines.append(f"- **Mode:** {stage1.mode} (horizon {config.forecast.horizon_hours} h)")
    lines.append(f"- **Algorithm:** {algo_label}")
    lines.append(f"- **Beats persistence on test?** {'yes' if stage1.beats_persistence else '**NO — flag**'}")
    lines.append("")
    lines.append("| Model | MAE (m/s) | RMSE (m/s) | MAPE (%) | R² |")
    lines.append("|:---|---:|---:|---:|---:|")
    for label, m in [
        ("Persistence", stage1.metrics_persistence),
        ("Climatology", stage1.metrics_climatology),
        (f"{algo_label} (train)", stage1.metrics_train),
        (f"{algo_label} (test)", stage1.metrics_test),
    ]:
        lines.append(f"| {label} | {m.mae:.3f} | {m.rmse:.3f} | {m.mape:.2f} | {m.r2:.3f} |")
    lines.append("")

    lines.append("### Top 15 features (by mean|SHAP|)")
    lines.append("")
    lines.append("| Rank | Feature | Pearson r | mean\\|SHAP\\| |")
    lines.append("|---:|:---|---:|---:|")
    for i, row in feature_ranking.head(15).iterrows():
        lines.append(
            f"| {i + 1} | `{row['feature']}` | {row['pearson_r']:+.3f} | {row['mean_abs_shap']:.4f} |"
        )
    lines.append("")

    lines.append("## Stage 1+2 comparison — three variants")
    lines.append(
        f"Rated power for normalization: **{variants_summary['rated_w'] / 1e6:.2f} MW**. "
        "Literature reference (Energies 2021 two-stage): ~8.8 % nMAE / ~13 % nRMSE."
    )
    lines.append("")
    lines.append("| Variant | MAE | RMSE | MAPE | R² | nMAE | nRMSE |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|")
    for row in variants_summary["rows"]:
        lines.append(_fmt_metrics_row(row))
    lines.append("")

    lines.append("## Figures")
    for label, p in figure_paths.items():
        rel = Path(p).as_posix()
        lines.append(f"- **{label}** — `{rel}`")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append(f"- seed: `{config.model.seed}`")
    lines.append(f"- config hash: `{config.hash()}`")
    lines.append(f"- git SHA: `{sha}`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
