"""End-to-end CLI orchestrator.

    python -m src.pipeline --config config.yaml

Loads config → ingests weather → builds features → trains Stage 1 →
runs Stage 2 → evaluates three variants → writes report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_ingest import load_weather
from .evaluate import (
    error_propagation_plot,
    evaluate_variants,
    summarize_variants,
    timeseries_plot,
)
from .features import build_feature_frame
from .report import write_report
from .stage1_windspeed import analyze_feature_influence, save_stage1, train_stage1
from .stage2_powercurve import load_turbine, turbine_spec
from .utils import ensure_dir, get_logger, load_config, set_seed


def run(config_path: str, mode: str = "forecast") -> Path:
    logger = get_logger("pipeline")
    config = load_config(config_path)
    set_seed(config.model.seed)
    outputs = ensure_dir(config.paths.outputs)
    ensure_dir(Path(outputs) / "models")

    logger.info("Loading weather for %s (%s)", config.site.name, config.data.source)
    weather = load_weather(config)

    logger.info("Building feature frame (mode=%s)", mode)
    X, y, _ = build_feature_frame(weather, config, mode=mode)

    logger.info("Feature-influence analysis (Pearson + SHAP)")
    ranking = analyze_feature_influence(X, y, outputs, seed=config.model.seed)

    logger.info("Training Stage 1 (%s)", config.model.stage1_algo)
    stage1 = train_stage1(X, y, config, mode=mode)
    stage1_path = save_stage1(stage1, Path(outputs) / "models" / "stage1.pkl", config)
    logger.info("Saved Stage 1 model → %s", stage1_path)

    logger.info("Loading turbine %s", config.turbine.name)
    turbine = load_turbine(config.turbine.name, config.site.hub_height_m)
    spec = turbine_spec(turbine, config.turbine.name)

    logger.info("Evaluating three variants")
    variants = evaluate_variants(weather, stage1, turbine, spec, config)
    summary = summarize_variants(variants, spec)

    figures: dict[str, Path] = {
        "Pearson correlation heatmap": Path(outputs) / "figures" / "pearson_heatmap.png",
        "SHAP summary": Path(outputs) / "figures" / "shap_summary.png",
    }

    ep_path = error_propagation_plot(
        stage1,
        variants["two_stage"],
        Path(outputs) / "figures" / "error_propagation.png",
    )
    figures["Error propagation (two-stage)"] = ep_path

    for name, v in variants.items():
        p = timeseries_plot(v, Path(outputs) / "figures" / f"timeseries_{name}.png")
        figures[f"Test-week power: {name}"] = p

    report_path = write_report(config, spec, stage1, ranking, summary, figures, outputs)
    logger.info("Wrote report → %s", report_path)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghana wind power pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="forecast", choices=["forecast", "nowcast"])
    args = parser.parse_args()
    run(args.config, mode=args.mode)


if __name__ == "__main__":
    main()
