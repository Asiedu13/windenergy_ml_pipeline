"""End-to-end CLI orchestrator.

    python -m src.pipeline --config config.yaml [--mode forecast|nowcast] [--compare]

Loads config → ingests weather → builds features → trains Stage 1 →
runs Stage 2 → evaluates three variants → writes report.

--compare trains BOTH the configured tree model and a deep LSTM, then puts a
two-row comparison table at the top of the report and uses the better
(lower test MAE) as the downstream two-stage forecaster.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_ingest import load_nwp_forecasts, load_weather
from .evaluate import (
    error_propagation_plot,
    evaluate_variants,
    summarize_variants,
    timeseries_plot,
)
from .features import build_feature_frame
from .report import write_report
from .stage1_windspeed import (
    Stage1Result,
    analyze_feature_influence,
    save_stage1,
    train_stage1,
)
from .stage2_powercurve import load_turbine, turbine_spec
from .utils import Config, ensure_dir, get_logger, load_config, set_seed

TREE_ALGOS = {"xgboost", "lightgbm"}
DEEP_ALGOS = {"lstm", "tft", "nbeats", "dlinear"}


def _train_stage1_by_algo(X, y, weather, config: Config, mode: str) -> Stage1Result:
    algo = config.model.stage1_algo.lower()
    if algo in TREE_ALGOS:
        return train_stage1(X, y, config, mode=mode)
    if algo in DEEP_ALGOS:
        from .stage1_deep import train_stage1_deep
        return train_stage1_deep(X, y, config, mode=mode, weather_df=weather)
    raise ValueError(f"Unknown stage1_algo: {algo!r}")


def run(config_path: str, mode: str = "forecast", compare: bool = False) -> Path:
    logger = get_logger("pipeline")
    config = load_config(config_path)
    set_seed(config.model.seed)
    outputs = ensure_dir(config.paths.outputs)
    ensure_dir(Path(outputs) / "models")

    logger.info("Loading weather for %s (%s)", config.site.name, config.data.source)
    weather = load_weather(config)

    # Archived NWP forecasts (empty frame unless config.nwp.enabled).
    nwp = load_nwp_forecasts(config)
    if not nwp.empty:
        logger.info("NWP forecasts: %d rows, %d cols from %s",
                    len(nwp), nwp.shape[1], ", ".join(config.nwp.models))

    logger.info("Building feature frame (mode=%s)", mode)
    X, y, _ = build_feature_frame(weather, config, mode=mode, nwp_df=nwp)

    logger.info("Feature-influence analysis (Pearson + SHAP)")
    ranking = analyze_feature_influence(X, y, outputs, seed=config.model.seed)

    comparison: list[tuple[str, Stage1Result]] = []

    if compare:
        # Compare-mode: iterate over config.model.compare_algos (default: primary
        # + LSTM sibling for backward compat).
        primary_algo = config.model.stage1_algo.lower()
        algos = list(config.model.compare_algos) if config.model.compare_algos else []
        if not algos:
            sibling = "lstm" if primary_algo in TREE_ALGOS else "xgboost"
            algos = [primary_algo, sibling]

        import copy
        for algo in algos:
            logger.info("Training Stage 1 (%s)", algo)
            sub_config = copy.deepcopy(config)
            sub_config.model.stage1_algo = algo
            try:
                res = _train_stage1_by_algo(X, y, weather, sub_config, mode)
                comparison.append((algo, res))
            except Exception as exc:
                logger.error("Model %s failed: %s (skipping)", algo, exc)

        if not comparison:
            raise RuntimeError("All models failed in --compare mode.")

        # Downstream variants use the model with lowest test MAE
        better_name, better_result = min(comparison, key=lambda t: t[1].metrics_test.mae)
        logger.info("Best model: %s (test MAE %.3f) — used for downstream two-stage",
                    better_name, better_result.metrics_test.mae)
        stage1 = better_result
        winning_algo = better_name
    else:
        # Non-compare mode: train only the primary
        logger.info("Training Stage 1 (%s)", config.model.stage1_algo)
        stage1 = _train_stage1_by_algo(X, y, weather, config, mode)
        comparison = [(config.model.stage1_algo, stage1)]
        winning_algo = config.model.stage1_algo

    stage1_path = save_stage1(stage1, Path(outputs) / "models" / "stage1.pkl", config)
    logger.info("Saved Stage 1 model → %s", stage1_path)

    logger.info("Loading turbine %s", config.turbine.name)
    turbine = load_turbine(config.turbine.name, config.site.hub_height_m)
    spec = turbine_spec(turbine, config.turbine.name)

    logger.info("Evaluating three variants")
    variants = evaluate_variants(weather, stage1, turbine, spec, config, nwp_df=nwp)
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

    report_path = write_report(
        config, spec, stage1, ranking, summary, figures, outputs,
        stage1_comparison=comparison,
        stage1_algo_label=winning_algo,
    )
    logger.info("Wrote report → %s", report_path)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghana wind power pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="forecast", choices=["forecast", "nowcast"])
    parser.add_argument("--compare", action="store_true",
                        help="Also train an LSTM sibling and report both.")
    args = parser.parse_args()
    run(args.config, mode=args.mode, compare=args.compare)


if __name__ == "__main__":
    main()
