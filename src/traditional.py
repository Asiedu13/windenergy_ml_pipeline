"""Classical / statistical forecasters, on the same interface as the ML models.

Why this module exists
----------------------
The project has always compared ML against *naive* persistence. That is the wrong
opponent. Evidence already in hand says the classical end is competitive:

- a two-parameter linear rescale of one NWP model scores 0.9743 m/s, beating the
  entire tuned 53-feature XGBoost built on reanalysis state (1.0513);
- a full linear model on state-only features came within 1.6% of tuned XGBoost.

So the honest comparison needs the classical ladder built properly. Every predictor
here exposes the same `fit_predict(ctx) -> np.ndarray` shape as the tree and deep
models, so `bakeoff.py` can treat them identically and no model gets an accidental
advantage from a different harness.

The alignment convention throughout
-----------------------------------
`X`/`y` are indexed by **feature time t**, and `y[t] = hub(t + horizon)`. So:

- `ctx.hub_train`/`ctx.hub_test` are the hub wind speed observed **at t** — the
  thing genuinely known when the forecast is issued.
- `ctx.y_*` are the values being predicted, valid at `t + horizon`.

True 24 h persistence is therefore `hub(t)`, not `hub(t-1)`. The project's existing
`PersistenceForecaster` uses the `_lag1` column and is really a *25 hour*
persistence — a 2.9% easier bar that was quoted as "24 h persistence" throughout
v1–v4 (docs/10 §4). `NaivePersistence` below fixes that by construction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .utils import get_logger

logger = get_logger(__name__)

# Spatial columns that are mathematically degenerate because the ERA5 request
# snaps to a 1 lat x 2 lon patch rather than the intended 3x3 (docs/03):
#   *_grad_ns  -> identically zero (north and south edge are the same row)
#   *_std      -> exactly |*_grad_ew| / 2, i.e. perfectly collinear
# Trees tolerate both; linear models do not, and neither carries information.
DEAD_COLUMN_SUFFIXES = ("_grad_ns", "_std")

CALENDAR_COLS = ("hour_sin", "hour_cos", "month_sin", "month_cos")

_LEAD_RE = re.compile(r"_d(\d+)$")


def is_dead_column(col: str) -> bool:
    """True for the degenerate spatial columns (never for rolling `rollstd*`)."""
    return col.endswith(DEAD_COLUMN_SUFFIXES)


def dead_columns(cols: Iterable[str], X: pd.DataFrame | None = None,
                 mode: str = "name") -> frozenset[str]:
    """The columns to withhold from every model, as a resolved set of names.

    Two modes, because the name-based rule stopped being true.

    ``name`` (default, and what every published figure was produced with)
        Suffix match on `DEAD_COLUMN_SUFFIXES`. Correct for the per-site ERA5
        request, whose 1 lat x 2 lon patch really does make `_grad_ns`
        identically zero and `_std` exactly `|_grad_ew| / 2`.

    ``value``
        Withhold a column only if it is degenerate **in the data at hand**:
        constant, or numerically equal to `|_grad_ew| / 2`. Measured on the real
        caches, the v6 regional 3x3 patch revives all ten — 5/5 `_grad_ns`
        non-zero and 0/5 `_std` still collinear, against 0/5 and 5/5 per-site —
        so `name` discards ten genuinely informative columns there. See
        docs/13 section 3.

    ⚠️ **`value` must be given training rows only.** Deciding which columns to
    keep by looking at a test fold would leak, and it would leak in the
    flattering direction. `run_phase_a` passes the earliest fold's training
    block, which is a prefix of every later fold's, so one decision is valid for
    all folds and the feature count stays constant across them — otherwise folds
    stop being comparable, which is the thing the protocol exists to protect.

    `name` is the default everywhere on purpose: flipping it re-bases published
    results, and that must be an explicit, recorded choice rather than a
    side effect of running on a different source.
    """
    cols = list(cols)
    if mode == "name":
        return frozenset(c for c in cols if is_dead_column(c))
    if mode != "value":
        raise ValueError(
            f"Unknown dead-column mode {mode!r}; expected 'name' or 'value'.")
    if X is None:
        raise ValueError("mode='value' needs training rows to judge deadness.")

    dead: set[str] = set()
    for col in cols:
        if col not in X.columns:
            continue
        s = pd.to_numeric(X[col], errors="coerce")
        if not np.isfinite(s.to_numpy(dtype=float)).any() or s.std() == 0:
            dead.add(col)
            continue
        if col.endswith("_std"):
            ew = col[: -len("_std")] + "_grad_ew"
            if ew in X.columns:
                other = pd.to_numeric(X[ew], errors="coerce").abs() / 2
                if (s - other).abs().max() < 1e-9:
                    dead.add(col)
    return frozenset(dead)


@dataclass
class FoldContext:
    """Everything a predictor may look at for one (fold, seed) combination.

    Passing a single object keeps every model on identical data — one of the
    comparability rules the original bakeoff broke.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    hub_train: pd.Series       # observed hub wind at feature time t
    hub_test: pd.Series
    nwp_cols: list[str]        # every nwp_* column present
    seed: int
    # Columns to withhold, already resolved by `dead_columns`. None means "decide
    # by name", which is what every published figure used.
    #
    # This is carried on the context rather than re-derived inside each model
    # because `is_dead_column` was being applied in THREE places — the feature
    # mask plus `ridge` and the tree models independently — and a mode that only
    # reached one of them would silently half-apply.
    dead_cols: frozenset[str] | None = None

    def live_columns(self) -> list[str]:
        """Training columns with the dead ones removed."""
        dead = (self.dead_cols if self.dead_cols is not None
                else {c for c in self.X_train.columns if is_dead_column(c)})
        return [c for c in self.X_train.columns if c not in dead]

    @property
    def primary_lead(self) -> int:
        """Shortest lead day still present — the freshest legitimate forecast.

        **Not hardcoded to `d1`.** `add_nwp_features` drops every lead with
        ``24*N < horizon``, because a shorter lead would have been issued after
        the forecast time. So at a 24 h horizon the shortest survivor is `d1`, at
        48 h it is `d2`, at 168 h it would be `d7` — and the freshest surviving
        lead is always the most informative one.

        This started as a hardcoded ``_d1`` and it silently broke the first 48 h
        run: every classical model fell back to calendar-only fits, and
        `mos_multi` scored *worse than persistence*. The sanity floor caught it.
        """
        leads = {
            int(m.group(1))
            for c in self.nwp_cols
            if (m := _LEAD_RE.search(c)) is not None
        }
        return min(leads) if leads else 0

    @property
    def nwp_primary_cols(self) -> list[str]:
        """Every forecast column at the primary lead."""
        d = self.primary_lead
        return [c for c in self.nwp_cols if c.endswith(f"_d{d}")] if d else []

    @property
    def nwp_primary_wind_cols(self) -> list[str]:
        """Per-model 100 m wind speed at the primary lead — the MOS predictors."""
        d = self.primary_lead
        if not d:
            return []
        return [c for c in self.nwp_cols if c.endswith(f"wind_speed_100m_d{d}")]


# ---------------------------------------------------------------------------
# Tier 0 — reference baselines
# ---------------------------------------------------------------------------

class NaivePersistence:
    """ŷ(t+h) = hub(t). The true horizon-h persistence, not the lag-1 variant."""

    name = "persistence"
    tier = "0 reference"

    def fit_predict(self, ctx: FoldContext) -> np.ndarray:
        return ctx.hub_test.to_numpy(dtype=float)


class DampedPersistence:
    """ŷ = a*hub(t) + b — persistence shrunk toward the mean by OLS.

    One parameter more than naive persistence and a much stronger opponent:
    on the 2020-2023 window this scored R2 0.267 against naive persistence's
    0.043. Naive persistence does not shrink toward the mean, which is why it
    flatters any model compared against it.
    """

    name = "damped_persistence"
    tier = "0 reference"

    def fit_predict(self, ctx: FoldContext) -> np.ndarray:
        a = ctx.hub_train.to_numpy(dtype=float).reshape(-1, 1)
        b = ctx.hub_test.to_numpy(dtype=float).reshape(-1, 1)
        model = LinearRegression().fit(a, ctx.y_train.to_numpy(dtype=float))
        return model.predict(b)


class Climatology:
    """Training mean of the target by (month, hour-of-day)."""

    name = "climatology"
    tier = "0 reference"

    def fit_predict(self, ctx: FoldContext) -> np.ndarray:
        from .stage1_windspeed import ClimatologyForecaster
        m = ClimatologyForecaster().fit(ctx.X_train, ctx.y_train)
        return m.predict(ctx.X_test)


# ---------------------------------------------------------------------------
# Tier 1 — classical NWP post-processing
# ---------------------------------------------------------------------------

class SingleModelMOS:
    """Model Output Statistics: OLS of observed wind on ONE model's forecast.

    Two parameters. This is what national weather services have done for decades,
    and it is the rung that shows how much of the skill is simply *having a
    forecast* rather than anything learned. The source model is chosen on the
    training fold only (highest |correlation| with the target), never on test.
    """

    name = "mos_single"
    tier = "1 classical"

    def __init__(self) -> None:
        self.chosen_: str | None = None

    def fit_predict(self, ctx: FoldContext) -> np.ndarray:
        cands = ctx.nwp_primary_wind_cols
        if not cands:
            raise ValueError("SingleModelMOS needs a *_wind_speed_100m_d1 column.")
        corrs = {c: abs(ctx.X_train[c].corr(ctx.y_train)) for c in cands}
        self.chosen_ = max(corrs, key=corrs.get)
        tr = ctx.X_train[[self.chosen_]].to_numpy(dtype=float)
        te = ctx.X_test[[self.chosen_]].to_numpy(dtype=float)
        model = LinearRegression().fit(tr, ctx.y_train.to_numpy(dtype=float))
        return model.predict(te)


class MultiModelMOS:
    """OLS on every day-1 forecast column plus calendar terms.

    **This is the cell nobody has ever measured.** If it lands near the tuned
    XGBoost score, then the project's ~30% headline gain is attributable to
    *information* rather than to machine learning — a stronger claim than the
    current one, not a weaker one. If it lands well short, the nonlinearity is
    doing real work and that can finally be said with evidence.
    """

    name = "mos_multi"
    tier = "1 classical"

    def fit_predict(self, ctx: FoldContext) -> np.ndarray:
        cols = ctx.nwp_primary_cols + [c for c in CALENDAR_COLS if c in ctx.X_train.columns]
        model = LinearRegression().fit(
            ctx.X_train[cols].to_numpy(dtype=float),
            ctx.y_train.to_numpy(dtype=float),
        )
        return model.predict(ctx.X_test[cols].to_numpy(dtype=float))


class KalmanMOS:
    """MOS with time-varying coefficients, updated by a Kalman filter.

    Why bother when `MultiModelMOS` exists: NWP bias drifts — with the season, and
    whenever the provider upgrades its model. Static regression coefficients fitted
    once on the training fold cannot track that; a random-walk state can. This is
    standard operational practice for wind power.

    **Observation availability is respected.** At issue time *t* the forecaster
    knows observations up to *t*, which means it knows `y[j] = hub(j + h)` only for
    `j <= t - h`. The filter is therefore updated with a deliberate `horizon`-step
    delay, so no test-period observation is used before it could have been known.
    Getting this wrong would make the filter a leakage vector rather than a model.
    """

    name = "kalman_mos"
    tier = "1 classical"

    def __init__(self, horizon: int = 24, q: float = 1e-5, r: float = 1.0) -> None:
        self.horizon = horizon
        self.q = q          # process noise: how fast coefficients may drift
        self.r = r          # observation noise

    def _design(self, X: pd.DataFrame, cols: list[str]) -> np.ndarray:
        block = X[cols].to_numpy(dtype=float)
        return np.column_stack([np.ones(len(block)), block])

    def fit_predict(self, ctx: FoldContext) -> np.ndarray:
        cols = ctx.nwp_primary_wind_cols
        A_tr = self._design(ctx.X_train, cols)
        A_te = self._design(ctx.X_test, cols)
        y_tr = ctx.y_train.to_numpy(dtype=float)
        y_te_len = len(A_te)

        # Warm start from the static OLS solution so the filter does not spend
        # the first weeks converging from zero.
        theta = np.linalg.lstsq(A_tr, y_tr, rcond=None)[0]
        P = np.eye(A_tr.shape[1]) * 0.1
        Q = np.eye(A_tr.shape[1]) * self.q

        def update(theta, P, x, y_obs):
            P = P + Q
            denom = float(x @ P @ x) + self.r
            K = (P @ x) / denom
            theta = theta + K * (y_obs - float(x @ theta))
            P = P - np.outer(K, x) @ P
            return theta, P

        # Run the filter over the training fold to settle the state.
        for i in range(len(A_tr)):
            theta, P = update(theta, P, A_tr[i], y_tr[i])

        # Test fold: predict first, then update with the observation that would
        # have arrived `horizon` steps later.
        preds = np.empty(y_te_len, dtype=float)
        y_te = ctx.y_test.to_numpy(dtype=float)
        for i in range(y_te_len):
            preds[i] = float(A_te[i] @ theta)
            j = i - self.horizon
            if j >= 0:
                theta, P = update(theta, P, A_te[j], y_te[j])
        return preds


class SarimaxNWP:
    """SARIMAX on the target with NWP forecasts as exogenous regressors.

    The classical time-series route. A pure AR model is capped by the target's
    24 h autocorrelation (0.571 -> R2 <= 0.326), so the exogenous block is what
    gives it any chance; the AR terms only mop up short-range structure.

    Forecasting is genuinely out-of-sample (`get_forecast` with test exog, no test
    observations fed back), so the AR contribution decays within a few steps and
    the model converges toward its exogenous regression. That is expected, and it
    is worth reporting rather than hiding — it is *why* ARIMA-family models add
    little at a 24 h horizon here.
    """

    name = "sarimax_nwp"
    tier = "1 classical"

    def __init__(self, order: tuple[int, int, int] = (2, 0, 0),
                 max_train_rows: int = 12000) -> None:
        self.order = order
        self.max_train_rows = max_train_rows
        self.truncated_ = False
        self.ar_roots_stable_: bool | None = None

    def fit_predict(self, ctx: FoldContext) -> np.ndarray:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        cols = ctx.nwp_primary_wind_cols + [
            c for c in CALENDAR_COLS if c in ctx.X_train.columns
        ]
        y_tr = ctx.y_train
        X_tr = ctx.X_train[cols]
        # Keep the fit affordable; state-space likelihood cost grows with length.
        if len(y_tr) > self.max_train_rows:
            self.truncated_ = True
            y_tr = y_tr.iloc[-self.max_train_rows:]
            X_tr = X_tr.iloc[-self.max_train_rows:]

        # `trend="c"` and `enforce_stationarity=True` are both load-bearing.
        # Without the constant the exogenous regression is forced through the
        # origin; without stationarity an AR root outside the unit circle makes a
        # 2000-step-ahead forecast diverge. The first run of this bakeoff had both
        # switched off and SARIMAX scored MAE 4.46 against persistence's 1.32 —
        # a numerically explosive forecast, not a model that "does badly".
        model = SARIMAX(
            y_tr.to_numpy(dtype=float),
            exog=X_tr.to_numpy(dtype=float),
            order=self.order,
            trend="c",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fitted = model.fit(disp=False)
        ar = np.asarray(getattr(fitted, "arroots", []), dtype=complex)
        self.ar_roots_stable_ = bool(ar.size == 0 or np.all(np.abs(ar) > 1.0))
        fc = fitted.get_forecast(
            steps=len(ctx.X_test),
            exog=ctx.X_test[cols].to_numpy(dtype=float),
        )
        return np.asarray(fc.predicted_mean, dtype=float)


class AnalogEnsemble:
    """Analog ensemble (AnEn): predict from historically similar forecast states.

    For each test hour, find the k training hours whose NWP forecast looks most
    like today's, and average what was actually observed on those occasions. It is
    cheap, needs no training, and yields a natural ensemble for uncertainty.

    **Data-limited here.** AnEn normally wants several years so that close analogues
    exist for unusual situations; the Open-Meteo archive starts 2024-03, leaving at
    most ~12.7k training hours. Report this as a data-constrained result, not as a
    fair test of the method.
    """

    name = "analog_ensemble"
    tier = "1 classical"

    def __init__(self, k: int = 25) -> None:
        self.k = k

    def fit_predict(self, ctx: FoldContext) -> np.ndarray:
        cols = ctx.nwp_primary_wind_cols + [
            c for c in CALENDAR_COLS if c in ctx.X_train.columns
        ]
        scaler = StandardScaler().fit(ctx.X_train[cols].to_numpy(dtype=float))
        tr = scaler.transform(ctx.X_train[cols].to_numpy(dtype=float))
        te = scaler.transform(ctx.X_test[cols].to_numpy(dtype=float))

        k = min(self.k, len(tr))
        nn = NearestNeighbors(n_neighbors=k).fit(tr)
        _, idx = nn.kneighbors(te)
        y_tr = ctx.y_train.to_numpy(dtype=float)
        return y_tr[idx].mean(axis=1)


# ---------------------------------------------------------------------------
# Tier 2 — linear ML on the full feature set
# ---------------------------------------------------------------------------

class RidgeAll:
    """Ridge on every (live) feature — the linear ceiling on this feature set.

    The alpha grid is resolved by RidgeCV's internal generalised cross-validation.
    That is regularisation selection, not the per-model hyperparameter search the
    protocol excludes: without it a linear model on ~100 partly-collinear columns
    is not a fair representative of its class.
    """

    name = "ridge"
    tier = "2 ML"

    def __init__(self) -> None:
        self.alpha_: float | None = None

    def fit_predict(self, ctx: FoldContext) -> np.ndarray:
        cols = ctx.live_columns()
        scaler = StandardScaler().fit(ctx.X_train[cols].to_numpy(dtype=float))
        tr = scaler.transform(ctx.X_train[cols].to_numpy(dtype=float))
        te = scaler.transform(ctx.X_test[cols].to_numpy(dtype=float))
        model = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(
            tr, ctx.y_train.to_numpy(dtype=float)
        )
        self.alpha_ = float(model.alpha_)
        return model.predict(te)


# ---------------------------------------------------------------------------
# Tier 2 — gradient boosting
# ---------------------------------------------------------------------------

class GradientBoosting:
    """XGBoost or LightGBM at the project's documented default settings.

    Deliberately untuned. On the authoritative window a 50-trial Optuna sweep and
    the untuned defaults both scored 0.6780 m/s — identical to four decimals — so
    equal documented settings matter more here than optimal ones.
    """

    tier = "2 ML"

    def __init__(self, algo: str = "xgboost") -> None:
        self.algo = algo
        self.name = algo

    def fit_predict(self, ctx: FoldContext) -> np.ndarray:
        from .stage1_windspeed import _make_regressor
        cols = ctx.live_columns()
        model = _make_regressor(self.algo, ctx.seed)
        model.fit(ctx.X_train[cols], ctx.y_train)
        return np.asarray(model.predict(ctx.X_test[cols]), dtype=float)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_registry() -> list:
    """Every non-deep predictor, cheapest first."""
    return [
        NaivePersistence(),
        DampedPersistence(),
        Climatology(),
        SingleModelMOS(),
        MultiModelMOS(),
        KalmanMOS(),
        AnalogEnsemble(),
        SarimaxNWP(),
        RidgeAll(),
        GradientBoosting("xgboost"),
        GradientBoosting("lightgbm"),
    ]


#: Predictors whose output does not depend on the seed — running them once per
#: fold is sufficient, and repeating them would fake a precision they do not have.
DETERMINISTIC = {
    "persistence", "damped_persistence", "climatology",
    "mos_single", "mos_multi", "kalman_mos", "analog_ensemble",
    "sarimax_nwp", "ridge",
}
