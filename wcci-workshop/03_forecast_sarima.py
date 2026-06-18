"""
03 – SARIMA forecast for a single bus stop.

Pipeline:
  1. Load & aggregate boarding data (hourly)
  2. Select the busiest stop
  3. Stationarity tests (ADF, KPSS)
  4. ACF / PACF plots
  5. Auto-select SARIMA order via pmdarima.auto_arima
  6. Fit, forecast, evaluate (MAE, RMSE, MAPE)
  7. Plot actual vs. forecast with confidence interval
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import load_timeseries

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs/sarima")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hourly data with daily seasonality M=24. Weekly M=168 is intractable for SARIMA.
FREQ         = "1h"       # hourly boardings per stop
SEASON_M     = 24         # daily seasonality
HORIZON      = 24         # test horizon — 1 day ahead (same as LSTM/GRU/Transformer/Chronos)
STOP_ID      = None       # None = busiest stop
START_DATE   = "2024-04-01"
PERIODS      = 60         # calendar days to load (2024-04-01 → 2024-05-30)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data ...")
ts_all = load_timeseries(start_date=START_DATE, n_days=PERIODS, freq=FREQ, top_n=20)

if STOP_ID is None:
    STOP_ID = ts_all.sum().idxmax()
print(f"Selected stop: {STOP_ID}")

series = ts_all[STOP_ID].asfreq(FREQ).fillna(0).astype(float)
print(f"Series length: {len(series)} hours  ({series.index[0].date()} → {series.index[-1].date()})")

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
test_steps = HORIZON        # 24 h — same test window as all other models
min_train  = SEASON_M * 6  # at least 6 seasonal cycles
if len(series) < test_steps + min_train:
    raise ValueError(
        f"Not enough data: {len(series)} hours but need "
        f"{test_steps + min_train}. Increase PERIODS (currently {PERIODS})."
    )
train = series.iloc[:-test_steps]
test  = series.iloc[-test_steps:]
print(f"Train: {len(train)} hours  |  Test: {len(test)} hours")

# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------
from statsmodels.tsa.stattools import adfuller, kpss

adf_stat, adf_p, *_ = adfuller(train.dropna())
print(f"\nADF test  – stat={adf_stat:.4f}, p={adf_p:.4f}  "
      f"({'stationary' if adf_p < 0.05 else 'non-stationary'})")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kpss_stat, kpss_p, *_ = kpss(train.dropna(), regression="c", nlags="auto")
print(f"KPSS test – stat={kpss_stat:.4f}, p={kpss_p:.4f}  "
      f"({'stationary' if kpss_p > 0.05 else 'non-stationary'})")

# ---------------------------------------------------------------------------
# ACF / PACF plot
# ---------------------------------------------------------------------------
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

max_lags = min(72, len(train) // 2 - 1)  # show 3 days of hourly lags
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
plot_acf( train, lags=max_lags, ax=axes[0], title=f"ACF – stop {STOP_ID} (hourly boardings)")
plot_pacf(train, lags=max_lags, ax=axes[1], title="PACF", method="ywm")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "acf_pacf.png", dpi=150)
plt.close(fig)
print("\nSaved acf_pacf.png")

# ---------------------------------------------------------------------------
# Auto-select SARIMA parameters
# ---------------------------------------------------------------------------
try:
    import pmdarima as pm
    print("\nRunning auto_arima (this may take a few minutes) ...")
    auto = pm.auto_arima(
        train,
        seasonal=True,
        m=SEASON_M,          # M=24: daily seasonality on hourly data
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aic",
        max_p=2, max_q=2, max_P=1, max_Q=1,
        d=None, D=1,
        trace=True,
    )
    order         = auto.order
    seasonal_order = auto.seasonal_order
    print(f"\nBest model: SARIMA{order}x{seasonal_order}")

except ImportError:
    # Fallback: reasonable defaults for hourly urban transit
    print(f"\npmdarima not installed – using default SARIMA(1,0,1)(1,1,1)[{SEASON_M}]")
    order          = (1, 0, 1)
    seasonal_order = (1, 1, 1, SEASON_M)

# ---------------------------------------------------------------------------
# Fit final model
# ---------------------------------------------------------------------------
from statsmodels.tsa.statespace.sarimax import SARIMAX

print(f"\nFitting SARIMA{order}x{seasonal_order} ...")
model = SARIMAX(
    train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fit = model.fit(disp=False)
print(fit.summary())

# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
forecast_res = fit.get_forecast(steps=len(test))
fc_mean = forecast_res.predicted_mean
fc_ci   = forecast_res.conf_int(alpha=0.05)

fc_mean.index = test.index
fc_ci.index   = test.index

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mae  = mean_absolute_error(test, fc_mean)
rmse = np.sqrt(mean_squared_error(test, fc_mean))
mape_val = mape(test.values, fc_mean.values)

print(f"\n{'='*40}")
print(f"  MAE  : {mae:.3f}")
print(f"  RMSE : {rmse:.3f}")
print(f"  MAPE : {mape_val:.2f}%")
print(f"{'='*40}")

# ---------------------------------------------------------------------------
# Plot actual vs. forecast
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))

# Show last 7 days of training for context
train_tail = train.iloc[-HORIZON * 7 :]
ax.plot(train_tail.index, train_tail.values, color="steelblue", lw=0.9, label="Train (tail)")
ax.plot(test.index, test.values, color="black", lw=1.2, label="Actual")
ax.plot(fc_mean.index, fc_mean.values, color="crimson", lw=1.5, linestyle="--", label="SARIMA forecast")
ax.fill_between(
    fc_ci.index,
    fc_ci.iloc[:, 0],
    fc_ci.iloc[:, 1],
    color="crimson", alpha=0.15, label="95% CI",
)
ax.set_title(
    f"SARIMA{order}x{seasonal_order} | Stop {STOP_ID} | "
    f"MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape_val:.1f}%",
    fontsize=11,
)
ax.set_ylabel("Boardings / hour")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "sarima_forecast.png", dpi=150)
plt.close(fig)
print(f"\nSaved sarima_forecast.png  →  {OUTPUT_DIR.resolve()}")
