"""
01 – Time-series visualization of bus-stop boardings.

Produces four figures:
  A) Daily boarding profile for the top 5 stops (line chart)
  B) Heatmap: hour-of-day × day-of-week for a single stop
  C) Weekly seasonality comparison (box-plot by day-of-week)
  D) Rolling 7-day average vs raw signal (trend decomposition preview)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

from utils import load_timeseries

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs/viz_timeseries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ = "1h"          # temporal resolution for the plots
TOP_N_STOPS = 5      # how many stops to highlight
SINGLE_STOP = None   # set to a specific stop_id to override auto-selection

sns.set_theme(style="whitegrid", palette="tab10")

# ---------------------------------------------------------------------------
# Load & aggregate
# ---------------------------------------------------------------------------
print("Loading boarding data ...")
ts = load_timeseries(freq=FREQ, top_n=50)
top_stops = ts.sum().nlargest(TOP_N_STOPS).index.tolist()
SINGLE_STOP = SINGLE_STOP or top_stops[0]

print(f"  Time steps : {len(ts)}  ({ts.index[0].date()} → {ts.index[-1].date()})")
print(f"  Stop count : {ts.shape[1]}")
print(f"  Top 5 stops: {top_stops}")

# ---------------------------------------------------------------------------
# Figure A – Daily boarding profiles for top 5 stops
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))

for stop in top_stops:
    ax.plot(ts.index, ts[stop], lw=0.8, alpha=0.85, label=f"Stop {stop}")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
fig.autofmt_xdate()
ax.set_title("Hourly boardings – top 5 stops", fontsize=13)
ax.set_ylabel("Boardings per hour")
ax.legend(loc="upper right", ncol=TOP_N_STOPS, fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "A_daily_profiles.png", dpi=150)
plt.close(fig)
print("Saved A_daily_profiles.png")

# ---------------------------------------------------------------------------
# Figure B – Heatmap hour-of-day × day-of-week for a single stop
# ---------------------------------------------------------------------------
stop_series = ts[SINGLE_STOP].copy()
stop_df = stop_series.to_frame("boardings")
stop_df["hour"] = stop_df.index.hour
stop_df["dow"]  = stop_df.index.day_name()

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pivot = (
    stop_df.groupby(["hour", "dow"])["boardings"]
    .mean()
    .unstack("dow")
    .reindex(columns=dow_order)
)

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(
    pivot,
    ax=ax,
    cmap="YlOrRd",
    linewidths=0.3,
    cbar_kws={"label": "Mean boardings / hour"},
)
ax.set_title(f"Hourly boarding heatmap – stop {SINGLE_STOP}", fontsize=13)
ax.set_xlabel("Day of week")
ax.set_ylabel("Hour of day")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "B_heatmap_stop.png", dpi=150)
plt.close(fig)
print("Saved B_heatmap_stop.png")

# ---------------------------------------------------------------------------
# Figure C – Weekly seasonality (box-plot by day-of-week)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, TOP_N_STOPS, figsize=(16, 4), sharey=False)

for ax, stop in zip(axes, top_stops):
    s = ts[stop].to_frame("boardings")
    s["dow"] = pd.Categorical(
        s.index.day_name(), categories=dow_order, ordered=True
    )
    sns.boxplot(data=s, x="dow", y="boardings", ax=ax,
                palette="pastel", flierprops={"ms": 2, "alpha": 0.4})
    ax.set_title(f"Stop {stop}", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_ylabel("Boardings/h" if stop == top_stops[0] else "")

fig.suptitle("Weekly seasonality per stop", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "C_weekly_seasonality.png", dpi=150)
plt.close(fig)
print("Saved C_weekly_seasonality.png")

# ---------------------------------------------------------------------------
# Figure D – Raw signal + rolling D-day trend for a single stop
# ---------------------------------------------------------------------------
s = ts[SINGLE_STOP]
D = 1
roll = s.rolling(window=24 * D, center=True, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(s.index, s, alpha=0.25, color="steelblue", label="Hourly")
ax.plot(roll.index, roll, color="navy", lw=1.5, label="7-day rolling mean")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
fig.autofmt_xdate()
ax.set_title(f"Boarding trend – stop {SINGLE_STOP}", fontsize=13)
ax.set_ylabel("Boardings per hour")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "D_moving_avg.png", dpi=150)
plt.close(fig)
print("Saved D_moving_avg.png")

print(f"\nAll figures saved to {OUTPUT_DIR.resolve()}")
