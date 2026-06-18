"""
SUNT Dataset loader — wraps the official suntdataset package.

API:
    from suntdataset import SUNTLoader, SUNTVisualizer
    loader.load_batch(dataset_type, start_date, periods, freq, day_type)

Dataset types available via load_batch():
    'boarding'       – AFC + LTI + AVL + GTFS integrated boarding events
    'alighting'      – estimated alighting events
    'od'             – Origin-Destination (loading per stop/trip)
    'gtfs-stops'     – bus stop metadata
    'gtfs-trips'     – trip definitions
    'gtfs-stop-times'– stop-time sequences per trip
    'gtfs-routes'    – route/line metadata
    'gtfs-shapes'    – geographic shapes
    'gtfs-agency'    – agency info
    'afc'            – raw fare-collection events  (from Mendeley)
    'avl-lines'      – static route/stop sequences (from Mendeley)
    'avl-vehicles'   – timestamped GPS vehicle positions (from Mendeley)
    'lti'            – trip start/end info          (from Mendeley)

Paper: https://www.nature.com/articles/s41597-025-05674-6
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import requests
from suntdataset import SUNTLoader, SUNTVisualizer
from tqdm import tqdm

_loader = SUNTLoader()

# OD data uses "n-boardings" / "n-alighting" (dashes, not underscores)
_COLUMN_ALIASES = {
    "n-boardings": "n_boardings",
    "n-alighting": "n_alighting",
}

# HuggingFace folder name for each dataset type
_HF_FOLDER = {
    "boarding": "Boarding",
    "od":       "OD",
    "alighting":"Alighting",
}

_HF_API     = "https://huggingface.co/api/datasets/labiaufba/PublicTransportationSunt/tree/main/{folder}"
_HF_RESOLVE = "https://huggingface.co/datasets/labiaufba/PublicTransportationSunt/resolve/main/{folder}/{fname}"

# Local cache directory for HuggingFace downloads.
# Override before importing if a different path is needed:
#   import utils.data_loader as dl; dl.CACHE_DIR = Path("/tmp/sunt")
CACHE_DIR: Path = Path.home() / ".cache" / "sunt_dataset"


@lru_cache(maxsize=8)
def get_available_dates(dataset_type: str) -> set[str]:
    """
    Query HuggingFace to find which dates actually exist for a given dataset type.
    Returns a set of date strings like {'2024-03-01', '2024-04-01', ...}.
    Falls back to an empty set (disabling the filter) if the request fails.
    """
    folder = _HF_FOLDER.get(dataset_type)
    if folder is None:
        return set()
    try:
        r = requests.get(_HF_API.format(folder=folder), timeout=10)
        r.raise_for_status()
        dates = set()
        for entry in r.json():
            fname = entry["path"].split("/")[-1]          # e.g. "boarding-2024-04-01.parquet"
            date_part = fname.split("-", 1)[-1]           # "2024-04-01.parquet"
            date_str  = date_part.replace(".parquet", "") # "2024-04-01"
            dates.add(date_str)
        return dates
    except Exception:
        return set()


def _load_one_hgface(dataset_type: str, date_str: str) -> pd.DataFrame:
    """Return a single day's parquet, reading from local cache or downloading and caching it."""
    folder = _HF_FOLDER.get(dataset_type)
    if folder is None:
        raise ValueError(f"HuggingFace loading not supported for dataset_type='{dataset_type}'")
    fname = f"{dataset_type}-{date_str}.parquet"
    cache_path = CACHE_DIR / folder / fname
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    url = _HF_RESOLVE.format(folder=folder, fname=fname)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(r.content)
    return pd.read_parquet(cache_path)


def _load_available(
    dataset_type: str,
    start_date: str,
    n_days: int,
    day_type: str,
    source: str = "pip",
) -> pd.DataFrame:
    """
    Load only the dates that actually exist in the HuggingFace repo,
    avoiding 404 errors for missing days.
    """
    available = get_available_dates(dataset_type)

    # Generate the requested date range
    all_dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    # Filter by day_type
    dow = all_dates.dayofweek
    if day_type == "workdays":
        all_dates = all_dates[dow < 5]
    elif day_type == "saturdays":
        all_dates = all_dates[dow == 5]
    elif day_type == "sundays":
        all_dates = all_dates[dow == 6]

    # Keep only dates present in the repo (skip 404s silently)
    if available:
        all_dates = [d for d in all_dates if d.strftime("%Y-%m-%d") in available]

    if not all_dates:
        raise ValueError(
            f"No available {dataset_type} data found between {start_date} "
            f"and {n_days} days later. Check get_available_dates('{dataset_type}')."
        )

    frames = []
    for dt in tqdm(all_dates, desc=f"Loading {dataset_type} [{source}]", unit="day"):
        date_str = dt.strftime("%Y-%m-%d")
        try:
            if source == "hgface":
                df = _load_one_hgface(dataset_type, date_str)
            else:
                df = _loader.load_batch(dataset_type, start_date=date_str, periods=1)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            tqdm.write(f"  Warning: skipped {date_str} ({e})")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _rename_dash_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=_COLUMN_ALIASES)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_boarding(
    start_date: str = "2024-04-01",
    n_days: int = 30,
    day_type: str = "all",
    source: str = "pip",
) -> pd.DataFrame:
    """
    Load the processed boarding table, skipping dates not in the dataset.

    Columns: tripuserid, type_bus, user_type, register_time,
             route_short_name, vehicle, stop_id, classification, date_ref

    Parameters
    ----------
    start_date : first date to load (YYYY-MM-DD)
    n_days     : number of calendar days to span
    day_type   : 'all' | 'workdays' | 'saturdays' | 'sundays'
    source     : 'pip' (suntdataset package) | 'hgface' (HuggingFace direct)
    """
    df = _load_available("boarding", start_date, n_days, day_type, source=source)
    if "register_time" in df.columns:
        df["register_time"] = pd.to_datetime(df["register_time"])
    return df


def load_alighting(
    start_date: str = "2024-04-01",
    n_days: int = 30,
    day_type: str = "all",
    source: str = "pip",
) -> pd.DataFrame:
    """
    Load the processed alighting table, skipping dates not in the dataset.

    Columns: tripuserid, stop_time_ali, stop_id_ali,
             walk_dis, trip_dis, target_alighting, date_ref

    Parameters
    ----------
    source : 'pip' (suntdataset package) | 'hgface' (HuggingFace direct)
    """
    df = _load_available("alighting", start_date, n_days, day_type, source=source)
    if "stop_time_ali" in df.columns:
        df["stop_time_ali"] = pd.to_datetime(df["stop_time_ali"])
    return df


def load_od(
    start_date: str = "2024-04-01",
    n_days: int = 7,
    day_type: str = "all",
    source: str = "pip",
) -> pd.DataFrame:
    """
    Load the Origin-Destination table, skipping dates not in the dataset.

    Columns: stop_id, trip_id, pt_sequence, stop_time, n_boardings,
             n_alighting, loading, balance, route_short_name,
             direction_id, vehicle, date_ref
    (n-boardings / n-alighting are renamed to snake_case on load)

    Parameters
    ----------
    source : 'pip' (suntdataset package) | 'hgface' (HuggingFace direct)
    """
    df = _load_available("od", start_date, n_days, day_type, source=source)
    df = _rename_dash_cols(df)
    if "stop_time" in df.columns:
        df["stop_time"] = pd.to_datetime(df["stop_time"])
    return df


def load_gtfs(table: str = "stops") -> pd.DataFrame:
    """
    Load one GTFS table. table ∈ {stops, trips, stop-times, routes, shapes, agency}
    """
    key = f"gtfs-{table}"
    return _loader.load_batch(key, periods=1)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def load_timeseries(
    start_date: str = "2024-04-01",
    n_days: int = 30,
    freq: str = "1h",
    top_n: int = 50,
    day_type: str = "all",
    source: str = "pip",
) -> pd.DataFrame:
    """
    Memory-efficient alternative to load_boarding() + create_stop_timeseries().

    Loads boarding data one day at a time, aggregates immediately to
    (time × stop) counts, then discards the raw rows. Peak memory is
    O(1 day of raw data) instead of O(n_days of raw data).

    Parameters
    ----------
    source : 'pip' (suntdataset package) | 'hgface' (HuggingFace direct)

    Returns
    -------
    DataFrame, shape (T, N), index = DatetimeIndex, columns = stop_id
    """
    available = get_available_dates("boarding")
    all_dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    dow = all_dates.dayofweek
    if day_type == "workdays":
        all_dates = all_dates[dow < 5]
    elif day_type == "saturdays":
        all_dates = all_dates[dow == 5]
    elif day_type == "sundays":
        all_dates = all_dates[dow == 6]

    if available:
        all_dates = [d for d in all_dates if d.strftime("%Y-%m-%d") in available]

    if not all_dates:
        raise ValueError(f"No available boarding data found starting {start_date}.")

    daily_aggs = []
    for dt in tqdm(all_dates, desc=f"Loading timeseries [{source}]", unit="day"):
        date_str = dt.strftime("%Y-%m-%d")
        try:
            if source == "hgface":
                df = _load_one_hgface("boarding", date_str)
            else:
                df = _loader.load_batch("boarding", start_date=date_str, periods=1)
            if df.empty:
                continue
            df["register_time"] = pd.to_datetime(df["register_time"])
            df = df[df["stop_id"].notna()]
            df = df[df["stop_id"].astype(str) != "nan"]

            # Clip to the calendar day to prevent out-of-range timestamps
            # from inflating the time axis (trips crossing midnight, bad GPS, etc.)
            day_start = pd.Timestamp(date_str)
            day_end   = day_start + pd.Timedelta(days=1)
            df = df[(df["register_time"] >= day_start) & (df["register_time"] < day_end)]
            if df.empty:
                continue

            agg = (
                df.set_index("register_time")
                .groupby([pd.Grouper(freq=freq), "stop_id"])
                .size()
                .unstack(fill_value=0)
            )
            agg = agg.loc[:, ~agg.columns.astype(str).isin(["nan", "None", ""])]
            daily_aggs.append(agg)
            del df  # free raw data immediately
        except Exception as e:
            tqdm.write(f"  Warning: skipped {date_str} ({e})")

    if not daily_aggs:
        raise RuntimeError("No data could be loaded.")

    ts = pd.concat(daily_aggs).fillna(0).sort_index()

    # Collapse duplicate timestamps (trips crossing midnight appear in two files)
    ts = ts.groupby(ts.index).sum()

    # Keep only top_n stops by total activity
    top_stops = ts.sum().nlargest(top_n).index
    ts = ts[top_stops]

    # Complete gap-free index
    full_index = pd.date_range(ts.index.min(), ts.index.max(), freq=freq)
    return ts.reindex(full_index, fill_value=0)


def create_stop_timeseries(
    boarding: pd.DataFrame,
    freq: str = "15min",
    top_n: int = 50,
    fillna: float = 0.0,
) -> pd.DataFrame:
    """
    Aggregate boarding events into a (time × stop) matrix.

    Parameters
    ----------
    boarding : DataFrame with columns register_time, stop_id
    freq     : pandas offset alias, e.g. '15min', '1h'
    top_n    : keep only the N stops with highest total boardings
    fillna   : value used to fill missing intervals

    Returns
    -------
    DataFrame, shape (T, N), index = DatetimeIndex (complete, no gaps),
    columns = stop_id
    """
    df = boarding.copy()
    # Drop rows with null or 'nan' stop_id
    df = df[df["stop_id"].notna()]
    df = df[df["stop_id"].astype(str) != "nan"]

    ts = (
        df.set_index("register_time")
        .groupby([pd.Grouper(freq=freq), "stop_id"])
        .size()
        .unstack(fill_value=fillna)
    )

    # Reindex to a complete, gap-free DatetimeIndex
    full_index = pd.date_range(ts.index.min(), ts.index.max(), freq=freq)
    ts = ts.reindex(full_index, fill_value=fillna)

    # Drop the 'nan' column if it survived
    ts = ts.loc[:, ~ts.columns.astype(str).isin(["nan", "None", ""])]

    top_stops = ts.sum().nlargest(top_n).index
    return ts[top_stops].sort_index()


def build_graph_from_od(
    od: pd.DataFrame,
    origin_col: str = "stop_id",
    trip_col: str = "trip_id",
    sequence_col: str = "pt_sequence",
    time_col: str = "stop_time",
    weight_col: str = "n-boardings",
) -> nx.DiGraph:
    """
    Build a directed NetworkX graph from OD data using SUNTVisualizer.

    Nodes  = unique stop_id values
    Edges  = consecutive stops within the same trip,
             weighted by total passenger boardings.

    Uses the official SUNTVisualizer.build_od_graph() under the hood.
    Falls back to raw column name 'n-boardings' because the visualizer
    expects the original (pre-rename) column.
    """
    od_raw = od.rename(columns={v: k for k, v in _COLUMN_ALIASES.items()})

    viz = SUNTVisualizer(od_raw)
    return viz.build_od_graph(
        origin_col=origin_col,
        trip_col=trip_col,
        sequence_col=sequence_col,
        time_col=time_col,
        weight_col=weight_col,
    )


def prepare_sequences(
    ts: pd.DataFrame,
    input_len: int = 96,
    horizon: int = 12,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict:
    """
    Slide a window over the time series to create (X, y) pairs.

    Parameters
    ----------
    ts         : (T, N) DataFrame
    input_len  : number of past steps used as input
    horizon    : number of future steps to predict
    train/val  : split ratios (remainder = test)

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                    scaler, columns, n_features
    """
    from sklearn.preprocessing import MinMaxScaler

    values = ts.values.astype(np.float32)  # (T, N)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for t in range(input_len, len(scaled) - horizon + 1):
        X.append(scaled[t - input_len : t])
        y.append(scaled[t : t + horizon])

    X = np.array(X)  # (samples, input_len, N)
    y = np.array(y)  # (samples, horizon, N)

    n = len(X)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    return {
        "X_train":    X[:n_train],
        "y_train":    y[:n_train],
        "X_val":      X[n_train : n_train + n_val],
        "y_val":      y[n_train : n_train + n_val],
        "X_test":     X[n_train + n_val :],
        "y_test":     y[n_train + n_val :],
        "scaler":     scaler,
        "columns":    ts.columns.tolist(),
        "n_features": ts.shape[1],
    }
