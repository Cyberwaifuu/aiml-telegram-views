"""Признаки v3: контекст публикаций канала и история его просмотров, с проверками утечек.

Все величины считаются по всем постам корпуса. Контекст использует только посты того же канала,
опубликованные строго раньше текущего. История просмотров использует только посты того же канала
и того же прогона сбора, опубликованные не позже чем за 24 часа до текущего.
"""

import numpy as np
import pandas as pd

from common import COLLECTION_TIMES_PATH, CORPUS_PATH, DATA_DIR, SEED

CHANNELS_PATH = DATA_DIR / "channels.csv"
WINDOWS = {"posts_prev_1h": "1h", "posts_prev_3h": "3h", "posts_prev_24h": "24h"}
HISTORY_LAG = pd.Timedelta(hours=24)
HISTORY_WINDOW = pd.Timedelta(days=30)
HISTORY_POSTS = 20
MIN_HISTORY = 5
CONTEXT_FEATURES = ["log_minutes_since_prev", *WINDOWS, "day_ordinal", "first_of_day"]
HISTORY_FEATURES = ["hist_last20_log", "hist_30d_log", "hist_diff"]
CHECK_SAMPLE = 300


def corpus_timeline() -> pd.DataFrame:
    columns = ["doc_id", "source", "channel", "date", "local_day", "views"]
    corpus = pd.read_parquet(CORPUS_PATH, columns=columns)
    posts = corpus[corpus["source"] == "telegram_post"].drop(columns="source")
    runs = pd.read_parquet(COLLECTION_TIMES_PATH, columns=["doc_id", "run_label"])
    posts = posts.merge(runs, on="doc_id", validate="one_to_one")
    timestamp = pd.to_datetime(posts["date"], utc=True, format="ISO8601")
    ordered = posts.assign(timestamp=timestamp).sort_values(["channel", "timestamp", "doc_id"])
    return ordered.reset_index(drop=True)


def nanoseconds(values: pd.Series) -> np.ndarray:
    return values.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").view(np.int64)


def channel_context(part: pd.DataFrame) -> pd.DataFrame:
    times = nanoseconds(part["timestamp"])
    earlier = np.searchsorted(times, times, side="left")
    previous = times[np.maximum(earlier - 1, 0)]
    minutes = np.where(earlier > 0, (times - previous) / 60e9, np.nan)
    counts = {
        name: earlier - np.searchsorted(times, times - pd.Timedelta(span).value, side="left")
        for name, span in WINDOWS.items()
    }
    day_first = (
        pd.Series(np.arange(len(part))).groupby(part["local_day"].to_numpy()).transform("min")
    )
    ordinal = earlier - day_first.to_numpy() + 1
    return pd.DataFrame(
        {
            "log_minutes_since_prev": np.log1p(minutes),
            **counts,
            "day_ordinal": ordinal,
            "first_of_day": (ordinal == 1).astype("int8"),
        },
        index=part.index,
    )


def median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) >= MIN_HISTORY else np.nan


def run_history(part: pd.DataFrame) -> pd.DataFrame:
    times, views = nanoseconds(part["timestamp"]), part["views"].to_numpy(dtype=float)
    end = np.searchsorted(times, times - HISTORY_LAG.value, side="right")
    start = np.searchsorted(times, times - HISTORY_WINDOW.value, side="left")
    last = [median_or_nan(views[max(0, k - HISTORY_POSTS) : k]) for k in end]
    window = [median_or_nan(views[s:k]) for s, k in zip(start, end, strict=True)]
    last_log, window_log = np.log1p(last), np.log1p(window)
    return pd.DataFrame(
        {
            "hist_last20_log": last_log,
            "hist_30d_log": window_log,
            "hist_diff": last_log - window_log,
        },
        index=part.index,
    )


def timeline_features(timeline: pd.DataFrame) -> pd.DataFrame:
    context = pd.concat(channel_context(part) for _, part in timeline.groupby("channel"))
    history = pd.concat(run_history(part) for _, part in timeline.groupby(["channel", "run_label"]))
    return timeline[["doc_id"]].join(context).join(history)


def subscribers() -> pd.DataFrame:
    channels = pd.read_csv(CHANNELS_PATH, usecols=["channel", "subscribers"])
    return channels.assign(log_subscribers=np.log1p(channels["subscribers"]))[
        ["channel", "log_subscribers"]
    ]


def brute_context(timeline: pd.DataFrame, row: pd.Series) -> dict[str, float]:
    earlier = timeline[
        (timeline["channel"] == row["channel"]) & (timeline["timestamp"] < row["timestamp"])
    ]
    gaps = row["timestamp"] - earlier["timestamp"]
    same_day = earlier["local_day"] == row["local_day"]
    minutes = gaps.min().total_seconds() / 60 if len(earlier) else np.nan
    return {
        "log_minutes_since_prev": np.log1p(minutes),
        **{name: int((gaps <= pd.Timedelta(span)).sum()) for name, span in WINDOWS.items()},
        "day_ordinal": int(same_day.sum()) + 1,
        "first_of_day": int(same_day.sum() == 0),
    }


def brute_history(timeline: pd.DataFrame, row: pd.Series) -> dict[str, float]:
    eligible = timeline[
        (timeline["channel"] == row["channel"])
        & (timeline["run_label"] == row["run_label"])
        & (timeline["timestamp"] <= row["timestamp"] - HISTORY_LAG)
    ]
    recent = eligible.tail(HISTORY_POSTS)["views"].to_numpy(dtype=float)
    window = eligible[eligible["timestamp"] >= row["timestamp"] - HISTORY_WINDOW]["views"]
    last, month = np.log1p(median_or_nan(recent)), np.log1p(median_or_nan(window.to_numpy(float)))
    return {"hist_last20_log": last, "hist_30d_log": month, "hist_diff": last - month}


def check_features(timeline: pd.DataFrame, features: pd.DataFrame) -> int:
    sample = timeline.sample(CHECK_SAMPLE, random_state=SEED)
    for index, row in sample.iterrows():
        expected = brute_context(timeline, row) | brute_history(timeline, row)
        actual = features.loc[index, list(expected)].to_numpy(dtype=float)
        if not np.allclose(actual, list(expected.values()), equal_nan=True):
            raise ValueError(f"Признаки поста {row['doc_id']} не совпадают с правилом: {expected}")
    return len(sample)
