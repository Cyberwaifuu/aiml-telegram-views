"""Частота публикаций: посты канала за тот же день и время до следующего поста канала.

Только описание, цель v2, без моделей; в признаки моделей эти величины не входят. Обе величины
считаются по всем постам корпуса (source = telegram_post), доли high_views = 1 — по постам
анализа v2. Пишет results/density_*.csv; таблицы входят в раздел «Диагностика» summary.md.
"""

import json
from pathlib import Path

import pandas as pd

from common import CORPUS_PATH, FEATURES_PATH, RESULTS_DIR, format_ru
from diagnostics import layer_parts
from report import (
    DIAGNOSTIC_LAYERS,
    WEEKDAY_NAMES,
    markdown_table,
    points,
    share,
)

TARGET = "high_views"
UTC_OFFSET = pd.Timedelta(hours=5)
MEASURES = {
    "posts_same_day": "постов канала в тот же день",
    "next_post_hours": "часов до следующего поста канала",
}
STRATIFIER = "next_post_hours"
N_QUARTILES = 4
MIN_SPREAD_POSTS = 100
STATS_PATH = RESULTS_DIR / "density_stats.json"


def table_path(name: str) -> Path:
    return RESULTS_DIR / f"density_{name}.csv"


def corpus_posts() -> pd.DataFrame:
    columns = ["doc_id", "source", "channel", "actor_type", "date", "local_day"]
    corpus = pd.read_parquet(CORPUS_PATH, columns=columns)
    posts = corpus[corpus["source"] == "telegram_post"].reset_index(drop=True)
    timestamp = pd.to_datetime(posts["date"], utc=True, format="ISO8601")
    local = timestamp + UTC_OFFSET
    return posts.assign(timestamp=timestamp, weekday=local.dt.dayofweek, hour=local.dt.hour)


def add_density(posts: pd.DataFrame) -> pd.DataFrame:
    ordered = posts.sort_values(["channel", "timestamp", "doc_id"])
    following = ordered.groupby("channel")["timestamp"].shift(-1)
    hours = (following - ordered["timestamp"]).dt.total_seconds() / 3600
    same_day = ordered.groupby(["channel", "local_day"])["doc_id"].transform("size")
    return ordered.assign(next_post_hours=hours, posts_same_day=same_day)


def analysis_posts(density: pd.DataFrame) -> pd.DataFrame:
    columns = ["doc_id", "channel", "actor_type", "weekday", "hour", TARGET]
    features = pd.read_parquet(FEATURES_PATH, columns=columns)
    measures = density[["doc_id", *MEASURES]]
    return features.merge(measures, on="doc_id", how="left", validate="one_to_one")


def quartiles(values: pd.Series) -> pd.Series:
    return pd.qcut(values, N_QUARTILES, labels=False, duplicates="drop") + 1


def quartile_rows(data: pd.DataFrame, measure: str) -> list[dict]:
    frame = data.assign(quartile=quartiles(data[measure])).dropna(subset=["quartile"])
    rows = []
    for layer, part in layer_parts(frame).items():
        stats = part.groupby("quartile").agg(
            n=(TARGET, "size"), share=(TARGET, "mean"), low=(measure, "min"), high=(measure, "max")
        )
        rows += [
            {
                "measure": measure,
                "quartile": int(q),
                "layer": layer,
                **row.to_dict(),
                "n": int(row["n"]),
            }
            for q, row in stats.iterrows()
        ]
    return rows


def calendar_days(posts: pd.DataFrame) -> pd.DatetimeIndex:
    days = pd.to_datetime(posts["local_day"])
    return pd.date_range(days.min(), days.max(), freq="D")


def rate_rows(posts: pd.DataFrame) -> list[dict]:
    days = calendar_days(posts)
    weekday_days = pd.Series(days.dayofweek).value_counts()
    rows = []
    for layer, part in layer_parts(posts).items():
        channels = part["channel"].nunique()
        for weekday, n in part["weekday"].value_counts().sort_index().items():
            rate = n / (channels * weekday_days[weekday])
            rows.append({"variable": "weekday", "value": weekday, "layer": layer, "rate": rate})
        for hour, n in part["hour"].value_counts().sort_index().items():
            rate = n / (channels * len(days))
            rows.append({"variable": "hour", "value": hour, "layer": layer, "rate": rate})
    return rows


def wait_rows(data: pd.DataFrame) -> list[dict]:
    rows = []
    for variable in ("weekday", "hour"):
        medians = data.groupby(variable)[STRATIFIER].agg(["count", "median"])
        rows += [
            {"variable": variable, "value": value, "n": int(n), "median_hours": median}
            for value, (n, median) in medians.iterrows()
        ]
    return rows


def stratified_rows(data: pd.DataFrame) -> list[dict]:
    frame = data.assign(quartile=quartiles(data[STRATIFIER])).dropna(subset=["quartile"])
    rows = []
    for variable in ("weekday", "hour"):
        stats = frame.groupby(["quartile", variable])[TARGET].agg(["size", "mean"])
        rows += [
            {"variable": variable, "quartile": int(q), "value": value, "n": int(n), "share": mean}
            for (q, value), (n, mean) in stats.iterrows()
        ]
        overall = data.groupby(variable)[TARGET].agg(["size", "mean"])
        rows += [
            {"variable": variable, "quartile": 0, "value": value, "n": int(n), "share": mean}
            for value, (n, mean) in overall.iterrows()
        ]
    return rows


def spread_rows(stratified: pd.DataFrame) -> list[dict]:
    enough = stratified[stratified["n"] >= MIN_SPREAD_POSTS]
    grouped = enough.groupby(["variable", "quartile"])["share"]
    spread = (grouped.max() - grouped.min()).rename("spread")
    values = grouped.size().rename("values")
    return pd.concat([spread, values], axis=1).reset_index().to_dict("records")


def density_stats(density: pd.DataFrame, data: pd.DataFrame) -> dict:
    days = calendar_days(density)
    return {
        "corpus_posts": len(density),
        "first_day": days.min().strftime("%Y-%m-%d"),
        "last_day": days.max().strftime("%Y-%m-%d"),
        "analysis_posts": len(data),
        "analysis_without_next_post": int(data[STRATIFIER].isna().sum()),
    }


def save_tables() -> None:
    density = add_density(corpus_posts())
    data = analysis_posts(density)
    STATS_PATH.write_text(json.dumps(density_stats(density, data), indent=2), "utf-8")
    stratified = pd.DataFrame(stratified_rows(data))
    tables = {
        "quartiles": [row for measure in MEASURES for row in quartile_rows(data, measure)],
        "rates": rate_rows(density),
        "wait": wait_rows(data),
        "stratified": stratified.to_dict("records"),
        "spread": spread_rows(stratified),
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(table_path(name), index=False, encoding="utf-8")
        print(f"Записано: {table_path(name)}")


def range_text(low: float, high: float, measure: str) -> str:
    digits = 0 if measure == "posts_same_day" else 2
    return f"{format_ru(low, digits)}–{format_ru(high, digits)}"


def quartile_table(measure: str) -> str:
    rows = pd.read_csv(table_path("quartiles"))
    rows = rows[rows["measure"] == measure]
    overall = rows[rows["layer"] == "all"].set_index("quartile")
    columns = {
        "квартиль": overall.index.tolist(),
        "диапазон": [
            range_text(low, high, measure)
            for low, high in zip(overall["low"], overall["high"], strict=True)
        ],
    }
    for layer, name in DIAGNOSTIC_LAYERS.items():
        part = rows[rows["layer"] == layer].set_index("quartile").reindex(overall.index)
        columns[f"{name}: n"] = part["n"].fillna(0).map(format_ru).tolist()
        columns[f"{name}: доля"] = part["share"].map(share).tolist()
    return markdown_table(pd.DataFrame(columns))


def rate_table(variable: str, first_column: str, labels: list[str]) -> str:
    rates = pd.read_csv(table_path("rates"))
    waits = pd.read_csv(table_path("wait"))
    rates = rates[rates["variable"] == variable]
    waits = waits[waits["variable"] == variable].set_index("value").reindex(range(len(labels)))
    columns = {first_column: labels}
    for layer, name in DIAGNOSTIC_LAYERS.items():
        part = rates[rates["layer"] == layer].set_index("value").reindex(range(len(labels)))
        columns[f"постов на канал в день: {name}"] = (
            part["rate"].fillna(0).map(lambda v: format_ru(v, 3)).tolist()
        )
    columns["медиана часов до следующего поста"] = (
        waits["median_hours"].map(lambda v: format_ru(v, 2)).tolist()
    )
    return markdown_table(pd.DataFrame(columns))


def stratified_table(variable: str, first_column: str, labels: list[str]) -> str:
    rows = pd.read_csv(table_path("stratified"))
    rows = rows[rows["variable"] == variable]
    columns = {first_column: labels}
    for quartile in sorted(rows["quartile"].unique()):
        part = rows[rows["quartile"] == quartile].set_index("value").reindex(range(len(labels)))
        name = "все посты" if quartile == 0 else f"квартиль {quartile}"
        columns[name] = part["share"].map(share).tolist()
    return markdown_table(pd.DataFrame(columns))


def spread_table() -> str:
    spread = pd.read_csv(table_path("spread"))
    names = {"weekday": "день недели", "hour": "час"}
    columns = {"переменная": list(names.values())}
    for quartile in sorted(spread["quartile"].unique()):
        part = spread[spread["quartile"] == quartile].set_index("variable").reindex(list(names))
        name = "все посты" if quartile == 0 else f"квартиль {quartile}"
        columns[f"{name}, п. п."] = part["spread"].map(points).tolist()
    return markdown_table(pd.DataFrame(columns))


def density_sections(level: str) -> list[str]:
    stats = json.loads(STATS_PATH.read_text("utf-8"))
    weekdays, hours = list(WEEKDAY_NAMES), [f"{hour:02d}" for hour in range(24)]
    period = f"с {stats['first_day']} по {stats['last_day']}"
    return [
        f"{level} е) Частота публикаций и время до следующего поста",
        "Только описание, цель high_views из раздела «Баланс классов»; в признаки моделей эти "
        "величины не входят. «Постов канала в тот же день» — число постов канала с тем же "
        "local_day, включая сам пост; «часов до следующего поста канала» — время до следующего "
        "поста того же канала. Обе величины считаются по всем постам корпуса "
        f"({format_ru(stats['corpus_posts'])}, {period}); доли high_views = 1 — по постам "
        f"анализа ({format_ru(stats['analysis_posts'])}). Квартили — по всем постам анализа; "
        f"без следующего поста в корпусе: {stats['analysis_without_next_post']} постов анализа.",
        f"{level}# Доля high_views = 1 по квартилям числа постов канала в тот же день",
        quartile_table("posts_same_day"),
        f"{level}# Доля high_views = 1 по квартилям времени до следующего поста канала, часов",
        quartile_table("next_post_hours"),
        f"{level}# Частота публикаций по дню недели",
        "Постов на канал в день — число постов в этот день недели, делённое на число каналов "
        f"слоя и число таких дней {period} (все посты корпуса). Медиана часов до следующего "
        "поста — по постам анализа.",
        rate_table("weekday", "день недели", weekdays),
        f"{level}# Частота публикаций по часу (UTC+5)",
        "Постов на канал в день в этот час — число постов в этот час, делённое на число каналов "
        "слоя и число дней периода.",
        rate_table("hour", "час", hours),
        f"{level}# Доля high_views = 1 по дню недели внутри квартилей времени до следующего поста",
        stratified_table("weekday", "день недели", weekdays),
        f"{level}# Доля high_views = 1 по часу внутри квартилей времени до следующего поста",
        stratified_table("hour", "час", hours),
        f"{level}# Разница максимальной и минимальной доли high_views = 1",
        f"По дням недели или часам с n не меньше {MIN_SPREAD_POSTS} постов в ячейке.",
        spread_table(),
    ]


if __name__ == "__main__":
    save_tables()
