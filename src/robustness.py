"""Устойчивость к окну медианы: цель high_views_week внутри «канал × ISO-неделя × прогон сбора».

Посты с age_days не меньше MIN_AGE_DAYS, группы меньше 10 постов исключены. Признаки те же, что
в v2 (тот же список ключевых слов); переобучается только выбранная конфигурация. Пишет таблицы
results/robustness_*.csv и results/robustness.md; раздел «Устойчивость» в summary.md собирается
из тех же таблиц.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from build_dataset import (
    add_targets,
    add_time_columns,
    build_features,
    drop_small_groups,
    drop_young_posts,
    load_posts,
)
from common import (
    DATASET_STATS_PATH,
    FEATURES_PATH,
    MIN_AGE_DAYS,
    MODEL_LABELS,
    RESULTS_DIR,
    TARGET_GROUP,
    TEST_START,
    TRAIN_END,
    feature_columns,
    feature_label,
    format_ru,
)
from diagnostics import layer_parts
from report import (
    DIAGNOSTIC_LAYERS,
    WEEKDAY_NAMES,
    interval_text,
    markdown_table,
    metric,
    p_value,
    points,
    share,
)
from stats import channel_differences, compare_shares, sign_test_row
from train_models import (
    Setup,
    check_split,
    classification_metrics,
    permutation_importance_by_group,
    prepare_split,
    run_final,
    set_columns,
)
from v1_target import LATE_DAY, QUARTERS, late_gap, relative_age_quarter

V2_TARGET = "high_views"
WEEK_TARGET = "high_views_week"
WEEK_GROUP = ("channel", "week", "run_label")
WEEK_MIN_GROUP_SIZE = 10
N_TOP_FEATURES = 5
VERSION_LABELS = {"v2": "v2 (месяц)", "week": "неделя"}
OUTPUT_PATH = RESULTS_DIR / "robustness.md"
STATS_PATH = RESULTS_DIR / "robustness_stats.json"


@dataclass
class Version:
    key: str
    data: pd.DataFrame
    group: tuple[str, ...]
    target: str


def table_path(name: str) -> Path:
    return RESULTS_DIR / f"robustness_{name}.csv"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def iso_week(local_day: pd.Series) -> pd.Series:
    calendar = pd.to_datetime(local_day).dt.isocalendar()
    return calendar["year"].astype(str) + "-W" + calendar["week"].astype(str).str.zfill(2)


def week_posts() -> tuple[pd.DataFrame, dict]:
    aged = drop_young_posts(add_time_columns(load_posts()))
    aged = aged.assign(week=iso_week(aged["local_day"]))
    kept, stats = drop_small_groups(aged, WEEK_GROUP, WEEK_MIN_GROUP_SIZE)
    groups_kept = int(kept.groupby(list(WEEK_GROUP)).ngroups)
    stats = {"posts_aged": len(aged), **stats, "posts_kept": len(kept), "groups_kept": groups_kept}
    return add_targets(kept, WEEK_GROUP), stats


def week_table(posts: pd.DataFrame) -> pd.DataFrame:
    top = read_json(DATASET_STATS_PATH)["top_keywords"]
    table, _ = build_features(posts, top)
    return table.rename(columns={V2_TARGET: WEEK_TARGET}).assign(week=posts["week"].to_numpy())


def split_stats(table: pd.DataFrame) -> dict:
    boundary = iso_week(pd.Series([TRAIN_END])).iloc[0]
    days = table.loc[table["week"] == boundary, "local_day"]
    return {
        "posts_train": int((table["local_day"] <= TRAIN_END).sum()),
        "posts_test": int((table["local_day"] >= TEST_START).sum()),
        "positive_share": float(table[WEEK_TARGET].mean()),
        "boundary_week": boundary,
        "boundary_first_day": days.min(),
        "boundary_last_day": days.max(),
    }


def quartile_rows(version: Version) -> list[dict]:
    quarters = relative_age_quarter(version.data, version.group)
    grouped = version.data.groupby(quarters, observed=False)[version.target]
    stats = grouped.agg(["size", "mean"])
    return [
        {
            "version": version.key,
            "quarter": q,
            "n": stats.loc[q, "size"],
            "share": stats.loc[q, "mean"],
        }
        for q in QUARTERS
    ]


def late_gap_rows(version: Version) -> list[dict]:
    data = version.data
    rows = [
        {"version": version.key, "month": month, "gap": late_gap(part, version.target)}
        for month, part in data.groupby("month")
    ]
    return rows + [{"version": version.key, "month": "все", "gap": late_gap(data, version.target)}]


def time_rows(version: Version) -> list[dict]:
    rows = []
    for column in ("weekday", "hour"):
        stats = version.data.groupby(column)[version.target].agg(["size", "mean"])
        rows += [
            {"version": version.key, "variable": column, "value": value, "n": int(n), "share": mean}
            for value, (n, mean) in stats.iterrows()
        ]
    return rows


def keyword_rows(version: Version) -> list[dict]:
    return [
        {"layer": layer, **compare_shares(part[version.target], part["keyword_pass"].astype(bool))}
        for layer, part in layer_parts(version.data).items()
    ]


def sign_rows(version: Version) -> list[dict]:
    return [
        {"layer": layer, **sign_test_row(channel_differences(part, version.target, "keyword_pass"))}
        for layer, part in layer_parts(version.data).items()
    ]


def retrain(table: pd.DataFrame, best: dict) -> tuple[dict, pd.DataFrame]:
    train = table[table["local_day"] <= TRAIN_END].reset_index(drop=True)
    test = table[table["local_day"] >= TEST_START].reset_index(drop=True)
    check_split(train, test, len(table))
    feature_set = best["feature_set"]
    columns = set_columns(feature_set, feature_columns(table))
    setup = Setup(feature_set, [], prepare_split(train, test, feature_set), columns)
    experiment = run_final(best["model"], best["params"], [], setup, WEEK_TARGET)
    y_test = setup.final.evaluate[WEEK_TARGET].to_numpy()
    scores = classification_metrics(y_test, experiment.predicted, experiment.scores)
    row = {"version": "week", "posts_test": len(y_test), "positive_share": y_test.mean(), **scores}
    importance = permutation_importance_by_group(
        experiment.pipeline, setup.final.evaluate, y_test, setup.columns
    )
    return row, importance


def v2_model_row(v2: pd.DataFrame, best: dict) -> dict:
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    chosen = metrics[
        (metrics["target"] == V2_TARGET)
        & (metrics["model"] == best["model"])
        & (metrics["feature_set"] == best["feature_set"])
    ].iloc[0]
    test = v2[v2["local_day"] >= TEST_START]
    return {
        "version": "v2",
        "posts_test": len(test),
        "positive_share": test[V2_TARGET].mean(),
        **{name: chosen[name] for name in ("accuracy", "f1", "roc_auc")},
    }


def save_tables(versions: list[Version]) -> None:
    week = versions[-1]
    tables = {
        "quartiles": [row for version in versions for row in quartile_rows(version)],
        "late_gap": [row for version in versions for row in late_gap_rows(version)],
        "time": [row for version in versions for row in time_rows(version)],
        "keyword": keyword_rows(week),
        "sign_test": sign_rows(week),
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(table_path(name), index=False, encoding="utf-8")


def facts(stats: dict) -> str:
    return "\n".join(
        [
            "- Цель high_views_week: просмотры строго выше медианы группы «канал × ISO-неделя × "
            "прогон сбора» (неделя по local_day, UTC+5, с понедельника по воскресенье); посты с "
            f"age_days не меньше {MIN_AGE_DAYS}; группы меньше {WEEK_MIN_GROUP_SIZE} постов "
            "исключены.",
            f"- Постов с age_days не меньше {MIN_AGE_DAYS}: {format_ru(stats['posts_aged'])}; "
            f"групп: {stats['groups_total']}; исключено групп меньше {WEEK_MIN_GROUP_SIZE} постов: "
            f"{stats['groups_excluded']}, в них {format_ru(stats['posts_excluded'])} постов.",
            f"- Осталось: {format_ru(stats['posts_kept'])} постов в {stats['groups_kept']} группах "
            f"(в анализе v2: {format_ru(stats['posts_v2'])} постов). Доля класса 1: "
            f"{share(stats['positive_share'])}.",
            "- Признаки те же, что в v2 (тот же список из 30 ключевых слов). Обучение: "
            f"{format_ru(stats['posts_train'])} постов, "
            f"проверка: {format_ru(stats['posts_test'])}; "
            f"неделя {stats['boundary_week']} ({stats['boundary_first_day']} — "
            f"{stats['boundary_last_day']}) содержит посты обеих частей.",
        ]
    )


def by_version(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    return frame[frame["version"] == key].reset_index(drop=True)


def quartile_table() -> str:
    quartiles, gaps = pd.read_csv(table_path("quartiles")), pd.read_csv(table_path("late_gap"))
    late = f"дни {LATE_DAY}–31 минус дни 1–{LATE_DAY - 1}, п. п."
    columns = {"показатель": [*(f"доля класса 1: {q}" for q in QUARTERS), late]}
    for key, label in VERSION_LABELS.items():
        overall = by_version(gaps, key).set_index("month").loc["все", "gap"]
        columns[label] = [*by_version(quartiles, key)["share"].map(share), points(overall)]
    return markdown_table(pd.DataFrame(columns))


def late_gap_table() -> str:
    gaps = pd.read_csv(table_path("late_gap"))
    columns = {"месяц": by_version(gaps, "v2")["month"]}
    for key, label in VERSION_LABELS.items():
        columns[f"{label}, п. п."] = by_version(gaps, key)["gap"].map(points)
    return markdown_table(pd.DataFrame(columns))


def time_table(variable: str, first_column: str, labels: list[str]) -> str:
    time = pd.read_csv(table_path("time"))
    rows = time[time["variable"] == variable]
    columns = {first_column: labels}
    for key, label in VERSION_LABELS.items():
        part = by_version(rows, key).set_index("value").reindex(range(len(labels)))
        columns[f"{label}: n"] = part["n"].fillna(0).map(format_ru).tolist()
        columns[f"{label}: доля"] = part["share"].map(share).tolist()
    return markdown_table(pd.DataFrame(columns))


def keyword_table() -> str:
    week = pd.read_csv(table_path("keyword"))
    v2 = pd.read_csv(RESULTS_DIR / "keyword_comparison.csv")
    v2 = v2[v2["definition"] == "keyword_pass"].set_index("layer").reindex(week["layer"])
    return markdown_table(
        pd.DataFrame(
            {
                "слой": week["layer"].map(DIAGNOSTIC_LAYERS),
                "n (1)": week["n1"].map(format_ru),
                "доля (1)": week["share1"].map(share),
                "n (0)": week["n0"].map(format_ru),
                "доля (0)": week["share0"].map(share),
                "разница, п. п.": week["diff"].map(points),
                "95 % ДИ, п. п.": [
                    interval_text(low, high)
                    for low, high in zip(week["ci_low"], week["ci_high"], strict=True)
                ],
                "z": week["z"].map(lambda v: format_ru(v, 2)),
                "p": week["p_value"].map(p_value),
                "v2: разница, п. п. [95 % ДИ]": [
                    f"{points(diff)} {interval_text(low, high)}"
                    for diff, low, high in zip(v2["diff"], v2["ci_low"], v2["ci_high"], strict=True)
                ],
            }
        )
    )


def sign_table() -> str:
    week = pd.read_csv(table_path("sign_test"))
    v2 = (
        pd.read_csv(RESULTS_DIR / "channel_sign_test.csv").set_index("layer").reindex(week["layer"])
    )
    return markdown_table(
        pd.DataFrame(
            {
                "слой": week["layer"].map(DIAGNOSTIC_LAYERS),
                "каналов": week["channels"],
                "разница < 0": week["negative"],
                "разница > 0": week["positive"],
                "медиана разницы, п. п.": week["median_diff"].map(points),
                "p": week["p_value"].map(p_value),
                "v2: разница < 0": [
                    f"{negative} из {channels}, p = {p_value(p)}"
                    for negative, channels, p in zip(
                        v2["negative"], v2["channels"], v2["p_value"], strict=True
                    )
                ],
            }
        )
    )


def model_table() -> str:
    rows = pd.read_csv(table_path("model"))
    return markdown_table(
        pd.DataFrame(
            {
                "версия цели": rows["version"].map(VERSION_LABELS),
                "постов в проверке": rows["posts_test"].map(format_ru),
                "доля класса 1": rows["positive_share"].map(share),
                "accuracy": rows["accuracy"].map(metric),
                "F1": rows["f1"].map(metric),
                "ROC-AUC": rows["roc_auc"].map(metric),
            }
        )
    )


def importance_table() -> str:
    columns = {"№": range(1, N_TOP_FEATURES + 1)}
    sources = {
        "v2": RESULTS_DIR / "permutation_importance.csv",
        "week": table_path("importance"),
    }
    for key, path in sources.items():
        top = pd.read_csv(path).head(N_TOP_FEATURES)
        label = VERSION_LABELS[key]
        columns[f"{label}: признак"] = top["feature"].map(feature_label).tolist()
        columns[f"{label}: снижение ROC-AUC"] = (
            top["importance_mean"].map(lambda v: format_ru(v, 4)).tolist()
        )
    return markdown_table(pd.DataFrame(columns))


def robustness_sections(level: str) -> list[str]:
    stats, best = read_json(STATS_PATH), read_json(RESULTS_DIR / "best_config.json")
    sub = level + "#"
    config = f"{MODEL_LABELS[best['model']]}, набор {best['feature_set']}"
    weekdays, hours = list(WEEKDAY_NAMES), [f"{hour:02d}" for hour in range(24)]
    return [
        f"{level} Устойчивость: медиана по неделе",
        facts(stats),
        f"{sub} Неделя и месяц: относительный возраст в группе",
        "Четверти — по рангу age_days внутри группы медианы: «канал × месяц × прогон» в v2, "
        "«канал × ISO-неделя × прогон» для недели.",
        quartile_table(),
        f"{sub} Неделя и месяц: дни {LATE_DAY}–31 минус дни 1–{LATE_DAY - 1} по месяцам",
        late_gap_table(),
        f"{sub} Неделя и месяц: доля класса 1 по дню недели",
        time_table("weekday", "день недели", weekdays),
        f"{sub} Неделя и месяц: доля класса 1 по часу публикации (UTC+5)",
        time_table("hour", "час", hours),
        f"{sub} Неделя: keyword_pass = 1 против 0",
        "z-тест для двух долей, 95 % ДИ разницы — интервал Вальда; посты считаются независимыми.",
        keyword_table(),
        f"{sub} Неделя: знаковый тест по каналам",
        "Разница долей класса 1 (keyword_pass = 1 минус 0) внутри каждого канала; двусторонний "
        "биномиальный тест, вероятность 0,5.",
        sign_table(),
        f"{sub} Неделя: {config}, проверочная часть",
        f"Те же параметры (`{json.dumps(best['params'])}`), разбиение по времени и seed; "
        "проверочные части v2 и недели состоят из разных постов.",
        model_table(),
        f"{sub} Неделя: {N_TOP_FEATURES} главных признаков (перестановочная важность)",
        importance_table(),
    ]


def main() -> None:
    posts, stats = week_posts()
    table = week_table(posts)
    v2 = pd.read_parquet(FEATURES_PATH)
    best = read_json(RESULTS_DIR / "best_config.json")
    stats |= split_stats(table) | {"posts_v2": len(v2)}
    save_tables(
        [
            Version("v2", v2, TARGET_GROUP, V2_TARGET),
            Version("week", table, WEEK_GROUP, WEEK_TARGET),
        ]
    )
    week_model, importance = retrain(table, best)
    pd.DataFrame([v2_model_row(v2, best), week_model]).to_csv(
        table_path("model"), index=False, encoding="utf-8"
    )
    importance.to_csv(table_path("importance"), index=False, encoding="utf-8")
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), "utf-8")
    OUTPUT_PATH.write_text("\n\n".join(robustness_sections("#")) + "\n", "utf-8")
    print(f"Записано: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
