"""Диагностика цели v1 и выбор минимального возраста поста; пишет results/v1_target_diagnostics.md.

Цель v1: медиана внутри «канал × месяц», все посты, группы меньше 20 постов исключены.
Цель v2: посты от MIN_AGE_DAYS дней на момент сбора, медиана внутри «канал × месяц × прогон».
"""

import pandas as pd

from build_dataset import (
    add_targets,
    add_time_columns,
    drop_small_groups,
    drop_young_posts,
    load_posts,
)
from common import AGE_BUCKETS, MIN_AGE_DAYS, RESULTS_DIR, TARGET_GROUP, age_bucket, format_ru
from report import markdown_table, points, share

V1_GROUP = ("channel", "month")
CANDIDATE_MIN_AGES = (0, 7, 14, 21, 30)
SPEARMAN_MIN_AGES = (0, 14, 30)
GROUP_AGE_EDGES = [0, 30, 60, 90, 180, 400]
LATE_DAY = 25
QUARTERS = ("младшая четверть", "2-я", "3-я", "старшая четверть")
OUTPUT_PATH = RESULTS_DIR / "v1_target_diagnostics.md"


def all_posts() -> pd.DataFrame:
    return add_time_columns(load_posts())


def v1_target(posts: pd.DataFrame) -> pd.DataFrame:
    kept, _ = drop_small_groups(posts, V1_GROUP)
    return add_targets(kept, V1_GROUP)


def v2_target(posts: pd.DataFrame, min_age: float = MIN_AGE_DAYS) -> pd.DataFrame:
    kept, _ = drop_small_groups(drop_young_posts(posts, min_age))
    return add_targets(kept)


def age_profile(v1: pd.DataFrame) -> str:
    ratio = v1["views"] / v1.groupby("channel")["views"].transform("median")
    grouped = v1.assign(bucket=age_bucket(v1["age_days"]), ratio=ratio).groupby(
        "bucket", observed=False
    )
    return markdown_table(
        pd.DataFrame(
            {
                "возраст при сборе, дней": list(AGE_BUCKETS),
                "n": grouped.size().map(format_ru).tolist(),
                "доля high_views = 1 (v1)": grouped["high_views"].mean().map(share).tolist(),
                "медиана просмотров / медиана канала": grouped["ratio"]
                .median()
                .map(lambda v: format_ru(v, 3))
                .tolist(),
                "доля keyword_pass = 1": grouped["keyword_pass"].mean().map(share).tolist(),
            }
        )
    )


def mixed_groups(posts: pd.DataFrame, v1: pd.DataFrame) -> str:
    columns = list(V1_GROUP)
    all_runs = posts.groupby(columns)["run_label"].nunique()
    kept_runs = v1.groupby(columns)["run_label"].nunique()
    keys = pd.MultiIndex.from_frame(v1[columns])
    in_mixed = keys.isin(kept_runs[kept_runs > 1].index)
    return "\n".join(
        [
            f"- Групп «канал × месяц» всего: {len(all_runs)}, из них с постами больше чем одного "
            f"прогона: {int((all_runs > 1).sum())}.",
            f"- Групп в анализе v1: {len(kept_runs)}, из них смешанных: "
            f"{int((kept_runs > 1).sum())}; в них {format_ru(in_mixed.sum())} постов "
            f"({share(in_mixed.mean())} постов анализа v1).",
        ]
    )


def run_table(v1: pd.DataFrame) -> str:
    grouped = v1.groupby("run_label")
    return markdown_table(
        pd.DataFrame(
            {
                "прогон": grouped.size().index,
                "n": grouped.size().map(format_ru).tolist(),
                "медиана возраста, дней": grouped["age_days"].median().map(format_ru).tolist(),
                "доля keyword_pass = 1": grouped["keyword_pass"].mean().map(share).tolist(),
                "доля high_views = 1 (v1)": grouped["high_views"].mean().map(share).tolist(),
            }
        )
    )


def spearman(frame: pd.DataFrame, first: str, second: str) -> float:
    return frame[first].corr(frame[second].astype(float), method="spearman")


def keyword_age_facts(v1: pd.DataFrame) -> str:
    within = v1.groupby("channel").apply(
        lambda part: spearman(part, "age_days", "keyword_pass"), include_groups=False
    )
    return (
        "Ранговая корреляция Спирмена между age_days и keyword_pass: по всем постам v1 "
        f"{format_ru(spearman(v1, 'age_days', 'keyword_pass'), 3)}; внутри каналов — медиана "
        f"{format_ru(within.median(), 3)}, от {format_ru(within.min(), 3)} "
        f"до {format_ru(within.max(), 3)}."
    )


def with_day(data: pd.DataFrame) -> pd.DataFrame:
    day = data["local_day"].str[8:].astype(int)
    return data.assign(day=day, late=day >= LATE_DAY)


def day_table(v1: pd.DataFrame, v2: pd.DataFrame) -> str:
    first, second = with_day(v1).groupby("day"), with_day(v2).groupby("day")
    days = range(1, 32)
    return markdown_table(
        pd.DataFrame(
            {
                "день месяца": list(days),
                "v1: n": first.size().reindex(days).map(format_ru).tolist(),
                "v1: доля high_views = 1": first["high_views"].mean().reindex(days).map(share),
                "v2: n": second.size().reindex(days).fillna(0).map(format_ru).tolist(),
                "v2: доля high_views = 1": second["high_views"].mean().reindex(days).map(share),
            }
        )
    )


def late_gap(data: pd.DataFrame, target: str = "high_views") -> float:
    marked = with_day(data)
    return marked.loc[marked["late"], target].mean() - marked.loc[~marked["late"], target].mean()


def late_gap_table(v1: pd.DataFrame, v2: pd.DataFrame) -> str:
    months = sorted(v1["month"].unique())
    rows = [
        {
            "месяц": month,
            "v1, п. п.": points(late_gap(v1[v1["month"] == month])),
            "v2, п. п.": points(late_gap(v2[v2["month"] == month])),
        }
        for month in months
    ]
    rows.append(
        {"месяц": "все", "v1, п. п.": points(late_gap(v1)), "v2, п. п.": points(late_gap(v2))}
    )
    return markdown_table(pd.DataFrame(rows))


def late_by_age_table(v1: pd.DataFrame) -> str:
    marked = with_day(v1).assign(bucket=age_bucket(v1["age_days"]))
    stats = marked.groupby(["bucket", "late"], observed=False)["high_views"].agg(["size", "mean"])
    return markdown_table(
        pd.DataFrame(
            {
                "возраст при сборе, дней": list(AGE_BUCKETS),
                f"дни 1–{LATE_DAY - 1}: n": [
                    format_ru(stats.loc[(b, False), "size"]) for b in AGE_BUCKETS
                ],
                f"дни 1–{LATE_DAY - 1}: доля": [
                    share(stats.loc[(b, False), "mean"]) for b in AGE_BUCKETS
                ],
                f"дни {LATE_DAY}–31: n": [
                    format_ru(stats.loc[(b, True), "size"]) for b in AGE_BUCKETS
                ],
                f"дни {LATE_DAY}–31: доля": [
                    share(stats.loc[(b, True), "mean"]) for b in AGE_BUCKETS
                ],
            }
        )
    )


def relative_age_quarter(data: pd.DataFrame, group: tuple[str, ...] = TARGET_GROUP) -> pd.Series:
    rank = data.groupby(list(group))["age_days"].rank(pct=True)
    return pd.cut(rank, [0, 0.25, 0.5, 0.75, 1.0], labels=list(QUARTERS))


def candidate_row(posts: pd.DataFrame, min_age: int) -> dict[str, str]:
    data = v2_target(posts, min_age)
    by_bucket = data.groupby(age_bucket(data["age_days"]), observed=False)["high_views"].mean()
    by_quarter = data.groupby(relative_age_quarter(data), observed=False)["high_views"].mean()
    return {
        "мин. возраст, дней": str(min_age),
        "постов": format_ru(len(data)),
        "групп": str(data.groupby(list(TARGET_GROUP)).ngroups),
        **{f"возраст {bucket}": share(by_bucket[bucket]) for bucket in AGE_BUCKETS},
        **{quarter: share(by_quarter[quarter]) for quarter in QUARTERS},
    }


def candidate_table(posts: pd.DataFrame) -> str:
    return markdown_table(pd.DataFrame([candidate_row(posts, age) for age in CANDIDATE_MIN_AGES]))


def group_spearman(data: pd.DataFrame) -> pd.DataFrame:
    groups = data.groupby(list(TARGET_GROUP))
    rho = groups.apply(lambda part: spearman(part, "age_days", "views"), include_groups=False)
    age_class = pd.cut(groups["age_days"].median(), GROUP_AGE_EDGES)
    return pd.DataFrame({"rho": rho, "age_class": age_class})


def spearman_table(posts: pd.DataFrame) -> str:
    rows = []
    for min_age in SPEARMAN_MIN_AGES:
        frame = group_spearman(v2_target(posts, min_age))
        for age_class, part in frame.groupby("age_class", observed=True):
            rows.append(
                {
                    "мин. возраст, дней": str(min_age),
                    "медиана возраста группы, дней": f"{age_class.left:.0f}–{age_class.right:.0f}",
                    "групп": str(len(part)),
                    "средний rho": format_ru(part["rho"].mean(), 3),
                    "доля групп с rho > 0": share((part["rho"] > 0).mean()),
                }
            )
    return markdown_table(pd.DataFrame(rows))


def build_report(posts: pd.DataFrame) -> str:
    v1, v2 = v1_target(posts), v2_target(posts)
    sections = [
        "# Диагностика цели v1 и выбор минимального возраста поста",
        "v1: медиана просмотров внутри «канал × месяц», все посты. v2: посты от "
        f"{MIN_AGE_DAYS} дней на момент сбора, медиана внутри «канал × месяц × прогон сбора». "
        "В обеих версиях группы меньше 20 постов исключены, класс 1 — строго выше медианы. "
        "Возрастные интервалы включают левую границу.",
        "## Возраст поста на момент сбора, цель v1",
        "Медиана канала — по всем постам канала в анализе v1.",
        age_profile(v1),
        "## Группы «канал × месяц» с постами разных прогонов",
        mixed_groups(posts, v1),
        "## keyword_pass по прогонам сбора",
        run_table(v1),
        keyword_age_facts(v1),
        "## День месяца",
        day_table(v1, v2),
        f"Разница долей high_views = 1: дни {LATE_DAY}–31 минус дни 1–{LATE_DAY - 1}.",
        late_gap_table(v1, v2),
        f"Цель v1, дни 1–{LATE_DAY - 1} и {LATE_DAY}–31 внутри интервалов возраста.",
        late_by_age_table(v1),
        "## Выбор минимального возраста",
        "Цель v2 при разных порогах. Доля high_views = 1 по интервалам возраста и по четвертям "
        "относительного возраста внутри группы «канал × месяц × прогон» (ранг age_days в группе).",
        candidate_table(posts),
        "Ранговая корреляция Спирмена между age_days и views внутри каждой группы "
        "«канал × месяц × прогон»; группы разбиты по медиане возраста.",
        spearman_table(posts),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    OUTPUT_PATH.write_text(build_report(all_posts()), "utf-8")
    print(f"Записано: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
