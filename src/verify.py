"""Независимая проверка чисел из results/summary.md по data/corpus.parquet и времени сбора.

Модули проекта не импортируются: отбор постов, исключение молодых постов и малых групп,
целевые переменные и диагностические таблицы определены здесь заново и посчитаны другим кодом.
Не проверяются метрики моделей (в том числе переобученной на недельной цели) и термины
логистической регрессии. Код выхода 1, если есть хотя бы один FAIL.
"""

import itertools
import math
import operator
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "corpus.parquet"
COLLECTION_PATH = ROOT / "data" / "collection_times.parquet"
FEATURES_PATH = ROOT / "data" / "posts_features.parquet"
SUMMARY_PATH = ROOT / "results" / "summary.md"
IMPORTANCE_PATH = ROOT / "results" / "permutation_importance.csv"

CORPUS_COLUMNS = [
    "doc_id",
    "source",
    "channel",
    "actor_type",
    "date",
    "local_day",
    "language",
    "keyword_pass",
    "p_on_topic",
    "views",
    "forwards",
]
COLLECTION_COLUMNS = ["doc_id", "channel", "collected_at_est", "run_label", "age_days"]
RAW_COLUMNS = ["views", "forwards", "keyword_pass", "p_on_topic", "run_label", "age_days"]
POST_SOURCE = "telegram_post"
MIN_AGE_DAYS = 14
MIN_GROUP_SIZE = 20
GROUP = ["channel", "month", "run_label"]
TRAIN_END = "2026-06-30"
TEST_START = "2026-07-01"
UTC_OFFSET = pd.Timedelta(hours=5)
TARGETS = {"high_views": "views", "high_forwards": "forwards"}
LAYER_NAMES = {"government": "государственные", "blogger": "блогеры"}
ALL_NAME = "всего"
PERIOD_NAMES = ("все посты", "обучение", "проверка")
AGE_LABELS = ["<7", "7-14", "14-30", "30-90", "90-180", ">180"]
AGE_EDGES = [0, 7, 14, 30, 90, 180, math.inf]
WEEKDAY_NAMES = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
TOP_FRACTION = 1 / 3
Z_95 = NormalDist().inv_cdf(0.975)
NOT_MODEL_FEATURES = ["age_days", "run_label", "views", "forwards", "channel", "doc_id"]
# (минимальный возраст, минимальный размер группы, строгое «>»); первая строка — основной вариант.
SENSITIVITY = [
    (14, 20, True),
    (0, 20, True),
    (7, 20, True),
    (21, 20, True),
    (30, 20, True),
    (14, 10, True),
    (14, 30, True),
    (14, 20, False),
]

WEEK_GROUP = ["channel", "week_start", "run_label"]
WEEK_MIN_GROUP_SIZE = 10
LATE_DAY = 25
QUARTER_NAMES = ["младшая четверть", "2-я", "3-я", "старшая четверть"]
MONTH_LABEL = "v2 (месяц)"
WEEK_LABEL = "неделя"
MIN_SPREAD_POSTS = 100
DENSITY_DIGITS = {"posts_same_day": 0, "next_post_hours": 2}

LANGUAGE_HEADING = "### Посты по слоям и языкам"
LAYER_HEADING = "### Каналы, посты и медианы по слоям"
BALANCE_HEADING = "### Баланс классов"
TOPIC_HEADING = "### Доля high_views = 1 среди экономических и остальных постов"
WEEKDAY_HEADING = "### а) Доля high_views = 1 по дню недели"
HOUR_HEADING = "### а) Доля high_views = 1 по часу публикации (UTC+5)"
AGE_HEADING = "### б) Доля high_views = 1 по возрасту поста на момент сбора"
COMPARISON_HEADING = "### г) Экономические и остальные посты"
SIGN_MARKER = "Знаковый тест по каналам, keyword_pass"
MONTHLY_HEADING = "### д) Медианы просмотров и пересылок по месяцам и слоям"
QUARTILE_HEADING = "### Неделя и месяц: относительный возраст в группе"
LATE_MONTH_HEADING = f"### Неделя и месяц: дни {LATE_DAY}–31 минус дни 1–{LATE_DAY - 1} по месяцам"
WEEK_WEEKDAY_HEADING = "### Неделя и месяц: доля класса 1 по дню недели"
WEEK_HOUR_HEADING = "### Неделя и месяц: доля класса 1 по часу публикации (UTC+5)"
WEEK_KEYWORD_HEADING = "### Неделя: keyword_pass = 1 против 0"
WEEK_SIGN_HEADING = "### Неделя: знаковый тест по каналам"
DENSITY_QUARTILE_HEADINGS = {
    "posts_same_day": "#### Доля high_views = 1 по квартилям числа постов канала в тот же день",
    "next_post_hours": "#### Доля high_views = 1 по квартилям времени до следующего поста канала",
}
RATE_HEADINGS = {
    "weekday": "#### Частота публикаций по дню недели",
    "hour": "#### Частота публикаций по часу (UTC+5)",
}
STRATIFIED_HEADINGS = {
    "weekday": "#### Доля high_views = 1 по дню недели внутри квартилей",
    "hour": "#### Доля high_views = 1 по часу внутри квартилей",
}
SPREAD_HEADING = "#### Разница максимальной и минимальной доли high_views = 1"
# Разделитель тысяч в summary.md — неразрывный пробел, поэтому в числах допускается любой \s.
INPUT_RE = r"на входе: (\d[\d\s]*)\."
YOUNG_RE = r"моложе (\d+) дней на момент сбора: (\d[\d\s]*)\."
SMALL_GROUPS_RE = (
    r"«канал × месяц × прогон сбора» меньше (\d+) постов: (\d+) из (\d+), в них (\d[\d\s]*) постов"
)
ANALYSED_RE = r"Постов в анализе: (\d[\d\s]*)\."
RUNS_RE = r"Постов по прогонам сбора \(run_label\): (.+)\."
CHANNELS_RE = r"Каналов: (\d+) \(государственные: (\d+), блогеры: (\d+)\)"
PERIOD_RE = r"Период \(local_day, UTC\+5\): (\S+) — (\S+)\."
PART_RE = r"{name}: (\d{{4}}-\d\d-\d\d) — (\d{{4}}-\d\d-\d\d), (\d[\d\s]*) постов"
WEEK_AGED_RE = (
    r"Постов с age_days не меньше (\d+): (\d[\d\s]*); групп: (\d+); исключено групп меньше "
    r"(\d+) постов: (\d+), в них (\d[\d\s]*) постов"
)
WEEK_KEPT_RE = (
    r"Осталось: (\d[\d\s]*) постов в (\d+) группах \(в анализе v2: (\d[\d\s]*) постов\)\. "
    r"Доля класса 1: ([\d,]+ %)"
)
WEEK_SPLIT_RE = (
    r"Обучение: (\d[\d\s]*) постов, проверка: (\d[\d\s]*); неделя (\S+) \((\S+) — (\S+)\)"
)
DENSITY_FACTS_RE = (
    r"по всем постам корпуса \((\d[\d\s]*), с (\S+) по (\S+)\); доли high_views = 1 — по постам "
    r"анализа \((\d[\d\s]*)\)\. Квартили — по всем постам анализа; без следующего поста "
    r"в корпусе: (\d+) постов анализа"
)


@dataclass
class Check:
    name: str
    expected: str
    actual: str
    passed: bool


@dataclass
class Filtered:
    aged: pd.DataFrame
    kept: pd.DataFrame
    group_sizes: pd.Series
    small_groups: pd.Series


def load_posts() -> pd.DataFrame:
    corpus = pd.read_parquet(CORPUS_PATH, columns=CORPUS_COLUMNS)
    posts = corpus[corpus["source"] == POST_SOURCE]
    times = pd.read_parquet(COLLECTION_PATH, columns=COLLECTION_COLUMNS)
    merged = posts.merge(times, on="doc_id", how="left", suffixes=("", "_collection"))
    timestamp = pd.to_datetime(merged["date"], utc=True, format="ISO8601")
    local = timestamp + UTC_OFFSET
    day = pd.to_datetime(merged["local_day"])
    # Неделя задаётся датой своего понедельника: то же разбиение, что и ISO-неделя.
    monday = (day - pd.to_timedelta(day.dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")
    return merged.assign(
        month=merged["local_day"].str[:7],
        weekday=local.dt.dayofweek,
        hour=local.dt.hour,
        timestamp=timestamp,
        week_start=monday,
    )


def filter_posts(
    posts: pd.DataFrame, min_age: int, min_size: int, group: list[str] = GROUP
) -> Filtered:
    aged = posts[posts["age_days"] >= min_age]
    sizes = aged.value_counts(group)
    small = sizes[sizes < min_size]
    keys = pd.MultiIndex.from_frame(aged[group])
    kept = aged[~keys.isin(small.index)].reset_index(drop=True)
    return Filtered(aged, kept, sizes, small)


def add_targets(posts: pd.DataFrame, strict: bool = True, group: list[str] = GROUP) -> pd.DataFrame:
    columns = list(TARGETS.values())
    medians = posts.groupby(group)[columns].median()
    merged = posts.join(medians, on=group, rsuffix="_median")
    above = operator.gt if strict else operator.ge
    flags = {
        target: above(merged[column], merged[f"{column}_median"]).astype("int8")
        for target, column in TARGETS.items()
    }
    return posts.assign(**flags)


def periods(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train = data[data["local_day"] <= TRAIN_END]
    test = data[data["local_day"] >= TEST_START]
    return dict(zip(PERIOD_NAMES, (data, train, test), strict=True))


def layers(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parts = {name: data[data["actor_type"] == code] for code, name in LAYER_NAMES.items()}
    return parts | {ALL_NAME: data}


def layers_all_first(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parts = layers(data)
    return {ALL_NAME: parts.pop(ALL_NAME)} | parts


def find(pattern: str, text: str) -> tuple[str, ...]:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"В summary.md не найдено: {pattern}")
    return match.groups()


def table_after(text: str, marker: str) -> list[dict[str, str]]:
    lines = text.split(marker, 1)[1].splitlines()
    block = itertools.dropwhile(lambda line: not line.startswith("|"), lines)
    rows = itertools.takewhile(lambda line: line.startswith("|"), block)
    cells = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    header, body = cells[0], cells[2:]
    return [dict(zip(header, row, strict=True)) for row in body]


def plain(cell: str) -> str:
    return re.sub(r"\s", "", cell)


def ru(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def count_check(name: str, cell: str, value: int) -> Check:
    return Check(name, cell, str(value), int(plain(cell)) == value)


def share_check(name: str, cell: str, value: float) -> Check:
    if pd.isna(value):
        return Check(name, cell, "—", cell == "—")
    stated = cell.removesuffix("%").strip().replace(",", ".")
    return Check(name, cell, f"{100 * value:.3f} %", f"{100 * value:.1f}" == stated)


def text_check(name: str, stated: str, value: str) -> Check:
    return Check(name, stated, value, plain(stated) == plain(value))


def median_check(name: str, cell: str, value: float) -> Check:
    expected = f"{value:.1f}" if value % 1 else f"{value:.0f}"
    return Check(name, cell, f"{value:g}", plain(cell).replace(",", ".") == expected)


def list_check(name: str, stated: list[object], expected: list[object]) -> Check:
    return Check(name, str(stated), str(expected), stated == expected)


def zero_check(name: str, count: int) -> Check:
    return Check(name, "0", str(count), count == 0)


def count_checks(text: str, posts: pd.DataFrame, filtered: Filtered) -> list[Check]:
    (n_input,) = find(INPUT_RE, text)
    min_age, n_young = find(YOUNG_RE, text)
    threshold, n_small, n_groups, n_small_posts = find(SMALL_GROUPS_RE, text)
    (n_kept,) = find(ANALYSED_RE, text)
    return [
        count_check("постов на входе", n_input, len(posts)),
        count_check("минимальный возраст, дней", min_age, MIN_AGE_DAYS),
        count_check("исключено молодых постов", n_young, len(posts) - len(filtered.aged)),
        count_check("порог размера группы", threshold, MIN_GROUP_SIZE),
        count_check("исключено групп", n_small, len(filtered.small_groups)),
        count_check("всего групп канал x месяц x прогон", n_groups, len(filtered.group_sizes)),
        count_check(
            "исключено постов в малых группах", n_small_posts, int(filtered.small_groups.sum())
        ),
        count_check("постов в анализе", n_kept, len(filtered.kept)),
    ]


def run_checks(text: str, data: pd.DataFrame) -> list[Check]:
    (line,) = find(RUNS_RE, text)
    stated = dict(re.findall(r"(\w+) — (\d[\d\s]*)", line))
    counts = data["run_label"].value_counts()
    checks = [list_check("прогоны сбора", sorted(stated), sorted(counts.index))]
    checks += [
        count_check(f"постов, прогон {run}", n, int(counts.get(run, 0)))
        for run, n in stated.items()
    ]
    return checks


def channel_checks(text: str, data: pd.DataFrame) -> list[Check]:
    n_all, n_government, n_blogger = find(CHANNELS_RE, text)
    channels = data.groupby("actor_type")["channel"].nunique()
    return [
        count_check("каналов", n_all, data["channel"].nunique()),
        count_check("каналов, государственные", n_government, int(channels["government"])),
        count_check("каналов, блогеры", n_blogger, int(channels["blogger"])),
    ]


def period_checks(text: str, data: pd.DataFrame) -> list[Check]:
    parts = periods(data)
    first, last = find(PERIOD_RE, text)
    checks = [
        text_check("период, начало", first, data["local_day"].min()),
        text_check("период, конец", last, data["local_day"].max()),
    ]
    for name in PERIOD_NAMES[1:]:
        part = parts[name]
        start, end, n = find(PART_RE.format(name=name.capitalize()), text)
        checks += [
            text_check(f"{name}, начало", start, part["local_day"].min()),
            text_check(f"{name}, конец", end, part["local_day"].max()),
            count_check(f"{name}, постов", n, len(part)),
        ]
    return checks


def language_checks(text: str, data: pd.DataFrame) -> list[Check]:
    rows = table_after(text, LANGUAGE_HEADING)
    parts = layers(data)
    languages = [column for column in rows[0] if column not in ("слой", ALL_NAME)]
    checks = [
        list_check("строки таблицы языков", [row["слой"] for row in rows], list(parts)),
        list_check("столбцы-языки", sorted(languages), sorted(data["language"].unique())),
    ]
    for row in rows:
        part = parts[row["слой"]]
        counts = part["language"].value_counts()
        label = f"посты, {row['слой']}"
        checks += [
            count_check(f"{label}, {lang}", row[lang], int(counts.get(lang, 0)))
            for lang in languages
        ]
        checks.append(count_check(f"{label}, всего", row["всего"], len(part)))
    return checks


def layer_checks(text: str, data: pd.DataFrame) -> list[Check]:
    rows = table_after(text, LAYER_HEADING)
    parts = layers(data)
    checks = [list_check("строки таблицы медиан", [row["слой"] for row in rows], list(parts))]
    for row in rows:
        part, label = parts[row["слой"]], f"слой {row['слой']}"
        checks += [
            count_check(f"{label}: каналов", row["каналов"], part["channel"].nunique()),
            count_check(f"{label}: постов", row["постов"], len(part)),
            median_check(
                f"{label}: медиана просмотров", row["медиана просмотров"], part["views"].median()
            ),
            median_check(
                f"{label}: медиана пересылок", row["медиана пересылок"], part["forwards"].median()
            ),
        ]
    return checks


def balance_checks(text: str, data: pd.DataFrame) -> list[Check]:
    rows = table_after(text, BALANCE_HEADING)
    expected = [(target, name) for target in TARGETS for name in PERIOD_NAMES]
    stated = [(row["цель"], row["часть"]) for row in rows]
    checks = [list_check("строки таблицы баланса", stated, expected)]
    parts = periods(data)
    for row in rows:
        part, target = parts[row["часть"]], row["цель"]
        label = f"баланс {target}, {row['часть']}"
        checks += [
            count_check(f"{label}: n", row["n"], len(part)),
            count_check(f"{label}: класс 1", row["класс 1"], int(part[target].sum())),
            share_check(f"{label}: доля класса 1", row["доля класса 1"], part[target].mean()),
        ]
    return checks


def topic_checks(text: str, data: pd.DataFrame) -> list[Check]:
    rows = table_after(text, TOPIC_HEADING)
    parts = layers(data)
    stated = [row["слой"] for row in rows]
    checks = [list_check("строки таблицы keyword_pass", stated, list(parts))]
    for row in rows:
        part = parts[row["слой"]]
        for flag, group in ((True, "экономические"), (False, "остальные")):
            subset = part[part["keyword_pass"] == flag]
            label = f"keyword_pass, {row['слой']}, {group}"
            checks += [
                count_check(f"{label}: n", row[f"{group}: n"], len(subset)),
                share_check(
                    f"{label}: доля high_views = 1",
                    row[f"{group}: доля high_views = 1"],
                    subset["high_views"].mean(),
                ),
            ]
    return checks


def layered_checks(
    text: str, heading: str, data: pd.DataFrame, column: str, keys: dict[str, object]
) -> list[Check]:
    rows = table_after(text, heading)
    first = next(iter(rows[0]))
    checks = [list_check(f"{first}: строки", [row[first] for row in rows], list(keys))]
    for row in rows:
        for name, part in layers_all_first(data).items():
            subset = part[part[column] == keys[row[first]]]
            label = f"{first} {row[first]}, {name}"
            checks += [
                count_check(f"{label}: n", row[f"{name}: n"], len(subset)),
                share_check(f"{label}: доля", row[f"{name}: доля"], subset["high_views"].mean()),
            ]
    return checks


def time_age_checks(text: str, data: pd.DataFrame) -> list[Check]:
    buckets = pd.cut(data["age_days"], AGE_EDGES, right=False, labels=AGE_LABELS).astype(str)
    with_bucket = data.assign(age_bucket=buckets)
    weekdays = dict(zip(WEEKDAY_NAMES, range(7), strict=True))
    hours = {f"{hour:02d}": hour for hour in range(24)}
    return [
        *layered_checks(text, WEEKDAY_HEADING, data, "weekday", weekdays),
        *layered_checks(text, HOUR_HEADING, data, "hour", hours),
        *layered_checks(
            text,
            AGE_HEADING,
            with_bucket,
            "age_bucket",
            dict(zip(AGE_LABELS, AGE_LABELS, strict=True)),
        ),
    ]


def z_test(k1: int, n1: int, k0: int, n0: int) -> tuple[float, float]:
    pooled = (k1 + k0) / (n1 + n0)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    z = (k1 / n1 - k0 / n0) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


# Для z-статистики берётся объединённая доля (гипотеза о равенстве долей), для интервала —
# раздельные доли групп (интервал Вальда для разности).
def wald_interval(k1: int, n1: int, k0: int, n0: int) -> tuple[float, float]:
    p1, p0 = k1 / n1, k0 / n0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return p1 - p0 - Z_95 * se, p1 - p0 + Z_95 * se


def compare(target: pd.Series, flag: pd.Series) -> dict[str, float]:
    k1, n1 = int(target[flag].sum()), int(flag.sum())
    k0, n0 = int(target[~flag].sum()), int((~flag).sum())
    z, p_value = z_test(k1, n1, k0, n0)
    low, high = wald_interval(k1, n1, k0, n0)
    return {
        "n1": n1,
        "share1": k1 / n1,
        "n0": n0,
        "share0": k0 / n0,
        "diff": k1 / n1 - k0 / n0,
        "ci_low": low,
        "ci_high": high,
        "z": z,
        "p": p_value,
    }


def p_text(value: float) -> str:
    return ru(value, 3) if value >= 0.001 else f"{value:.1e}".replace(".", ",")


def in_top_fraction(scores: pd.Series) -> pd.Series:
    return scores >= scores.quantile(1 - TOP_FRACTION)


def topic_definitions(data: pd.DataFrame) -> dict[str, pd.Series]:
    by_layer = data.groupby("actor_type")["p_on_topic"].transform(in_top_fraction)
    return {
        "keyword_pass = 1 против 0": data["keyword_pass"].astype(bool),
        "p_on_topic: верхняя треть, общий порог": in_top_fraction(data["p_on_topic"]),
        "p_on_topic: верхняя треть, порог внутри слоя": by_layer.astype(bool),
    }


def threshold_text(definition: str, data: pd.DataFrame, part: pd.DataFrame, layer: str) -> str:
    quantile = 1 - TOP_FRACTION
    if definition.endswith("общий порог"):
        return ru(data["p_on_topic"].quantile(quantile), 3)
    if definition.endswith("внутри слоя") and layer != ALL_NAME:
        return ru(part["p_on_topic"].quantile(quantile), 3)
    return "—"


def comparison_row_checks(row: dict[str, str], result: dict[str, float], label: str) -> list[Check]:
    interval = f"[{ru(100 * result['ci_low'], 2)}; {ru(100 * result['ci_high'], 2)}]"
    return [
        count_check(f"{label}: n (1)", row["n (1)"], result["n1"]),
        share_check(f"{label}: доля (1)", row["доля (1)"], result["share1"]),
        count_check(f"{label}: n (0)", row["n (0)"], result["n0"]),
        share_check(f"{label}: доля (0)", row["доля (0)"], result["share0"]),
        text_check(f"{label}: разница", row["разница, п. п."], ru(100 * result["diff"], 2)),
        text_check(f"{label}: 95% ДИ", row["95 % ДИ, п. п."], interval),
        text_check(f"{label}: z", row["z"], ru(result["z"], 2)),
        text_check(f"{label}: p", row["p"], p_text(result["p"])),
    ]


def comparison_checks(text: str, data: pd.DataFrame) -> list[Check]:
    rows = table_after(text, COMPARISON_HEADING)
    definitions, parts = topic_definitions(data), layers_all_first(data)
    expected = [(d, layer) for d in definitions for layer in parts]
    checks = [
        list_check("строки таблицы г)", [(r["определение"], r["слой"]) for r in rows], expected)
    ]
    for row in rows:
        definition, layer = row["определение"], row["слой"]
        part = parts[layer]
        result = compare(part["high_views"], definitions[definition].loc[part.index])
        label = f"г) {definition}, {layer}"
        threshold = threshold_text(definition, data, part, layer)
        checks.append(text_check(f"{label}: порог", row["порог p_on_topic"], threshold))
        checks += comparison_row_checks(row, result, label)
    return checks


def sign_test(negative: int, total: int) -> float:
    tail = sum(math.comb(total, i) for i in range(min(negative, total - negative) + 1))
    return min(1.0, 2 * tail / 2**total)


def channel_differences(data: pd.DataFrame) -> pd.Series:
    flag = data["keyword_pass"].astype(bool)
    shares = data.groupby(["channel", flag])["high_views"].mean().unstack()
    return (shares[True] - shares[False]).dropna()


def sign_checks(text: str, data: pd.DataFrame) -> list[Check]:
    rows = table_after(text, SIGN_MARKER)
    parts = layers_all_first(data)
    checks = [list_check("строки знакового теста", [row["слой"] for row in rows], list(parts))]
    for row in rows:
        diffs = channel_differences(parts[row["слой"]])
        checks += sign_row_checks(row, diffs, f"знаковый тест, {row['слой']}")
    return checks


def monthly_checks(text: str, data: pd.DataFrame) -> list[Check]:
    rows = table_after(text, MONTHLY_HEADING)
    parts = layers_all_first(data)
    expected = [
        (layer, month) for layer, part in parts.items() for month in sorted(part["month"].unique())
    ]
    checks = [list_check("строки таблицы д)", [(r["слой"], r["месяц"]) for r in rows], expected)]
    for row in rows:
        part = parts[row["слой"]]
        month = part[part["month"] == row["месяц"]]
        label = f"д) {row['слой']}, {row['месяц']}"
        checks += [
            count_check(f"{label}: n", row["n"], len(month)),
            median_check(f"{label}: просмотры", row["медиана просмотров"], month["views"].median()),
            median_check(
                f"{label}: пересылки", row["медиана пересылок"], month["forwards"].median()
            ),
            text_check(
                f"{label}: возраст",
                row["медиана возраста при сборе, дней"],
                f"{month['age_days'].median():.0f}",
            ),
        ]
    return checks


def week_data(posts: pd.DataFrame) -> tuple[Filtered, pd.DataFrame]:
    filtered = filter_posts(posts, MIN_AGE_DAYS, WEEK_MIN_GROUP_SIZE, WEEK_GROUP)
    return filtered, add_targets(filtered.kept, group=WEEK_GROUP)


def monday_of(day: str) -> str:
    current = date.fromisoformat(day)
    return (current - timedelta(days=current.weekday())).isoformat()


def iso_label(day: str) -> str:
    year, week, _ = date.fromisoformat(day).isocalendar()
    return f"{year}-W{week:02d}"


def week_fact_checks(
    text: str, filtered: Filtered, week: pd.DataFrame, data: pd.DataFrame
) -> list[Check]:
    min_age, n_aged, n_groups, min_size, n_small, n_small_posts = find(WEEK_AGED_RE, text)
    n_kept, n_kept_groups, n_v2, positive = find(WEEK_KEPT_RE, text)
    n_train, n_test, boundary, first, last = find(WEEK_SPLIT_RE, text)
    boundary_days = week.loc[week["week_start"] == monday_of(TRAIN_END), "local_day"]
    parts = periods(week)
    return [
        count_check("неделя: минимальный возраст", min_age, MIN_AGE_DAYS),
        count_check("неделя: постов с age_days >= 14", n_aged, len(filtered.aged)),
        count_check("неделя: групп", n_groups, len(filtered.group_sizes)),
        count_check("неделя: порог размера группы", min_size, WEEK_MIN_GROUP_SIZE),
        count_check("неделя: исключено групп", n_small, len(filtered.small_groups)),
        count_check("неделя: исключено постов", n_small_posts, int(filtered.small_groups.sum())),
        count_check("неделя: осталось постов", n_kept, len(week)),
        count_check(
            "неделя: осталось групп",
            n_kept_groups,
            len(filtered.group_sizes) - len(filtered.small_groups),
        ),
        count_check("неделя: постов в v2", n_v2, len(data)),
        share_check("неделя: доля класса 1", positive, week["high_views"].mean()),
        count_check("неделя: обучение", n_train, len(parts["обучение"])),
        count_check("неделя: проверка", n_test, len(parts["проверка"])),
        text_check("неделя: граничная неделя", boundary, iso_label(TRAIN_END)),
        text_check("неделя: граничная неделя, первый день", first, boundary_days.min()),
        text_check("неделя: граничная неделя, последний день", last, boundary_days.max()),
    ]


def quarter_shares(data: pd.DataFrame, group: list[str]) -> list[float]:
    rank = data.groupby(group)["age_days"].rank(pct=True)
    quarter = (rank * 4).map(math.ceil)
    return [data.loc[quarter == q, "high_views"].mean() for q in range(1, 5)]


def late_gap(data: pd.DataFrame) -> float:
    late = data["local_day"].str[8:].astype(int) >= LATE_DAY
    return data.loc[late, "high_views"].mean() - data.loc[~late, "high_views"].mean()


def quartile_checks(text: str, frames: dict[str, tuple[pd.DataFrame, list[str]]]) -> list[Check]:
    rows = table_after(text, QUARTILE_HEADING)
    late_label = f"дни {LATE_DAY}–31 минус дни 1–{LATE_DAY - 1}, п. п."
    expected = [f"доля класса 1: {q}" for q in QUARTER_NAMES] + [late_label]
    checks = [
        list_check("устойчивость: строки четвертей", [r["показатель"] for r in rows], expected)
    ]
    for name, (frame, group) in frames.items():
        for row, value in zip(rows[:4], quarter_shares(frame, group), strict=True):
            checks.append(share_check(f"{name}: {row['показатель']}", row[name], value))
        checks.append(
            text_check(f"{name}: {late_label}", rows[4][name], ru(100 * late_gap(frame), 2))
        )
    return checks


def late_month_checks(text: str, frames: dict[str, tuple[pd.DataFrame, list[str]]]) -> list[Check]:
    rows = table_after(text, LATE_MONTH_HEADING)
    months = sorted(frames[MONTH_LABEL][0]["month"].unique()) + ["все"]
    checks = [list_check("устойчивость: месяцы", [row["месяц"] for row in rows], months)]
    for name, (frame, _) in frames.items():
        for row in rows:
            part = frame if row["месяц"] == "все" else frame[frame["month"] == row["месяц"]]
            label = f"{name}: дни 25–31 минус 1–24, {row['месяц']}"
            checks.append(text_check(label, row[f"{name}, п. п."], ru(100 * late_gap(part), 2)))
    return checks


def version_time_checks(
    text: str,
    heading: str,
    frames: dict[str, tuple[pd.DataFrame, list[str]]],
    column: str,
    keys: dict[str, int],
) -> list[Check]:
    rows = table_after(text, heading)
    first = next(iter(rows[0]))
    checks = [
        list_check(f"устойчивость, {first}: строки", [row[first] for row in rows], list(keys))
    ]
    for row in rows:
        for name, (frame, _) in frames.items():
            subset = frame[frame[column] == keys[row[first]]]
            label = f"{name}, {first} {row[first]}"
            checks += [
                count_check(f"{label}: n", row[f"{name}: n"], len(subset)),
                share_check(f"{label}: доля", row[f"{name}: доля"], subset["high_views"].mean()),
            ]
    return checks


def interval_cells(result: dict[str, float]) -> str:
    return f"[{ru(100 * result['ci_low'], 2)}; {ru(100 * result['ci_high'], 2)}]"


def week_keyword_checks(text: str, week: pd.DataFrame, data: pd.DataFrame) -> list[Check]:
    rows = table_after(text, WEEK_KEYWORD_HEADING)
    parts, v2_parts = layers_all_first(week), layers_all_first(data)
    checks = [
        list_check("неделя, keyword_pass: строки", [row["слой"] for row in rows], list(parts))
    ]
    for row in rows:
        layer = row["слой"]
        result = compare(parts[layer]["high_views"], parts[layer]["keyword_pass"].astype(bool))
        v2 = compare(v2_parts[layer]["high_views"], v2_parts[layer]["keyword_pass"].astype(bool))
        checks += comparison_row_checks(row, result, f"неделя, keyword_pass, {layer}")
        v2_text = f"{ru(100 * v2['diff'], 2)} {interval_cells(v2)}"
        checks.append(
            text_check(
                f"неделя, keyword_pass, {layer}: v2", row["v2: разница, п. п. [95 % ДИ]"], v2_text
            )
        )
    return checks


def sign_row_checks(row: dict[str, str], diffs: pd.Series, label: str) -> list[Check]:
    negative = int((diffs < 0).sum())
    return [
        count_check(f"{label}: каналов", row["каналов"], len(diffs)),
        count_check(f"{label}: разница < 0", row["разница < 0"], negative),
        count_check(f"{label}: разница > 0", row["разница > 0"], int((diffs > 0).sum())),
        text_check(f"{label}: медиана", row["медиана разницы, п. п."], ru(100 * diffs.median(), 2)),
        text_check(f"{label}: p", row["p"], p_text(sign_test(negative, len(diffs)))),
    ]


def week_sign_checks(text: str, week: pd.DataFrame, data: pd.DataFrame) -> list[Check]:
    rows = table_after(text, WEEK_SIGN_HEADING)
    parts, v2_parts = layers_all_first(week), layers_all_first(data)
    checks = [
        list_check("неделя, знаковый тест: строки", [row["слой"] for row in rows], list(parts))
    ]
    for row in rows:
        layer = row["слой"]
        checks += sign_row_checks(
            row, channel_differences(parts[layer]), f"неделя, знаковый тест, {layer}"
        )
        v2 = channel_differences(v2_parts[layer])
        negative = int((v2 < 0).sum())
        v2_text = f"{negative} из {len(v2)}, p = {p_text(sign_test(negative, len(v2)))}"
        checks.append(
            text_check(f"неделя, знаковый тест, {layer}: v2", row["v2: разница < 0"], v2_text)
        )
    return checks


def robustness_checks(text: str, posts: pd.DataFrame, data: pd.DataFrame) -> list[Check]:
    filtered, week = week_data(posts)
    frames = {MONTH_LABEL: (data, GROUP), WEEK_LABEL: (week, WEEK_GROUP)}
    weekdays = dict(zip(WEEKDAY_NAMES, range(7), strict=True))
    hours = {f"{hour:02d}": hour for hour in range(24)}
    return [
        *week_fact_checks(text, filtered, week, data),
        *quartile_checks(text, frames),
        *late_month_checks(text, frames),
        *version_time_checks(text, WEEK_WEEKDAY_HEADING, frames, "weekday", weekdays),
        *version_time_checks(text, WEEK_HOUR_HEADING, frames, "hour", hours),
        *week_keyword_checks(text, week, data),
        *week_sign_checks(text, week, data),
    ]


def density_measures(posts: pd.DataFrame) -> pd.DataFrame:
    ordered = posts.sort_values(["channel", "timestamp", "doc_id"])
    following = ordered.groupby("channel")["timestamp"].shift(-1)
    same_day = ordered[["channel", "local_day"]].merge(
        ordered.value_counts(["channel", "local_day"]).rename("posts_same_day").reset_index(),
        on=["channel", "local_day"],
        how="left",
    )
    return pd.DataFrame(
        {
            "doc_id": ordered["doc_id"].to_numpy(),
            "next_post_hours": (
                (following - ordered["timestamp"]).dt.total_seconds() / 3600
            ).to_numpy(),
            "posts_same_day": same_day["posts_same_day"].to_numpy(),
        }
    )


def quartile_of(values: pd.Series) -> pd.Series:
    edges = values.quantile([0.25, 0.5, 0.75]).to_list()
    quartile = 1 + sum((values > edge).astype(int) for edge in edges)
    return quartile.where(values.notna())


def density_fact_checks(text: str, posts: pd.DataFrame, data: pd.DataFrame) -> list[Check]:
    n_corpus, first, last, n_analysis, n_missing = find(DENSITY_FACTS_RE, text)
    return [
        count_check("частота: постов корпуса", n_corpus, len(posts)),
        text_check("частота: первый день", first, posts["local_day"].min()),
        text_check("частота: последний день", last, posts["local_day"].max()),
        count_check("частота: постов анализа", n_analysis, len(data)),
        count_check(
            "частота: без следующего поста", n_missing, int(data["next_post_hours"].isna().sum())
        ),
    ]


def density_quartile_checks(text: str, data: pd.DataFrame, measure: str) -> list[Check]:
    rows = table_after(text, DENSITY_QUARTILE_HEADINGS[measure])
    frame = data.assign(quartile=quartile_of(data[measure]))
    checks = [
        list_check(f"{measure}: квартили", [row["квартиль"] for row in rows], ["1", "2", "3", "4"])
    ]
    digits = DENSITY_DIGITS[measure]
    for row in rows:
        part = frame[frame["quartile"] == int(row["квартиль"])]
        span = f"{ru(part[measure].min(), digits)}–{ru(part[measure].max(), digits)}"
        checks.append(
            text_check(f"{measure}, квартиль {row['квартиль']}: диапазон", row["диапазон"], span)
        )
        for name, layer_part in layers_all_first(part).items():
            label = f"{measure}, квартиль {row['квартиль']}, {name}"
            checks += [
                count_check(f"{label}: n", row[f"{name}: n"], len(layer_part)),
                share_check(
                    f"{label}: доля", row[f"{name}: доля"], layer_part["high_views"].mean()
                ),
            ]
    return checks


def rate_checks(
    text: str, posts: pd.DataFrame, data: pd.DataFrame, variable: str, keys: dict[str, int]
) -> list[Check]:
    rows = table_after(text, RATE_HEADINGS[variable])
    first = next(iter(rows[0]))
    days = pd.date_range(posts["local_day"].min(), posts["local_day"].max(), freq="D")
    checks = [list_check(f"частота, {first}: строки", [row[first] for row in rows], list(keys))]
    for row in rows:
        value = keys[row[first]]
        n_days = int((days.dayofweek == value).sum()) if variable == "weekday" else len(days)
        for name, part in layers_all_first(posts).items():
            rate = int((part[variable] == value).sum()) / (part["channel"].nunique() * n_days)
            column = f"постов на канал в день: {name}"
            checks.append(
                text_check(f"частота, {first} {row[first]}, {name}", row[column], ru(rate, 3))
            )
        median = data.loc[data[variable] == value, "next_post_hours"].median()
        checks.append(
            text_check(
                f"частота, {first} {row[first]}: медиана часов",
                row["медиана часов до следующего поста"],
                ru(median, 2),
            )
        )
    return checks


def stratified_shares(data: pd.DataFrame, variable: str) -> dict[int, pd.DataFrame]:
    frame = data.assign(quartile=quartile_of(data["next_post_hours"]))
    parts = {0: frame} | {q: frame[frame["quartile"] == q] for q in range(1, 5)}
    return {
        q: part.groupby(variable)["high_views"].agg(["size", "mean"]) for q, part in parts.items()
    }


def stratified_checks(
    text: str, data: pd.DataFrame, variable: str, keys: dict[str, int]
) -> list[Check]:
    rows = table_after(text, STRATIFIED_HEADINGS[variable])
    first = next(iter(rows[0]))
    shares = stratified_shares(data, variable)
    checks = [list_check(f"расслоение, {first}: строки", [row[first] for row in rows], list(keys))]
    for row in rows:
        for quartile, stats in shares.items():
            column = "все посты" if quartile == 0 else f"квартиль {quartile}"
            value = stats["mean"].get(keys[row[first]], float("nan"))
            checks.append(
                share_check(f"расслоение, {first} {row[first]}, {column}", row[column], value)
            )
    return checks


def spread_checks(text: str, data: pd.DataFrame) -> list[Check]:
    rows = table_after(text, SPREAD_HEADING)
    names = {"день недели": "weekday", "час": "hour"}
    checks = [list_check("разброс: строки", [row["переменная"] for row in rows], list(names))]
    for row in rows:
        for quartile, stats in stratified_shares(data, names[row["переменная"]]).items():
            enough = stats.loc[stats["size"] >= MIN_SPREAD_POSTS, "mean"]
            column = "все посты, п. п." if quartile == 0 else f"квартиль {quartile}, п. п."
            spread = ru(100 * (enough.max() - enough.min()), 2)
            checks.append(
                text_check(f"разброс, {row['переменная']}, {column}", row[column], spread)
            )
    return checks


def density_checks(text: str, posts: pd.DataFrame, data: pd.DataFrame) -> list[Check]:
    measured = data.merge(density_measures(posts), on="doc_id", how="left")
    weekdays = dict(zip(WEEKDAY_NAMES, range(7), strict=True))
    hours = {f"{hour:02d}": hour for hour in range(24)}
    return [
        *density_fact_checks(text, posts, measured),
        *density_quartile_checks(text, measured, "posts_same_day"),
        *density_quartile_checks(text, measured, "next_post_hours"),
        *rate_checks(text, posts, measured, "weekday", weekdays),
        *rate_checks(text, posts, measured, "hour", hours),
        *stratified_checks(text, measured, "weekday", weekdays),
        *stratified_checks(text, measured, "hour", hours),
        *spread_checks(text, measured),
    ]


def age_mismatch(posts: pd.DataFrame) -> int:
    published = pd.to_datetime(posts["date"], utc=True, format="ISO8601")
    age = (posts["collected_at_est"] - published).dt.total_seconds() / 86400
    return int(((age - posts["age_days"]).abs() > 1e-6).sum())


def local_day_mismatches(posts: pd.DataFrame) -> int:
    utc = pd.to_datetime(posts["date"], utc=True, format="ISO8601")
    return int(((utc + UTC_OFFSET).dt.strftime("%Y-%m-%d") != posts["local_day"]).sum())


def integrity_checks(posts: pd.DataFrame) -> list[Check]:
    checks = [
        zero_check("повторов doc_id среди постов", int(posts["doc_id"].duplicated().sum())),
        zero_check("local_day не совпадает с date в UTC+5", local_day_mismatches(posts)),
        zero_check(
            "каналов с двумя значениями actor_type",
            int((posts.groupby("channel")["actor_type"].nunique() > 1).sum()),
        ),
        zero_check(
            "канал в collection_times не совпадает с корпусом",
            int((posts["channel"] != posts["channel_collection"]).sum()),
        ),
        zero_check("age_days не равно collected_at_est минус date", age_mismatch(posts)),
    ]
    checks += [zero_check(f"пропусков в {c}", int(posts[c].isna().sum())) for c in RAW_COLUMNS]
    checks += [
        zero_check(f"отрицательных {c}", int((posts[c] < 0).sum())) for c in TARGETS.values()
    ]
    return checks


def features_checks(data: pd.DataFrame) -> list[Check]:
    columns = ["doc_id", *RAW_COLUMNS, *TARGETS]
    features = pd.read_parquet(FEATURES_PATH, columns=columns)
    merged = data.merge(features, on="doc_id", how="outer", suffixes=("", "_file"), indicator=True)
    checks = [
        zero_check(
            "posts_features: doc_id только в одном из наборов",
            int((merged["_merge"] != "both").sum()),
        ),
        zero_check(
            f"posts_features: постов моложе {MIN_AGE_DAYS} дней",
            int((features["age_days"] < MIN_AGE_DAYS).sum()),
        ),
    ]
    for column in columns[1:]:
        differs = merged[column] != merged[f"{column}_file"]
        checks.append(zero_check(f"posts_features: расхождений в {column}", int(differs.sum())))
    return checks


def model_feature_checks() -> list[Check]:
    features = set(pd.read_csv(IMPORTANCE_PATH)["feature"])
    leaked = sorted(features.intersection(NOT_MODEL_FEATURES))
    name = "признаки модели (permutation_importance.csv) без " + ", ".join(NOT_MODEL_FEATURES)
    return [Check(name, "[]", str(leaked), not leaked)]


def percent(value: float) -> str:
    return f"{100 * value:.2f} %"


def points(value: float) -> str:
    return f"{100 * value:+.2f}"


def sensitivity_row(
    posts: pd.DataFrame, min_age: int, min_size: int, strict: bool
) -> dict[str, str]:
    data = add_targets(filter_posts(posts, min_age, min_size).kept, strict)
    result = compare(data["high_views"], data["keyword_pass"].astype(bool))
    return {
        "мин. возраст": str(min_age),
        "мин. группа": str(min_size),
        "класс 1": "> медианы" if strict else ">= медианы",
        "постов": str(len(data)),
        "доля high_views": percent(data["high_views"].mean()),
        "доля high_forwards": percent(data["high_forwards"].mean()),
        "разница kw, п.п.": points(result["diff"]),
        "95% ДИ, п.п.": f"[{points(result['ci_low'])}; {points(result['ci_high'])}]",
    }


def print_checks(title: str, checks: list[Check], reference: str) -> None:
    print(f"\n{title}\n")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status}  {check.name}: {reference} = {check.expected} | пересчёт = {check.actual}")


def print_sensitivity(posts: pd.DataFrame) -> None:
    rows = [sensitivity_row(posts, *variant) for variant in SENSITIVITY]
    print("\nЧувствительность к выбору (не из summary.md); первая строка — основной вариант\n")
    print(pd.DataFrame(rows).to_string())


def summary_checks(
    text: str, posts: pd.DataFrame, filtered: Filtered, data: pd.DataFrame
) -> list[Check]:
    sections: list[Callable[[str, pd.DataFrame], list[Check]]] = [
        run_checks,
        channel_checks,
        period_checks,
        language_checks,
        layer_checks,
        balance_checks,
        topic_checks,
        time_age_checks,
        comparison_checks,
        sign_checks,
        monthly_checks,
    ]
    checks = count_checks(text, posts, filtered)
    for section in sections:
        checks += section(text, data)
    return checks


def main() -> int:
    text = SUMMARY_PATH.read_text("utf-8")
    posts = load_posts()
    filtered = filter_posts(posts, MIN_AGE_DAYS, MIN_GROUP_SIZE)
    data = add_targets(filtered.kept)
    stated = [
        *summary_checks(text, posts, filtered, data),
        *robustness_checks(text, posts, data),
        *density_checks(text, posts, data),
    ]
    print_checks("Сверка с results/summary.md", stated, "summary.md")
    own = integrity_checks(posts) + features_checks(data) + model_feature_checks()
    print_checks("Целостность данных, промежуточного файла и списка признаков", own, "ожидается")
    print_sensitivity(posts)

    checks = stated + own
    failed = sum(not check.passed for check in checks)
    print(f"\nИтого: {len(checks) - failed} PASS, {failed} FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
