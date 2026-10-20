"""Сравнивает расчёты v1 (results_v1/) и v2 (results/), пишет results/v1_vs_v2.md."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common import (
    FEATURES_PATH,
    MIN_AGE_DAYS,
    MIN_GROUP_SIZE,
    MODEL_LABELS,
    RESULTS_DIR,
    ROOT,
    TARGET_GROUP,
    TARGETS,
    feature_label,
    format_ru,
)
from diagnostics import channel_sign_tests, keyword_comparison
from report import (
    DEFINITION_LABELS,
    DIAGNOSTIC_LAYERS,
    markdown_table,
    metric,
    p_value,
    period_parts,
    points,
    share,
)
from v1_target import (
    LATE_DAY,
    QUARTERS,
    V1_GROUP,
    all_posts,
    late_gap,
    relative_age_quarter,
    v1_target,
)

V1_DIR = ROOT / "results_v1"
OUTPUT_PATH = RESULTS_DIR / "v1_vs_v2.md"
BASELINE_MODEL = "DummyClassifier"
BEST_V1 = ("LightGBM", "B")
TOP_THREE_V1 = {"text", "weekday", "hour"}
N_TOP_FEATURES = 5
SIGNIFICANCE = 0.05
HIGH_VIEWS_BAND = (0.49, 0.51)
HIGH_FORWARDS_BAND = (0.39, 0.42)


@dataclass
class Version:
    name: str
    data: pd.DataFrame
    stats: dict
    best: dict
    metrics: pd.DataFrame
    importance: pd.DataFrame
    comparison: pd.DataFrame
    signs: pd.DataFrame


def read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def load_version(name: str, directory: Path, data: pd.DataFrame) -> Version:
    stats = read_json(directory / "dataset_stats.json")
    if stats["posts_output"] != len(data):
        raise ValueError(f"{name}: {len(data)} постов вместо {stats['posts_output']}")
    return Version(
        name,
        data,
        stats,
        read_json(directory / "best_config.json"),
        pd.read_csv(directory / "metrics.csv"),
        pd.read_csv(directory / "permutation_importance.csv"),
        keyword_comparison(data),
        channel_sign_tests(data),
    )


def count_rows(version: Version) -> list[str]:
    stats, parts = version.stats, period_parts(version.data)
    young = stats.get("posts_young_excluded")
    return [
        format_ru(stats["posts_input"]),
        "не применялось" if young is None else format_ru(young),
        str(stats["groups_total"]),
        str(stats["groups_excluded"]),
        format_ru(stats["posts_excluded"]),
        format_ru(len(version.data)),
        format_ru(len(parts["обучение"])),
        format_ru(len(parts["проверка"])),
        f"{version.data['local_day'].min()} — {version.data['local_day'].max()}",
    ]


def posts_table(v1: Version, v2: Version) -> str:
    labels = [
        "постов на входе",
        f"исключено постов моложе {MIN_AGE_DAYS} дней на момент сбора",
        "групп для медианы",
        f"исключено групп меньше {MIN_GROUP_SIZE} постов",
        "исключено постов в этих группах",
        "постов в анализе",
        "обучение, постов",
        "проверка, постов",
        "период (local_day)",
    ]
    return markdown_table(
        pd.DataFrame({"показатель": labels, "v1": count_rows(v1), "v2": count_rows(v2)})
    )


def balance_rows(version: Version) -> pd.DataFrame:
    rows = [
        {"цель": target, "часть": name, "n": len(part), "share": part[target].mean()}
        for target in TARGETS
        for name, part in period_parts(version.data).items()
    ]
    return pd.DataFrame(rows)


def balance_table(v1: Version, v2: Version) -> str:
    first, second = balance_rows(v1), balance_rows(v2)
    return markdown_table(
        pd.DataFrame(
            {
                "цель": first["цель"],
                "часть": first["часть"],
                "v1: n": first["n"].map(format_ru),
                "v1: доля класса 1": first["share"].map(share),
                "v2: n": second["n"].map(format_ru),
                "v2: доля класса 1": second["share"].map(share),
            }
        )
    )


def merged_metrics(v1: Version, v2: Version, target: str) -> pd.DataFrame:
    first = v1.metrics[v1.metrics["target"] == target]
    second = v2.metrics[v2.metrics["target"] == target]
    merged = first.merge(second, on=["feature_set", "model"], how="outer", suffixes=("_v1", "_v2"))
    order = merged["model"].map(list(MODEL_LABELS).index)
    return merged.assign(order=order).sort_values(["feature_set", "order"], ignore_index=True)


# Округление до 3 знаков до вывода, чтобы не печатать «-0,000».
def difference(value: float) -> str:
    return metric(round(value, 3) + 0.0)


def metrics_table(v1: Version, v2: Version, target: str) -> str:
    merged = merged_metrics(v1, v2, target)
    columns = {"набор": merged["feature_set"], "модель": merged["model"].map(MODEL_LABELS)}
    for column, label in (("cv_roc_auc", "ROC-AUC, CV"), ("roc_auc", "ROC-AUC")):
        columns[f"{label}: v1"] = merged[f"{column}_v1"].map(metric)
        columns[f"{label}: v2"] = merged[f"{column}_v2"].map(metric)
    columns["ROC-AUC: v2 − v1"] = (merged["roc_auc_v2"] - merged["roc_auc_v1"]).map(difference)
    for column, label in (("accuracy", "accuracy"), ("f1", "F1")):
        columns[f"{label}: v1"] = merged[f"{column}_v1"].map(metric)
        columns[f"{label}: v2"] = merged[f"{column}_v2"].map(metric)
    columns["параметры v1"] = merged["params_v1"].map(lambda p: "—" if pd.isna(p) else f"`{p}`")
    columns["параметры v2"] = merged["params_v2"].map(lambda p: "—" if pd.isna(p) else f"`{p}`")
    return markdown_table(pd.DataFrame(columns))


def best_line(version: Version) -> str:
    best = version.best
    return (
        f"- {version.name}: {MODEL_LABELS[best['model']]}, набор {best['feature_set']}, "
        f"`{json.dumps(best['params'])}`, ROC-AUC CV {metric(best['cv_roc_auc'])}."
    )


def importance_table(v1: Version, v2: Version) -> str:
    columns = {"№": range(1, N_TOP_FEATURES + 1)}
    for version in (v1, v2):
        top = version.importance.head(N_TOP_FEATURES)
        columns[f"{version.name}: признак"] = top["feature"].map(feature_label).tolist()
        columns[f"{version.name}: снижение ROC-AUC"] = (
            top["importance_mean"].map(lambda v: format_ru(v, 4)).tolist()
        )
    return markdown_table(pd.DataFrame(columns))


def interval_text(row: pd.Series) -> str:
    return f"{points(row['diff'])} [{points(row['ci_low'])}; {points(row['ci_high'])}]"


def gap_table(v1: Version, v2: Version) -> str:
    first, second = v1.comparison, v2.comparison
    return markdown_table(
        pd.DataFrame(
            {
                "определение": first["definition"].map(DEFINITION_LABELS),
                "слой": first["layer"].map(DIAGNOSTIC_LAYERS),
                "v1: разница, п. п. [95 % ДИ]": first.apply(interval_text, axis=1),
                "v2: разница, п. п. [95 % ДИ]": second.apply(interval_text, axis=1),
                "v1: p": first["p_value"].map(p_value),
                "v2: p": second["p_value"].map(p_value),
            }
        )
    )


def sign_text(row: pd.Series) -> str:
    return f"{row['negative']} из {row['channels']}, p = {p_value(row['p_value'])}"


def sign_table(v1: Version, v2: Version) -> str:
    return markdown_table(
        pd.DataFrame(
            {
                "слой": v1.signs["layer"].map(DIAGNOSTIC_LAYERS),
                "v1: каналов с разницей < 0": v1.signs.apply(sign_text, axis=1),
                "v2: каналов с разницей < 0": v2.signs.apply(sign_text, axis=1),
            }
        )
    )


def gap(version: Version, definition: str, layer: str) -> pd.Series:
    rows = version.comparison
    return rows[(rows["definition"] == definition) & (rows["layer"] == layer)].iloc[0]


def keyword_below(layer: str) -> Callable[[Version], tuple[bool, str]]:
    def check(version: Version) -> tuple[bool, str]:
        row = gap(version, "keyword_pass", layer)
        return bool(row["ci_high"] < 0), interval_text(row)

    return check


def government_gap_larger(definition: str) -> Callable[[Version], tuple[bool, str]]:
    def check(version: Version) -> tuple[bool, str]:
        government = gap(version, definition, "government")["diff"]
        blogger = gap(version, definition, "blogger")["diff"]
        text = f"{points(government)} против {points(blogger)}"
        return bool(government < blogger), text

    return check


def topic_rows(version: Version) -> pd.DataFrame:
    return version.comparison[version.comparison["definition"] != "keyword_pass"]


def topic_same_sign(version: Version) -> tuple[bool, str]:
    rows = topic_rows(version)
    return bool((rows["diff"] < 0).all()), f"{int((rows['diff'] < 0).sum())} из {len(rows)}"


def topic_significant(version: Version) -> tuple[bool, str]:
    rows = topic_rows(version)
    below = rows["ci_high"] < 0
    return bool(below.all()), f"{int(below.sum())} из {len(rows)}"


def sign_significant(layer: str) -> Callable[[Version], tuple[bool, str]]:
    def check(version: Version) -> tuple[bool, str]:
        row = version.signs[version.signs["layer"] == layer].iloc[0]
        return bool(row["p_value"] < SIGNIFICANCE), sign_text(row)

    return check


def best_is_v1_choice(version: Version) -> tuple[bool, str]:
    best = (version.best["model"], version.best["feature_set"])
    text = f"{MODEL_LABELS[best[0]]}, набор {best[1]}"
    return best == BEST_V1, text


def primary_metrics(version: Version) -> pd.DataFrame:
    return version.metrics[version.metrics["target"] == TARGETS[0]]


def models_above_chance(version: Version) -> tuple[bool, str]:
    rows = primary_metrics(version)
    models = rows[rows["model"] != BASELINE_MODEL]
    above = models["roc_auc"] > 0.5
    return bool(
        above.all()
    ), f"{int(above.sum())} из {len(models)}, мин. {metric(models['roc_auc'].min())}"


def set_b_better(version: Version) -> tuple[bool, str]:
    rows = primary_metrics(version)
    wide = rows.pivot(index="model", columns="feature_set", values="roc_auc").drop(BASELINE_MODEL)
    better = int((wide["B"] > wide["A"]).sum())
    return better > len(wide) / 2, f"{better} из {len(wide)}"


def top_three(version: Version) -> tuple[bool, str]:
    top = version.importance["feature"].head(3)
    return set(top) == TOP_THREE_V1, ", ".join(top.map(feature_label))


def forwards_above_baseline(version: Version) -> tuple[bool, str]:
    rows = version.metrics[version.metrics["target"] == TARGETS[1]]
    baseline = rows[rows["model"] == BASELINE_MODEL]["roc_auc"].iloc[0]
    best = rows[rows["model"] != BASELINE_MODEL]["roc_auc"].iloc[0]
    return bool(best > baseline), f"{metric(best)} против {metric(baseline)}"


def class_balance(version: Version) -> tuple[bool, str]:
    views, forwards = (version.data[target].mean() for target in TARGETS)
    inside = HIGH_VIEWS_BAND[0] <= views <= HIGH_VIEWS_BAND[1]
    inside &= HIGH_FORWARDS_BAND[0] <= forwards <= HIGH_FORWARDS_BAND[1]
    return bool(inside), f"{share(views)} и {share(forwards)}"


CLAIMS: list[tuple[str, Callable[[Version], tuple[bool, str]]]] = [
    ("Доля класса 1 (все посты): high_views 49–51 %, high_forwards 39–42 %", class_balance),
    (
        "keyword_pass = 1: доля high_views = 1 ниже, 95 % ДИ разницы не содержит 0, все посты",
        keyword_below("all"),
    ),
    (
        "keyword_pass = 1: доля high_views = 1 ниже, 95 % ДИ не содержит 0, государственные",
        keyword_below("government"),
    ),
    (
        "keyword_pass = 1: доля high_views = 1 ниже, 95 % ДИ не содержит 0, блогеры",
        keyword_below("blogger"),
    ),
    (
        "Разница по keyword_pass у государственных каналов больше (отрицательнее), чем у блогеров",
        government_gap_larger("keyword_pass"),
    ),
    (
        "Разница по p_on_topic (верхняя треть, общий порог) у государственных каналов больше, "
        "чем у блогеров",
        government_gap_larger("p_on_topic_top_global"),
    ),
    ("p_on_topic: разница того же знака во всех шести сравнениях", topic_same_sign),
    ("p_on_topic: 95 % ДИ не содержит 0 во всех шести сравнениях", topic_significant),
    ("Знаковый тест по каналам (keyword_pass) p < 0,05: все каналы", sign_significant("all")),
    (
        "Знаковый тест по каналам (keyword_pass) p < 0,05: государственные",
        sign_significant("government"),
    ),
    ("Знаковый тест по каналам (keyword_pass) p < 0,05: блогеры", sign_significant("blogger")),
    ("Лучшая конфигурация по CV — LightGBM, набор B", best_is_v1_choice),
    ("ROC-AUC на проверке выше 0,5 у всех моделей, кроме DummyClassifier", models_above_chance),
    ("Набор B лучше набора A по ROC-AUC на проверке у большинства моделей", set_b_better),
    ("Три главных признака — текст, день недели, час публикации", top_three),
    ("high_forwards: выбранная модель лучше DummyClassifier по ROC-AUC", forwards_above_baseline),
]


def verdict(first: bool, second: bool) -> str:
    if not first:
        return "в v1 не выполнялось"
    return "да" if second else "нет"


def claim_rows(v1: Version, v2: Version) -> pd.DataFrame:
    rows = []
    for statement, check in CLAIMS:
        (first, first_text), (second, second_text) = check(v1), check(v2)
        rows.append(
            {
                "утверждение": statement,
                "v1": first_text,
                "v2": second_text,
                "сохранилось": verdict(first, second),
            }
        )
    return pd.DataFrame(rows)


def claims_line(claims: pd.DataFrame) -> str:
    held = claims[claims["сохранилось"] != "в v1 не выполнялось"]
    lost = held.loc[held["сохранилось"] == "нет", "утверждение"]
    lost_text = "; ".join(f"«{statement}»" for statement in lost) or "нет"
    return (
        f"Из {len(held)} утверждений, выполнявшихся в v1, в v2 сохранились "
        f"{len(held) - len(lost)}. Не сохранились: {lost_text}."
    )


def quarter_shares(data: pd.DataFrame, group: tuple[str, ...]) -> pd.Series:
    quarters = relative_age_quarter(data, group)
    return data.groupby(quarters, observed=False)["high_views"].mean()


def drift_table(v1: Version, v2: Version) -> str:
    first = quarter_shares(v1.data, V1_GROUP)
    second = quarter_shares(v2.data, TARGET_GROUP)
    late = f"дни {LATE_DAY}–31 минус дни 1–{LATE_DAY - 1}, п. п."
    return markdown_table(
        pd.DataFrame(
            {
                "показатель": [*(f"доля high_views = 1: {q}" for q in QUARTERS), late],
                "v1": [*first.map(share), points(late_gap(v1.data))],
                "v2": [*second.map(share), points(late_gap(v2.data))],
            }
        )
    )


def definitions() -> str:
    return "\n".join(
        [
            "- v1: медиана внутри «канал × месяц», все посты.",
            f"- v2: посты от {MIN_AGE_DAYS} дней на момент сбора, медиана внутри "
            "«канал × месяц × прогон сбора».",
            f"- В обеих версиях: группы меньше {MIN_GROUP_SIZE} постов исключены, класс 1 — "
            "строго выше медианы; признаки, разбиение по времени, модели, сетки параметров "
            "и seed одинаковы.",
            "- Наборы постов, в том числе проверочная часть, в v1 и v2 разные, поэтому метрики "
            "посчитаны на разных постах.",
        ]
    )


def build_report(v1: Version, v2: Version) -> str:
    claims = claim_rows(v1, v2)
    sections = [
        "# Сравнение v1 и v2",
        definitions(),
        "## Посты",
        posts_table(v1, v2),
        "## Баланс классов",
        balance_table(v1, v2),
        "## Метрики моделей, цель high_views, проверочная часть",
        metrics_table(v1, v2, TARGETS[0]),
        "Выбранная конфигурация (максимальный средний ROC-AUC на CV):",
        "\n".join([best_line(v1), best_line(v2)]),
        "## Повтор выбранной конфигурации, цель high_forwards, проверочная часть",
        metrics_table(v1, v2, TARGETS[1]),
        f"## {N_TOP_FEATURES} главных признаков (перестановочная важность выбранной модели)",
        importance_table(v1, v2),
        "## Разница долей high_views = 1: экономические минус остальные",
        gap_table(v1, v2),
        "Знаковый тест по каналам, keyword_pass:",
        sign_table(v1, v2),
        "## Доля high_views = 1 по относительному возрасту внутри группы",
        "Четверти — по рангу age_days внутри группы медианы (v1: «канал × месяц», v2: «канал × "
        "месяц × прогон»). Подробнее: v1_target_diagnostics.md.",
        drift_table(v1, v2),
        "## Что сохранилось",
        claims_line(claims),
        "«в v1 не выполнялось» — утверждение было ложным уже в v1.",
        markdown_table(claims),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    v1 = load_version("v1", V1_DIR, v1_target(all_posts()))
    v2 = load_version("v2", RESULTS_DIR, pd.read_parquet(FEATURES_PATH))
    OUTPUT_PATH.write_text(build_report(v1, v2), "utf-8")
    print(f"Записано: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
