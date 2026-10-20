"""Форматирование чисел и markdown-таблиц для отчётов в results/."""

import re

import pandas as pd

from common import LAYER_LABELS, TEST_START, TRAIN_END, format_ru

DIAGNOSTIC_LAYERS = {"all": "всего", **LAYER_LABELS}
WEEKDAY_NAMES = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)
DEFINITION_LABELS = {
    "keyword_pass": "keyword_pass = 1 против 0",
    "p_on_topic_top_global": "p_on_topic: верхняя треть, общий порог",
    "p_on_topic_top_layer": "p_on_topic: верхняя треть, порог внутри слоя",
}
NUMBER_CELL_RE = re.compile(r"^[\d\s,%—−+;\[\]e-]+$")


def share(value: float) -> str:
    return "—" if pd.isna(value) else f"{format_ru(100 * value, 1)} %"


def points(value: float) -> str:
    return format_ru(100 * value, 2)


def p_value(value: float) -> str:
    return format_ru(value, 3) if value >= 0.001 else f"{value:.1e}".replace(".", ",")


def metric(value: float) -> str:
    return "—" if pd.isna(value) else format_ru(value, 3)


# Медиана целых чисел бывает полуцелой; дробная часть показывается, чтобы округление её не прятало.
def median_text(value: float) -> str:
    return format_ru(value, 1) if value % 1 else format_ru(value)


def interval_text(low: float, high: float) -> str:
    return f"[{points(low)}; {points(high)}]"


def column_rule(values: pd.Series) -> str:
    is_number = values.astype(str).map(lambda cell: bool(NUMBER_CELL_RE.match(cell))).all()
    return "---:" if is_number else "---"


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join(column_rule(frame[column]) for column in frame.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def period_parts(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "все посты": data,
        "обучение": data[data["local_day"] <= TRAIN_END],
        "проверка": data[data["local_day"] >= TEST_START],
    }


def layered_share_table(
    frame: pd.DataFrame, first_column: str, order: list[object], labels: list[str]
) -> str:
    counts = frame.pivot(index="value", columns="layer", values="n").reindex(order)
    shares = frame.pivot(index="value", columns="layer", values="share").reindex(order)
    columns = {first_column: labels}
    for layer, name in DIAGNOSTIC_LAYERS.items():
        columns[f"{name}: n"] = counts[layer].fillna(0).map(format_ru).tolist()
        columns[f"{name}: доля"] = shares[layer].map(share).tolist()
    return markdown_table(pd.DataFrame(columns))
