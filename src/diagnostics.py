"""Считает таблицы и график для раздела «Диагностика» в results/summary.md."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.image import AxesImage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import (
    FEATURES_PATH,
    LAYER_LABELS,
    RESULTS_DIR,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TRAIN_END,
    age_bucket,
    format_ru,
)
from plotting import AXIS, DIVERGING, GRID, apply_style, save_figure
from stats import channel_differences, compare_shares, sign_test_row
from term_text import boilerplate_lines, clean_texts

TARGET = "high_views"
ALL_LAYERS = "all"
LAYERS = ("government", "blogger")
LAYER_TITLES = {ALL_LAYERS: "Все посты"} | {k: v.capitalize() for k, v in LAYER_LABELS.items()}
WEEKDAYS = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")
MIN_CELL_POSTS = 30
TERM_LANGUAGES = ("ru", "kz")
N_TERMS = 20
TERM_MODEL_C = 0.1
TERM_MIN_DF = 50
TERM_MAX_DF = 0.2
TOP_FRACTION = 1 / 3
TOPIC_DEFINITIONS = ("keyword_pass", "p_on_topic_top_global", "p_on_topic_top_layer")


def layer_parts(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {ALL_LAYERS: data} | {layer: data[data["actor_type"] == layer] for layer in LAYERS}


def share_by(data: pd.DataFrame, column: str) -> pd.DataFrame:
    tables = []
    for layer, part in layer_parts(data).items():
        grouped = part.groupby(column, observed=False)[TARGET].agg(n="size", share="mean")
        tables.append(grouped.reset_index().rename(columns={column: "value"}).assign(layer=layer))
    return pd.concat(tables, ignore_index=True).assign(variable=column)


def time_shares(data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([share_by(data, "weekday"), share_by(data, "hour")], ignore_index=True)


def age_shares(data: pd.DataFrame) -> pd.DataFrame:
    return share_by(data.assign(age_bucket=age_bucket(data["age_days"])), "age_bucket")


def weekday_hour_grid(data: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for layer, part in layer_parts(data).items():
        cells = part.groupby(["weekday", "hour"])[TARGET].agg(n="size", share="mean")
        tables.append(cells.reset_index().assign(layer=layer))
    return pd.concat(tables, ignore_index=True)


def grid_matrix(cells: pd.DataFrame, values: str) -> pd.DataFrame:
    matrix = cells.pivot(index="weekday", columns="hour", values=values)
    return matrix.reindex(index=range(len(WEEKDAYS)), columns=range(24))


def plot_panel(ax: Axes, cells: pd.DataFrame, title: str, norm: Normalize) -> AxesImage:
    shares = grid_matrix(cells, "share").where(grid_matrix(cells, "n") >= MIN_CELL_POSTS)
    # Клетки с малым n не закрашиваются: сквозь них видна штриховка фона.
    ax.patch.set(facecolor="white", hatch="////", edgecolor=GRID)
    image = ax.imshow(shares, cmap=DIVERGING, norm=norm, aspect="auto")
    ax.grid(False)
    ax.set_yticks(range(len(WEEKDAYS)), WEEKDAYS)
    ax.set_xticks(range(0, 24, 2))
    ax.set_title(title, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_weekday_hour(grid: pd.DataFrame) -> None:
    enough = grid[grid["n"] >= MIN_CELL_POSTS]
    spread = float(np.abs(enough["share"] - 0.5).max())
    norm = TwoSlopeNorm(vcenter=0.5, vmin=0.5 - spread, vmax=0.5 + spread)
    fig, axes = plt.subplots(len(LAYER_TITLES), 1, figsize=(9, 8.5), sharex=True)
    images = [
        plot_panel(ax, grid[grid["layer"] == layer], title, norm)
        for ax, (layer, title) in zip(axes, LAYER_TITLES.items(), strict=True)
    ]
    axes[-1].set_xlabel("Час публикации (UTC+5)")
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.6)
    colorbar.outline.set_visible(False)
    colorbar.ax.yaxis.set_major_formatter(lambda value, _: f"{format_ru(100 * value)} %")
    colorbar.ax.tick_params(color=AXIS)
    colorbar.set_label("Доля high_views = 1")
    fig.suptitle(
        "Доля high_views = 1 по дню недели и часу публикации\n"
        f"штриховка: меньше {MIN_CELL_POSTS} постов в клетке",
        x=0.06,
        ha="left",
    )
    save_figure(fig, "weekday_hour_heatmap.png")


def term_coefficients(texts: pd.Series, target: pd.Series) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TERM_MIN_DF,
        max_df=TERM_MAX_DF,
        max_features=TFIDF_MAX_FEATURES,
    )
    matrix = vectorizer.fit_transform(texts)
    model = LogisticRegression(C=TERM_MODEL_C, max_iter=2000).fit(matrix, target)
    return pd.DataFrame(
        {
            "term": vectorizer.get_feature_names_out(),
            "coefficient": model.coef_[0],
            "posts": np.asarray((matrix > 0).sum(axis=0)).ravel(),
        }
    )


def extreme_terms(terms: pd.DataFrame) -> pd.DataFrame:
    ordered = terms.sort_values("coefficient", ascending=False, ignore_index=True)
    positive = ordered.head(N_TERMS).assign(direction="positive")
    negative = ordered.tail(N_TERMS).iloc[::-1].assign(direction="negative")
    return pd.concat([positive, negative]).assign(rank=list(range(1, N_TERMS + 1)) * 2)


def top_terms(train: pd.DataFrame, texts: pd.Series) -> pd.DataFrame:
    tables = []
    for language in TERM_LANGUAGES:
        part = train[train["language"] == language]
        coefficients = term_coefficients(texts[part.index], part[TARGET])
        terms = extreme_terms(coefficients)
        tables.append(
            terms.assign(language=language, train_posts=len(part), vocabulary=len(coefficients))
        )
    columns = [
        "language",
        "train_posts",
        "vocabulary",
        "direction",
        "rank",
        "term",
        "coefficient",
        "posts",
    ]
    return pd.concat(tables, ignore_index=True)[columns]


def term_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data[data["local_day"] <= TRAIN_END]
    boilerplate = boilerplate_lines(train)
    return top_terms(train, clean_texts(train, boilerplate)), boilerplate


def in_top_fraction(scores: pd.Series) -> pd.Series:
    return scores >= scores.quantile(1 - TOP_FRACTION)


def topic_flags(data: pd.DataFrame) -> pd.DataFrame:
    by_layer = data.groupby("actor_type")["p_on_topic"].transform(in_top_fraction)
    return pd.DataFrame(
        dict(
            zip(
                TOPIC_DEFINITIONS,
                (data["keyword_pass"].astype(bool), in_top_fraction(data["p_on_topic"]), by_layer),
                strict=True,
            )
        )
    )


def topic_threshold(data: pd.DataFrame, part: pd.DataFrame, definition: str, layer: str) -> float:
    if definition == "p_on_topic_top_global":
        return data["p_on_topic"].quantile(1 - TOP_FRACTION)
    if definition == "p_on_topic_top_layer" and layer != ALL_LAYERS:
        return part["p_on_topic"].quantile(1 - TOP_FRACTION)
    return np.nan


def keyword_comparison(data: pd.DataFrame) -> pd.DataFrame:
    flags = topic_flags(data)
    rows = [
        {
            "definition": definition,
            "layer": layer,
            "threshold": topic_threshold(data, part, definition, layer),
            **compare_shares(part[TARGET], flags.loc[part.index, definition]),
        }
        for definition in TOPIC_DEFINITIONS
        for layer, part in layer_parts(data).items()
    ]
    return pd.DataFrame(rows)


def channel_sign_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"layer": layer, **sign_test_row(channel_differences(part, TARGET, "keyword_pass"))}
        for layer, part in layer_parts(data).items()
    ]
    return pd.DataFrame(rows)


def monthly_medians(data: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for layer, part in layer_parts(data).items():
        medians = part.groupby("month").agg(
            n=("views", "size"),
            median_views=("views", "median"),
            median_forwards=("forwards", "median"),
            median_age_days=("age_days", "median"),
        )
        tables.append(medians.reset_index().assign(layer=layer))
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    apply_style()
    data = pd.read_parquet(FEATURES_PATH)
    grid = weekday_hour_grid(data)
    plot_weekday_hour(grid)
    terms, boilerplate = term_tables(data)
    tables = {
        "high_views_by_time.csv": time_shares(data),
        "high_views_weekday_hour.csv": grid,
        "high_views_by_age.csv": age_shares(data),
        "top_terms.csv": terms,
        "term_boilerplate.csv": boilerplate,
        "keyword_comparison.csv": keyword_comparison(data),
        "channel_sign_test.csv": channel_sign_tests(data),
        "monthly_medians.csv": monthly_medians(data),
    }
    for name, table in tables.items():
        table.to_csv(RESULTS_DIR / name, index=False, encoding="utf-8")
        print(f"Записано: {RESULTS_DIR / name}")


if __name__ == "__main__":
    main()
