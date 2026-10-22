"""Таблица 1 статьи: состав выборки по слоям и доля класса 1 (high_views = 1).

Для государственных каналов, блогеров и всех постов: число каналов, посты по языкам, всего
постов, медианы просмотров и пересылок. Для всех постов: число и доля класса 1 во всей выборке,
в обучающей и в проверочной частях. Читает data/posts_features.parquet, пишет article/table1.csv.
"""

import pandas as pd

from common import FEATURES_PATH, ROOT, TEST_START, TRAIN_END

ARTICLE_DIR = ROOT / "article"
TABLE_PATH = ARTICLE_DIR / "table1.csv"
TARGET = "high_views"
LAYERS = ("government", "blogger")
LANGUAGES = ("ru", "kz", "parallel", "mixed", "other")
COLUMNS = ["channel", "actor_type", "language", "local_day", "views", "forwards", TARGET]


def load_posts() -> pd.DataFrame:
    posts = pd.read_parquet(FEATURES_PATH, columns=COLUMNS)
    unknown = set(posts["language"]) - set(LANGUAGES)
    if unknown:
        raise ValueError(f"Неизвестные значения language: {sorted(unknown)}")
    return posts


def layer_parts(posts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {layer: posts[posts["actor_type"] == layer] for layer in LAYERS} | {"all": posts}


def period_parts(posts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "all": posts,
        "train": posts[posts["local_day"] <= TRAIN_END],
        "test": posts[posts["local_day"] >= TEST_START],
    }


def layer_summary(posts: pd.DataFrame) -> dict[str, float]:
    languages = posts["language"].value_counts().reindex(LANGUAGES, fill_value=0)
    return {
        "channels": posts["channel"].nunique(),
        **{f"posts_{language}": int(n) for language, n in languages.items()},
        "posts_total": len(posts),
        "median_views": posts["views"].median(),
        "median_forwards": posts["forwards"].median(),
    }


def class_summary(posts: pd.DataFrame) -> dict[str, float]:
    rows = {}
    for period, part in period_parts(posts).items():
        rows[f"posts_{period}"] = len(part)
        rows[f"class1_{period}"] = int(part[TARGET].sum())
        rows[f"class1_share_{period}"] = part[TARGET].mean()
    return rows


# Series с dtype=object не даёт pandas привести счётчики к float рядом с медианами и долями.
def column(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, dtype=object)


def table1(posts: pd.DataFrame) -> pd.DataFrame:
    layers = {name: column(layer_summary(part)) for name, part in layer_parts(posts).items()}
    classes = {"all": column(class_summary(posts))}
    return pd.concat([pd.DataFrame(layers), pd.DataFrame(classes)]).rename_axis("row")


def main() -> None:
    table = table1(load_posts())
    ARTICLE_DIR.mkdir(exist_ok=True)
    table.to_csv(TABLE_PATH, encoding="utf-8")
    print(table.to_string())


if __name__ == "__main__":
    main()
