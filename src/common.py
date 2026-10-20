"""Общие константы, списки признаков и проверки для всех скриптов проекта."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CORPUS_PATH = DATA_DIR / "corpus.parquet"
COLLECTION_TIMES_PATH = DATA_DIR / "collection_times.parquet"
FEATURES_PATH = DATA_DIR / "posts_features.parquet"
DATASET_STATS_PATH = RESULTS_DIR / "dataset_stats.json"

SEED = 42
TRAIN_END = "2026-06-30"
TEST_START = "2026-07-01"
TARGETS = ("high_views", "high_forwards")
# Просмотры сняты один раз, в одном из прогонов сбора, поэтому медиана берётся внутри прогона,
# а слишком молодые на момент сбора посты исключаются.
MIN_AGE_DAYS = 14
TARGET_GROUP = ("channel", "month", "run_label")
MIN_GROUP_SIZE = 20
AGE_BUCKETS = {
    "<7": (0, 7),
    "7-14": (7, 14),
    "14-30": (14, 30),
    "30-90": (30, 90),
    "90-180": (90, 180),
    ">180": (180, float("inf")),
}
FORBIDDEN_FEATURES = frozenset(
    {"views", "forwards", "replies_count", "channel", "doc_id", "age_days", "run_label"}
)
VALIDATION_MONTHS = ("2026-04", "2026-05", "2026-06")
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 20
TFIDF_MAX_FEATURES = 20_000
N_SVD = 100
N_PERMUTATION_REPEATS = 10

KEYWORD_FLAG_PREFIX = "kw_"
CATEGORICAL_FEATURES = ["actor_type", "language"]
CYCLIC_PERIODS = {"hour": 24, "weekday": 7}
COUNT_FEATURES = [
    "n_chars",
    "n_words",
    "n_lines",
    "n_links",
    "n_hashtags",
    "n_emoji",
    "n_digits",
    "n_question",
    "n_exclaim",
    "n_kw_ru",
    "n_kw_kz",
]
BINARY_FEATURES = ["has_percent", "has_currency", "keyword_pass"]
CONTINUOUS_FEATURES = ["p_on_topic"]
BASE_FEATURES = (
    CATEGORICAL_FEATURES
    + list(CYCLIC_PERIODS)
    + COUNT_FEATURES
    + BINARY_FEATURES
    + CONTINUOUS_FEATURES
)

FEATURE_LABELS = {
    "actor_type": "слой (actor_type)",
    "language": "язык",
    "hour": "час публикации",
    "weekday": "день недели",
    "n_chars": "длина, символов",
    "n_words": "длина, слов",
    "n_lines": "число строк",
    "n_links": "число ссылок",
    "n_hashtags": "число хэштегов",
    "n_emoji": "число эмодзи",
    "n_digits": "число цифр",
    "n_question": "число знаков «?»",
    "n_exclaim": "число знаков «!»",
    "n_kw_ru": "число ключевых слов (ru)",
    "n_kw_kz": "число ключевых слов (kz)",
    "has_percent": "есть проценты",
    "has_currency": "есть упоминание валюты",
    "keyword_pass": "keyword_pass",
    "p_on_topic": "p_on_topic",
    "text": "текст (TF-IDF + SVD)",
}
LAYER_LABELS = {"government": "государственные", "blogger": "блогеры"}
MODEL_LABELS = {
    "DummyClassifier": "DummyClassifier",
    "LogisticRegression": "логистическая регрессия",
    "kNN": "kNN",
    "DecisionTree": "дерево решений",
    "RandomForest": "случайный лес",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "MLP": "MLPClassifier",
}


def format_ru(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def age_bucket(age_days: pd.Series) -> pd.Series:
    edges = [low for low, _ in AGE_BUCKETS.values()] + [float("inf")]
    return pd.cut(age_days, edges, right=False, labels=list(AGE_BUCKETS))


def keyword_flag_column(keyword: str) -> str:
    return KEYWORD_FLAG_PREFIX + keyword.replace(" ", "_")


def feature_label(name: str) -> str:
    if name.startswith(KEYWORD_FLAG_PREFIX):
        return f"ключевое слово «{name.removeprefix(KEYWORD_FLAG_PREFIX).replace('_', ' ')}»"
    return FEATURE_LABELS.get(name, name)


def check_no_forbidden(columns: list[str]) -> None:
    leaked = FORBIDDEN_FEATURES.intersection(columns)
    if leaked:
        raise ValueError(f"В признаках запрещённые колонки: {sorted(leaked)}")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    flags = [c for c in frame.columns if c.startswith(KEYWORD_FLAG_PREFIX)]
    columns = BASE_FEATURES + flags
    check_no_forbidden(columns)
    return columns
