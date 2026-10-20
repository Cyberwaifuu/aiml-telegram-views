"""Строит таблицу признаков постов data/posts_features.parquet из data/corpus.parquet."""

import json
import re
from collections import Counter
from datetime import timedelta, timezone

import pandas as pd

from common import (
    COLLECTION_TIMES_PATH,
    CORPUS_PATH,
    DATASET_STATS_PATH,
    FEATURES_PATH,
    MIN_AGE_DAYS,
    MIN_GROUP_SIZE,
    TARGET_GROUP,
    TARGETS,
    TRAIN_END,
    feature_columns,
    keyword_flag_column,
)

EXPECTED_ROWS = 71_351
N_TOP_KEYWORDS = 30
ASTANA_TZ = timezone(timedelta(hours=5))
COLLECTION_COLUMNS = ["doc_id", "channel", "run_label", "age_days"]
SERVICE_COLUMNS = [
    "doc_id",
    "channel",
    "local_day",
    "month",
    "run_label",
    "age_days",
    "views",
    "forwards",
    "text",
]

URL_RE = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#\w+")
# Флаг страны состоит из двух региональных символов и считается как два эмодзи.
EMOJI_RE = re.compile("[\U0001f000-\U0001faff⌀-⏿☀-➿⬀-⯿]")
DIGIT_RE = re.compile(r"\d")
PERCENT_RE = re.compile(r"%|процент|пайыз", re.IGNORECASE)
# «евро» и «тенге» не склоняются, поэтому ищутся целым словом: иначе совпадут «Европа» и т. п.
CURRENCY_RE = re.compile(
    r"[₸$€₽¥]|\b(?:тенге|евро|еуро|kzt|usd|eur|rub|cny)\b|\b(?:теңге|доллар|рубл|юан)",
    re.IGNORECASE,
)


def load_posts() -> pd.DataFrame:
    corpus = pd.read_parquet(CORPUS_PATH)
    posts = corpus[corpus["source"] == "telegram_post"].reset_index(drop=True)
    if len(posts) != EXPECTED_ROWS:
        raise ValueError(f"Ожидалось {EXPECTED_ROWS} постов, получено {len(posts)}")
    return add_collection_times(posts)


def add_collection_times(posts: pd.DataFrame) -> pd.DataFrame:
    times = pd.read_parquet(COLLECTION_TIMES_PATH, columns=COLLECTION_COLUMNS)
    merged = posts.merge(times, on=["doc_id", "channel"], how="left", validate="one_to_one")
    if merged["run_label"].isna().any():
        raise ValueError("Не у всех постов есть время сбора")
    return merged


def add_time_columns(posts: pd.DataFrame) -> pd.DataFrame:
    local = pd.to_datetime(posts["date"], utc=True, format="ISO8601").dt.tz_convert(ASTANA_TZ)
    if (local.dt.strftime("%Y-%m-%d") != posts["local_day"]).any():
        raise ValueError("Дата в UTC+5 не совпадает с local_day")
    return posts.assign(
        hour=local.dt.hour,
        weekday=local.dt.dayofweek,
        month=posts["local_day"].str[:7],
    )


def drop_young_posts(posts: pd.DataFrame, min_age: float = MIN_AGE_DAYS) -> pd.DataFrame:
    return posts[posts["age_days"] >= min_age].reset_index(drop=True)


def drop_small_groups(
    posts: pd.DataFrame, group: tuple[str, ...] = TARGET_GROUP, min_size: int = MIN_GROUP_SIZE
) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = list(group)
    sizes = posts.groupby(columns)["doc_id"].transform("size")
    small = sizes < min_size
    stats = {
        "groups_total": int(posts.groupby(columns).ngroups),
        "groups_excluded": int(posts[small].groupby(columns).ngroups),
        "posts_excluded": int(small.sum()),
    }
    return posts[~small].reset_index(drop=True), stats


def add_targets(posts: pd.DataFrame, group: tuple[str, ...] = TARGET_GROUP) -> pd.DataFrame:
    targets = {}
    for target, column in zip(TARGETS, ("views", "forwards"), strict=True):
        median = posts.groupby(list(group))[column].transform("median")
        targets[target] = (posts[column] > median).astype("int8")
    return posts.assign(**targets)


# Строки pandas 3 хранятся в Arrow, а его движок регулярных выражений понимает \b и \w
# только для латиницы, поэтому поиск идёт через модуль re.
def count_matches(text: pd.Series, pattern: re.Pattern[str]) -> pd.Series:
    return text.map(lambda t: len(pattern.findall(t))).astype("int32")


def has_match(text: pd.Series, pattern: re.Pattern[str]) -> pd.Series:
    return text.map(lambda t: pattern.search(t) is not None).astype("int8")


def text_shape_features(text: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_chars": text.str.len(),
            "n_words": text.map(lambda t: len(t.split())),
            "n_lines": text.map(lambda t: sum(1 for line in t.splitlines() if line.strip())),
            "n_links": count_matches(text, URL_RE),
            "n_hashtags": count_matches(text, HASHTAG_RE),
            "n_emoji": count_matches(text, EMOJI_RE),
            "n_digits": count_matches(text, DIGIT_RE),
            "n_question": text.map(lambda t: t.count("?")),
            "n_exclaim": text.map(lambda t: t.count("!")),
            "has_percent": has_match(text, PERCENT_RE),
            "has_currency": has_match(text, CURRENCY_RE),
        }
    )


def post_keywords(posts: pd.DataFrame) -> pd.Series:
    return posts.apply(
        lambda row: set(row["matched_keywords_ru"]) | set(row["matched_keywords_kz"]), axis=1
    )


def top_train_keywords(keywords: pd.Series, local_day: pd.Series) -> list[str]:
    counts = Counter()
    for words in keywords[local_day <= TRAIN_END]:
        counts.update(words)
    return [word for word, _ in counts.most_common(N_TOP_KEYWORDS)]


def keyword_features(
    posts: pd.DataFrame, top: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    keywords = post_keywords(posts)
    top = top or top_train_keywords(keywords, posts["local_day"])
    flags = {
        keyword_flag_column(word): keywords.map(lambda found, w=word: w in found).astype("int8")
        for word in top
    }
    counts = {
        "n_kw_ru": posts["matched_keywords_ru"].map(len),
        "n_kw_kz": posts["matched_keywords_kz"].map(len),
    }
    return pd.DataFrame(counts | flags), top


def build_features(
    posts: pd.DataFrame, top: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    keyword_part, top = keyword_features(posts, top)
    base = posts.assign(keyword_pass=posts["keyword_pass"].astype("int8"))
    table = pd.concat([base, text_shape_features(posts["text"]), keyword_part], axis=1)
    columns = SERVICE_COLUMNS + list(TARGETS) + feature_columns(table)
    return table[columns], top


def main() -> None:
    posts = add_time_columns(load_posts())
    aged = drop_young_posts(posts)
    filtered, stats = drop_small_groups(aged)
    table, top = build_features(add_targets(filtered))
    table.to_parquet(FEATURES_PATH, index=False)

    stats = {
        "posts_input": len(posts),
        "min_age_days": MIN_AGE_DAYS,
        "posts_young_excluded": len(posts) - len(aged),
        "min_group_size": MIN_GROUP_SIZE,
        "target_group": list(TARGET_GROUP),
        **stats,
        "posts_output": len(table),
        "top_keywords": top,
    }
    DATASET_STATS_PATH.parent.mkdir(exist_ok=True)
    DATASET_STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "top_keywords"}, indent=2))
    print("Ключевые слова:", ", ".join(top))
    print(table[list(TARGETS)].mean().round(4).to_string())


if __name__ == "__main__":
    main()
