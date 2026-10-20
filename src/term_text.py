"""Очистка текстов постов для таблиц терминов: ссылки, @, #, подписи и названия каналов."""

import re

import pandas as pd

from common import DATA_DIR

URL_RE = re.compile(
    r"https?://\S+|www\.\S+|\b(?:t\.me|youtu\.be)/\S*"
    r"|\b[\w.-]+\.(?:kz|com|ru|media|me|org|net|io|tv|news)\b(?:/\S*)?",
    re.IGNORECASE,
)
HANDLE_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
SOCIAL_RE = re.compile(
    r"telegram|телеграм|instagram|инстаграм|facebook|фейсбук|youtube|ютуб|tiktok|тикток"
    r"|threads|twitter|вконтакте|whatsapp",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[^\W_]+")
# Строка считается подписью канала, если повторяется не реже чем в 5 % постов канала
# или не менее чем в 50 постах (столько же, сколько min_df у TF-IDF терминов).
BOILERPLATE_SHARE = 0.05
BOILERPLATE_POSTS = 50
GENERIC_HANDLE_PARTS = frozenset({"resmi", "official", "news", "finance", "business"})
# Названия каналов, которые не выводятся из их хэндлов, по одному в строке.
CHANNEL_NAME_EXTRAS_PATH = DATA_DIR / "channel_name_extras.txt"


def channel_name_extras() -> list[str]:
    lines = CHANNEL_NAME_EXTRAS_PATH.read_text("utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def strip_marks(text: str) -> str:
    return HASHTAG_RE.sub(" ", HANDLE_RE.sub(" ", URL_RE.sub(" ", text)))


def line_key(line: str) -> str:
    return " ".join(WORD_RE.findall(strip_marks(line).lower()))


def post_line_keys(posts: pd.DataFrame) -> pd.DataFrame:
    lines = posts[["channel", "text"]].assign(line=posts["text"].str.split("\n")).explode("line")
    keys = lines.assign(key=lines["line"].map(line_key)).reset_index(names="post")
    return keys[keys["key"] != ""].drop_duplicates(["post", "key"])


def boilerplate_lines(posts: pd.DataFrame) -> pd.DataFrame:
    keys = post_line_keys(posts)
    counts = keys.groupby(["channel", "key"]).size().rename("posts").reset_index()
    totals = posts.groupby("channel").size()
    shares = counts["posts"] / counts["channel"].map(totals)
    recurring = (shares >= BOILERPLATE_SHARE) | (counts["posts"] >= BOILERPLATE_POSTS)
    return counts[recurring].assign(share=shares[recurring]).reset_index(drop=True)


def channel_names(channels: pd.Series) -> list[str]:
    names = set(channel_name_extras())
    for handle in channels.str.lstrip("@").str.lower().unique():
        parts = {part for part in handle.split("_") if len(part) >= 4}
        names |= {handle, handle.replace("_", "")} | (parts - GENERIC_HANDLE_PARTS)
    return sorted(names, key=len, reverse=True)


def channel_name_pattern(channels: pd.Series) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(name) for name in channel_names(channels))
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


def clean_text(text: str, recurring: set[str], names: re.Pattern[str]) -> str:
    kept = [
        strip_marks(line)
        for line in text.split("\n")
        if not SOCIAL_RE.search(line) and line_key(line) not in recurring
    ]
    return names.sub(" ", "\n".join(kept))


def clean_texts(posts: pd.DataFrame, boilerplate: pd.DataFrame) -> pd.Series:
    names = channel_name_pattern(posts["channel"])
    by_channel = boilerplate.groupby("channel")["key"].agg(set).to_dict()
    cleaned = [
        clean_text(text, by_channel.get(channel, set()), names)
        for channel, text in zip(posts["channel"], posts["text"], strict=True)
    ]
    return pd.Series(cleaned, index=posts.index)
