import re
from typing import Iterable

EMOJI_RE = re.compile(r"[\U0001F300-\U0001F6FF\U0001F700-\U0001FAFF\u2600-\u27BF\uFE0F]")


def _strip_emojis(text: str) -> str:
    return EMOJI_RE.sub(" ", text).lower()


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    text = _strip_emojis(text)
    text = _normalize_spaces(text)
    return text


def preprocess_many(texts: Iterable[str]) -> list[str]:
    return [preprocess_text(t) for t in texts]
