from __future__ import annotations

import re
import string


_MULTISPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, lowercase: bool = True, strip_punctuation: bool = False) -> str:
    normalized = text.strip()
    if lowercase:
        normalized = normalized.lower()
    if strip_punctuation:
        normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = _MULTISPACE_RE.sub(" ", normalized)
    return normalized


def exact_match(prediction: str, target: str) -> bool:
    return prediction == target


def normalized_exact_match(
    prediction: str,
    target: str,
    lowercase: bool = True,
    strip_punctuation: bool = False,
) -> bool:
    return normalize_text(prediction, lowercase, strip_punctuation) == normalize_text(
        target,
        lowercase,
        strip_punctuation,
    )


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def char_error_rate(prediction: str, target: str) -> float:
    if not target:
        return 0.0 if not prediction else 1.0
    return levenshtein_distance(prediction, target) / len(target)
