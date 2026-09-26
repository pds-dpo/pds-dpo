"""Versioned MMHal parser correction; never chooses an arbitrary last/highest score."""
import re

PARSER_VERSION = "explicit-final-rating-v1"


def rating(content: str) -> int:
    # Markdown emphasis is presentation, not part of the rating label.
    normalized = content.replace("**", "").replace("__", "")
    finals = re.findall(r"(?im)^\s*(?:[-*]\s+|#{1,6}\s+)?final\s+rating\s*:\s*([^\r\n]*)", normalized)
    if finals:
        if len(finals) != 1:
            raise ValueError("expected a single explicit final rating")
        match = re.fullmatch(r"([0-6])(?:\s*[,;:\u2014-]\s*[^\r\n]+|\s*\.)?\s*", finals[0])
        if not match or re.search(r"(?i)\brating\s*:", finals[0]):
            raise ValueError("invalid or ambiguous explicit final rating")
        return int(match.group(1))
    # Preserve the legacy rule for judgments without an explicit final verdict.
    found = {int(value) for value in re.findall(r"(?i)rating\s*:\s*([0-6])\b", content)}
    if len(found) != 1:
        raise ValueError(f"expected exactly one rating, found {sorted(found)}")
    return found.pop()
