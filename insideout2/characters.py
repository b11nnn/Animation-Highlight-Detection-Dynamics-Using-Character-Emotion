from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from insideout2.config import DATA_DIR

CharacterProfile = dict[str, Any]

DEFAULT_CHARACTERS: dict[str, CharacterProfile] = {
    "Joy": {
        "korean": "기쁨이",
        "expected_emotion": "happy",
        "queries": [
            "Joy from Inside Out 2",
            "blue hair yellow dress cartoon character",
            "yellow happy emotion character with blue hair",
        ],
    },
    "Sadness": {
        "korean": "슬픔이",
        "expected_emotion": "sad",
        "queries": [
            "Sadness from Inside Out 2",
            "blue sad cartoon character with glasses",
            "blue emotion character with glasses and sweater",
        ],
    },
    "Anger": {
        "korean": "버럭이",
        "expected_emotion": "angry",
        "queries": [
            "Anger from Inside Out 2",
            "red angry cartoon character",
            "red square angry emotion character",
        ],
    },
    "Fear": {
        "korean": "소심이",
        "expected_emotion": "fearful",
        "queries": [
            "Fear from Inside Out 2",
            "purple fearful cartoon character",
            "purple scared emotion character",
        ],
    },
    "Disgust": {
        "korean": "까칠이",
        "expected_emotion": "disgusted",
        "queries": [
            "Disgust from Inside Out 2",
            "green disgusted cartoon character",
            "green emotion character disgust face",
        ],
    },
    "Anxiety": {
        "korean": "불안이",
        "expected_emotion": "fearful",
        "queries": [
            "Anxiety from Inside Out 2",
            "orange anxious cartoon character Inside Out 2",
            "orange nervous emotion character",
        ],
    },
    "Envy": {
        "korean": "부럽이",
        "expected_emotion": "angry",
        "queries": [
            "Envy from Inside Out 2",
            "small teal jealous cartoon character Inside Out 2",
            "turquoise envy emotion character",
        ],
    },
    "Ennui": {
        "korean": "따분이",
        "expected_emotion": "neutral",
        "queries": [
            "Ennui from Inside Out 2",
            "purple bored cartoon character Inside Out 2",
            "bored tired emotion character",
        ],
    },
    "Embarrassment": {
        "korean": "당황이",
        "expected_emotion": "fearful",
        "queries": [
            "Embarrassment from Inside Out 2",
            "large pink embarrassed cartoon character Inside Out 2",
            "pink shy emotion character with hoodie",
        ],
    },
}

_KOREAN_NAMES = {
    "Joy": "기쁨이",
    "Sadness": "슬픔이",
    "Anger": "버럭이",
    "Fear": "소심이",
    "Disgust": "까칠이",
    "Anxiety": "불안이",
    "Envy": "부럽이",
    "Ennui": "따분이",
    "Embarrassment": "당황이",
}


def load_characters_from_csv(csv_path: Path | None = None) -> dict[str, CharacterProfile]:
    """data/character_queries.csv → CHARACTERS 딕셔너리."""
    path = csv_path or DATA_DIR / "character_queries.csv"
    if not path.exists():
        return dict(DEFAULT_CHARACTERS)

    characters: dict[str, CharacterProfile] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Character"].strip()
            queries = [
                row["OWL-ViT Query 1"].strip(),
                row["OWL-ViT Query 2"].strip(),
            ]
            queries = [q for q in queries if q]
            base = DEFAULT_CHARACTERS.get(name, {})
            characters[name] = {
                "korean": base.get("korean", _KOREAN_NAMES.get(name, name)),
                "expected_emotion": base.get("expected_emotion", "neutral"),
                "queries": queries or base.get("queries", [name]),
            }
    return characters
