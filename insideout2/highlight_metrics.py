"""
Inside Out 2 하이라이트 지표 연산 (웹 서버용).

노트북 `(260603) insideout2.ipynb`의 다음 로직을 단일 진입점으로 통합했습니다.
- `calculate_highlight_scores` (슬라이딩 윈도우 데모 점수)
- `make_enhanced_highlight_dataset` (5초 구간 특성 + 시간 맥락)

표준 라이브러리만 사용합니다 (numpy/pandas 불필요).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

DEFAULT_EMOTIONS: tuple[str, ...] = (
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised",
    "neutral",
)

TIME_COLUMN_CANDIDATES: tuple[str, ...] = (
    "second",
    "time_sec",
    "timestamp",
    "frame_time",
)

# valence: 부정(-1) ~ 긍정(+1), arousal: 차분(0) ~ 격앙(1)
EMOTION_COORDS: dict[str, tuple[float, float]] = {
    "happy": (1.0, 0.7),
    "sad": (-0.8, 0.3),
    "angry": (-0.9, 1.0),
    "fearful": (-0.8, 0.9),
    "disgusted": (-0.7, 0.6),
    "surprised": (0.1, 1.0),
    "neutral": (0.0, 0.0),
}


@dataclass(frozen=True)
class HighlightMetricsConfig:
    """하이라이트 지표 계산 파라미터."""

    segment_sec: int = 5
    highlight_threshold: float = 0.6
    high_mismatch_threshold: float = 0.6
    temporal_context_sec: int = 15
    emotions: tuple[str, ...] = DEFAULT_EMOTIONS
    # calculate_highlight_scores 데모 공식 가중치
    window_size: int = 10
    window_step: int = 5
    demo_weights: tuple[float, float, float, float] = (0.45, 0.25, 0.20, 0.10)


@dataclass
class HighlightMetricsResult:
    """계산 결과."""

    segments: list[dict[str, Any]]
    sliding_windows: list[dict[str, Any]] = field(default_factory=list)


def compute_highlight_metrics(
    detections: Sequence[Mapping[str, Any]],
    youtube_replay_by_second: Mapping[int, float] | Sequence[float] | None = None,
    *,
    video_id: str = "",
    config: HighlightMetricsConfig | None = None,
    include_sliding_windows: bool = False,
) -> HighlightMetricsResult:
    """
    웹 서버용 단일 진입점: 감정 탐지 결과 → 구간별 하이라이트 지표.

    Parameters
    ----------
    detections
        프레임/초 단위 탐지 레코드. 필수 키:
        character, expected_emotion, predicted_emotion, mismatch_score
        시간 키(하나): second | time_sec | timestamp | frame_time
        권장: emotion_confidence, prob_{emotion}
    youtube_replay_by_second
        초 단위 YouTube replay 점수.
        dict {second: score} 또는 duration 길이의 sequence.
    video_id
        응답 메타데이터용 영상 ID.
    config
        구간 길이, 임계값 등.
    include_sliding_windows
        True면 노트북 `calculate_highlight_scores` 슬라이딩 윈도우 결과도 포함.

    Returns
    -------
    HighlightMetricsResult
        segments: 5초(기본) 구간 특성 + is_highlight
        sliding_windows: (옵션) 데모 highlight_score 윈도우
    """
    cfg = config or HighlightMetricsConfig()
    normalized = _normalize_detections(detections, cfg.emotions)
    youtube_by_second = _normalize_youtube_scores(youtube_replay_by_second)

    segments = _build_segment_rows(
        normalized,
        youtube_by_second,
        video_id=video_id,
        cfg=cfg,
    )
    _apply_temporal_context(segments, cfg)

    sliding_windows: list[dict[str, Any]] = []
    if include_sliding_windows:
        sliding_windows = _build_sliding_window_scores(normalized, cfg)

    return HighlightMetricsResult(segments=segments, sliding_windows=sliding_windows)


def segments_to_jsonable(result: HighlightMetricsResult) -> dict[str, Any]:
    """FastAPI/Flask 응답용 JSON 직렬화 헬퍼."""
    return {
        "segments": [_json_sanitize_row(row) for row in result.segments],
        "sliding_windows": [_json_sanitize_row(row) for row in result.sliding_windows],
    }


def compute_highlight_metrics_from_dataframe(
    emotion_df: Any,
    youtube_second_df: Any | None = None,
    *,
    video_id: str = "",
    config: HighlightMetricsConfig | None = None,
    include_sliding_windows: bool = False,
) -> HighlightMetricsResult:
    """노트북 DataFrame 파이프라인과 호환되는 래퍼 (pandas 필요)."""
    detections = emotion_df.to_dict(orient="records")
    youtube_map: dict[int, float] | None = None
    if youtube_second_df is not None:
        youtube_map = {
            int(row["second"]): float(row["youtube_replay_score"])
            for row in youtube_second_df.to_dict(orient="records")
        }
    return compute_highlight_metrics(
        detections,
        youtube_map,
        video_id=video_id,
        config=config,
        include_sliding_windows=include_sliding_windows,
    )


# ---------------------------------------------------------------------------
# Detection normalization
# ---------------------------------------------------------------------------


def _normalize_detections(
    detections: Sequence[Mapping[str, Any]],
    emotions: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not detections:
        return []

    sample = detections[0]
    time_col = next((c for c in TIME_COLUMN_CANDIDATES if c in sample), None)
    if time_col is None:
        raise ValueError(
            f"No time column found. Expected one of {TIME_COLUMN_CANDIDATES}. "
            f"Got keys: {list(sample.keys())}"
        )

    required = {"character", "expected_emotion", "predicted_emotion", "mismatch_score"}
    if not required.issubset(sample.keys()):
        raise ValueError(
            f"detections must contain {required}. Got keys: {list(sample.keys())}"
        )

    rows: list[dict[str, Any]] = []
    for raw in detections:
        row = dict(raw)
        row["time_sec"] = float(raw[time_col])
        row["emotion_confidence"] = float(raw.get("emotion_confidence", 0.0))
        row["mismatch_score"] = float(raw["mismatch_score"])
        row["expected_emotion_probability"] = _probability_for_emotion(
            row, str(row["expected_emotion"])
        )
        row["emotion_intensity_score"] = _emotion_intensity(row, emotions)
        row["emotion_distance_score"] = _emotion_distance(
            str(row["expected_emotion"]), str(row["predicted_emotion"])
        )
        row["valence_flip"] = _valence_flip(
            str(row["expected_emotion"]), str(row["predicted_emotion"])
        )
        row["arousal_change_score"] = _arousal_change(
            str(row["expected_emotion"]), str(row["predicted_emotion"])
        )
        rows.append(row)

    rows.sort(key=lambda r: r["time_sec"])
    return rows


def _normalize_youtube_scores(
    youtube_replay_by_second: Mapping[int, float] | Sequence[float] | None,
) -> dict[int, float]:
    if youtube_replay_by_second is None:
        return {}
    if isinstance(youtube_replay_by_second, Mapping):
        return {int(k): float(v) for k, v in youtube_replay_by_second.items()}
    return {i: float(v) for i, v in enumerate(youtube_replay_by_second)}


# ---------------------------------------------------------------------------
# Segment-level metrics (make_enhanced_highlight_dataset)
# ---------------------------------------------------------------------------


def _build_segment_rows(
    detections: list[dict[str, Any]],
    youtube_by_second: dict[int, float],
    *,
    video_id: str,
    cfg: HighlightMetricsConfig,
) -> list[dict[str, Any]]:
    if detections:
        max_detection_time = max(d["time_sec"] for d in detections)
    else:
        max_detection_time = 0.0
    max_youtube_time = max(youtube_by_second.keys(), default=0)
    duration_sec = int(math.ceil(max(max_detection_time, max_youtube_time)))
    if duration_sec <= 0:
        return []

    total_characters = max(1, len({d["character"] for d in detections}))
    total_emotions = max(1, len(cfg.emotions))
    max_entropy = math.log2(total_emotions)

    segments: list[dict[str, Any]] = []
    prev_emotion_dist: dict[str, float] | None = None
    character_emotion_history: dict[str, Counter[str]] = {}

    for start in range(0, duration_sec, cfg.segment_sec):
        end = start + cfg.segment_sec
        seg = [d for d in detections if start <= d["time_sec"] < end]
        youtube_score = _mean_youtube_score(youtube_by_second, start, end)

        row: dict[str, Any] = {
            "video_id": video_id,
            "segment_id": start // cfg.segment_sec + 1,
            "segment_sec": cfg.segment_sec,
            "start_time": start,
            "end_time": end,
            "character": None,
            "expected_emotion": None,
            "predicted_emotion": None,
            "detection_count": len(seg),
            "character_count": 0,
            "emotion_count": 0,
            "mismatch_score": 0.0,
            "entropy_score": 0.0,
            "transition_score": 0.0,
            "diversity_score": 0.0,
            "emotion_confidence_score": 0.0,
            "expected_emotion_probability": 0.0,
            "emotion_intensity_score": 0.0,
            "emotion_distance_score": 0.0,
            "valence_flip_rate": 0.0,
            "arousal_change_score": 0.0,
            "emotion_change_count": 0,
            "emotion_change_rate": 0.0,
            "character_emotion_rarity_score": 0.0,
            "youtube_highlight_score": youtube_score,
            "is_highlight": (
                int(youtube_score >= cfg.highlight_threshold)
                if youtube_score is not None and not math.isnan(youtube_score)
                else 0
            ),
        }

        if not seg:
            segments.append(row)
            prev_emotion_dist = None
            continue

        main_character = _mode([d["character"] for d in seg])
        main_seg = [d for d in seg if d["character"] == main_character]

        row["character"] = main_character
        row["expected_emotion"] = _mode([d["expected_emotion"] for d in main_seg])
        row["predicted_emotion"] = _mode([d["predicted_emotion"] for d in seg])
        row["character_count"] = len({d["character"] for d in seg})
        row["emotion_count"] = len({d["predicted_emotion"] for d in seg})
        row["mismatch_score"] = _mean_field(seg, "mismatch_score")
        row["emotion_confidence_score"] = _mean_field(seg, "emotion_confidence")
        row["expected_emotion_probability"] = _mean_field(
            seg, "expected_emotion_probability"
        )
        row["emotion_intensity_score"] = _mean_field(seg, "emotion_intensity_score")
        row["emotion_distance_score"] = _mean_field(seg, "emotion_distance_score")
        row["valence_flip_rate"] = _mean_field(seg, "valence_flip")
        row["arousal_change_score"] = _mean_field(seg, "arousal_change_score")

        emotion_counts = Counter(d["predicted_emotion"] for d in seg)
        raw_entropy = _entropy_from_counts(list(emotion_counts.values()))
        row["entropy_score"] = (
            raw_entropy / max_entropy if emotion_counts and max_entropy > 0 else 0.0
        )
        row["diversity_score"] = _clip(
            (row["character_count"] / total_characters + row["emotion_count"] / total_emotions)
            / 2,
            0.0,
            1.0,
        )

        total_in_seg = sum(emotion_counts.values())
        current_emotion_dist = {
            emo: count / total_in_seg for emo, count in emotion_counts.items()
        }
        if prev_emotion_dist is not None:
            all_emotions = sorted(
                set(current_emotion_dist) | set(prev_emotion_dist)
            )
            transition = sum(
                abs(current_emotion_dist.get(e, 0.0) - prev_emotion_dist.get(e, 0.0))
                for e in all_emotions
            )
            row["transition_score"] = transition / 2
        prev_emotion_dist = current_emotion_dist

        change_count = 0
        by_character: dict[str, list[dict[str, Any]]] = {}
        for d in seg:
            by_character.setdefault(d["character"], []).append(d)
        for char_rows in by_character.values():
            char_rows.sort(key=lambda r: r["time_sec"])
            emotions = [r["predicted_emotion"] for r in char_rows]
            change_count += sum(
                emotions[i] != emotions[i - 1] for i in range(1, len(emotions))
            )
        row["emotion_change_count"] = change_count
        row["emotion_change_rate"] = change_count / max(1, len(seg) - row["character_count"])

        rarity_values: list[float] = []
        for d in seg:
            history = character_emotion_history.get(d["character"], Counter())
            history_count = sum(history.values())
            if history_count:
                rarity_values.append(
                    1.0 - history[d["predicted_emotion"]] / history_count
                )
            else:
                rarity_values.append(0.0)
        row["character_emotion_rarity_score"] = mean(rarity_values) if rarity_values else 0.0

        for d in seg:
            history = character_emotion_history.setdefault(d["character"], Counter())
            history[d["predicted_emotion"]] += 1

        segments.append(row)

    return segments


def _apply_temporal_context(segments: list[dict[str, Any]], cfg: HighlightMetricsConfig) -> None:
    """구간 간 시간 맥락 특성 (과거·현재만 사용, 미래 누수 없음)."""
    if not segments:
        return

    context_segments = max(1, int(math.ceil(cfg.temporal_context_sec / cfg.segment_sec)))

    mismatch_scores = [s["mismatch_score"] for s in segments]
    entropy_scores = [s["entropy_score"] for s in segments]
    predicted_emotions = [s["predicted_emotion"] for s in segments]

    consecutive_duration = 0
    for i, seg in enumerate(segments):
        if i > 0:
            seg["mismatch_delta"] = mismatch_scores[i] - mismatch_scores[i - 1]
            seg["entropy_delta"] = entropy_scores[i] - entropy_scores[i - 1]
        else:
            seg["mismatch_delta"] = 0.0
            seg["entropy_delta"] = 0.0

        prev_emo = predicted_emotions[i - 1] if i > 0 else None
        cur_emo = predicted_emotions[i]
        seg["dominant_emotion_changed"] = int(
            cur_emo is not None and prev_emo is not None and cur_emo != prev_emo
        )

        past_slice = mismatch_scores[max(0, i - context_segments) : i]
        past_mean = mean(past_slice) if past_slice else 0.0
        seg["surprise_spike"] = max(0.0, mismatch_scores[i] - past_mean)

        roll_slice = mismatch_scores[max(0, i - context_segments + 1) : i + 1]
        seg["rolling_mismatch_mean_15sec"] = mean(roll_slice) if roll_slice else 0.0

        ent_slice = entropy_scores[max(0, i - context_segments + 1) : i + 1]
        seg["rolling_entropy_std_15sec"] = pstdev(ent_slice) if len(ent_slice) >= 2 else 0.0

        consecutive_duration = (
            consecutive_duration + cfg.segment_sec
            if mismatch_scores[i] >= cfg.high_mismatch_threshold
            else 0
        )
        seg["high_mismatch_duration_sec"] = consecutive_duration


# ---------------------------------------------------------------------------
# Sliding window demo scores (calculate_highlight_scores)
# ---------------------------------------------------------------------------


def _build_sliding_window_scores(
    detections: list[dict[str, Any]],
    cfg: HighlightMetricsConfig,
) -> list[dict[str, Any]]:
    if not detections:
        raise ValueError(
            "분석 결과가 없습니다. detections가 비어 있으면 슬라이딩 윈도우 점수를 계산할 수 없습니다."
        )

    max_time = max(d["time_sec"] for d in detections)
    w_mismatch, w_entropy, w_transition, w_character = cfg.demo_weights
    windows: list[dict[str, Any]] = []
    start = 0.0

    while start <= max_time:
        end = start + cfg.window_size
        sub = [d for d in detections if start <= d["time_sec"] < end]
        if sub:
            avg_mismatch = _mean_field(sub, "mismatch_score")
            avg_conf = _mean_field(sub, "emotion_confidence")
            character_count = len({d["character"] for d in sub})
            emotion_entropy = _label_entropy(
                [d["predicted_emotion"] for d in sub], len(cfg.emotions)
            )
            transition_count = _emotion_transition_count(sub)
            transition_score = min(1.0, transition_count / 5)
            highlight_score = (
                w_mismatch * avg_mismatch
                + w_entropy * emotion_entropy
                + w_transition * transition_score
                + w_character * min(1.0, character_count / 5)
            )
            windows.append({
                "start_sec": start,
                "end_sec": end,
                "start_time": _seconds_to_time(start),
                "end_time": _seconds_to_time(end),
                "num_detections": len(sub),
                "character_count": character_count,
                "emotion_count": len({d["predicted_emotion"] for d in sub}),
                "avg_mismatch": avg_mismatch,
                "emotion_entropy": emotion_entropy,
                "transition_score": transition_score,
                "avg_emotion_confidence": avg_conf,
                "highlight_score": highlight_score,
            })
        start += cfg.window_step

    windows.sort(key=lambda w: w["highlight_score"], reverse=True)
    return windows


# ---------------------------------------------------------------------------
# Emotion geometry helpers
# ---------------------------------------------------------------------------


def _probability_for_emotion(row: Mapping[str, Any], emotion: str) -> float:
    return float(row.get(f"prob_{emotion}", 0.0))


def _emotion_intensity(row: Mapping[str, Any], emotions: tuple[str, ...]) -> float:
    return float(sum(
        _probability_for_emotion(row, emotion) * EMOTION_COORDS[emotion][1]
        for emotion in emotions
        if emotion in EMOTION_COORDS
    ))


def _emotion_distance(expected: str, predicted: str) -> float:
    ex = EMOTION_COORDS.get(expected, (0.0, 0.0))
    pr = EMOTION_COORDS.get(predicted, (0.0, 0.0))
    distance = math.sqrt((ex[0] - pr[0]) ** 2 + (ex[1] - pr[1]) ** 2)
    return _clip(distance / math.sqrt(5), 0.0, 1.0)


def _valence_flip(expected: str, predicted: str) -> float:
    expected_valence = EMOTION_COORDS.get(expected, (0.0, 0.0))[0]
    predicted_valence = EMOTION_COORDS.get(predicted, (0.0, 0.0))[0]
    return float(expected_valence * predicted_valence < 0)


def _arousal_change(expected: str, predicted: str) -> float:
    expected_arousal = EMOTION_COORDS.get(expected, (0.0, 0.0))[1]
    predicted_arousal = EMOTION_COORDS.get(predicted, (0.0, 0.0))[1]
    return float(abs(expected_arousal - predicted_arousal))


def _entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _label_entropy(labels: list[str], num_emotions: int) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    raw = _entropy_from_counts(list(counts.values()))
    max_ent = math.log2(max(1, num_emotions))
    return raw / max_ent if max_ent > 0 else 0.0


def _emotion_transition_count(sub: list[dict[str, Any]]) -> int:
    by_character: dict[str, list[dict[str, Any]]] = {}
    for d in sub:
        by_character.setdefault(d["character"], []).append(d)
    transition_count = 0
    for char_rows in by_character.values():
        char_rows.sort(key=lambda r: r["time_sec"])
        emotions = [r["predicted_emotion"] for r in char_rows]
        transition_count += sum(
            emotions[i] != emotions[i - 1] for i in range(1, len(emotions))
        )
    return transition_count


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _mean_field(rows: list[dict[str, Any]], key: str) -> float:
    return mean(r[key] for r in rows) if rows else 0.0


def _mode(values: list[Any]) -> Any:
    return Counter(values).most_common(1)[0][0]


def _mean_youtube_score(
    youtube_by_second: dict[int, float],
    start: int,
    end: int,
) -> float | None:
    scores = [youtube_by_second[s] for s in range(start, end) if s in youtube_by_second]
    if not scores:
        return None
    return mean(scores)


def _seconds_to_time(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _json_sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float) and math.isnan(value):
            sanitized[key] = None
        elif isinstance(value, float):
            sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized
