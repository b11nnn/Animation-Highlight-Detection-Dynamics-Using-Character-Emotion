from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from insideout2.config import Settings
from insideout2.io_paths import output_file
from insideout2.utils import seconds_to_time

FINAL_CLIP_FILENAME = "final_highlight_10sec.mp4"
FINAL_CLIP_META_FILENAME = "final_highlight_10sec.json"


def _ffmpeg_executable() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _video_duration_sec(video_path: str) -> float | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps and fps > 0 and frames > 0:
        return float(frames / fps)
    return None


def compute_top_highlight_segment(
    score_df: pd.DataFrame,
    *,
    clip_duration: float = 10.0,
    video_duration: float | None = None,
) -> dict[str, Any]:
    """하이라이트 점수 1위 윈도우 중심으로 clip_duration(기본 10초) 구간 계산."""
    top = score_df.iloc[0]
    window_start = float(top["start_sec"])
    window_end = float(top["end_sec"])
    center = (window_start + window_end) / 2.0
    start = max(0.0, center - clip_duration / 2.0)

    if video_duration is not None and video_duration > 0:
        if start + clip_duration > video_duration:
            start = max(0.0, video_duration - clip_duration)
        clip_duration = min(clip_duration, video_duration - start)

    end = start + clip_duration
    return {
        "clip_start_sec": start,
        "clip_end_sec": end,
        "clip_duration_sec": clip_duration,
        "highlight_score": float(top["highlight_score"]),
        "window_start_sec": window_start,
        "window_end_sec": window_end,
        "clip_start_time": seconds_to_time(start),
        "clip_end_time": seconds_to_time(end),
        "window_start_time": str(top.get("start_time", seconds_to_time(window_start))),
        "window_end_time": str(top.get("end_time", seconds_to_time(window_end))),
    }


def final_highlight_clip_path(settings: Settings) -> Path:
    return output_file(settings, "highlight_clips", FINAL_CLIP_FILENAME)


def final_highlight_meta_path(settings: Settings) -> Path:
    return output_file(settings, "highlight_clips", FINAL_CLIP_META_FILENAME)


def save_highlight_meta(settings: Settings, meta: dict[str, Any]) -> Path:
    path = final_highlight_meta_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_highlight_meta(settings: Settings) -> dict[str, Any]:
    path = final_highlight_meta_path(settings)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_final_highlight_clip(
    settings: Settings,
    score_df: pd.DataFrame,
    *,
    video_path: str | None = None,
) -> Path | None:
    """
    하이라이트 점수 1위 구간을 중심으로 원본 영상에서 10초 클립을 ffmpeg로 추출.
    결과: insideout2_output/highlight_clips/final_highlight_10sec.mp4
    """
    if score_df is None or len(score_df) == 0:
        return None

    src = video_path or settings.video_path
    if not src or not Path(src).exists():
        return None

    duration_setting = float(getattr(settings, "highlight_clip_duration", 10))
    video_duration = _video_duration_sec(src)
    segment = compute_top_highlight_segment(
        score_df,
        clip_duration=duration_setting,
        video_duration=video_duration,
    )

    out_path = final_highlight_clip_path(settings)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _ffmpeg_executable()
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        str(segment["clip_start_sec"]),
        "-i",
        src,
        "-t",
        str(segment["clip_duration_sec"]),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    meta = {
        **segment,
        "source_video": str(Path(src).resolve()),
        "output_clip": str(out_path.resolve()),
    }
    save_highlight_meta(settings, meta)

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    return None


def ensure_final_highlight_clip(
    settings: Settings,
    score_df: pd.DataFrame | None,
    *,
    video_path: str | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """
    대시보드용: 저장된 10초 클립을 반환하거나, 없으면 score_df로 새로 추출.
    """
    clip_path = final_highlight_clip_path(settings)
    meta = load_highlight_meta(settings)

    if clip_path.exists() and clip_path.stat().st_size > 0:
        return clip_path, meta

    if score_df is None or len(score_df) == 0:
        return None, meta

    src = video_path or settings.video_path or meta.get("source_video", "")
    if src and Path(src).exists():
        extracted = extract_final_highlight_clip(settings, score_df, video_path=str(src))
        if extracted:
            return extracted, load_highlight_meta(settings)

    return None, meta


def extract_highlight_clips(
    settings: Settings,
    score_df: pd.DataFrame,
) -> pd.DataFrame:
    """파이프라인 호환: 최종 10초 하이라이트 클립 1개 추출."""
    extract_final_highlight_clip(settings, score_df)
    return score_df.head(1).copy()
