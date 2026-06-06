from __future__ import annotations

from typing import Callable

import cv2
import pandas as pd

from insideout2.config import Settings
from insideout2.io_paths import output_file, prepare_dirs
from insideout2.utils import is_blurry, is_duplicate, is_poor_lighting, seconds_to_time


def extract_frames(
    settings: Settings,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    video_path = settings.video_path
    sample_fps = settings.sample_fps

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    prepare_dirs(settings)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    frame_interval = max(1, int(original_fps / sample_fps))

    frame_infos: list[dict] = []
    dropped_blur_infos: list[dict] = []
    dropped_light_infos: list[dict] = []
    dropped_dup_infos: list[dict] = []

    frame_idx = saved_idx = blur_idx = light_idx = dup_idx = 0
    dropped_blur = dropped_light = dropped_dup = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / original_fps
            timestamp_str = seconds_to_time(timestamp)

            if is_blurry(frame):
                dropped_blur += 1
                blur_path = output_file(
                    settings, "dropped_blur_frames", f"blur_{blur_idx:05d}.jpg"
                )
                cv2.imwrite(str(blur_path), frame)
                dropped_blur_infos.append({
                    "drop_id": blur_idx,
                    "original_frame_idx": frame_idx,
                    "frame_path": str(blur_path),
                    "timestamp": timestamp,
                    "timestamp_str": timestamp_str,
                    "drop_reason": "blur",
                })
                blur_idx += 1
                frame_idx += 1
                continue

            if is_poor_lighting(frame):
                dropped_light += 1
                light_path = output_file(
                    settings, "dropped_light_frames", f"light_{light_idx:05d}.jpg"
                )
                cv2.imwrite(str(light_path), frame)
                dropped_light_infos.append({
                    "drop_id": light_idx,
                    "original_frame_idx": frame_idx,
                    "frame_path": str(light_path),
                    "timestamp": timestamp,
                    "timestamp_str": timestamp_str,
                    "drop_reason": "poor_lighting",
                })
                light_idx += 1
                frame_idx += 1
                continue

            if is_duplicate(frame, prev_frame):
                dropped_dup += 1
                dup_path = output_file(
                    settings, "dropped_duplicate_frames", f"dup_{dup_idx:05d}.jpg"
                )
                cv2.imwrite(str(dup_path), frame)
                dropped_dup_infos.append({
                    "drop_id": dup_idx,
                    "original_frame_idx": frame_idx,
                    "frame_path": str(dup_path),
                    "timestamp": timestamp,
                    "timestamp_str": timestamp_str,
                    "drop_reason": "duplicate",
                })
                dup_idx += 1
                frame_idx += 1
                continue

            frame_path = output_file(settings, "frames", f"frame_{saved_idx:05d}.jpg")
            cv2.imwrite(str(frame_path), frame)
            prev_frame = frame.copy()
            frame_infos.append({
                "frame_id": saved_idx,
                "original_frame_idx": frame_idx,
                "frame_path": str(frame_path),
                "timestamp": timestamp,
                "timestamp_str": timestamp_str,
            })
            saved_idx += 1

            if progress_callback and total_frames > 0:
                progress_callback(frame_idx / total_frames, f"Extracting frames — {saved_idx} saved")

        frame_idx += 1

    cap.release()

    frame_df = pd.DataFrame(frame_infos)
    blur_df = pd.DataFrame(dropped_blur_infos)
    light_df = pd.DataFrame(dropped_light_infos)
    dup_df = pd.DataFrame(dropped_dup_infos)

    frame_df.to_csv(output_file(settings, "extracted_frames.csv"), index=False, encoding="utf-8-sig")
    blur_df.to_csv(output_file(settings, "dropped_blur_frames.csv"), index=False, encoding="utf-8-sig")
    light_df.to_csv(output_file(settings, "dropped_light_frames.csv"), index=False, encoding="utf-8-sig")
    dup_df.to_csv(output_file(settings, "dropped_duplicate_frames.csv"), index=False, encoding="utf-8-sig")

    return frame_df, blur_df, light_df, dup_df
