from __future__ import annotations

import cv2
import pandas as pd
from tqdm import tqdm

from insideout2.config import Settings
from insideout2.io_paths import output_file

COLOR_MAP = {
    "Joy": (0, 255, 255),
    "Sadness": (255, 100, 0),
    "Anger": (0, 0, 255),
    "Fear": (255, 0, 255),
    "Disgust": (0, 255, 0),
    "Anxiety": (0, 140, 255),
    "Envy": (255, 255, 0),
    "Ennui": (180, 0, 180),
    "Embarrassment": (203, 192, 255),
}


def draw_annotated_frames(
    result_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    settings: Settings,
) -> None:
    for _, frame_row in tqdm(frame_df.iterrows(), total=len(frame_df)):
        frame_id = int(frame_row["frame_id"])
        frame_path = frame_row["frame_path"]
        img = cv2.imread(frame_path)
        if img is None:
            continue

        sub = result_df[result_df["frame_id"] == frame_id]
        for _, r in sub.iterrows():
            xmin, ymin, xmax, ymax = int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"])
            color = COLOR_MAP.get(r["character"], (255, 255, 255))
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{r['character']} | {r['predicted_emotion']} | mismatch {r['mismatch_score']:.2f}"
            cv2.putText(
                img, label, (xmin, max(20, ymin - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

        out_path = output_file(settings, "annotated_frames", f"frame_{frame_id:05d}.jpg")
        cv2.imwrite(str(out_path), img)
