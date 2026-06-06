from __future__ import annotations

import pandas as pd

from insideout2.characters import CharacterProfile
from insideout2.config import Settings
from insideout2.io_paths import output_file


def make_summary_report(
    result_df: pd.DataFrame,
    score_df: pd.DataFrame,
    settings: Settings,
    characters: dict[str, CharacterProfile],
) -> str:
    report_path = output_file(settings, "summary_report.txt")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("Animation Character Emotion Analysis — Demo Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("[Characters Analyzed]\n")
        for ch, info in characters.items():
            f.write(f"- {ch}: expected emotion = {info['expected_emotion']}\n")

        f.write("\n[Detection Summary]\n")
        f.write(f"- Total crop detections: {len(result_df)}\n")
        f.write(f"- Unique characters: {result_df['character'].nunique() if len(result_df) else 0}\n")

        if len(result_df) > 0:
            f.write("\n[Detections by Character]\n")
            f.write(result_df["character"].value_counts().to_string())
            f.write("\n\n[Predicted Emotion Distribution]\n")
            f.write(result_df["predicted_emotion"].value_counts().to_string())
            f.write("\n\n[Average Mismatch Score]\n")
            f.write(str(result_df["mismatch_score"].mean()))
            f.write("\n\n")

        f.write("[Top Highlight Candidates]\n")
        for _, row in score_df.head(5).iterrows():
            f.write(
                f"- {row['start_time']} ~ {row['end_time']} | "
                f"Highlight Score={row['highlight_score']:.3f}, "
                f"Mismatch={row['avg_mismatch']:.3f}, "
                f"Emotion Entropy={row['emotion_entropy']:.3f}\n"
            )

        f.write(
            "\n[Note]\n"
            "This is a zero-shot OWL-ViT + CLIP emotion classification demo. "
            "Accuracy is limited without task-specific fine-tuning.\n"
        )

    return str(report_path)
