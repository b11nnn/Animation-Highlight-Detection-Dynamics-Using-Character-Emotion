from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from insideout2.config import Settings
from insideout2.io_paths import output_file


def make_visualizations(
    result_df: pd.DataFrame,
    score_df: pd.DataFrame,
    settings: Settings,
) -> dict[str, str]:
    paths: dict[str, str] = {}

    if len(result_df) > 0:
        pivot = pd.crosstab(result_df["character"], result_df["predicted_emotion"])
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_title("Predicted Emotion Distribution by Character")
        ax.set_xlabel("Character")
        ax.set_ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        p1 = output_file(settings, "emotion_distribution_by_character.png")
        plt.savefig(p1, dpi=200)
        plt.close()
        paths["emotion_distribution"] = str(p1)

        timeline = result_df.groupby("timestamp")["mismatch_score"].mean().reset_index()
        plt.figure(figsize=(12, 4))
        plt.plot(timeline["timestamp"], timeline["mismatch_score"], marker="o")
        plt.title("Mismatch Score Over Time")
        plt.xlabel("Time (sec)")
        plt.ylabel("Average Mismatch Score")
        plt.tight_layout()
        p2 = output_file(settings, "mismatch_score_timeline.png")
        plt.savefig(p2, dpi=200)
        plt.close()
        paths["mismatch_timeline"] = str(p2)

    sorted_score = score_df.sort_values("start_sec")
    plt.figure(figsize=(12, 4))
    plt.plot(sorted_score["start_sec"], sorted_score["highlight_score"], marker="o")
    plt.title("Highlight Score Over Time")
    plt.xlabel("Window Start Time (sec)")
    plt.ylabel("Highlight Score")
    plt.tight_layout()
    p3 = output_file(settings, "highlight_score_timeline.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    paths["highlight_timeline"] = str(p3)

    return paths
