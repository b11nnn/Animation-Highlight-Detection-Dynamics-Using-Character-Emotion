from __future__ import annotations

import pandas as pd

from insideout2.config import Settings
from insideout2.highlight_metrics import HighlightMetricsConfig, compute_highlight_metrics
from insideout2.io_paths import output_file


def calculate_highlight_scores(result_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if len(result_df) == 0:
        raise ValueError(
            "No analysis results. Try lowering DETECTION_THRESHOLD to 0.05 or 0.03."
        )

    cfg = HighlightMetricsConfig(
        window_size=settings.window_size,
        window_step=settings.step_size,
    )
    detections = result_df.to_dict(orient="records")
    result = compute_highlight_metrics(
        detections,
        include_sliding_windows=True,
        config=cfg,
    )
    score_df = pd.DataFrame(result.sliding_windows)
    score_df.to_csv(output_file(settings, "highlight_scores.csv"), index=False, encoding="utf-8-sig")
    return score_df.sort_values("highlight_score", ascending=False)


def make_enhanced_highlight_dataset(
    result_df: pd.DataFrame,
    youtube_second_df: pd.DataFrame,
    video_id: str,
    settings: Settings,
) -> pd.DataFrame:
    from insideout2.highlight_metrics import compute_highlight_metrics_from_dataframe

    cfg = HighlightMetricsConfig(
        segment_sec=settings.segment_sec,
        highlight_threshold=settings.highlight_threshold,
        high_mismatch_threshold=settings.high_mismatch_threshold,
    )
    metrics = compute_highlight_metrics_from_dataframe(
        result_df,
        youtube_second_df,
        video_id=video_id,
        config=cfg,
    )
    final_df = pd.DataFrame(metrics.segments)
    out_path = output_file(settings, "final_highlight_dataset_5sec_enhanced.csv")
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return final_df
