from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from insideout2.annotation import draw_annotated_frames
from insideout2.characters import CharacterProfile, load_characters_from_csv
from insideout2.clips import extract_final_highlight_clip, final_highlight_clip_path, load_highlight_meta
from insideout2.config import Settings
from insideout2.detection import analyze_frames
from insideout2.frames import extract_frames
from insideout2.highlights import calculate_highlight_scores, make_enhanced_highlight_dataset
from insideout2.io_paths import prepare_dirs
from insideout2.models import load_models
from insideout2.report import make_summary_report
from insideout2.visualize import make_visualizations
from insideout2.youtube import (
    extract_video_id,
    get_heatmap_svg_path_with_selenium,
    svg_path_d_to_second_level_replay_df,
)

ProgressCallback = Callable[[str, float, str], None]


@dataclass
class PipelineState:
    """Streamlit session_state에 저장할 파이프라인 중간 결과."""

    frame_df: pd.DataFrame | None = None
    result_df: pd.DataFrame | None = None
    score_df: pd.DataFrame | None = None
    youtube_second_df: pd.DataFrame | None = None
    enhanced_df: pd.DataFrame | None = None
    video_id: str = ""
    chart_paths: dict[str, str] = field(default_factory=dict)
    report_path: str = ""
    final_highlight_path: str = ""
    final_highlight_meta: dict[str, Any] = field(default_factory=dict)
    models_loaded: bool = False
    detector: Any = None
    clip_model: Any = None
    clip_processor: Any = None


class AnalysisPipeline:
    """노트북 전체 흐름을 단계별 API로 노출."""

    def __init__(
        self,
        settings: Settings,
        characters: dict[str, CharacterProfile] | None = None,
    ):
        self.settings = settings
        self.characters = characters or load_characters_from_csv()
        self.state = PipelineState()

    def _notify(
        self,
        callback: ProgressCallback | None,
        stage: str,
        progress: float,
        message: str,
    ) -> None:
        if callback:
            callback(stage, progress, message)

    def setup(self) -> None:
        prepare_dirs(self.settings)

    def load_models(self, callback: ProgressCallback | None = None) -> None:
        self._notify(callback, "models", 0.1, "Loading models...")
        detector, clip_model, clip_processor = load_models(self.settings)
        self.state.detector = detector
        self.state.clip_model = clip_model
        self.state.clip_processor = clip_processor
        self.state.models_loaded = True
        self._notify(callback, "models", 1.0, "Models loaded")

    def run_frame_extraction(self, callback: ProgressCallback | None = None) -> pd.DataFrame:
        self.setup()
        self._notify(callback, "frames", 0.0, "Starting frame extraction")

        def inner(p: float, msg: str) -> None:
            self._notify(callback, "frames", p, msg)

        frame_df, _, _, _ = extract_frames(self.settings, progress_callback=inner)
        self.state.frame_df = frame_df
        self._notify(callback, "frames", 1.0, f"Extracted {len(frame_df)} frames")
        return frame_df

    def run_analysis(self, callback: ProgressCallback | None = None) -> pd.DataFrame:
        if self.state.frame_df is None:
            raise RuntimeError("Run frame extraction first.")
        if not self.state.models_loaded:
            self.load_models(callback)

        def inner(p: float, msg: str) -> None:
            self._notify(callback, "analysis", p, msg)

        result_df = analyze_frames(
            self.state.frame_df,
            self.state.detector,
            self.state.clip_model,
            self.state.clip_processor,
            self.settings,
            self.characters,
            progress_callback=inner,
        )
        self.state.result_df = result_df
        self._notify(callback, "analysis", 1.0, f"Emotion analysis complete — {len(result_df)} detections")
        return result_df

    def run_highlight_scoring(self, callback: ProgressCallback | None = None) -> pd.DataFrame:
        if self.state.result_df is None:
            raise RuntimeError("Run frame analysis first.")
        self._notify(callback, "highlights", 0.5, "Computing highlight scores...")
        score_df = calculate_highlight_scores(self.state.result_df, self.settings)
        self.state.score_df = score_df
        self._notify(callback, "highlights", 1.0, "Highlight scoring complete")
        return score_df

    def run_youtube_replay(
        self,
        youtube_url: str | None = None,
        callback: ProgressCallback | None = None,
    ) -> pd.DataFrame | None:
        url = youtube_url or self.settings.youtube_url
        if not url:
            return None

        self._notify(callback, "youtube", 0.2, "Collecting YouTube heatmap...")
        video_id = extract_video_id(url)
        self.state.video_id = video_id

        svg_d = get_heatmap_svg_path_with_selenium(url)
        if svg_d is None:
            self._notify(callback, "youtube", 1.0, "No heatmap found — skipped")
            return None

        if self.state.result_df is not None and len(self.state.result_df) > 0:
            duration = float(self.state.result_df["timestamp"].max())
        elif self.state.frame_df is not None and len(self.state.frame_df) > 0:
            duration = float(self.state.frame_df["timestamp"].max())
        else:
            duration = 600.0

        youtube_df = svg_path_d_to_second_level_replay_df(svg_d, duration)
        self.state.youtube_second_df = youtube_df
        youtube_df.to_csv(
            self.settings.output_path / "youtube_replay_by_second.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self._notify(callback, "youtube", 1.0, "YouTube replay scores extracted")
        return youtube_df

    def run_enhanced_dataset(self, callback: ProgressCallback | None = None) -> pd.DataFrame:
        if self.state.result_df is None:
            raise RuntimeError("Run frame analysis first.")
        if self.state.youtube_second_df is None:
            raise RuntimeError("Collect YouTube replay data first.")

        self._notify(callback, "enhanced", 0.5, "Building 5-second enhanced dataset...")
        enhanced = make_enhanced_highlight_dataset(
            self.state.result_df,
            self.state.youtube_second_df,
            self.state.video_id,
            self.settings,
        )
        self.state.enhanced_df = enhanced
        self._notify(callback, "enhanced", 1.0, f"Dataset ready — {len(enhanced)} segments")
        return enhanced

    def run_visualization(self, callback: ProgressCallback | None = None) -> dict[str, str]:
        if self.state.result_df is None or self.state.score_df is None:
            raise RuntimeError("Analysis and highlight scoring are required.")
        self._notify(callback, "viz", 0.5, "Generating visualizations...")
        paths = make_visualizations(self.state.result_df, self.state.score_df, self.settings)
        self.state.chart_paths = paths
        self._notify(callback, "viz", 1.0, "Visualizations complete")
        return paths

    def run_annotation_and_clips(self, callback: ProgressCallback | None = None) -> None:
        if self.state.frame_df is None or self.state.result_df is None:
            raise RuntimeError("Frame extraction and analysis are required.")
        if self.state.score_df is None:
            raise RuntimeError("Highlight scoring is required.")

        self._notify(callback, "export", 0.2, "Generating annotated frames...")
        draw_annotated_frames(self.state.result_df, self.state.frame_df, self.settings)

        self._notify(callback, "export", 0.5, "Extracting 10s highlight clip from source video...")
        clip_path = extract_final_highlight_clip(self.settings, self.state.score_df)
        if clip_path:
            self.state.final_highlight_path = str(clip_path)
            self.state.final_highlight_meta = load_highlight_meta(self.settings)

        self._notify(callback, "export", 0.8, "Writing summary report...")
        self.state.report_path = make_summary_report(
            self.state.result_df,
            self.state.score_df,
            self.settings,
            self.characters,
        )
        self._notify(callback, "export", 1.0, "Export complete")

    def run_full(
        self,
        *,
        include_youtube: bool = True,
        callback: ProgressCallback | None = None,
    ) -> PipelineState:
        self.run_frame_extraction(callback)
        self.run_analysis(callback)
        self.run_highlight_scoring(callback)
        if include_youtube and self.settings.youtube_url:
            self.run_youtube_replay(callback=callback)
            if self.state.youtube_second_df is not None:
                self.run_enhanced_dataset(callback)
        self.run_visualization(callback)
        self.run_annotation_and_clips(callback)
        return self.state

    def load_existing_results(self) -> bool:
        """output_dir에 저장된 CSV가 있으면 session에 복원."""
        out = self.settings.output_path
        loaded = False
        frame_csv = out / "extracted_frames.csv"
        result_csv = out / "character_emotion_results.csv"
        score_csv = out / "highlight_scores.csv"
        yt_csv = out / "youtube_replay_by_second.csv"
        enhanced_csv = out / "final_highlight_dataset_5sec_enhanced.csv"

        if frame_csv.exists():
            self.state.frame_df = pd.read_csv(frame_csv)
            loaded = True
        if result_csv.exists():
            self.state.result_df = pd.read_csv(result_csv)
            loaded = True
        if score_csv.exists():
            self.state.score_df = pd.read_csv(score_csv)
            loaded = True
        if yt_csv.exists():
            self.state.youtube_second_df = pd.read_csv(yt_csv)
            loaded = True
        if enhanced_csv.exists():
            self.state.enhanced_df = pd.read_csv(enhanced_csv)
            loaded = True

        clip = final_highlight_clip_path(self.settings)
        if clip.exists() and clip.stat().st_size > 0:
            self.state.final_highlight_path = str(clip)
            self.state.final_highlight_meta = load_highlight_meta(self.settings)
            loaded = True

        return loaded
