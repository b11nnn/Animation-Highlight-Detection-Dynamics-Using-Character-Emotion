from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from insideout2.characters import load_characters_from_csv
from insideout2.config import Settings, available_devices, get_best_device, resolve_device
from insideout2.pipeline import AnalysisPipeline


def init_session() -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


def get_pipeline(settings: Settings) -> AnalysisPipeline:
    characters = load_characters_from_csv()
    pipeline: AnalysisPipeline = (
        st.session_state.pipeline or AnalysisPipeline(settings, characters)
    )
    pipeline.settings = settings
    pipeline.characters = characters
    st.session_state.pipeline = pipeline
    return pipeline


def build_settings_from_sidebar(*, show_path_input: bool = True) -> Settings:
    with st.sidebar:
        st.markdown('<p class="sidebar-brand">Highlight Extractor</p>', unsafe_allow_html=True)
        st.caption("Emotion Analysis Web Service")
        st.divider()
        st.header("Input Settings")

        uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        video_path_input = ""
        if show_path_input:
            video_path_input = st.text_input(
                "Enter Input Video Path for Demo",
                placeholder="/path/to/video.mp4",
                help="Use when running on a server with local file access.",
            )

        youtube_url = st.text_input(
            "YouTube URL (optional)",
            value="https://www.youtube.com/watch?v=vW0IQoSweVI",
            help="Used to collect Most Replayed heatmap data.",
        )
        output_dir = st.text_input("Output Folder", value="insideout2_output")

        with st.expander("Advanced Parameters", expanded=False):
            sample_fps = st.slider("Frame Sampling FPS", 0.5, 3.0, 1.0, 0.5)
            detection_threshold = st.slider("Detection Threshold", 0.03, 0.20, 0.08, 0.01)
            window_size = st.number_input("Highlight Window (sec)", 3, 30, 5)
            step_size = st.number_input("Window Step (sec)", 1, 30, 5)
            top_k = st.number_input("Top Highlight Clips", 1, 10, 3)
            min_box_area = st.number_input("Min Face Box Area (px²)", 1000, 10000, 3600, 100)
            device_options = available_devices()
            default_device = get_best_device()
            device = st.selectbox(
                "Compute Device",
                device_options,
                index=device_options.index(default_device),
                help="Apple Silicon: mps (Metal) recommended. Falls back to CPU when CUDA is unavailable.",
            )
            st.caption(f"Auto-detected: **{default_device}**")

        characters = load_characters_from_csv()
        st.divider()
        st.caption(f"{len(characters)} characters · data/character_queries.csv")

    video_path = ""
    if uploaded is not None:
        upload_dir = Path(tempfile.gettempdir()) / "insideout2_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        video_path = str(dest)
        st.sidebar.success("🎬 Video successfully loaded!")
    elif video_path_input.strip():
        video_path = video_path_input.strip()
        if Path(video_path).exists():
            st.sidebar.success("🎬 Video successfully loaded!")
        else:
            st.sidebar.error("⚠️ Video file not found. Please check the path.")

    return Settings(
        video_path=video_path,
        output_dir=output_dir,
        youtube_url=youtube_url.strip(),
        sample_fps=sample_fps,
        detection_threshold=detection_threshold,
        window_size=int(window_size),
        step_size=int(step_size),
        top_k_highlights=int(top_k),
        min_box_area=int(min_box_area),
        device=resolve_device(device),
    )
