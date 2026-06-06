"""
Animated Character Emotion Analysis & Highlight Extraction — Home (landing page).

Run:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from insideout2.ui.components import feature_cards, inject_site_style, render_reset_button, site_footer
from insideout2.ui.session import init_session

st.set_page_config(
    page_title="Animation Highlight Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session()
inject_site_style()

render_reset_button(key="reset_home")

st.markdown(
    """
    <div class="hero">
        <h1>Animation Character Emotion Analysis & Highlight Extractor</h1>
        <p>
            An AI-powered web service that detects character facial expressions in animation videos,
            combines 7-emotion mismatch metrics with YouTube replay data,
            and identifies the most compelling highlight segments.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/01_Analysis.py", label="✨ Start Highlight Extraction →", icon="🚀")
with col2:
    st.page_link("pages/02_Analytics_Dashboard.py", label="📊 Open Analytics Dashboard", icon="📊")

feature_cards([
    ("🎯", "Character Detection", "Frame-level detection of main animation characters using Google OWL-ViT zero-shot."),
    ("😊", "Emotion Classification", "7-emotion zero-shot labeling (happy, sad, angry, etc.) via OpenAI CLIP."),
    ("⚡", "Highlight Scoring", "5-second window candidates scored by mismatch, entropy, and transition metrics."),
    ("📈", "YouTube Integration", "Parse Most Replayed heatmap SVG and compare against viewer replay data."),
    ("🎞️", "Deliverables", "Highlight clips, CSV datasets, visualization charts, and summary reports."),
    ("🌐", "Web UI", "Upload a video in the browser and run the pipeline step-by-step or end-to-end."),
])

st.markdown('<p class="section-title">Analysis Pipeline</p>', unsafe_allow_html=True)

steps = [
    ("Frame Extraction", "1 fps sampling + blur, lighting & duplicate filters"),
    ("Model Inference", "OWL-ViT detection → CLIP emotion classification"),
    ("Score Computation", "5-second sliding-window highlight scores"),
    ("YouTube", "Replay heatmap → 5-second enhanced dataset"),
    ("Export", "Clips, charts, reports & CSV files"),
]
cols = st.columns(len(steps))
for col, (title, desc) in zip(cols, steps):
    with col:
        st.info(f"**{title}**\n\n{desc}")

with st.expander("Tech Stack & Notes"):
    st.markdown(
        """
        - **Detection**: `google/owlvit-base-patch32`
        - **Emotion**: `openai/clip-vit-base-patch32` (7 emotions)
        - **Metrics**: mismatch score, entropy, transition, YouTube replay
        - **Note**: Zero-shot demo; GPU (CUDA/MPS) significantly speeds up inference.
        - **YouTube**: Requires Selenium + Chrome headless (included in Docker image).
        """
    )

site_footer()
