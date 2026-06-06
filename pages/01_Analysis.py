"""Analysis page — video upload and pipeline control."""

from __future__ import annotations

import streamlit as st

from insideout2.ui.components import inject_site_style, page_header, render_pipeline_steps, render_reset_button, site_footer
from insideout2.ui.results import render_results
from insideout2.ui.session import build_settings_from_sidebar, get_pipeline, init_session

st.set_page_config(page_title="Run Analysis", page_icon="🚀", layout="wide")

init_session()
inject_site_style()
page_header(
    "Run Analysis",
    "Upload a video and run the pipeline step-by-step or end-to-end.",
)

settings = build_settings_from_sidebar()
pipeline = get_pipeline(settings)

render_pipeline_steps()
st.divider()

col_run, col_load, col_reset = st.columns([3, 1, 1])
with col_run:
    run_full = st.button("▶ Run Full Pipeline", type="primary", use_container_width=True)
    include_youtube = st.checkbox("Include YouTube replay", value=bool(settings.youtube_url))
with col_load:
    load_existing = st.button("📂 Load Saved Results", use_container_width=True)
with col_reset:
    render_reset_button(key="reset_analysis", output_dir=settings.output_dir)

step_cols = st.columns(6)
btn_frames = step_cols[0].button("1️⃣ Frames", use_container_width=True)
btn_analyze = step_cols[1].button("2️⃣ Analyze", use_container_width=True)
btn_score = step_cols[2].button("3️⃣ Highlights", use_container_width=True)
btn_yt = step_cols[3].button("4️⃣ YouTube", use_container_width=True)
btn_viz = step_cols[4].button("5️⃣ Visualize", use_container_width=True)
btn_export = step_cols[5].button("6️⃣ Export", use_container_width=True)

has_video = bool(settings.video_path)

if not has_video and st.session_state.pipeline is None:
    st.warning("Upload a video or enter a server-side path in the left sidebar.")
elif not has_video:
    st.info("You can load saved results or continue viewing a previous session without a video.")

bar = st.progress(0.0, text="Idle")


def callback(stage: str, progress: float, message: str) -> None:
    bar.progress(min(1.0, progress), text=f"[{stage}] {message}")


action_taken = False

if load_existing:
    if pipeline.load_existing_results():
        st.success("Saved results loaded successfully.")
        action_taken = True
    else:
        st.error("No result files found to load.")

try:
    if run_full:
        if not has_video:
            st.error("A video is required to run the full pipeline.")
        else:
            with st.spinner("Analyzing video frames… (this may take several minutes)"):
                pipeline.run_full(include_youtube=include_youtube, callback=callback)
            bar.progress(1.0, text="Complete")
            st.success("✅ Analysis Completed")
            action_taken = True

    elif btn_frames and has_video:
        pipeline.run_frame_extraction(callback)
        bar.progress(1.0, text="Frame extraction complete")
        action_taken = True

    elif btn_analyze:
        pipeline.run_analysis(callback)
        bar.progress(1.0, text="Emotion analysis complete")
        action_taken = True

    elif btn_score:
        pipeline.run_highlight_scoring(callback)
        bar.progress(1.0, text="Highlight scoring complete")
        action_taken = True

    elif btn_yt:
        pipeline.run_youtube_replay(callback=callback)
        if pipeline.state.youtube_second_df is not None:
            pipeline.run_enhanced_dataset(callback)
        bar.progress(1.0, text="YouTube processing complete")
        action_taken = True

    elif btn_viz:
        pipeline.run_visualization(callback)
        bar.progress(1.0, text="Visualization complete")
        action_taken = True

    elif btn_export:
        pipeline.run_annotation_and_clips(callback)
        bar.progress(1.0, text="Export complete")
        action_taken = True

except Exception as exc:
    st.error(f"Error: {exc}")
else:
    if action_taken or pipeline.state.frame_df is not None:
        st.divider()
        render_results(pipeline)

site_footer()
