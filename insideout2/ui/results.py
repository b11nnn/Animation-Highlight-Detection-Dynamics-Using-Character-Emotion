from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from insideout2.clips import ensure_final_highlight_clip, load_highlight_meta
from insideout2.pipeline import AnalysisPipeline
from insideout2.ui.components import inject_site_style


def _render_final_highlight_media(pipeline: AnalysisPipeline) -> None:
    """Display the final 5-second original highlight clip only."""
    st.markdown(
        '<p class="section-title">🎬 Extracted Highlight Clip</p>',
        unsafe_allow_html=True,
    )

    settings = pipeline.settings
    score_df = pipeline.state.score_df

    meta: dict = dict(pipeline.state.final_highlight_meta or {})
    if not meta:
        try:
            meta = load_highlight_meta(settings) or {}
        except Exception:
            meta = {}

    video_path = settings.video_path or meta.get("source_video") or None
    if video_path and not Path(str(video_path)).exists():
        video_path = None

    try:
        clip_path, loaded_meta = ensure_final_highlight_clip(
            settings,
            score_df,
            video_path=video_path,
        )
    except Exception as exc:
        st.warning(f"Failed to load highlight clip: {exc}")
        clip_path, loaded_meta = None, {}

    if loaded_meta:
        meta = loaded_meta

    if meta:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Highlight Score",
            f"{float(meta.get('highlight_score') or 0):.3f}",
        )
        c2.metric(
            "Extracted Segment",
            f"{meta.get('clip_start_time', '?')} ~ {meta.get('clip_end_time', '?')}",
        )
        c3.metric(
            "Analysis Window",
            f"{meta.get('window_start_time', '?')} ~ {meta.get('window_end_time', '?')}",
        )

    if clip_path is not None and clip_path.exists() and clip_path.stat().st_size > 0:
        st.video(str(clip_path))
        st.caption(f"Trimmed from source video · `{clip_path.name}`")
        pipeline.state.final_highlight_path = str(clip_path)
        pipeline.state.final_highlight_meta = meta or {}
        return

    if score_df is not None and len(score_df) > 0:
        from insideout2.clips import compute_top_highlight_segment

        segment = compute_top_highlight_segment(
            score_df,
            clip_duration=getattr(settings, "highlight_clip_duration", 10.0),
        )
        st.warning(
            "The highlight clip has not been generated yet. "
            "Run **6️⃣ Export**, or verify that the source video path is valid."
        )
        st.info(
            f"Predicted highlight segment: **{segment['clip_start_time']} ~ {segment['clip_end_time']}** "
            f"(score {segment['highlight_score']:.3f})"
        )
        if not settings.video_path:
            st.caption(
                "Re-upload the source video on the Analysis page and run Export. "
                "ffmpeg will trim the segment and save it as `final_highlight_5sec.mp4`."
            )
        return

    st.info(
        "No highlight video available. Run Analysis → Highlight Scoring → Export in order."
    )


def render_results(pipeline: AnalysisPipeline) -> None:
    inject_site_style()
    state = pipeline.state

    if state.result_df is not None and len(state.result_df) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detections", len(state.result_df))
        c2.metric("Characters", state.result_df["character"].nunique())
        c3.metric("Avg. Mismatch", f"{state.result_df['mismatch_score'].mean():.2f}")
        if state.frame_df is not None:
            c4.metric("Frames", len(state.frame_df))

    tabs = st.tabs([
        "Emotion Analysis",
        "Highlight Scores",
        "YouTube / Dataset",
        "Visualizations",
        "🎬 Highlight Clip",
        "Files",
    ])

    with tabs[0]:
        if state.frame_df is not None:
            st.markdown('<p class="section-title">Extracted Frames</p>', unsafe_allow_html=True)
            st.dataframe(state.frame_df.head(15), use_container_width=True, hide_index=True)

        if state.result_df is not None and len(state.result_df) > 0:
            st.markdown('<p class="section-title">Emotion Detections by Character</p>', unsafe_allow_html=True)
            st.dataframe(
                state.result_df[
                    ["timestamp_str", "character", "expected_emotion",
                     "predicted_emotion", "mismatch_score", "emotion_confidence"]
                ].head(40),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No emotion analysis results yet.")

    with tabs[1]:
        if state.score_df is not None:
            st.dataframe(state.score_df.head(20), use_container_width=True, hide_index=True)
            if len(state.score_df) > 0:
                chart_df = state.score_df.sort_values("start_sec")
                st.line_chart(
                    chart_df.set_index("start_sec")["highlight_score"],
                    height=280,
                )
        else:
            st.info("No highlight scores yet.")

    with tabs[2]:
        if state.youtube_second_df is not None:
            st.markdown('<p class="section-title">YouTube Replay Score</p>', unsafe_allow_html=True)
            st.line_chart(
                state.youtube_second_df.set_index("second")["youtube_replay_score"],
                height=280,
            )
        if state.enhanced_df is not None:
            st.markdown('<p class="section-title">5-Second Enhanced Highlight Dataset</p>', unsafe_allow_html=True)
            st.dataframe(state.enhanced_df.head(25), use_container_width=True, hide_index=True)
        if state.youtube_second_df is None and state.enhanced_df is None:
            st.info("No YouTube replay or enhanced dataset available.")

    with tabs[3]:
        if state.chart_paths:
            cols = st.columns(min(3, len(state.chart_paths)))
            for col, (name, path) in zip(cols, state.chart_paths.items()):
                with col:
                    st.caption(name.replace("_", " "))
                    if Path(path).exists():
                        st.image(path, use_container_width=True)
        else:
            st.info("No visualization images yet.")

    with tabs[4]:
        _render_final_highlight_media(pipeline)

    with tabs[5]:
        out = pipeline.settings.output_path
        st.code(str(out.resolve()), language=None)
        if out.exists():
            file_rows = [
                {
                    "File": str(p.relative_to(out)),
                    "Size (KB)": round(p.stat().st_size / 1024, 1),
                }
                for p in sorted(out.rglob("*"))
                if p.is_file()
            ]
            if file_rows:
                st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)

        if state.report_path and Path(state.report_path).exists():
            st.markdown('<p class="section-title">Summary Report</p>', unsafe_allow_html=True)
            st.text(Path(state.report_path).read_text(encoding="utf-8"))
