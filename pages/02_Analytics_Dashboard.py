"""Analytics Dashboard — view saved analysis results."""

from __future__ import annotations

import streamlit as st

from insideout2.ui.components import inject_site_style, page_header, render_reset_button, site_footer
from insideout2.ui.results import render_results
from insideout2.ui.session import build_settings_from_sidebar, get_pipeline, init_session

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

init_session()
inject_site_style()
page_header(
    "📊 Analytics Dashboard",
    "Load saved results and explore charts, clips, and CSV outputs.",
)

settings = build_settings_from_sidebar(show_path_input=False)
pipeline = get_pipeline(settings)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🔄 Refresh Results", type="primary", use_container_width=True):
        if pipeline.load_existing_results():
            st.rerun()
        else:
            st.error(f"⚠️ No data found in output folder: `{settings.output_dir}`")
with col2:
    render_reset_button(key="reset_dashboard", output_dir=settings.output_dir)

with col3:
    st.caption(f"Output path: `{settings.output_path.resolve()}`")

if pipeline.state.frame_df is None:
    pipeline.load_existing_results()

if pipeline.state.frame_df is not None or pipeline.state.result_df is not None:
    render_results(pipeline)
else:
    st.info(
        "No results yet. Run the pipeline on the **Analysis** page, "
        "or verify your output folder path."
    )
    st.page_link("pages/01_Analysis.py", label="✨ Go to Analysis →", icon="🚀")

site_footer()
