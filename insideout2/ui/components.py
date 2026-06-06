from __future__ import annotations

import streamlit as st

from insideout2.io_paths import clear_all_results, clear_upload_cache

DEFAULT_OUTPUT_DIR = "insideout2_output"

SITE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero {
    background: linear-gradient(135deg, #312E81 0%, #4338CA 45%, #6366F1 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 10px 40px rgba(99, 102, 241, 0.25);
}

.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    color: white !important;
}

.hero p {
    font-size: 1.05rem;
    opacity: 0.92;
    margin: 0;
    line-height: 1.6;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.feature-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem;
    transition: border-color 0.2s, transform 0.2s;
}

.feature-card:hover {
    border-color: #6366F1;
    transform: translateY(-2px);
}

.feature-card h3 {
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 0.4rem 0;
    color: #E2E8F0;
}

.feature-card p {
    font-size: 0.875rem;
    color: #94A3B8;
    margin: 0;
    line-height: 1.5;
}

.step-badge {
    display: inline-block;
    background: #4338CA;
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    margin-right: 0.4rem;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #E2E8F0;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #334155;
}

.sidebar-brand {
    font-size: 1.1rem;
    font-weight: 700;
    color: #A5B4FC;
    margin-bottom: 0.25rem;
}

footer.site-footer {
    text-align: center;
    color: #64748B;
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
    margin-top: 2rem;
    border-top: 1px solid #334155;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
}

[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
}
</style>
"""


def inject_site_style() -> None:
    st.markdown(SITE_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, *, hero: bool = False) -> None:
    inject_site_style()
    if hero:
        st.markdown(
            f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"## {title}")
        st.caption(subtitle)


def feature_cards(items: list[tuple[str, str, str]]) -> None:
    cards_html = '<div class="feature-grid">'
    for icon, title, desc in items:
        cards_html += (
            f'<div class="feature-card">'
            f'<h3>{icon} {title}</h3><p>{desc}</p></div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def render_pipeline_steps() -> None:
    steps = [
        ("1", "Frame Extraction", "Quality filtering + 1 fps sampling"),
        ("2", "Character & Emotion", "OWL-ViT detection + CLIP 7-emotion labels"),
        ("3", "Highlight Scores", "5s window mismatch & entropy metrics"),
        ("4", "YouTube Replay", "Most Replayed heatmap integration"),
        ("5", "Export", "Charts, clips, reports & CSV"),
    ]
    cols = st.columns(len(steps))
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"**<span class='step-badge'>{num}</span>{title}**", unsafe_allow_html=True)
            st.caption(desc)


def site_footer() -> None:
    st.markdown(
        '<footer class="site-footer">Animated Characters Emotion Analysis Demo · '
        "OWL-ViT + CLIP + Highlight Metrics</footer>",
        unsafe_allow_html=True,
    )


def collect_output_dirs(output_dir: str | None = None) -> set[str]:
    """초기화 대상 결과 폴더 목록 (중복 제거)."""
    dirs: set[str] = set()
    if output_dir:
        dirs.add(output_dir)
    pipeline = st.session_state.get("pipeline")
    if pipeline is not None and hasattr(pipeline, "settings"):
        dirs.add(pipeline.settings.output_dir)
    if not dirs:
        dirs.add(DEFAULT_OUTPUT_DIR)
    return dirs


def reset_workspace(*, output_dir: str | None = None) -> None:
    """세션·디스크 결과·업로드 캐시를 모두 초기화."""
    for path in collect_output_dirs(output_dir):
        clear_all_results(path)
    clear_upload_cache()
    for session_key in list(st.session_state.keys()):
        del st.session_state[session_key]


def render_reset_button(*, key: str = "reset_all", output_dir: str | None = None) -> None:
    """Clear session + on-disk outputs, then rerun the page."""
    if st.button("🔄 Reset Application", key=key):
        reset_workspace(output_dir=output_dir)
        st.rerun()
