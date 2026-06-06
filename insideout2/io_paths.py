from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from insideout2.config import Settings

OUTPUT_SUBDIRS = (
    "frames",
    "crops",
    "annotated_frames",
    "highlight_clips",
    "dropped_blur_frames",
    "dropped_light_frames",
    "dropped_duplicate_frames",
)


def prepare_dirs(settings: Settings) -> None:
    base = settings.output_path
    for sub in OUTPUT_SUBDIRS:
        (base / sub).mkdir(parents=True, exist_ok=True)


def output_file(settings: Settings, *parts: str) -> Path:
    return settings.output_path.joinpath(*parts)


def clear_all_results(output_dir: str | Path) -> None:
    """결과 폴더의 CSV·이미지·클립 등 추출물을 모두 삭제하고 빈 구조만 남깁니다."""
    path = Path(output_dir)
    if path.exists():
        shutil.rmtree(path)
    prepare_dirs(Settings(output_dir=str(path)))


def clear_upload_cache() -> None:
    """업로드된 임시 영상 파일 삭제."""
    upload_dir = Path(tempfile.gettempdir()) / "insideout2_uploads"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
