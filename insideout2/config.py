from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def is_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def is_mps_available() -> bool:
    try:
        import torch

        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        return False


def get_best_device() -> str:
    """CUDA → MPS(Apple Silicon) → CPU 순으로 자동 선택."""
    if is_cuda_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def available_devices() -> list[str]:
    """UI·설정용: 현재 환경에서 선택 가능한 장치 목록."""
    options: list[str] = []
    if is_cuda_available():
        options.append("cuda")
    if is_mps_available():
        options.append("mps")
    options.append("cpu")
    return options


def resolve_device(requested: str | None = None) -> str:
    """
    요청 장치를 검증해 실제 사용 가능한 torch device 문자열을 반환.
    Mac에서 cuda를 고르면 MPS 또는 CPU로 안전하게 폴백.
    """
    if not requested:
        return get_best_device()

    key = requested.lower().strip()
    if key == "cuda":
        return "cuda" if is_cuda_available() else get_best_device()
    if key == "mps":
        return "mps" if is_mps_available() else "cpu"
    if key == "cpu":
        return "cpu"
    return get_best_device()


def _default_device() -> str:
    return get_best_device()

EMOTIONS: tuple[str, ...] = (
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised",
    "neutral",
)

CLIP_EMOTION_PROMPTS: tuple[str, ...] = (
    "a close-up of a cartoon character with a happy smiling face",
    "a close-up of a cartoon character with a sad crying face",
    "a close-up of a cartoon character with an angry furious face",
    "a close-up of a cartoon character with a fearful scared face",
    "a close-up of a cartoon character with a disgusted face",
    "a close-up of a cartoon character with a surprised shocked face",
    "a close-up of a cartoon character with a neutral calm face",
)

MISMATCH_TABLE: dict[str, dict[str, float]] = {
    "happy": {
        "happy": 0.0,
        "neutral": 0.4,
        "surprised": 0.5,
        "fearful": 0.8,
        "sad": 1.0,
        "angry": 1.0,
        "disgusted": 0.9,
    },
    "sad": {
        "sad": 0.0,
        "neutral": 0.4,
        "fearful": 0.7,
        "surprised": 0.7,
        "angry": 0.8,
        "disgusted": 0.8,
        "happy": 1.0,
    },
    "angry": {
        "angry": 0.0,
        "neutral": 0.4,
        "disgusted": 0.5,
        "surprised": 0.6,
        "fearful": 0.8,
        "sad": 0.8,
        "happy": 1.0,
    },
    "fearful": {
        "fearful": 0.0,
        "surprised": 0.4,
        "neutral": 0.5,
        "sad": 0.7,
        "disgusted": 0.8,
        "angry": 0.9,
        "happy": 1.0,
    },
    "disgusted": {
        "disgusted": 0.0,
        "neutral": 0.4,
        "angry": 0.5,
        "sad": 0.7,
        "surprised": 0.7,
        "fearful": 0.8,
        "happy": 1.0,
    },
    "surprised": {
        "surprised": 0.0,
        "fearful": 0.4,
        "happy": 0.6,
        "neutral": 0.6,
        "sad": 0.7,
        "disgusted": 0.8,
        "angry": 0.8,
    },
    "neutral": {
        "neutral": 0.0,
        "happy": 0.5,
        "sad": 0.5,
        "disgusted": 0.5,
        "surprised": 0.6,
        "fearful": 0.7,
        "angry": 0.8,
    },
}


@dataclass
class Settings:
    """노트북 전역 설정을 Streamlit/서버에서 주입 가능한 형태로 캡슐화."""

    video_path: str = ""
    output_dir: str = "insideout2_output"
    youtube_url: str = ""
    sample_fps: float = 1.0
    detection_threshold: float = 0.08
    window_size: int = 5
    step_size: int = 5
    top_k_highlights: int = 3
    segment_sec: int = 5
    highlight_threshold: float = 0.6
    high_mismatch_threshold: float = 0.6
    min_box_area: int = 3600
    clip_margin: int = 2
    highlight_clip_duration: float = 10.0
    device: str = field(default_factory=_default_device)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)

    @property
    def pipeline_device(self) -> int | str:
        """HuggingFace pipeline용 device 인자 (cuda=0, mps='mps', cpu=-1)."""
        if self.device == "cuda":
            return 0
        if self.device == "mps":
            return "mps"
        return -1

    @property
    def device_id(self) -> int:
        """하위 호환: cuda GPU index 또는 CPU(-1). MPS는 pipeline_device 사용."""
        return 0 if self.device == "cuda" else -1

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
