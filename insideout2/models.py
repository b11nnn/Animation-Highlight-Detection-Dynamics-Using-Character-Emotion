from __future__ import annotations

from typing import Any

from transformers import CLIPModel, CLIPProcessor, pipeline

from insideout2.config import Settings, resolve_device


def load_models(settings: Settings) -> tuple[Any, CLIPModel, CLIPProcessor]:
    """OWL-ViT + CLIP 로딩. MPS 미지원 시 CPU로 자동 폴백."""
    try:
        return _load_models_on_device(settings)
    except RuntimeError as exc:
        if settings.device == "mps":
            settings.device = resolve_device("cpu")
            return _load_models_on_device(settings)
        raise RuntimeError(
            f"Model loading failed (device={settings.device}): {exc}"
        ) from exc


def _load_models_on_device(settings: Settings) -> tuple[Any, CLIPModel, CLIPProcessor]:
    detector = pipeline(
        task="zero-shot-object-detection",
        model="google/owlvit-base-patch32",
        device=settings.pipeline_device,
    )
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(settings.device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return detector, clip_model, clip_processor
