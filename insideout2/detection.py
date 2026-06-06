from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from insideout2.characters import CharacterProfile
from insideout2.config import CLIP_EMOTION_PROMPTS, EMOTIONS, MISMATCH_TABLE, Settings
from insideout2.io_paths import output_file


def detect_characters_in_frame(
    frame_path: str,
    detector: Any,
    characters: dict[str, CharacterProfile],
    threshold: float,
) -> list[dict]:
    image = Image.open(frame_path).convert("RGB")
    all_results: list[dict] = []
    tokenizer = detector.tokenizer

    for character_name, info in characters.items():
        candidate_boxes: list[dict] = []

        for query in info["queries"]:
            clean_query = str(query).strip()
            if not clean_query:
                continue

            sub_queries = [q.strip() for q in clean_query.split(",") if q.strip()]
            for sub_query in sub_queries:
                tokens = tokenizer(sub_query)["input_ids"]
                original_token_count = len(tokens)

                if original_token_count > 16:
                    original_query = sub_query
                    words = sub_query.split()
                    while len(tokens) > 16 and len(words) > 1:
                        words.pop()
                        sub_query = " ".join(words)
                        tokens = tokenizer(sub_query)["input_ids"]

                try:
                    results = detector(image, candidate_labels=[sub_query], threshold=threshold)
                except Exception:
                    results = []

                for r in results:
                    candidate_boxes.append({
                        "character": character_name,
                        "character_ko": info["korean"],
                        "score": float(r["score"]),
                        "box": r["box"],
                        "query": query,
                    })

        if candidate_boxes:
            best = max(candidate_boxes, key=lambda x: x["score"])
            all_results.append(best)

    return all_results


def crop_box_from_image(frame_path: str, box: dict, padding: float = 0.08):
    image = Image.open(frame_path).convert("RGB")
    w, h = image.size
    xmin, ymin, xmax, ymax = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
    bw, bh = xmax - xmin, ymax - ymin
    pad_x, pad_y = int(bw * padding), int(bh * padding)
    xmin = max(0, xmin - pad_x)
    ymin = max(0, ymin - pad_y)
    xmax = min(w, xmax + pad_x)
    ymax = min(h, ymax + pad_y)
    return image.crop((xmin, ymin, xmax, ymax)), (xmin, ymin, xmax, ymax)


def classify_emotion_with_clip(
    image_pil: Image.Image,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
) -> tuple[str, float, dict[str, float]]:
    inputs = clip_processor(
        text=list(CLIP_EMOTION_PROMPTS),
        images=image_pil,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_emotion = EMOTIONS[pred_idx]
    confidence = float(probs[pred_idx])
    prob_dict = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, probs)}
    return pred_emotion, confidence, prob_dict


def get_mismatch_score(
    character: str,
    predicted_emotion: str,
    characters: dict[str, CharacterProfile],
) -> tuple[float, str]:
    expected = characters[character]["expected_emotion"]
    score = MISMATCH_TABLE.get(expected, {}).get(predicted_emotion, 0.5)
    return float(score), expected


def analyze_frames(
    frame_df: pd.DataFrame,
    detector: Any,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    settings: Settings,
    characters: dict[str, CharacterProfile],
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    dropped_small_faces = 0
    total = len(frame_df)

    for i, (_, row) in enumerate(tqdm(frame_df.iterrows(), total=total)):
        frame_id = int(row["frame_id"])
        frame_path = row["frame_path"]
        timestamp = float(row["timestamp"])
        timestamp_str = row["timestamp_str"]

        detections = detect_characters_in_frame(
            frame_path,
            detector,
            characters,
            settings.detection_threshold,
        )

        for det_idx, det in enumerate(detections):
            character = det["character"]
            box = det["box"]
            detect_score = det["score"]
            width = box["xmax"] - box["xmin"]
            height = box["ymax"] - box["ymin"]
            box_area = width * height

            if box_area < settings.min_box_area:
                dropped_small_faces += 1
                continue

            crop_img, fixed_box = crop_box_from_image(frame_path, box)
            crop_path = output_file(
                settings, "crops", f"frame_{frame_id:05d}_{character}_{det_idx}.jpg"
            )
            crop_img.save(crop_path)

            pred_emotion, emotion_conf, prob_dict = classify_emotion_with_clip(
                crop_img, clip_model, clip_processor, settings.device
            )
            mismatch_score, expected_emotion = get_mismatch_score(
                character, pred_emotion, characters
            )

            result: dict = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "timestamp_str": timestamp_str,
                "frame_path": frame_path,
                "crop_path": str(crop_path),
                "character": character,
                "character_ko": characters[character]["korean"],
                "expected_emotion": expected_emotion,
                "predicted_emotion": pred_emotion,
                "detection_score": detect_score,
                "emotion_confidence": emotion_conf,
                "mismatch_score": mismatch_score,
                "xmin": fixed_box[0],
                "ymin": fixed_box[1],
                "xmax": fixed_box[2],
                "ymax": fixed_box[3],
                "box_area": box_area,
            }
            for emo in EMOTIONS:
                result[f"prob_{emo}"] = prob_dict.get(emo, 0.0)
            rows.append(result)

        if progress_callback and total > 0:
            progress_callback((i + 1) / total, f"Analyzing frames — {i + 1}/{total}")

    result_df = pd.DataFrame(rows)
    if len(result_df) > 0:
        result_df.to_csv(
            output_file(settings, "character_emotion_results.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    return result_df
