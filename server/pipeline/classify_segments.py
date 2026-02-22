"""Zero-shot segment classification using CLIP ViT-B/32."""

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

INDOOR_VOCABULARY = [
    "wall", "floor", "ceiling", "door", "window",
    "couch", "sofa", "chair", "armchair", "office chair",
    "table", "desk", "coffee table", "dining table",
    "bed", "pillow", "blanket", "mattress",
    "shelf", "bookshelf", "cabinet", "dresser", "wardrobe",
    "lamp", "ceiling light", "chandelier",
    "tv", "monitor", "screen", "computer",
    "plant", "potted plant", "vase", "flower",
    "rug", "carpet", "curtain", "blinds",
    "refrigerator", "oven", "sink", "microwave",
    "painting", "picture frame", "mirror", "clock",
    "box", "bag", "basket", "trash can",
]

_clip_model = None
_clip_processor = None


def _load_clip():
    """Lazy-load CLIP model."""
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor

    import torch
    from transformers import CLIPModel, CLIPProcessor

    logger.info("Loading CLIP ViT-B/32...")
    model_name = "openai/clip-vit-base-patch32"
    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    _clip_model = CLIPModel.from_pretrained(model_name)
    _clip_model.eval()

    if torch.cuda.is_available():
        _clip_model = _clip_model.to("cuda")

    logger.info("CLIP loaded")
    return _clip_model, _clip_processor


def classify_segments(
    manifest_path: Path,
    frames_dir: Path,
    vocabulary: list[str] | None = None,
    confidence_threshold: float = 0.15,
) -> dict:
    """Classify each segment using zero-shot CLIP on cropped bounding boxes.

    Updates the manifest in place and returns it.
    """
    import torch

    model, processor = _load_clip()
    device = next(model.parameters()).device

    if vocabulary is None:
        vocabulary = INDOOR_VOCABULARY

    text_labels = [f"a photo of a {v}" for v in vocabulary]

    manifest = json.loads(manifest_path.read_text())
    segments = manifest.get("segments", [])

    if not segments:
        return manifest

    # Pre-compute text features
    text_inputs = processor(text=text_labels, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for seg in segments:
        primary_frame = seg.get("primary_frame", "")
        bbox = seg.get("bbox", [0, 0, 0, 0])

        if not primary_frame or bbox == [0, 0, 0, 0]:
            continue

        frame_path = frames_dir / primary_frame
        if not frame_path.exists():
            continue

        try:
            img = Image.open(frame_path).convert("RGB")
            x1, y1, x2, y2 = bbox
            # Add margin
            w, h = img.size
            margin_x = max(10, (x2 - x1) // 5)
            margin_y = max(10, (y2 - y1) // 5)
            crop = img.crop((
                max(0, x1 - margin_x),
                max(0, y1 - margin_y),
                min(w, x2 + margin_x),
                min(h, y2 + margin_y),
            ))

            image_inputs = processor(images=crop, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

            with torch.no_grad():
                image_features = model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).squeeze(0)
            probs = similarities.softmax(dim=0).cpu().numpy()

            best_idx = int(np.argmax(probs))
            best_conf = float(probs[best_idx])

            if best_conf >= confidence_threshold:
                seg["label"] = vocabulary[best_idx]
                seg["semantic_confidence"] = round(best_conf, 3)
            else:
                seg["semantic_confidence"] = round(best_conf, 3)

        except Exception as e:
            logger.warning(f"Failed to classify segment {seg.get('id')}: {e}")
            continue

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
