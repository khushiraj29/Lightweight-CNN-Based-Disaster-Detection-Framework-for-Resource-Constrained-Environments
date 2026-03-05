"""
Single-image inference script for the Lightweight CNN Disaster Detection Framework.

Usage::

    python src/inference.py --image path/to/image.jpg \\
                            --checkpoint checkpoints/best_model.pth

Run ``python src/inference.py --help`` for all options.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_eval_transforms
from model import LightweightDisasterCNN, build_model


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load an RGB image from disk."""
    img = Image.open(image_path).convert("RGB")
    return img


def predict_image(
    model: LightweightDisasterCNN,
    image: Image.Image,
    class_names: List[str],
    device: torch.device,
) -> Dict:
    """
    Run inference on a single PIL image.

    Returns a dictionary with:
        - ``predicted_class``: The predicted class name.
        - ``confidence``: Confidence score (0–100 %) for the predicted class.
        - ``probabilities``: Dict mapping each class name to its probability (%).
    """
    transform = get_eval_transforms()
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = probs.argmax().item()
    return {
        "predicted_class": class_names[pred_idx],
        "confidence": round(probs[pred_idx].item() * 100, 2),
        "probabilities": {
            name: round(prob.item() * 100, 2)
            for name, prob in zip(class_names, probs)
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-image inference with the Lightweight Disaster Detection CNN"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image (JPEG, PNG, etc.)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--width_multiplier", type=float, default=1.0,
        help="Must match the value used during training",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device override (e.g. 'cpu', 'cuda'). Auto-detected if empty.",
    )
    parser.add_argument(
        "--output_json", type=str, default="",
        help="Optional path to save the prediction result as JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Checkpoint & class names ───────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "class_names" in state:
        class_names = state["class_names"]
    else:
        class_names = LightweightDisasterCNN.CLASS_NAMES

    num_classes = len(class_names)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=num_classes,
        width_multiplier=args.width_multiplier,
        checkpoint_path=str(checkpoint_path),
        device=str(device),
    )

    # ── Image ─────────────────────────────────────────────────────────────────
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = load_image(image_path)

    # ── Inference ─────────────────────────────────────────────────────────────
    result = predict_image(model, image, class_names, device)

    print(f"\nImage: {image_path.name}")
    print(f"Predicted class : {result['predicted_class']}")
    print(f"Confidence      : {result['confidence']:.2f}%")
    print("\nClass probabilities:")
    for cls_name, prob in sorted(result["probabilities"].items(),
                                 key=lambda kv: kv[1], reverse=True):
        bar = "█" * int(prob / 5)
        print(f"  {cls_name:<15} {prob:6.2f}%  {bar}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to {output_path}")

    return result


if __name__ == "__main__":
    main()
