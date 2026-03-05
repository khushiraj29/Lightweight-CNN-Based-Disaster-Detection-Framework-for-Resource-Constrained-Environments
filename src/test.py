"""
Testing / evaluation script for the Lightweight CNN Disaster Detection Framework.

This script evaluates a trained model on a held-out test set and prints a full
classification report (precision, recall, F1, accuracy) plus an optional
confusion matrix.

Usage::

    # Evaluate using the best checkpoint on the test split
    python src/test.py --data_dir data/ --checkpoint checkpoints/best_model.pth

    # Also save a confusion matrix image
    python src/test.py --data_dir data/ --checkpoint checkpoints/best_model.pth \\
                       --save_confusion_matrix results/confusion_matrix.png

Run ``python src/test.py --help`` for all options.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from dataset import build_dataloaders, get_class_names
from model import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    """Run inference on a DataLoader and return (all_labels, all_preds, all_probs)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu()
        preds = probs.argmax(dim=1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
    return all_labels, all_preds, all_probs


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save a labelled confusion matrix using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed; skipping confusion matrix plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the Lightweight Disaster Detection CNN on a test set"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Root data directory containing a 'test/' subdirectory",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the .pth checkpoint file (saved by train.py)",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"],
        help="Which data split to evaluate on",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
    )
    parser.add_argument(
        "--width_multiplier", type=float, default=1.0,
        help="Must match the value used during training",
    )
    parser.add_argument(
        "--save_confusion_matrix", type=str, default="",
        help="Optional path to save a confusion matrix PNG (e.g. results/cm.png)",
    )
    parser.add_argument(
        "--save_report", type=str, default="",
        help="Optional path to save the classification report as JSON",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device override (e.g. 'cpu', 'cuda'). Auto-detected if empty.",
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
    print(f"Using device: {device}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    # Resolve class names: prefer stored names, fall back to filesystem scan
    if isinstance(state, dict) and "class_names" in state:
        class_names = state["class_names"]
    else:
        class_names = get_class_names(args.data_dir, split=args.split)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=num_classes,
        width_multiplier=args.width_multiplier,
        checkpoint_path=str(checkpoint_path),
        device=str(device),
    )
    model.eval()
    print(f"Model parameters: {model.count_parameters():,}")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.split not in loaders:
        raise FileNotFoundError(
            f"No '{args.split}' split found under '{args.data_dir}'. "
            "Ensure the directory contains a subfolder named after the split."
        )
    loader = loaders[args.split]
    print(f"Evaluating on '{args.split}' split ({len(loader.dataset)} images) …")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_labels, all_preds, _ = predict(model, loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print("\n── Classification Report ──────────────────────────────────────")
    print(report_str)

    # Overall accuracy (also present in the report, shown separately for clarity)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100.0 * correct / len(all_labels)
    print(f"Overall Accuracy: {accuracy:.2f}%")

    # ── Optional outputs ──────────────────────────────────────────────────────
    if args.save_confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, class_names, args.save_confusion_matrix)

    if args.save_report:
        report_dict = classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        )
        report_dict["overall_accuracy_pct"] = accuracy
        save_path = Path(args.save_report)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"Classification report saved to {save_path}")


if __name__ == "__main__":
    main()
