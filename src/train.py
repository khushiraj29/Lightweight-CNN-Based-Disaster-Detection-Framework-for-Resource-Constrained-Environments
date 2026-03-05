"""
Training script for the Lightweight CNN Disaster Detection Framework.

Usage::

    python src/train.py --data_dir data/ --epochs 30 --batch_size 32

Run ``python src/train.py --help`` for the full list of options.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import build_dataloaders, get_class_names
from model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Lightweight Disaster Detection CNN"
    )
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root directory with train/ and val/ subdirectories")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--width_multiplier", type=float, default=1.0,
                        help="Channel width multiplier (use <1.0 for smaller model)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="",
                        help="Device to use (auto-detected if empty)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data
    loaders = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if "train" not in loaders:
        raise FileNotFoundError(
            f"No 'train' split found under '{args.data_dir}'. "
            "Ensure the directory structure is: data/train/<class_name>/..."
        )
    class_names = get_class_names(args.data_dir, split="train")
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Model
    model = build_model(
        num_classes=num_classes,
        width_multiplier=args.width_multiplier,
        device=str(device),
    )
    print(f"Model parameters: {model.count_parameters():,}")

    # Training components
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        scheduler.step()

        val_loss, val_acc = 0.0, 0.0
        if "val" in loaders:
            val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
            f"({elapsed:.1f}s)"
        )

        # Save best model
        if "val" in loaders and val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_names": class_names,
                },
                checkpoint_path,
            )
            print(f"  → Saved best model (val_acc={val_acc:.2f}%) to {checkpoint_path}")

    # Always save the final model
    final_path = output_dir / "final_model.pth"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
        },
        final_path,
    )
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
