"""
Linear Probing for VideoMAE
===========================
Evaluate pretrained VideoMAE encoder with frozen features + linear classifier.

Usage:
    python linear_probe.py --data_dir path/to/dataset --pretrained_path checkpoint.pt

Author: VLoad Project
Date: 2026
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import (
    create_videomae_for_classification,
    load_pretrained_encoder,
    get_model_info,
)
from dataset import create_classification_dataloaders
from utils import (
    set_seed,
    get_device,
    AverageMeter,
    compute_accuracy,
    compute_confusion_matrix,
    compute_metrics_from_confusion_matrix,
)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple:
    """Train one epoch."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        preds = outputs.logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        
        loss_meter.update(loss.item(), pixel_values.size(0))
        acc_meter.update(acc.item(), pixel_values.size(0))
        
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})
    
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: list = None,
) -> dict:
    """Evaluate model and compute metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    loss_meter = AverageMeter()
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        
        preds = outputs.logits.argmax(dim=-1)
        
        all_preds.append(preds)
        all_labels.append(labels)
        loss_meter.update(outputs.loss.item(), pixel_values.size(0))
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Compute metrics
    accuracy = (all_preds == all_labels).float().mean().item()
    
    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)
    metrics = compute_metrics_from_confusion_matrix(cm, class_names)
    
    metrics["loss"] = loss_meter.avg
    metrics["accuracy"] = accuracy
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Linear Probing for VideoMAE")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--class_names", type=str, default="empty,light,heavy")
    
    # Model
    parser.add_argument("--model_name", type=str, default="videomae-base")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--pretrained_path", type=str, default=None)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_operators", type=str, default="ope7,ope8",
                        help="Comma-separated operators for validation (e.g., 'ope7,ope8')")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./linear_probe_results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = args.class_names.split(",")
    
    # Parse validation operators
    val_operators = args.val_operators.split(",") if args.val_operators else None
    
    # Create dataloaders with operator-based split (no data leakage)
    train_loader, val_loader = create_classification_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        image_size=args.image_size,
        val_operators=val_operators,
        class_names=class_names,
    )
    
    # Create model with frozen encoder
    model = create_videomae_for_classification(
        num_classes=args.num_classes,
        model_name=args.model_name,
        from_pretrained=not args.pretrained_path,
        freeze_encoder=True,  # Key: freeze encoder for linear probing
    )
    
    if args.pretrained_path:
        model = load_pretrained_encoder(model, args.pretrained_path)
        # Re-verify encoder is frozen after loading
        for param in model.videomae.parameters():
            param.requires_grad = False
    
    model = model.to(device)
    
    # Print trainable parameters explicitly
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for _, p in trainable_params)
    print(f"\n=== LINEAR PROBING ===")
    print(f"Trainable parameters ({total_trainable:,} total):")
    for name, param in trainable_params:
        print(f"  - {name}: {param.numel():,}")
    
    if total_trainable == 0:
        raise RuntimeError("ERROR: No trainable parameters! Check freeze_encoder logic.")
    
    # Only optimize classifier
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        
        metrics = evaluate(model, val_loader, device, args.num_classes, class_names)
        
        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={metrics['accuracy']:.4f}, "
              f"macro_f1={metrics['macro_f1']:.4f}")
        
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save(model.state_dict(), output_dir / "best_linear_probe.pt")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    model.load_state_dict(torch.load(output_dir / "best_linear_probe.pt", weights_only=True))
    final_metrics = evaluate(model, val_loader, device, args.num_classes, class_names)
    
    print(f"Best Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Macro F1: {final_metrics['macro_f1']:.4f}")
    
    for name in class_names:
        print(f"  {name}: P={final_metrics[f'{name}_precision']:.3f}, "
              f"R={final_metrics[f'{name}_recall']:.3f}, "
              f"F1={final_metrics[f'{name}_f1']:.3f}")


if __name__ == "__main__":
    main()
