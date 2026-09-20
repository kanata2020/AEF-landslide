import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .metrics import add_counts, confusion_counts, detailed_metrics, empty_counts


def bce_dice_loss(logits, targets, pos_weight=None, dice_weight=0.3):
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()
    weight = None if pos_weight is None else torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).flatten(1).sum(1)
    union = probabilities.flatten(1).sum(1) + targets.flatten(1).sum(1)
    dice = 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
    return (1 - dice_weight) * bce + dice_weight * dice


def run_epoch(model, loader, optimizer, device, pos_weight, dice_weight, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, samples, counts = 0.0, 0, empty_counts()
    progress = tqdm(loader, desc="train" if training else "val", leave=False)
    for inputs, targets in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp = scaler is not None
        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=amp):
            output = model(inputs)
            logits = output["segmentation"]
            loss = bce_dice_loss(logits, targets, pos_weight, dice_weight)
        if training:
            if scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        batch_size = inputs.shape[0]
        total_loss += float(loss.detach()) * batch_size
        samples += batch_size
        metric_targets = targets.unsqueeze(1) if targets.ndim == 3 else targets
        add_counts(
            counts,
            confusion_counts(torch.sigmoid(logits.detach()) >= 0.5, metric_targets),
        )
        progress.set_postfix(loss=total_loss / samples)
    result = detailed_metrics(counts)["landslide"]
    return {**result, "loss": total_loss / max(samples, 1)}


def save_checkpoint(
    path, model, optimizer, epoch, best_iou, model_cfg, train_cfg,
    scheduler=None, scaler=None, bad_epochs=0,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "bad_epochs": bad_epochs,
            "epoch": epoch,
            "best_val_iou": best_iou,
            "model_cfg": model_cfg,
            "train_cfg": train_cfg,
            "in_channels": model_cfg["in_channels"],
            "target_size": (128, 128),
            "config": model_cfg,
        },
        path,
    )


def load_history(path: Path, completed_epoch: int):
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in rows if row["epoch"] <= completed_epoch]
