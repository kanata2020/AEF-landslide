from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import ConvAct, StableResidualSEBlock


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return inputs

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return gradient * ctx.scale, None


def scale_gradient(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    """Keep the forward values unchanged while scaling gradients to the shared stem."""
    return _ScaleGradient.apply(inputs, scale)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TemporalDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TemporalBranch(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float):
        super().__init__()
        middle, output = 128, 192
        self.features = nn.Sequential(
            TemporalDownBlock(in_channels, middle, dropout * 0.5),
            TemporalDownBlock(middle, output, dropout),
            TemporalDownBlock(output, output, dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(output * 2),
            nn.Linear(output * 2, output),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout + 0.10),
            nn.Linear(output, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = torch.cat((features.mean((2, 3)), features.amax((2, 3))), dim=1)
        return self.classifier(pooled)


class AEFSpatioTemporalNet(nn.Module):
    """Full-resolution AEF segmenter with an auxiliary, gradient-limited temporal branch."""
    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 96,
        residual_scale: float = 0.1,
        temporal_classes: int = 13,
        temporal_dropout: float = 0.15,
        temporal_shared_gradient_scale: float = 0.1,
    ):
        super().__init__()
        self.temporal_shared_gradient_scale = temporal_shared_gradient_scale

        # Shared stem and full-resolution spatial prediction path.
        self.stem = nn.Sequential(
            ConvAct(in_channels, hidden_channels, kernel_size=1, padding=0),
            ConvAct(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self.body = nn.Sequential(
            StableResidualSEBlock(hidden_channels, 1, residual_scale),
            StableResidualSEBlock(hidden_channels, 1, residual_scale),
            StableResidualSEBlock(hidden_channels, 2, residual_scale),
            StableResidualSEBlock(hidden_channels, 2, residual_scale),
            StableResidualSEBlock(hidden_channels, 1, residual_scale),
        )
        self.head = nn.Sequential(
            ConvAct(hidden_channels, hidden_channels // 2),
            nn.Dropout2d(0.05),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )
        self.temporal_branch = TemporalBranch(hidden_channels, temporal_classes, temporal_dropout)
        self.temporal_branch.apply(self._init_temporal_weights)

    @staticmethod
    def _init_temporal_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor):
        shared = self.stem(inputs)
        segmentation_logits = self.head(self.body(shared))
        temporal_features = scale_gradient(shared, self.temporal_shared_gradient_scale)
        temporal_logits = self.temporal_branch(temporal_features)
        return {
            "segmentation_logits": segmentation_logits,
            "temporal_logits": temporal_logits,
        }

    def load_spatial_state(self, state_dict: dict) -> None:
        spatial = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("stem.", "body.", "head."))
        }
        expected = {
            key for key in self.state_dict() if key.startswith(("stem.", "body.", "head."))
        }
        if set(spatial) != expected:
            missing = sorted(expected - set(spatial))
            unexpected = sorted(set(spatial) - expected)
            raise ValueError(
                f"Spatial checkpoint is incompatible; missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        self.load_state_dict(spatial, strict=False)


class JointLoss(nn.Module):
    def __init__(
        self,
        temporal_class_weights: Iterable[float],
        temporal_weight: float = 0.25,
        bce_weight: float = 0.7,
        dice_weight: float = 0.3,
        circular_smoothing: float = 0.10,
        no_event_class_index: int = 12,
    ):
        super().__init__()
        if abs(bce_weight + dice_weight - 1.0) > 1e-6:
            raise ValueError("BCE and Dice weights must sum to 1.")
        self.register_buffer("temporal_class_weights", torch.as_tensor(temporal_class_weights, dtype=torch.float32))
        self.temporal_weight = temporal_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.circular_smoothing = circular_smoothing
        self.no_event_class_index = no_event_class_index

    def forward(self, outputs, masks, months):
        logits = outputs["segmentation_logits"]
        masks = masks.float()
        bce = F.binary_cross_entropy_with_logits(logits, masks)
        probabilities = torch.sigmoid(logits).flatten(1)
        flat_masks = masks.flatten(1)
        intersection = (probabilities * flat_masks).sum(1)
        dice = 1.0 - ((2 * intersection + 1e-6) / (probabilities.sum(1) + flat_masks.sum(1) + 1e-6)).mean()
        spatial = self.bce_weight * bce + self.dice_weight * dice

        if torch.any((months < 0) | (months >= outputs["temporal_logits"].shape[1])):
            raise ValueError("Temporal targets must be class indices in [0, 12].")
        log_probs = F.log_softmax(outputs["temporal_logits"], dim=1)
        target = torch.zeros_like(log_probs)
        target.scatter_(1, months[:, None], 1.0)
        calendar = months != self.no_event_class_index
        if calendar.any() and self.circular_smoothing > 0:
            calendar_months = months[calendar]
            center = 1.0 - self.circular_smoothing
            adjacent = self.circular_smoothing / 2.0
            rows = torch.arange(calendar_months.shape[0], device=months.device)
            calendar_target = torch.zeros_like(target[calendar])
            calendar_target[rows, calendar_months] = center
            calendar_target[rows, (calendar_months - 1) % 12] += adjacent
            calendar_target[rows, (calendar_months + 1) % 12] += adjacent
            target[calendar] = calendar_target
        sample_weights = self.temporal_class_weights[months]
        temporal = ((-(target * log_probs).sum(1)) * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
        return {
            "total": spatial + self.temporal_weight * temporal,
            "spatial": spatial,
            "bce": bce,
            "dice": dice,
            "temporal": temporal,
        }
