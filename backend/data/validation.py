"""Strict observed metadata validation and training-only fitted normalization."""

import math

import torch
from torch import nn

from backend.config import FEATURES


class MetadataTransform(nn.Module):
    def __init__(self, fields=("age", "sex")):
        super().__init__()
        if not fields or len(set(fields)) != len(fields) or set(fields) - FEATURES.keys():
            raise ValueError("Unsupported or duplicate metadata fields")
        self.fields = tuple(fields)
        self.names = [
            name
            for f in self.fields
            for name in ([f"{f}={c}" for c in FEATURES[f]["categories"]] if f == "sex" else [f])
        ]
        self.register_buffer("mean", torch.zeros(len(self.names)))
        self.register_buffer("std", torch.ones(len(self.names)))
        self.register_buffer("fitted", torch.tensor(False))

    def encode(self, values):
        if not isinstance(values, dict) or set(values) != set(self.fields):
            raise ValueError(f"metadata must contain exactly these observed fields: {list(self.fields)}")
        result = []
        for field in self.fields:
            value = values[field]
            if field == "sex":
                if value not in FEATURES[field]["categories"]:
                    raise ValueError("sex must be female, male, other, or explicitly unknown")
                result.extend(float(value == c) for c in FEATURES[field]["categories"])
            else:
                if isinstance(value, bool) or value is None or str(value).strip() == "":
                    raise ValueError(f"{field} is required and must be observed")
                try:
                    number = float(value)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{field} must be numeric") from exc
                spec = FEATURES[field]
                if not math.isfinite(number) or not spec["min"] <= number <= spec["max"]:
                    raise ValueError(f"{field} must be in [{spec['min']}, {spec['max']}] {spec['unit']}")
                result.append(number)
        return torch.tensor(result, dtype=torch.float32)

    def fit(self, rows):
        values = torch.stack([self.encode(r) for r in rows])
        self.mean.copy_(values.mean(0))
        self.std.copy_(values.std(0, unbiased=False).clamp_min(1e-6))
        for i, name in enumerate(self.names):
            if "=" in name:
                self.mean[i] = 0
                self.std[i] = 1
        self.fitted.fill_(True)
        return self

    def forward(self, values):
        if not self.fitted.item():
            raise ValueError("Metadata normalization must be fitted on training patients first")
        if values.ndim != 2 or values.shape[1] != len(self.names) or not torch.isfinite(values).all():
            raise ValueError("Invalid encoded metadata tensor")
        return (values - self.mean) / self.std


def validate_note(note):
    if not isinstance(note, str) or not note.strip():
        raise ValueError("Observed clinical notes from this examination are required")
    if len(note) > 20000:
        raise ValueError("Clinical notes must be at most 20000 characters")
    return " ".join(note.split())
