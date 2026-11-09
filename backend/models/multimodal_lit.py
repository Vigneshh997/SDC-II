"""Legacy Lightning entry point retained with an explicit migration error."""

from backend.models.losses import AsymmetricLoss  # noqa: F401


class MultimodalLitModel:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "Legacy label-leaking training disabled. Use python -m backend.pipelines.train --config config.yaml"
        )
