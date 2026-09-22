#!/usr/bin/env python3
"""Export the calibration-only MobileNet encoder to a compact ONNX file."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models import mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(torch.hub.get_dir()) / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
DESTINATION = ROOT / "event_cull" / "models" / "mobilenet_v3_small_visual.onnx"


class VisualEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = mobilenet_v3_small(weights=None)
        model.load_state_dict(torch.load(SOURCE, map_location="cpu", weights_only=True))
        self.features = model.features.eval()
        self.pool = model.avgpool.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(images)).flatten(1)


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"Missing source checkpoint: {SOURCE}")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    model = VisualEncoder().eval()
    example = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    torch.onnx.export(
        model,
        (example,),
        DESTINATION,
        input_names=["images"],
        output_names=["features"],
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    print(f"Exported {DESTINATION} ({DESTINATION.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
