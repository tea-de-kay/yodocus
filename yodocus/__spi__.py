from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL.Image import Image


@dataclass
class YoloModelConfig:
    model_name: str
    nc: int
    class_names: list[str]
    model_description: str | None = None
    input_size: tuple[int, int] = (640, 640)
    preserve_aspect_ratio: bool = False
    """Whether to preserve the aspect ratio by scaling and padding to input size."""

    @classmethod
    def from_yaml(cls, path: Path) -> YoloModelConfig:
        data = yaml.safe_load(path.read_text())

        return YoloModelConfig(**data)


@dataclass
class DetectionConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    visualize: bool = False
    """Whether to draw detected bounding boxes on input image."""


@dataclass
class DetectedBox:
    class_id: int
    class_name: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    def calc_intersection_area(self, other: DetectedBox) -> float:
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return (x1 - x0) * (y1 - y0)

    def merge(self, other: DetectedBox) -> DetectedBox:
        return DetectedBox(
            class_id=self.class_id,
            class_name=self.class_name,
            score=max(self.score, other.score),
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )


@dataclass
class DetectionResult:
    boxes: list[DetectedBox]
    visual: Image | None = None
    """Visual representation of the detection results."""

    def copy_with(self, boxes: list[DetectedBox]) -> DetectionResult:
        return DetectionResult(boxes=boxes, visual=self.visual)


@dataclass
class PostprocessorConfig:
    containment_threshold: float = 0.9
    """Fraction of box to be contained in surrounding box in order to merge."""
