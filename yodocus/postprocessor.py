import numpy as np
from PIL import Image

from yodocus.__spi__ import DetectionResult, PostprocessorConfig
from yodocus.visualizer import Visualizer


class HeuristicPostprocessor:
    def __init__(self, config: PostprocessorConfig) -> None:
        self._config = config

    def process(
        self,
        result: DetectionResult,
        original_image: Image.Image | None,
    ) -> DetectionResult:
        changed = True
        boxes = result.boxes[:]
        while changed:
            changed = False
            merged = []
            skip = set()

            for i, box1 in enumerate(boxes):
                if i in skip:
                    continue
                for j, box2 in enumerate(boxes[i + 1 :], start=i + 1):
                    if j in skip:
                        continue
                    if box1.class_id != box2.class_id:
                        continue

                    inter = box1.calc_intersection_area(box2)
                    if inter > 0:
                        smaller_area = min(box1.area, box2.area)
                        if inter / smaller_area >= self._config.containment_threshold:
                            box1 = box1.merge(box2)
                            skip.add(j)
                            changed = True
                merged.append(box1)

            boxes = merged

        visual = None
        if result.visual is not None:
            visualizer = Visualizer(classes={b.class_id: b.class_name for b in boxes})
            visual = visualizer.visualize(
                np.array(original_image or result.visual), boxes
            )

        return DetectionResult(boxes=boxes, visual=visual)
