import cv2
import numpy as np
from PIL import Image

from yodocus.__spi__ import DetectedBox


class Visualizer:
    def __init__(self, classes: dict[int, str]) -> None:
        self._classes = classes
        self._color_palette = np.random.uniform(0, 255, size=(len(classes), 3))

    def visualize(self, img: np.ndarray, boxes: list[DetectedBox]) -> Image.Image:
        for box in boxes:
            self.draw_box(img, box)

        return Image.fromarray(img)

    def draw_box(
        self,
        img: np.ndarray,
        box: DetectedBox,
    ) -> None:
        color_palette = np.random.uniform(0, 255, size=(len(self._classes), 3))
        color = color_palette[box.class_id]

        cv2.rectangle(
            img, (int(box.x0), int(box.y0)), (int(box.x1), int(box.y1)), color, 2
        )

        label = f"{box.class_name}: {box.score:.2f}"

        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        label_x = int(box.x0)
        label_y = int(box.y0 - 10 if box.y0 - 10 > label_height else box.y0 + 10)

        cv2.rectangle(
            img,
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height),
            color,
            cv2.FILLED,
        )

        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
