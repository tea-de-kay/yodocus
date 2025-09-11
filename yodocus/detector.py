# Inspired by https://github.com/ultralytics/ultralytics/blob/d346c132a87a7193639af1a7c6977460958e737b/examples/YOLOv8-ONNXRuntime/main.py

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from yodocus.__spi__ import (
    DetectedBox,
    DetectionConfig,
    DetectionResult,
    YoloModelConfig,
)
from yodocus.visualizer import Visualizer


class Detector:
    """YOLO object detection handling ONNX inference and visualization."""

    def __init__(self, model: str):
        self._model_path, self._model_config = self._load(model)
        self._classes = {
            idx: name for idx, name in enumerate(self._model_config.class_names)
        }

        self._session = ort.InferenceSession(
            self._model_path, providers=self._get_ort_providers()
        )
        self._model_inputs = self._session.get_inputs()

    @property
    def classes(self) -> dict[int, str]:
        return self._classes

    @property
    def config(self) -> YoloModelConfig:
        return self._model_config

    @property
    def input_size(self) -> tuple[int, int]:
        return self._model_config.input_size

    @property
    def input_height(self) -> int:
        return self._model_config.input_size[0]

    @property
    def input_width(self) -> int:
        return self._model_config.input_size[1]

    @staticmethod
    def _load(model: str) -> tuple[str, YoloModelConfig]:
        dir = Path(model)
        model_path = dir / "model.onnx"
        config_path = dir / "config.yaml"

        if not model_path.exists() or not config_path.exists():
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(repo_id=model, filename="model.onnx")
            config_path = hf_hub_download(repo_id=model, filename="config.yaml")

        config = YoloModelConfig.from_yaml(Path(config_path))
        return str(model_path), config

    @staticmethod
    def _get_ort_providers() -> list[str]:
        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider"]

        providers.append("CPUExecutionProvider")

        return providers

    def resize(self, img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = img.shape[:2]
        new_h, new_w = self._model_config.input_size

        if self._model_config.preserve_aspect_ratio:
            ratio = min(new_h / h, new_w / w)
            new_unpad_h = int(round(h * ratio))
            new_unpad_w = int(round(w * ratio))
            dh = (new_h - new_unpad_h) / 2
            dw = (new_w - new_unpad_w) / 2

            if h != new_unpad_h or w != new_unpad_w:
                img = cv2.resize(
                    img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR
                )
            top_pad, bottom_pad = int(round(dh - 0.1)), int(round(dh + 0.1))
            left_pad, right_pad = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(
                img,
                top_pad,
                bottom_pad,
                left_pad,
                right_pad,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
        else:
            top_pad = 0
            left_pad = 0
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return img, (top_pad, left_pad)

    def preprocess(self, image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
        img = np.array(image)

        img, pad = self.resize(img)

        # Normalize
        image_data = img / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data, pad

    def postprocess(
        self,
        model_output: list[np.ndarray],
        input_image: np.ndarray,
        padding: tuple[int, int],
        config: DetectionConfig,
    ) -> DetectionResult:
        # Transpose and squeeze the output to match the expected shape
        model_outputs = np.transpose(np.squeeze(model_output[0]))

        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        orig_h, orig_w = input_image.shape[:2]
        new_h, new_w = self._model_config.input_size
        gain_h = new_h / orig_h
        gain_w = new_w / orig_w

        model_outputs[:, 0] -= padding[1]
        model_outputs[:, 1] -= padding[0]

        for row in model_outputs:
            # Extract the class scores from the current row
            classes_scores = row[4:]
            max_score = np.amax(classes_scores)

            if max_score >= config.conf_threshold:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = row[0], row[1], row[2], row[3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) / gain_w)
                top = int((y - h / 2) / gain_h)
                width = int(w / gain_w)
                height = int(h / gain_h)

                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, config.conf_threshold, config.iou_threshold
        )

        detected_boxes: list[DetectedBox] = []
        for i in indices:
            box = boxes[i]
            score = float(scores[i])
            class_id = int(class_ids[i])
            class_name = self._classes[class_id]
            x0, y0, w, h = box
            x1 = x0 + w
            y1 = y0 + h
            detected_box = DetectedBox(
                class_id=class_id,
                class_name=class_name,
                score=score,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
            )
            detected_boxes.append(detected_box)

        visual = None
        if config.visualize:
            visualizer = Visualizer(classes=self._classes)
            visual = visualizer.visualize(input_image, detected_boxes)

        return DetectionResult(boxes=detected_boxes, visual=visual)

    def detect(self, image: Image.Image, config: DetectionConfig) -> DetectionResult:
        img_data, padding = self.preprocess(image)

        outputs = cast(
            list[np.ndarray],
            self._session.run(None, {self._model_inputs[0].name: img_data}),
        )

        result = self.postprocess(
            model_output=outputs,
            input_image=np.array(image),
            padding=padding,
            config=config,
        )

        return result
