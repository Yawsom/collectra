__all__ = ["OCRSuraya", "LineDetectorSurya"]

from dotenv import load_dotenv

from collectra.tasks.base import Task
from collectra.tasks.text_output import TextOutputTask
from collectra.types.images import Image, ImageCrop
from collectra.types.texts import Text

load_dotenv()


class OCRSuraya(TextOutputTask):
    def __init__(self, name: str, **kwargs) -> None:
        """Initialize the task without loading Surya model weights."""
        super().__init__(name, **kwargs)
        self.detection_predictor = None
        self.recognition_predictor = None

    def _init_predictors(self) -> None:
        """Load Surya predictors on first use."""
        if self.detection_predictor is not None and self.recognition_predictor is not None:
            return
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        self.detection_predictor = DetectionPredictor()
        foundation_predictor = FoundationPredictor()
        self.recognition_predictor = RecognitionPredictor(foundation_predictor)

    def input_type(self) -> type | tuple:
        """Define the expected input types for this LLM task.

        Returns:
            type | tuple: Tuple of str and Path types for text and file inputs.
        """
        return Image

    def output_type(self) -> type | tuple:
        """Define the expected output types for this LLM task.

        Returns:
            type | tuple: String type for generated text outputs.
        """
        return Text

    def run(self, *images: Image) -> list[Text]:
        """Recognize text in each image and return Text artefacts."""
        self._init_predictors()
        results: list[Text] = []
        for image in images:
            if not isinstance(image, Image):
                raise TypeError(f"image must be Image, got {type(image)}")
            predictions = self.recognition_predictor(
                [image.pil()], det_predictor=self.detection_predictor
            )
            text = "\n".join(line.text for line in predictions[0].text_lines)
            results.append(self.text_output(text))
        return results


class LineDetectorSurya(Task):
    def __init__(
        self,
        name: str,
        merge_horizontal: bool = False,
        min_height: int = 0,
        **kwargs,
    ) -> None:
        """Initialize the task without loading Surya model weights."""
        super().__init__(
            name,
            merge_horizontal=merge_horizontal,
            min_height=min_height or 0,
            **kwargs,
        )
        self.detection_predictor = None

    def _init_predictor(self) -> None:
        """Load the Surya detection predictor on first use."""
        if self.detection_predictor is not None:
            return
        from surya.detection import DetectionPredictor

        self.detection_predictor = DetectionPredictor()

    def input_type(self) -> type | tuple:
        """Define the expected input types for this LLM task.

        Returns:
            type | tuple: Tuple of str and Path types for text and file inputs.
        """
        return Image

    def output_type(self) -> type | tuple:
        """Define the expected output types for this LLM task.

        Returns:
            type | tuple: String type for generated text outputs.
        """
        return ImageCrop

    def run(self, *images: Image) -> list[ImageCrop]:
        """Detect text lines and return their image crops."""
        self._init_predictor()
        results: list[ImageCrop] = []
        for image in images:
            if not isinstance(image, Image):
                raise TypeError(f"image must be Image, got {type(image)}")
            pil_image = image.pil()
            predictions = self.detection_predictor([pil_image])

            # Sort bounding boxes vertically
            bounding_boxes = sorted(
                [polygon_box.bbox for polygon_box in predictions[0].bboxes],
                key=lambda bbox: bbox[1],
            )

            if len(bounding_boxes) == 0 or (
                self.min_height and image.height <= self.min_height
            ):
                crop = image.make_crop(0.5, 0.5, 1.0, 1.0)
                results.append(crop)
            elif self.merge_horizontal:
                bbox = bounding_boxes[0]
                top = bbox[1]
                bottom = bbox[3]

                for bbox in bounding_boxes:
                    if bbox[1] >= bottom:
                        crop = image.make_crop_bounding_box(
                            0, top, image.width, bottom, min_height=self.min_height
                        )
                        results.append(crop)
                        top = bbox[1]
                        bottom = bbox[3]
                    else:
                        top = min(top, bbox[1])
                        bottom = max(bottom, bbox[3])

                crop = image.make_crop_bounding_box(0, top, image.width, bottom)
                results.append(crop)
            else:
                for bbox in bounding_boxes:
                    crop = image.make_crop_bounding_box(
                        *bbox, min_height=self.min_height
                    )
                    results.append(crop)

        return results
