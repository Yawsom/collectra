"""Exercise TIFF inference through the real Ultralytics preprocessing/backend."""

import numpy as np
import pytest
import tifffile
import yaml

from collectra import Image, ObjectDetectionYOLO
from collectra.types.images import Orientation


@pytest.fixture(scope="module")
def multichannel_yolo(tmp_path_factory):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    config = yaml_model_load("yolo11n.yaml")
    config["channels"] = 75
    path = tmp_path_factory.mktemp("yolo") / "yolo11n.yaml"
    path.write_text(yaml.safe_dump(config))
    model = YOLO(path)
    args = model.model.args
    model.model = DetectionModel(config, ch=75, nc=1, verbose=False)
    model.model.args = args
    model.model.task = "detect"
    model.overrides.update(device="cpu", verbose=False)
    return model


@pytest.mark.parametrize("axes", ["CYX", "YXC", "QYX"])
@pytest.mark.parametrize("cropped", [False, True])
def test_tiff_inference_preserves_channels(
    tmp_path, monkeypatch, multichannel_yolo, axes, cropped
):
    from ultralytics.models.yolo.detect import DetectionPredictor

    # Distinct bands reveal dropped/reversed channels after preprocessing.
    pixels = np.broadcast_to(np.arange(75, dtype=np.uint8), (48, 64, 75)).copy()
    path = tmp_path / "views.tif"
    tifffile.imwrite(
        path,
        pixels if axes == "YXC" else pixels.transpose(2, 0, 1),
        photometric="minisblack",
        metadata={"axes": axes},
    )
    image = Image(name="views", data=path)
    expected_shape = (48, 64)
    if cropped:
        image = image.make_crop(0.5, 0.5, 0.5, 0.5)
        image.orientation = Orientation.WEST
        expected_shape = (32, 24)

    observed = []
    preprocess = DetectionPredictor.preprocess

    def capture_preprocess(predictor, images):
        assert images[0].shape == (*expected_shape, 75)
        tensor = preprocess(predictor, images)
        observed.append(tensor.shape)
        np.testing.assert_allclose(
            tensor[0, :, tensor.shape[2] // 2, tensor.shape[3] // 2].cpu().numpy(),
            np.arange(75) / 255,
            atol=1e-7,
        )
        return tensor

    monkeypatch.setattr(DetectionPredictor, "preprocess", capture_preprocess)
    task = ObjectDetectionYOLO("detector", model=multichannel_yolo)
    results = task.run(image, resolution=64)

    assert len(observed) == 1
    assert observed[0][0:2] == (1, 75)
    assert multichannel_yolo.predictor.results[0].orig_shape == expected_shape
    assert isinstance(results, list)


def test_tiff_inference_reports_channel_mismatch(tmp_path, multichannel_yolo):
    path = tmp_path / "gray.tif"
    tifffile.imwrite(path, np.zeros((48, 64), dtype=np.uint8))
    task = ObjectDetectionYOLO("detector", model=multichannel_yolo)

    with pytest.raises(ValueError, match="1 image channels.*expects 75"):
        task.run(Image(name="gray", data=path), resolution=64)
