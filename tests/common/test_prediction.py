"""The shared result type and the image loader in common.preprocessing."""

import io

import numpy as np
from PIL import Image

from common import ACCEPTED, Prediction
from common.preprocessing.image import open_image


def test_prediction_to_dict_is_json_ready():
    result = Prediction(organ="heart", modality="tabular", status=ACCEPTED, label="Low Risk", probability=0.1,
                        positive=False, details={"threshold": np.float64(0.3)})
    payload = result.to_dict(arrays=False)
    assert payload["status"] == ACCEPTED and payload["probability"] == 0.1
    assert isinstance(payload["details"]["threshold"], float)


def test_open_image_accepts_every_source(tmp_path):
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    path = tmp_path / "x.png"
    path.write_bytes(buffer.getvalue())

    for source in (image, array, buffer.getvalue(), io.BytesIO(buffer.getvalue()), str(path), path):
        assert open_image(source).size == (8, 8)
