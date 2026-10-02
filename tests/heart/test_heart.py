"""Heart models through common.registry: the tabular model on its example patient, the image model on a test image."""

from common.registry import predict

from conftest import STATUSES, example_patient, first_test_image, run_image_tests


def test_heart_tabular_example_patient():
    result = predict("heart", "tabular", example_patient("heart"))
    assert result.organ == "heart" and result.modality == "tabular"
    assert result.status in STATUSES
    if result.probability is not None:
        assert 0.0 <= result.probability <= 1.0


@run_image_tests
def test_heart_image_test_split_sample():
    result = predict("heart", "image", first_test_image("heart"), explain=False)
    assert result.organ == "heart" and result.modality == "image"
    assert result.status in STATUSES
