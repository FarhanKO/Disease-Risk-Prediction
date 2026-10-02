"""Lung models through common.registry: the tabular model on its example patient, the image model on a test image."""

from common.registry import predict

from conftest import STATUSES, example_patient, first_test_image, run_image_tests


def test_lung_tabular_example_patient():
    result = predict("lung", "tabular", example_patient("lung"))
    assert result.organ == "lung" and result.modality == "tabular"
    assert result.status in STATUSES
    if result.probability is not None:
        assert 0.0 <= result.probability <= 1.0


@run_image_tests
def test_lung_image_test_split_sample():
    result = predict("lung", "image", first_test_image("lung"), explain=False)
    assert result.organ == "lung" and result.modality == "image"
    assert result.status in STATUSES
