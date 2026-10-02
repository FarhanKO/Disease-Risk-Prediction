"""Kidney models through common.registry: the tabular model on its example patient, the image model on a test image."""

from common.registry import predict

from conftest import STATUSES, example_patient, first_test_image, run_image_tests


def test_kidney_tabular_example_patient():
    result = predict("kidney", "tabular", example_patient("kidney"))
    assert result.organ == "kidney" and result.modality == "tabular"
    assert result.status in STATUSES
    if result.probability is not None:
        assert 0.0 <= result.probability <= 1.0


@run_image_tests
def test_kidney_image_test_split_sample():
    result = predict("kidney", "image", first_test_image("kidney"), explain=False)
    assert result.organ == "kidney" and result.modality == "image"
    assert result.status in STATUSES
