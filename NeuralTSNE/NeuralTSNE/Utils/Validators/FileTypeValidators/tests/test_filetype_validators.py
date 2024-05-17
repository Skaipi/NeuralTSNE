import argparse
import os
import tempfile
from typing import List

import pytest

from NeuralTSNE.Utils.Validators.FileTypeValidators import (
    FileTypeWithExtensionCheck,
    FileTypeWithExtensionCheckWithPredefinedDatasets,
)


@pytest.fixture
def valid_temp_file(request):
    file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    temp_file_path = file.name
    file.close()
    yield temp_file_path
    os.remove(temp_file_path)


# TODO: Parametrize fixture to test multiple valid extensions through request


@pytest.fixture
def invalid_temp_file(request):
    file = tempfile.NamedTemporaryFile(suffix=".bat", delete=False)
    temp_file_path = file.name
    file.close()
    yield temp_file_path
    os.remove(temp_file_path)


def test_valid_extension(valid_temp_file: str):
    file_type = FileTypeWithExtensionCheck(valid_extensions="txt")
    result = file_type(valid_temp_file)
    assert result.name == valid_temp_file


def test_invalid_extension(invalid_temp_file: str):
    file_type = FileTypeWithExtensionCheck(valid_extensions="txt")
    with pytest.raises(argparse.ArgumentTypeError):
        file_type(invalid_temp_file)


@pytest.mark.parametrize(
    "temp_file_fixture",
    [
        pytest.param("valid_temp_file", marks=pytest.mark.valid_file),
        pytest.param("invalid_temp_file", marks=pytest.mark.invalid_file),
    ],
)
def test_no_extension_check(temp_file_fixture: str, request):
    file_type = FileTypeWithExtensionCheck()
    temp_file_path = request.getfixturevalue(temp_file_fixture)
    result = file_type(temp_file_path)
    assert result.name == temp_file_path


@pytest.fixture
def file_type_with_datasets(request):
    available_datasets = request.param.get("available_datasets", [])
    return FileTypeWithExtensionCheckWithPredefinedDatasets(
        valid_extensions="txt", available_datasets=available_datasets
    )


def test_valid_extension_with_datasets(valid_temp_file: str):
    file_type = FileTypeWithExtensionCheckWithPredefinedDatasets(valid_extensions="txt")
    result = file_type(valid_temp_file)
    assert result.name == valid_temp_file


def test_invalid_extension_with_datasets(invalid_temp_file: str):
    file_type = FileTypeWithExtensionCheckWithPredefinedDatasets(valid_extensions="txt")
    with pytest.raises(argparse.ArgumentTypeError):
        file_type(invalid_temp_file)


@pytest.mark.parametrize(
    "temp_file_fixture",
    [
        pytest.param("valid_temp_file", marks=pytest.mark.valid_file),
        pytest.param("invalid_temp_file", marks=pytest.mark.invalid_file),
    ],
)
def test_no_extension_check_with_datasets(temp_file_fixture: str, request):
    file_type = FileTypeWithExtensionCheckWithPredefinedDatasets()
    temp_file_path = request.getfixturevalue(temp_file_fixture)
    result = file_type(temp_file_path)
    assert result.name == temp_file_path


@pytest.mark.parametrize("dataset", ["dataset1", "dataset2"])
@pytest.mark.parametrize("available_datasets", [["dataset1", "dataset2"]])
def test_predefined_dataset(available_datasets: List[str], dataset: str):
    file_type = FileTypeWithExtensionCheckWithPredefinedDatasets(
        available_datasets=available_datasets
    )
    result = file_type(dataset)
    assert result == dataset


@pytest.mark.parametrize("dataset", ["dataset3", "invalid_dataset"])
@pytest.mark.parametrize("available_datasets", [["dataset1", "dataset2"]])
def test_invalid_dataset(available_datasets: List[str], dataset: str):
    file_type = FileTypeWithExtensionCheckWithPredefinedDatasets(
        valid_extensions="txt", available_datasets=available_datasets
    )
    with pytest.raises(argparse.ArgumentTypeError):
        file_type(dataset)


@pytest.mark.parametrize("dataset", ["dataset3", "invalid_dataset"])
@pytest.mark.parametrize("available_datasets", [["dataset1", "dataset2"]])
def test_invalid_dataset_with_no_extension(available_datasets: List[str], dataset: str):
    file_type = FileTypeWithExtensionCheckWithPredefinedDatasets(
        available_datasets=available_datasets
    )
    with pytest.raises(argparse.ArgumentTypeError):
        file_type(dataset)
