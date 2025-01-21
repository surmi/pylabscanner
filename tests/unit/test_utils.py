from enum import Enum
from pathlib import Path
import re

import h5py
import numpy as np
import pytest
from numpy.typing import ArrayLike
from pandas import DataFrame, read_csv

from pylabscanner import utils


def test_saving_with_metadata(
    tmp_path: Path, mock_measurement_data: DataFrame, mock_metadata: dict[str, any]
):
    path = tmp_path / "temp_measurement.h5"
    utils.saving(mock_measurement_data, metadata=mock_metadata, path=path)
    with h5py.File(path, mode="r") as h5f:
        # check data
        file_data = h5f["data"]
        for i in range(file_data.size):
            file_row = file_data[i]
            for descr in file_data.dtype.descr:
                k = descr[0]
                if k == "MEASUREMENT":
                    file_measurement = file_row[k]
                    mock_measurement = mock_measurement_data[k][i]
                    assert np.array_equal(file_measurement, mock_measurement)
                else:
                    assert file_row[k] == mock_measurement_data[k][i]
        # check metadata
        for k in mock_metadata:
            if isinstance(mock_metadata[k], Enum):
                assert h5f.attrs[k] == mock_metadata[k].name
            else:
                assert h5f.attrs[k] == mock_metadata[k]


def test_loading_with_metadata(
    tmp_path: Path, mock_measurement_data: DataFrame, mock_metadata: dict[str, any]
):
    path = tmp_path / "temp_measurement.h5"
    utils.saving(mock_measurement_data, metadata=mock_metadata, path=path)
    loaded_data, loated_metadata = utils.load_data(path)
    for k in mock_measurement_data.columns:
        if k == "MEASUREMENT":
            for i in range(mock_measurement_data[k].shape[0]):
                assert (mock_measurement_data[k][i] == loaded_data[k][i]).all()
        else:
            mock_column = mock_measurement_data[k]
            loaded_column = loaded_data[k]
            assert (mock_column == loaded_column).all()
    for k in mock_metadata:
        assert mock_metadata[k] == loated_metadata[k]


def _arrays_match_within_threshold(arr1: ArrayLike, arr2: ArrayLike, threshold: float):
    arr_diff = np.abs(arr1 - arr2)
    return (arr_diff < threshold).all()


def test_saving_without_metadata(tmp_path: Path, mock_measurement_data: DataFrame):
    path = tmp_path / "temp_measurement.csv"
    utils.saving(mock_measurement_data, path=path)
    saved_data = read_csv(path, index_col=0)
    for k in mock_measurement_data.columns:
        if k == "MEASUREMENT":
            saved_col = saved_data["MEASUREMENT"]
            for i, ar_str in enumerate(saved_col):
                loaded_ar = ar_str[1:-1]
                loaded_ar = np.fromstring(loaded_ar, sep=" ")
                mock_ar = mock_measurement_data["MEASUREMENT"][i]
                assert _arrays_match_within_threshold(loaded_ar, mock_ar, 10 ** (-8))
        else:
            mock_column = mock_measurement_data[k]
            saved_column = saved_data[k]
            assert (mock_column == saved_column).all()


def test_loading_without_metadata(tmp_path: Path, mock_measurement_data: DataFrame):
    path = tmp_path / "temp_measurement.csv"
    utils.saving(mock_measurement_data, path=path)
    loaded_data, loaded_metadata = utils.load_data(path)
    assert loaded_metadata is None
    for k in mock_measurement_data.columns:
        if k == "MEASUREMENT":
            measurement_col = loaded_data["MEASUREMENT"]
            for i, loaded_measurement in enumerate(measurement_col):
                mock_measurement = mock_measurement_data["MEASUREMENT"][i]
                assert _arrays_match_within_threshold(
                    loaded_measurement, mock_measurement, 10 ** (-8)
                )
        else:
            mock_column = mock_measurement_data[k]
            loaded_column = loaded_data[k]
            assert (mock_column == loaded_column).all()


def test_loading_unknown_extension(tmp_path: Path):
    path = tmp_path / "temp_measurement.json"
    with pytest.raises(NotImplementedError):
        utils.load_data(path)


def test_parse_filepath(tmp_path: Path):
    # raises exception on path pointing to directory
    dir_path = tmp_path / "subdir"
    dir_path.mkdir()
    with pytest.raises(ValueError):
        utils.parse_filepath(dir_path)

    # returns extensions as in a constant
    valid_file_paths = [dir_path / f"filename.{ext}" for ext in utils.FILE_EXTENSIONS]
    valid_file_paths.append(dir_path / "filename.CSV")
    valid_file_paths.append(dir_path / "filename.txt.csv")
    valid_file_paths.append(dir_path / "filename.png.csv")
    for p in valid_file_paths:
        _, ext = utils.parse_filepath(p)
        assert ext.lower() in utils.FILE_EXTENSIONS

    # recovers from unknown extensions
    invalid_file_paths = [dir_path / "filepath.doc", dir_path / "filepath.png"]
    for p in invalid_file_paths:
        _, ext = utils.parse_filepath(p)
        assert ext == "csv"

    # adds timestamp if file already exists
    existing_file_path = dir_path / "filename.csv"
    existing_file_path.write_text("content", encoding="utf-8")
    parsed_path, _ = utils.parse_filepath(existing_file_path)
    result = re.fullmatch(
        r"[0-9]{8}[0-9]*_[0-9]{6}\Z", parsed_path.stem.replace("filename_", "")
    )
    assert result is not None

    # adds timestamp if provided
    notexisting_file_path = dir_path / "filename_notexisting.csv"
    parsed_path, _ = utils.parse_filepath(notexisting_file_path)
    print(parsed_path)
    print(notexisting_file_path.exists())
    notmatching_result = re.fullmatch(
        r"[0-9]{8}[0-9]*_[0-9]{6}\Z",
        parsed_path.stem.replace("filename_notexisting_", ""),
    )
    assert notmatching_result is None
    parsed_path, _ = utils.parse_filepath(notexisting_file_path, timestamp=True)
    matching_result = re.fullmatch(
        r"[0-9]{8}[0-9]*_[0-9]{6}\Z",
        parsed_path.stem.replace("filename_notexisting_", ""),
    )
    assert matching_result is not None
