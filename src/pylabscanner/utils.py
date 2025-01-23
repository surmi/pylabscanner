import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Tuple
import csv

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

from .devices import BoloMsgFreq, BoloMsgSamples, BoloMsgSensor
from .devices.LTS import LTS, mm2steps
from .devices.manager import DetectorInitParams, DeviceManager, StageInitParams
from .scheduler.commands import LineStart, LineType

FILE_EXTENSIONS = ("h5", "he5", "hdf5", "hdf", "csv", "txt")


def setup_manager(
    stage_sn: dict[str, str],
    initialize: bool = True,
    is_mockup: bool = False,
    sensor: BoloMsgSensor | None = None,
    samples: BoloMsgSamples | None = None,
    freq: BoloMsgFreq | None = None,
) -> DeviceManager:
    stage_parameters = {}
    for axis in stage_sn:
        stage_parameters[axis] = StageInitParams(
            serial_number=stage_sn[axis],
            rev="LTSC" if axis == "z" else "LTS",
            initialize=initialize,
            is_mockup=is_mockup,
        )

    if sensor is not None and samples is not None and freq is not None:
        detector_parameters = DetectorInitParams(
            sensor=sensor,
            samples=samples,
            freq=freq,
            initialize=initialize,
            is_mockup=is_mockup,
        )
    else:
        detector_parameters = None

    manager = DeviceManager(
        stage_init_params=stage_parameters,
        detector_init_params=detector_parameters,
    )

    return manager


def conv_to_steps(
    stages: LTS | list[LTS], pos: int | float | list[int] | list[float]
) -> int | List[int]:
    """Convert required position to microsteps (used by LTS devices).

    Args:
        stages (LTS | list[LTS]): LTS object representing given stage (or whole
            list of them)
        pos (int | float | list[int] | list[float]): requested positions for
            corresponding LTS stages
    """
    if type(stages) is not list:
        stages = [stages]
    if type(pos) is not list:
        pos = [pos]

    retPos = []
    for s, p in zip(stages, pos):
        retPos.append(mm2steps(p, s.convunits["pos"]))
    if len(retPos) == 1:
        return retPos[0]
    else:
        return retPos


def parse_scan_parameters(linestart: str):
    if linestart == "snake":
        linestart = LineStart.SNAKE
    elif linestart == "cr":
        linestart = LineStart.CR
    return linestart


def parse_range(range: str) -> npt.NDArray | None:
    """Parse range option. Accepts single float or 3 floats separated with ':'.

    Args:
        range (str): range option

    Raises:
        ValueError: raised if wrong format of the string

    Returns:
        NDArray: array with generated positions to visit
    """
    if range is None:
        return None
    r = [float(i) for i in range.split(":")]
    if len(r) == 1:
        return np.linspace(r[0], r[0], 1)
    elif len(r) == 3:
        return np.linspace(r[0], r[1], int(r[2]))
    else:
        raise ValueError(
            "Wrong number of elements in the range string. Provide 1 or 3 numbers"
        )


def parse_detector_settings(
    detsens: str, detsamp: str, detfreq: str
) -> tuple[BoloMsgSamples, BoloMsgSamples, BoloMsgFreq]:
    """Parse detector settings.

    Args:
        detsens (int): sensor number
        detsamp (int): number of samples
        detfreq (int): sampling frequency

    Returns:
        tuple[BoloMsgSamples, BoloMsgSamples, BoloMsgFreq]: enum objects
            corresponding to the detector settings
    """
    # for el in BoloMsgFreq:
    #     if str(el.value[1]) == detfreq:
    #         detfreq = el
    detfreq = _parse_detector_frequency(detfreq=detfreq)
    for el in BoloMsgSamples:
        if str(el.value[1]) == detsamp:
            detsamp = el
    if detsens == "1":
        detsens = BoloMsgSensor.FIRST
    elif detsens == "2":
        detsens = BoloMsgSensor.SECOND
    elif detsens == "3":
        detsens = BoloMsgSensor.THIRD
    elif detsens == "4":
        detsens = BoloMsgSensor.FOURTH
    return (detsens, detsamp, detfreq)


def _parse_detector_frequency(detfreq: str) -> BoloMsgFreq:
    for el in BoloMsgFreq:
        if str(el.value[1]) == detfreq:
            return el


def parse_filepath(filepath: Path, timestamp: bool = False) -> Tuple[Path, str]:
    """Parse file path, timestamp, and extension options.

    Args:
        filepath (Path): output path.
        timestamp (bool): whether to add timestamp to the file name. Defaults to True.
        extension (None | str, optional): force specific extension. Defaults to None.

    Returns:
        Tuple[Path, str]: tuple containing output path and extension of the output file.
    """
    if filepath.is_dir():
        raise ValueError("'filepath' should contain path to a file")
        # TODO: add handling in CLI

    parent = filepath.parent
    stem = filepath.stem
    suffix = filepath.suffix

    if (
        suffix.split(".")[-1] not in FILE_EXTENSIONS
        or suffix.split(".")[-1].lower() == "txt"
    ):
        extension = "csv"
        suffix = suffix + f".{extension}"
    else:
        extension = suffix.split(".")[-1]

    if timestamp or filepath.exists():
        stem += "_{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    return (parent / f"{stem}{suffix}", extension)


def filepath_add_label(path: Path, label: str) -> Path:
    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    return parent / f"{stem}_{label}{suffix}"


def _closest_val(data: npt.ArrayLike, x: float) -> Tuple[int, float]:
    """Find value closest to 'x' in the 'data' and its index.

    Args:
        data (npt.ArrayLike): data to find 'x' in
        x (float): value to find

    Returns:
        Tuple[int, float]: index and value of the closest element
    """
    ind = np.argmin(np.abs(data - x))
    return ind, data[ind]


def postprocessing(
    data: pd.DataFrame,
    modulation_frequency: float = None,
    det_freq: float | int = None,
):
    """In-place FFT processing of raw data from measurements.
    Data after processing is appended as a new column.

    Args:
        data (pd.DataFrame): dataframe with raw data
        modulation_frequency (float, optional): frequency of signal modulation
        to read value from after application of FFT. Defaults to None.
        det_freq (float,int, optional): sampling frequency of the detector

    Raises:
        ValueError: raised if signal frequency or sampling frequency are not
            provided (only for 'fft' mode)
    """
    # If data is read from file, the type of elements in the 'MEASUREMENT'
    # column will be of string type. Deal with it here.
    if isinstance(data["MEASUREMENT"].at[0], str):
        workdata = (
            data["MEASUREMENT"]
            .str.strip("[]")
            .str.split(",")
            .map(np.array)
            .map(lambda x: x.astype(float))
        )
    else:
        workdata = data["MEASUREMENT"]

    # DATA PROCESSING
    if modulation_frequency is None:
        raise ValueError("Signal frequency required")
    if det_freq is None:
        raise ValueError("Signal sample spacing required")

    fft = workdata.map(lambda x: np.fft.rfft(x))

    # calculate closes frequency bin
    sample_spacing = 1 / det_freq
    freqs = []
    if isinstance(workdata[0], list):
        freqs = np.fft.rfftfreq(len(workdata[0]), sample_spacing)
    else:
        freqs = np.fft.rfftfreq(workdata[0].size, sample_spacing)
    ind, freq = _closest_val(freqs, modulation_frequency)
    # TODO: check if normalization is correct (1/N)
    val = fft.map(lambda x: np.abs(x[ind]) / x.size)
    data["FFT"] = val
    data["FFT_freq"] = freq
    return freq, val


def _predict_plot(data: pd.DataFrame, silent=False) -> Tuple[str, List[str]]:
    """Predict what kind of plot should be made based on the data properties.

    Args:
        data (pd.DataFrame): input data.
        silent (bool, optional): wheter to raise error on fail. Defaults to False.

    Raises:
        ValueError: raised on failed attempt to predict type of the plot. Only
            raised if 'silent' is equal to False.

    Returns:
        Tuple[str, List[str]]: type of the plot ('2D' or '3D') and order of
            axis (horizontal and vertical correspondingly).
    """
    ux = data["x"].unique()
    uy = data["y"].unique()
    uz = data["z"].unique()
    nx = ux.size
    ny = uy.size
    nz = uz.size

    testar = np.array([nx, ny, nz])
    if np.count_nonzero(np.equal(testar, 1)) == 1:
        # only one axis doesn't change
        # we need a 3D plot (2 axis + val)
        # e.g. imshow(), pcolormesh(), contour(), contourf()
        if nx == 1:
            axorder = ["y", "z"]
        elif ny == 1:
            axorder = ["x", "z"]
        else:
            axorder = ["x", "y"]
        return "3D", axorder
    elif np.count_nonzero(np.equal(testar, 1)) == 2:
        # two axis don't change
        # we need a 2D plot (1 axis + val)
        # e.g. scatter(), plot()
        if nx != 1:
            axorder = ["x"]
        elif ny != 1:
            axorder = ["y"]
        else:
            axorder = ["z"]
        return "2D", axorder
    elif not silent:
        raise ValueError("Can't predict type of the plot")
    return None


def plotting(
    data: pd.DataFrame,
    path: Path = None,
    save=False,
    show=True,
    plt_config: dict | None = None,
) -> Tuple[Figure, Any] | None:
    """Plot processed measurements.
    The out path in 'path' argument is assumed to be a full path to the output file
    including the file name.
    If 'save' argument is set to True, the plot will be saved to the file. If
    'show' argument is set to True, the plot will be displayed. Both options can
    be used at the same time. For both flags set to False the function skips
    execution and returns early.

    Args:
        data (pd.DataFrame): data.
        path (Path, optional): output path. Defaults to None.
        save (bool, optional): whether to save the plot instead of displaying
            it. Defaults to False.
        show (bool, optional): whether to save the plot instead of displaying
            it. Defaults to False.

    Raises:
        ValueError: raised when input data frame does not contain columns with
            processed data.

    Returns:
        Tuple[Figure, Any]: figure and axes objects with created plot(s).
    """
    # plt.set_loglevel('error')
    # logging.getLogger('PIL').setLevel(logging.ERROR)
    if not (save or show):
        return None
    # if processed data available create two axes to display both
    data.sort_values(by=["x", "y", "z"], inplace=True)
    n = 0
    labels = []
    if "FFT" in data.columns:
        n += 1
        labels.append("FFT")
    if n <= 0:
        raise ValueError("No data to plot")

    collim = 3
    ncols = collim if n >= collim else n
    nrows = np.ceil(n / collim).astype(int)
    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(3 * (ncols + 0.5), nrows * 3),
        layout="constrained",
    )

    # define type of plots
    label_axis = None
    if plt_config is None:
        pltype, axorder = _predict_plot(data)
    else:
        pltype = plt_config["pltype"]
        axorder = plt_config["axorder"]
    if isinstance(axs, plt.Axes):
        axs = np.array([axs])
    for label, ax in zip(labels, axs.flat):
        if pltype == "2D":
            x = data[axorder[0]]
            y = data[label]
            ax.plot(x, y, "o", ms=4)
            ax.set_title(label)
            label_axis = "_" + axorder[0]
        elif pltype == "3D":
            ux = data[axorder[0]].unique()
            x = data[axorder[0]]
            uy = data[axorder[1]].unique()
            y = data[axorder[1]]
            val = data[label].to_numpy().reshape((uy.size, ux.size))
            img = ax.imshow(
                val,
                cmap="inferno",
                aspect="equal",
                origin="lower",
                extent=[x.min(), x.max(), y.min(), y.max()],
            )
            fig.colorbar(img, ax=ax, orientation="horizontal")
            ax.set_title(label)
            label_axis = "_" + axorder[0] + axorder[1]
    if save:
        path.parent.mkdir(exist_ok=True)

        # add label to the file name
        parent = path.parent
        stem = path.stem + "_plot"
        if label_axis is not None:
            stem += label_axis
        format = "png"
        path = parent / f"{stem}.{format}"

        plt.savefig(fname=path, format=format, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axs


def _match_enum(key: str, value: Any) -> Any:
    """_summary_

    Args:
        key (str): string matching enum.
        value (Any): string matching specific value of enum.

    Returns:
        Any: matched enum value or `value`.
    """
    if key == "detector sensor number":
        return BoloMsgSensor[value]
    elif key == "detector sampling":
        return BoloMsgSamples[value]
    elif key == "detector sampling frequency [kHz]":
        return BoloMsgFreq[value]
    elif key == "scanning line start":
        return LineStart[value]
    elif key == "scanning mode":
        return LineType[value]
    return value


def load_data(path: Path) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Load data from file.
    Handled extensions: `.csv` and `.h5`.
    Metadata is only returned for `.h5` files.

    Args:
        path (Path): path to the file.

    Raises:
        NotImplementedError: raised for unhandled extensions.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]|None]: data and metadata for `.h5`;
            otherwise data and `None`.
    """
    if path.suffix == ".h5":
        data = {}
        metadata = {}
        with h5py.File(path, mode="r") as f:
            for descr in f["data"].dtype.descr:
                k = descr[0]
                if k == "MEASUREMENT":
                    measurements = []
                    for i in range(f["data"][k].shape[0]):
                        measurements.append(f["data"][k][i])
                    data[k] = measurements
                else:
                    data[k] = f["data"][k]
            for k in f.attrs.keys():
                metadata[k] = _match_enum(k, f.attrs[k])
            data = pd.DataFrame(data)
        return data, metadata
    elif path.suffix == ".csv":
        metadata = None
        data = pd.read_csv(path, index_col=0)
        if "MEASUREMENT" in data.columns:
            measurement_col = data["MEASUREMENT"]
            measurement = []
            for _, measurement_str in enumerate(measurement_col):
                measurement_str = measurement_str[1:-1]
                measurement.append(np.fromstring(measurement_str, sep=" "))
            data["MEASUREMENT"] = measurement
            return data, metadata
    else:
        raise NotImplementedError(f"Files with {path.suffix} extension are not handled")


def saving(
    data: pd.DataFrame,
    path: Path,
    extension: str,
    metadata: dict | None = None,
    label: str = None,
) -> None:
    """Save data to a file.
    Choice to what format of file is made based on the `extension` parameter.
    Two choices are available currently: `csv` and `hdf5`.

    NOTE: saving to `csv` reduces precision of floating numbers in
    `MEASUREMENT` column.

    Args:
        data (pd.DataFrame): data frame with measurements.
        path (Path): path to where the data should be saved.
        extension (str): requested output file extension.
        metadata (dict | None, optional): measurement metadata to attach. Defaults to None.
        label (str, optional): if provided, will be attached to the file name.
            Defaults to None.
    """
    if label is not None:
        # modify filename
        parent = path.parent
        stem = path.stem + f"_{label}"
        suffix = path.suffix
        path = parent / f"{stem}{suffix}"
    path.parent.mkdir(exist_ok=True)
    if extension in ("h5", "he5", "hdf5", "hdf"):
        ds_dtype = [
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("MEASUREMENT", np.float64, data["MEASUREMENT"][0].shape),
        ]
        ds_arr = np.recarray(data["x"].shape, dtype=ds_dtype)
        ds_arr["x"] = data["x"]
        ds_arr["y"] = data["y"]
        ds_arr["z"] = data["z"]
        ds_arr["MEASUREMENT"] = np.zeros(
            (data["MEASUREMENT"].size, data["MEASUREMENT"][0].size)
        )
        for i, v in enumerate(data["MEASUREMENT"]):
            ds_arr["MEASUREMENT"][i, :] = v

        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=ds_arr, maxshape=(None))
            if metadata is not None:
                for key, value in metadata.items():
                    if isinstance(value, Enum):
                        f.attrs.create(key, value.name)
                    else:
                        f.attrs[key] = value if value is not None else "None"
    elif extension == "csv":
        with path.open("w+") as f:
            data.to_csv(f, index_label="no.")
        if metadata is not None:
            metadata_path = filepath_add_label(path, "meta")
            with metadata_path.open("w+") as f:
                field_names = list(metadata.keys())
                writer = csv.DictWriter(f, fieldnames=field_names)
                writer.writeheader()
                writer.writerow(metadata)
