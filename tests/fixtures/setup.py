import numpy as np
import pandas as pd
import pytest

from pylabscanner.devices import BoloMsgFreq, BoloMsgSamples, BoloMsgSensor
from pylabscanner.scheduler.commands import LineStart, LineType
from pylabscanner.utils import parse_range


def _generate_fake_line_data(no_line_pts: int, no_samp: int):
    data = []
    for _ in range(no_line_pts):
        data.append(np.random.uniform(high=3.3, size=no_samp))
    return data


@pytest.fixture
def mock_measurement_data() -> pd.DataFrame:
    measurement_range_x = parse_range("10.0:20.0:2")
    measurement_range_y = parse_range("10.0:12.0:2")
    measurement_range_z = parse_range("10.0")
    is_reverse = False
    no_samp = 1000
    data = pd.DataFrame({"x": [], "y": [], "z": [], "MEASUREMENT": []})
    for z in measurement_range_z:
        for y in measurement_range_y:
            if is_reverse:
                x_data = np.flip(measurement_range_x)
            else:
                x_data = measurement_range_x
            data_line = {
                "x": x_data,
                "y": np.full(measurement_range_x.shape, y),
                "z": np.full(measurement_range_x.shape, z),
                "MEASUREMENT": _generate_fake_line_data(
                    measurement_range_x.size, no_samp
                ),
            }
            is_reverse = not is_reverse
            data = pd.concat([data, pd.DataFrame(data=data_line)], ignore_index=True)
    return data


@pytest.fixture
def mock_metadata() -> dict[str, any]:
    metadata = {
        "detector name": "Luvitera THz Mini, 4 sensor bolometer line",
        "detector sensor number": BoloMsgSensor.FIRST,
        "detector sampling": BoloMsgSamples.S1000,
        "detector sampling frequency [kHz]": BoloMsgFreq.F10,
        "signal modulation frequency [Hz]": 500,
        "x axis range [beg:end:no pts|pos]": "10.0:20.0:10",
        "y axis range [beg:end:no pts|pos]": "10.0:12.0:2",
        "z axis range [beg:end:no pts|pos]": "10.0",
        "scanning mode": LineType.PTBYPT,
        "scanning line start": LineStart.SNAKE,
    }
    return metadata
