import numpy as np
import pytest

from pylabscanner.devices.devices import DetectorInitParams
from pylabscanner.devices.manager import DeviceManager, StageInitParams
from pylabscanner.scheduler.commands import LineStart, LineType
from pylabscanner.scheduler.scheduler import ScanScheduler
from fixtures.setup import *


def pytest_addoption(parser):
    parser.addoption(
        "--use-mocks",
        action="store_true",
        help="Use mocks devices instead of actual devices.",
    )


@pytest.fixture
def mock_devices(request):
    return request.config.getoption("--use-mocks")


@pytest.fixture
def default_homed_position() -> dict[str, float]:
    return {"x": 0, "y": 0, "z": 0}


@pytest.fixture
def default_init_params(
    mock_devices: bool,
) -> tuple[DetectorInitParams, dict[str, StageInitParams]]:
    return (
        DetectorInitParams(is_mockup=mock_devices),
        {
            "x": StageInitParams(serial_number="123", is_mockup=mock_devices),
            "y": StageInitParams(serial_number="123", is_mockup=mock_devices),
            "z": StageInitParams(serial_number="123", is_mockup=mock_devices),
        },
    )


@pytest.fixture
def default_manager(
    default_init_params: tuple[DetectorInitParams, dict[str, StageInitParams]],
) -> DeviceManager:
    detector_init_params, stage_init_params = default_init_params
    manager = DeviceManager(
        stage_init_params=stage_init_params,
        detector_init_params=detector_init_params,
    )
    manager.initialize()
    manager.home("all")
    return manager


@pytest.fixture
def default_scheduler(default_manager: DeviceManager) -> ScanScheduler:
    ranges = {
        "x": np.linspace(0, 100, 3, endpoint=True),
        "y": np.linspace(0, 100, 3, endpoint=True),
        "z": np.linspace(0, 100, 3, endpoint=True),
    }
    line_type = LineType.PTBYPT
    line_start = LineStart.SNAKE
    fin_home = True
    use_tqdm = True
    scheduler = ScanScheduler(
        manager=default_manager,
        ranges=ranges,
        line_type=line_type,
        line_start=line_start,
        fin_home=fin_home,
        use_tqdm=use_tqdm,
    )
    return scheduler
