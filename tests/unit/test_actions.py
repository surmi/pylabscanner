import numpy as np
import pytest

from pylabscanner.devices.manager import DeviceManager
from pylabscanner.scheduler.commands import (
    ActionHome,
    ActionMoveTo,
    ActionPtByPt,
    StageAxis,
)

from .test_devices import _define_init_params


@pytest.mark.detector
@pytest.mark.stage
@pytest.mark.device
class TestAction:
    _homed_position = {"x": 0, "y": 0, "z": 0}

    def _setup_manager(self, mock_devices):
        detector_init_params, stage_init_params = _define_init_params(mock_devices)
        manager = DeviceManager(
            stage_init_params=stage_init_params,
            detector_init_params=detector_init_params,
        )
        manager.initialize()
        manager.home("all")
        return manager

    def _assert_measurement_data(
        self,
        data: list | np.ndarray,
        no_measurements: int | None = None,
        min: float | np.float64 | None = None,
        max: float | np.float64 | None = None,
        arr: np.ndarray | None = None,
    ):
        if no_measurements is not None:
            assert len(data) == no_measurements
        if max is not None:
            assert np.max(data) == max
        if min is not None:
            assert np.min(data) == min
        if arr is not None:
            data_sorted = np.sort(data)
            arr_sorted = np.sort(arr)
            # NOTE: this will only work for exact values
            assert np.all(data_sorted == arr_sorted)

    def test_move_to(self, mock_devices: bool):
        manager = self._setup_manager(mock_devices=mock_devices)
        destination = {"x": 50.0, "y": 35.0, "z": 10.0}
        action = ActionMoveTo(manager=manager, destination=destination)
        action.run()
        assert manager.current_position == destination
        assert action.ta(prev_position=self._homed_position) > 0.0

    def test_home(self, mock_devices: bool):
        manager = self._setup_manager(mock_devices=mock_devices)
        destination = {"x": 50.0, "y": 35.0, "z": 10.0}
        ActionMoveTo(manager=manager, destination=destination).run()
        assert manager.current_position == destination
        action = ActionHome(manager=manager)
        assert action.ta() > 0.0
        action.run()
        assert manager.current_position == self._homed_position

    def test_fly_by(self, mock_devices: bool):
        pass

    def test_pt_by_pt(self, mock_devices: bool):
        manager = self._setup_manager(mock_devices=mock_devices)
        no_measurements = 10
        measuring_range = np.linspace(10.0, 100.0, num=no_measurements)
        starting_position = {"x": np.min(measuring_range), "y": 0, "z": 0}
        final_position = {"x": np.max(measuring_range), "y": 0, "z": 0}
        assert manager.current_position == self._homed_position
        action = ActionPtByPt(
            movement_axis=StageAxis.x,
            measuring_range=measuring_range,
            manager=manager,
            starting_position=starting_position,
        )
        data = action.run()

        assert manager.current_position == final_position
        self._assert_measurement_data(
            data["x"],
            no_measurements=no_measurements,
            min=min(measuring_range),
            max=max(measuring_range),
            arr=measuring_range,
        )
        for axis_label in ["y", "z"]:
            self._assert_measurement_data(
                data[axis_label],
                no_measurements=no_measurements,
                min=starting_position[axis_label],
                max=starting_position[axis_label],
            )
        self._assert_measurement_data(
            data=data["MEASUREMENT"], no_measurements=no_measurements
        )
        ta = action.get_ta()
        assert ta > 0.0
