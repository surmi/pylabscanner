import pytest

from pylabscanner.devices.manager import DeviceManager
from pylabscanner.scheduler.commands import ActionHome, ActionMoveTo

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
