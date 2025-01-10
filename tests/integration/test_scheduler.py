import numpy as np
import pytest

from pylabscanner.devices.manager import DeviceManager
from pylabscanner.scheduler.commands import (
    Action,
    ActionHome,
    ActionMoveTo,
    ActionPtByPt,
    LineStart,
    LineType,
)
from pylabscanner.scheduler.scheduler import ScanScheduler


@pytest.mark.detector
@pytest.mark.stage
@pytest.mark.device
class TestScheduler:
    def _count_action_type(self, action_type: Action, actions: list[Action]):
        counter = 0
        for action in actions:
            if isinstance(action, action_type):
                counter += 1
        return counter

    def test_scheduler_initializes(self, default_manager: DeviceManager):
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
        assert len(scheduler.data["MEASUREMENT"]) == 0
        assert len(scheduler.data["x"]) == 0
        assert len(scheduler.data["y"]) == 0
        assert len(scheduler.data["z"]) == 0
        assert len(scheduler.actions) == 0
        assert len(scheduler.history) == 0
        assert not scheduler.is_built
        assert scheduler.ta == 0
        assert scheduler.ta_act == 0

    def test_schedule_ptbypt(self, default_manager: DeviceManager):
        ranges = {
            "x": np.linspace(0, 100, 3, endpoint=True),
            "y": np.linspace(0, 100, 3, endpoint=True),
            "z": np.linspace(100, 100, 1, endpoint=True),
        }
        scheduler = ScanScheduler(
            manager=default_manager,
            ranges=ranges,
            line_start=LineStart.SNAKE,
            line_type=LineType.PTBYPT,
        )
        scheduler.make_schedule()
        assert len(scheduler.actions) > 0
        assert len(scheduler.history) == 0
        assert scheduler.is_built
        assert scheduler.ta > 0

        # assert the created path is correct
        assert (
            self._count_action_type(action_type=ActionPtByPt, actions=scheduler.actions)
            == 3
        )
        assert (
            self._count_action_type(action_type=ActionMoveTo, actions=scheduler.actions)
            == 3
        )
        assert (
            self._count_action_type(action_type=ActionHome, actions=scheduler.actions)
            == 1
        )
        scan_actions = [x for x in scheduler.actions if isinstance(x, ActionPtByPt)]
        forward_last_position = scan_actions[0].last_position
        assert forward_last_position["x"] == ranges["x"][-1]
        assert forward_last_position["y"] == ranges["y"][0]
        assert forward_last_position["z"] == ranges["z"][0]
        backward_last_position = scan_actions[1].last_position
        assert backward_last_position["x"] == ranges["x"][0]
        assert backward_last_position["y"] == ranges["y"][1]
        assert backward_last_position["z"] == ranges["z"][0]

    @pytest.mark.skip(reason="Functionality not implemented yet")
    def test_schedule_flyby(self, default_manager: DeviceManager):
        pass

    def test_schedule_run(self, default_manager: DeviceManager):
        ranges = {
            "x": np.linspace(0, 100, 3, endpoint=True),
            "y": np.linspace(0, 100, 3, endpoint=True),
            "z": np.linspace(100, 100, 1, endpoint=True),
        }
        scheduler = ScanScheduler(
            manager=default_manager,
            ranges=ranges,
            line_start=LineStart.SNAKE,
            line_type=LineType.PTBYPT,
        )
        scheduler.make_schedule()
        assert len(scheduler.actions) > 0
        assert len(scheduler.history) == 0
        assert scheduler.is_built
        assert scheduler.ta > 0
        assert scheduler.ta_act == 0.0
        scheduler.run()
        assert len(scheduler.history) > 0
        assert scheduler.ta_act > 0.0
