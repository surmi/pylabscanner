#!/usr/bin/env python
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from time import sleep
from typing import Dict, List

import numpy as np
from numpy import ndarray

from ..devices.devices import Detector, calc_startposmod
from ..devices.LTS import LTS, aso_home_devs, aso_move_devs, mm2steps, steps2mm
from ..devices.manager import DeviceManager


# Message parts for the bolometer
class LineType(Enum):
    """Enum for the type of line to be scanned."""

    FLYBY = 1  # measure while moving
    PTBYPT = 2  # move to position, measure, and repeat


class LineStart(Enum):
    """Enum for the start position of the line to be scanned."""

    CR = 1  # return to the beginning of the line before starting next line
    SNAKE = 2  # start at the end of the previous line


class AxisOrder(Enum):
    """Enum for the order of the axes to be scanned."""

    X = 0
    Y = 1
    Z = 2


class Action(ABC):
    """Abstract class for actions to be performed in a lab routines."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def run(self):
        pass


class ActionMoveTo(Action):
    """Move stage(s) to position(s)."""

    def __init__(self, manager: DeviceManager, destination: Dict[str, float]):
        self.manager = manager
        self.destination = destination

    def __str__(self) -> str:
        stage_str = ", ".join([axis_name for axis_name in self.destination])
        pos_str = ", ".join(
            [str(self.destination[axis_name]) for axis_name in self.destination]
        )
        return f"Moving stage(s) {stage_str} to position(s) {pos_str}"

    def run(self):
        self.manager.move_stage(self.destination)

    def ta(self, prev_position: List[float] | List[int] | None = None):
        return self.manager.ta_move_stage(
            destination=self.destination, previous_position=prev_position
        )


class ActionHome(Action):
    """Home stage(s)."""

    def __init__(self, manager: DeviceManager, stage_label: str | list[str] = "all"):
        self.manager = manager
        self.stage_label = stage_label

    def __str__(self) -> str:
        stage_str = ", ".join([axis_name for axis_name in self.destination])
        return f"Homing stage(s) {stage_str}"

    def run(self):
        self.manager.home(self.stage_label)

    def ta(
        self,
        position_from: dict[str, float] | None = None,
    ):
        return self.manager.ta_home(
            position_from=position_from, stage_label=self.stage_label
        )


class ActionFlyBy(Action):
    """Single line fly-by scan on a stage."""

    def __init__(
        self,
        stage: LTS,
        detector: Detector,
        measrng: ndarray,
        order: List[AxisOrder],
        other_pos: List[float],
        t_ru: float,
        s_ru: float,
        reverse: bool = False,
    ):
        self.stage = stage
        self.detector = detector
        self.measrng = measrng
        self.t_ru = t_ru
        self.reverse = reverse

        vel_params = self.stage.velparams
        vel = steps2mm(vel_params["max_velocity"], self.stage.convunits["vel"])
        ds = self.measrng[-1] - self.measrng[-2]
        self.dt = ds / vel

        if self.reverse:
            self.final_pos = mm2steps(measrng.min() - s_ru, self.stage.convunits["pos"])
            self.data = {
                order[0].name: np.flip(self.measrng),  # TODO: testing
                # order[0].name: [i for i in reversed(self.measrng)],
            }
        else:
            self.final_pos = mm2steps(measrng.max() + s_ru, self.stage.convunits["pos"])
            self.data = {
                order[0].name: self.measrng,
            }
        self.data[order[1].name] = np.full(
            self.measrng.shape, other_pos[0]
        )  # TODO: testing
        # self.data[order[1].name] = [other_pos[0]]*len(self.measrng)
        self.data[order[2].name] = np.full(
            self.measrng.shape, other_pos[1]
        )  # TODO: testing
        # self.data[order[2].name] = [other_pos[1]]*len(self.measrng)

    def __str__(self) -> str:
        if self.reverse:
            return (
                f"Fly-by scan on stage {self.stage} over range "
                f"{self.measrng} (reversed)"
            )
        else:
            return f"Fly-by scan on stage {self.stage} over range " f"{self.measrng}"

    def run(self):
        # start movement without waiting
        asyncio.run(aso_move_devs(self.stage, pos=self.final_pos, waitfinished=False))
        # start measurement
        measurement = self.detector.measure_series(
            len(self.measrng), interval=self.dt, start_delay=self.t_ru
        )
        # after measurement, wait for movement to finish
        timeout = 10
        interval = 0.1
        elapsed = 0
        while not self.stage.status["move_completed"]:
            sleep(interval)
            elapsed += interval
            if timeout < elapsed:
                break
        self.data["MEASUREMENT"] = measurement
        return self.data


class ActionPtByPt(Action):
    """Perform a point-by-point line scan on a stage."""

    def __init__(
        self,
        stage: LTS,
        detector: Detector,
        measrng: ndarray,
        order: List[AxisOrder],
        other_pos: List[float],
        reverse: bool = False,
    ):
        self.stage = stage
        self.detector = detector
        self.measrng = measrng
        self.reverse = reverse
        self.order = order
        self.ta = None
        self.last_position = [0.0, 0.0, 0.0]

        if self.reverse:
            self.data = {
                order[0].name: np.flip(self.measrng),  # TODO: testing
                # order[0].name: [i for i in reversed(self.measrng)],
            }
            self.last_position[self.order[0].value] = np.flip(self.measrng)[-1]
        else:
            self.data = {
                order[0].name: self.measrng,
            }
            self.last_position[self.order[0].value] = self.measrng[-1]
        self.data[order[1].name] = np.full(self.measrng.shape, other_pos[0])
        self.last_position[self.order[1].value] = self.data[order[1].name]
        self.data[order[2].name] = np.full(self.measrng.shape, other_pos[1])
        self.last_position[self.order[2].value] = self.data[order[2].name]
        # self.data[order[1].name] = [other_pos[0]]*len(self.measrng)
        # self.data[order[2].name] = [other_pos[1]]*len(self.measrng)

    def __str__(self):
        if self.reverse:
            return (
                f"Point-by-point line scan on stage {self.stage} over range "
                f"{self.measrng} (reversed)"
            )
        else:
            return (
                f"Point-by-point line scan on stage {self.stage} over range "
                f"{self.measrng}"
            )

    def run(self):
        # move to next point and measure
        meas_val = []
        meas_pos = []
        if self.reverse:
            for i in reversed(self.measrng):
                pos = mm2steps(i, self.stage.convunits["pos"])
                asyncio.run(aso_move_devs(self.stage, pos=pos, waitfinished=True))
                meas_val.append(self.detector.measure())
                meas_pos.append(
                    self.stage.status["position"]
                )  # TODO: require testing; most probably require conversion
                print(self.stage.status["position"])
                # meas_pos.append(i)
        else:
            for i in self.measrng:
                pos = mm2steps(i, self.stage.convunits["pos"])
                asyncio.run(aso_move_devs(self.stage, pos=pos, waitfinished=True))
                meas_val.append(self.detector.measure())
                meas_pos.append(i)

        self.data[self.order[0].name] = meas_pos
        self.data["MEASUREMENT"] = meas_val
        return self.data

    def get_ta(self):
        # TODO: the same for flyby?
        if self.ta is None:
            pts_per_line = self.measrng.size - 1
            dist_btw_meas = (
                self.measrng.max() - self.measrng.min()
            )  # TODO: shouldn't be here a division by the number of points?
            self.ta = pts_per_line * calc_movetime(stage=self.stage, dist=dist_btw_meas)
            self.ta += self.detector.acquisition_time() * self.measrng.size
        return self.ta


# def calc_startposmod(stage: LTS) -> Tuple[float, float]:
#     """Calculate modification of position due to the stage needing to ramp up
#     to constant velocity.

#     Args:
#         stage (APTDevice_Motor): stage object from which the velocity
#             parameters are taken.

#     Returns:
#         tuple(float, float): (ramp up distance, ramp up time).
#     """
#     vel_params = stage.velparams
#     max_velocity = steps2mm(vel_params["max_velocity"], stage.convunits["vel"])
#     acceleration = steps2mm(vel_params["acceleration"], stage.convunits["acc"])

#     t_ru = max_velocity / acceleration
#     s_ru = 1 / 2 * float(str(acceleration)) * float(str(t_ru)) ** 2
#     s_ru = math.ceil(s_ru)

#     return s_ru, t_ru


def calc_movetime(stage: LTS, dist: float) -> float:
    """Calculate time of movement based on the distance and parameters of the
    stage.

    Args:
        stage (LTS): stage under control
        dist (float): full distance to move over

    Returns:
        float: time of the movement
    """
    s_ru, t_ru = calc_startposmod(stage=stage)
    max_velocity = steps2mm(stage.velparams["max_velocity"], stage.convunits["vel"])
    acceleration = steps2mm(stage.velparams["acceleration"], stage.convunits["acc"])

    if dist <= 2 * s_ru:
        return 2 * np.sqrt(dist / acceleration)
    else:
        return 2 * t_ru * (dist - 2 * s_ru) / max_velocity
