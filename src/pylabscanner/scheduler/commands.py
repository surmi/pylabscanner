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


class StageAxis(Enum):
    """Enum for the order of the axes to be scanned."""

    x = 0
    y = 1
    z = 2

    def other_names(self):
        return [i for i in self._member_names_ if i != self.name]


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

    def ta(self, prev_position: Dict[str, float] | None = None):
        """`prev_position` set to `None` will use current position"""
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
        # stage: LTS,
        # detector: Detector,
        movement_axis: StageAxis,
        measuring_range: ndarray,
        manager: DeviceManager,
        starting_position: Dict[str, float],
        # measrng: ndarray,
        # order: List[StageAxis],
        # other_pos: List[float],
        t_ru: float,
        s_ru: float,
        # reverse: bool = False,
    ):
        # raise NotImplementedError
        # TODO: responsibility for reversing direction of scanning is on
        # the caller. For this `measuring_range` and `s_ru` need to be adjusted
        # ==============================================================

        # self.stage = stage
        # self.detector = detector
        self.manager = manager
        self.measuring_range = measuring_range
        self.t_ru = t_ru
        # self.reverse = reverse
        self.movement_axis = movement_axis
        self.ta = None
        self.last_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.data = {}

        vel_params = self.stage.velparams
        vel = steps2mm(vel_params["max_velocity"], self.stage.convunits["vel"])
        ds = self.measuring_range[-1] - self.measuring_range[-2]
        self.dt = ds / vel

        # prefill data
        for ax in self.movement_axis.other_names():
            self.data[ax] = np.full(self.measuring_range.shape, starting_position[ax])
        for ax in starting_position:
            self.last_position[ax] = (
                self.measuring_range[-1] + self.s_ru
                if ax == self.movement_axis.name
                else starting_position[ax]
            )

        # if self.reverse:
        #     self.final_pos = mm2steps(measrng.min() - s_ru, self.stage.convunits["pos"])
        #     self.data = {
        #         order[0].name: np.flip(self.measuring_range),  # TODO: testing
        #         # order[0].name: [i for i in reversed(self.measrng)],
        #     }
        # else:
        #     self.final_pos = mm2steps(measrng.max() + s_ru, self.stage.convunits["pos"])
        #     self.data = {
        #         order[0].name: self.measuring_range,
        #     }

        # self.data[order[1].name] = np.full(
        #     self.measuring_range.shape, other_pos[0]
        # )  # TODO: testing
        # self.data[order[1].name] = [other_pos[0]]*len(self.measrng)
        # self.data[order[2].name] = np.full(
        #     self.measuring_range.shape, other_pos[1]
        # )  # TODO: testing
        # self.data[order[2].name] = [other_pos[1]]*len(self.measrng)

    def __str__(self) -> str:
        return (
            f"Fly-by line scan on {self.movement_axis.name} axis over range "
            f"{self.measuring_range}"
        )
        # if self.reverse:
        #     return (
        #         f"Fly-by scan on stage {self.stage} over range "
        #         f"{self.measuring_range} (reversed)"
        #     )
        # else:
        #     return (
        #         f"Fly-by scan on stage {self.stage} over range "
        #         f"{self.measuring_range}"
        #     )

    def run(self):
        # start movement without waiting
        self.manager.move_stage_async(stage_destination=self.last_position)
        # asyncio.run(aso_move_devs(self.stage, pos=self.final_pos, waitfinished=False))

        # start measurement
        measurement = self.manager.measure_series(
            no_measurements=len(self.measuring_range),
            interval=self.dt,
            start_delay=self.t_ru,
        )
        # measurement = self.detector.measure_series(
        #     len(self.measuring_range), interval=self.dt, start_delay=self.t_ru
        # )
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

    # TODO: add the ta function as in pt_by_pt


class ActionPtByPt(Action):
    """Perform a point-by-point line scan on a stage."""

    def __init__(
        self,
        movement_axis: StageAxis,
        measuring_range: ndarray,
        manager: DeviceManager,
        starting_position: Dict[str, float],
    ):
        self.manager = manager
        self.measuring_range = measuring_range
        self.movement_axis = movement_axis
        self.ta = None
        self.last_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.data = {}

        # prefill data
        for ax in self.movement_axis.other_names():
            self.data[ax] = np.full(self.measuring_range.shape, starting_position[ax])
        for ax in starting_position:
            self.last_position[ax] = (
                self.measuring_range[-1]
                if ax == self.movement_axis.name
                else starting_position[ax]
            )

    def __str__(self):
        return (
            f"Point-by-point line scan on {self.movement_axis.name} axis over range "
            f"{self.measuring_range}"
        )

    def run(self):
        spm = self.manager.samples_per_measurement
        meas_val = np.zeros(shape=(len(self.measuring_range), spm))
        meas_pos = np.zeros(self.measuring_range.shape)
        for i, position in enumerate(self.measuring_range):
            # move to next point and measure
            self.manager.move_stage({self.movement_axis.name: position})
            meas_val[i, :] = self.manager.measure()
            meas_pos[i] = self.manager.current_position[self.movement_axis.name]

        self.data[self.movement_axis.name] = meas_pos
        self.data["MEASUREMENT"] = meas_val
        return self.data

    def get_ta(self):
        if self.ta is None:
            measurement_acquisition_time = self.manager.ta_measurement
            self.ta = measurement_acquisition_time
            previous_position = self.measuring_range[0]
            for destination in self.measuring_range[1:]:
                self.ta += self.manager.ta_move_stage(
                    destination=destination, previous_position=previous_position
                )
                self.ta += measurement_acquisition_time
                previous_position = destination
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
