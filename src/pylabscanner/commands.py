#!/usr/bin/env python
import asyncio
import logging
import math
import threading
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from queue import Empty, Queue
from time import sleep, time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linspace, ndarray, sqrt
from thorlabs_apt_device.devices.aptdevice_motor import APTDevice_Motor
from tqdm import tqdm

from .devices import (
    BoloLine,
    BoloMsgFreq,
    BoloMsgSamples,
    BoloMsgSensor,
    Detector,
    Source,
)
from .LTS import LTS, LTSC, aso_home_devs, aso_move_devs, mm2steps, steps2mm


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

    def __init__(self, stages: List[LTS], pos: List[float]):
        self.stages = stages
        self.pos = pos
        self.step_pos = [
            mm2steps(pos[i], stages[i].convunits["pos"]) for i in range(len(pos))
        ]
        self.ta = None

    def __str__(self) -> str:
        stage_str = ", ".join([str(i) for i in self.stages])
        pos_str = ", ".join([str(i) for i in self.pos])
        return f"Move stage(s) {stage_str} to position(s) {pos_str}"

    def run(self):
        asyncio.run(aso_move_devs(self.stages, pos=self.step_pos))

    def get_ta(self, prev_position: List[float] | List[int]):
        max_ta = 0
        for stage, prev_pos, destination in zip(self.stages, prev_position, self.pos):
            distance = abs(prev_pos - destination)
            print(distance)
            print(type(distance))
            max_ta = max(max_ta, calc_movetime(stage, distance))
        self.ta = max_ta
        return self.ta


class ActionHome(Action):
    """Home stage(s)."""

    def __init__(self, stages: List[LTS]):
        self.stages = stages

    def __str__(self) -> str:
        stage_str = ", ".join([str(i) for i in self.stages])
        return f"Home stage(s) {stage_str}"

    def run(self):
        asyncio.run(aso_home_devs(self.stages))


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
            return f"Fly-by scan on stage {self.stage} over range {self.measrng} (reversed)"
        else:
            return f"Fly-by scan on stage {self.stage} over range {self.measrng}"

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
            return f"Point-by-point line scan on stage {self.stage} over range {self.measrng} (reversed)"
        else:
            return f"Point-by-point line scan on stage {self.stage} over range {self.measrng}"

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
            self.ta += self.detector.get_ta() * self.measrng.size
        return self.ta


def calc_startposmod(stage: LTS) -> Tuple[float, float]:
    """Calculate modification of position due to the stage needing to ramp up to constant velocity.

    Args:
        stage (APTDevice_Motor): stage object from which the velocity parameters are taken.

    Returns:
        tuple(float, float): (ramp up distance, ramp up time).
    """
    vel_params = stage.velparams
    max_velocity = steps2mm(vel_params["max_velocity"], stage.convunits["vel"])
    acceleration = steps2mm(vel_params["acceleration"], stage.convunits["acc"])

    t_ru = max_velocity / acceleration
    s_ru = 1 / 2 * float(str(acceleration)) * float(str(t_ru)) ** 2
    s_ru = math.ceil(s_ru)

    return s_ru, t_ru


def calc_movetime(stage: LTS, dist: float) -> float:
    """Calculate time of movement based on the distance and parameters of the stage.

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
        return 2 * sqrt(dist / acceleration)
    else:
        return 2 * t_ru * (dist - 2 * s_ru) / max_velocity


class ScanRoutine:
    """Routine for scanning with stages.

    Call `build()` before running (`run()` method) the routine to build list of steps to perform.
    """

    # TODO: validate fly-by range has at least 2 points
    # TODO: validate fly-by range can fit in the stage range
    def __init__(
        self,
        stages: List[LTS],
        detector: Detector,
        source: Source,
        ranges: List[ndarray],
        order: List[AxisOrder] = [AxisOrder.X, AxisOrder.Y, AxisOrder.Z],
        line_type: LineType = LineType.FLYBY,
        line_start: LineStart = LineStart.SNAKE,
        fin_home: bool = True,
        use_tqdm: bool = True,
    ):
        self.stages = stages
        self.detector = detector
        self.source = source
        self.ranges = ranges
        # in case if range is passed backwards
        for r in self.ranges:
            r.sort()
        self.order = order
        self.line_type = line_type
        self.line_start = line_start
        self.fin_home = fin_home
        self.use_tqdm = use_tqdm
        self.data = pd.DataFrame({"X": [], "Y": [], "Z": [], "MEASUREMENT": []})

        self.actions = []
        self.history = []
        self.is_built = False
        self.ta = 0.0  # acqusition time estimaiton (whole scan)
        self.ta_act = 0.0  # actual asqusition time

    def build(self):
        """Build the scan routine."""
        # line scan always starts at min
        start_pos = [range.min() for range in self.ranges]
        # vels = [
        #     steps2mm(stage.velparams["max_velocity"], stage.convunits["vel"])
        #     for stage in self.stages
        # ]
        t_ru = []
        s_ru = []
        for s in self.stages:
            ss, tt = calc_startposmod(s)
            s_ru.append(ss)
            t_ru.append(tt)

        # FLYBY ---------------------------------------------------------------
        if self.line_type == LineType.FLYBY:
            raise NotImplementedError(
                "FlyBy scanning not implemented yet"
            )  # TODO: require update after new changes to pt-by-pt scan will be tested
            # go to start position
            # s_ru, t_ru = calc_startposmod(self.stages[0])
            start_pos[0] = start_pos[0] - s_ru[0]
            self.actions.append(ActionMoveTo(self.stages, start_pos))
            maxind = max(range(len(start_pos)), key=start_pos.__getitem__)
            self.ta += calc_movetime(self.stages[maxind], max(start_pos))

            # do a single line
            reverse = False
            self.actions.append(
                ActionFlyBy(
                    self.stages[self.order[0].value],
                    self.detector,
                    self.ranges[self.order[0].value],
                    self.order,
                    start_pos[1:],
                    t_ru[self.order[0].value],
                    s_ru[self.order[0].value],
                    reverse=reverse,
                )
            )
            self.ta += calc_movetime(
                stage=self.stages[self.order[0].value],
                dist=2 * s_ru[self.order[0].value]
                + self.ranges[self.order[0].value].max()
                - self.ranges[self.order[0].value].min(),
            )

            # if order[1] or order[2] is bigger than 1,
            # move correct iterator by one and do a for loop.
            # Otherwise this is a single line scan and loop is not necessary.
            if (
                len(self.ranges[self.order[1].value]) > 1
                or len(self.ranges[self.order[2].value]) > 1
            ):
                # reduce range by one (one line already done)
                range_2 = self.ranges[self.order[2].value]
                range_1 = self.ranges[self.order[1].value]
                if len(range_1) > 1:
                    range_1 = range_1[1:]
                else:
                    range_2 = range_2[1:]

                # for loop: move to the beginning of the next line, do the line
                for i in range_2:
                    for j in range_1:
                        # go to a new line
                        new_line_pos = [0.0] * 3
                        if self.line_start == LineStart.SNAKE:
                            if reverse:
                                # stage is at min of order[0] range
                                new_line_pos[self.order[0].value] = (
                                    self.ranges[self.order[0].value].min()
                                    - s_ru[self.order[0].value]
                                )
                                pass
                            else:
                                # stage is at max of order[0] range
                                new_line_pos[self.order[0].value] = (
                                    self.ranges[self.order[0].value].max()
                                    + s_ru[self.order[0].value]
                                )

                            new_line_pos[self.order[1].value] = j
                            new_line_pos[self.order[2].value] = i

                            reverse = not reverse
                        elif self.line_start == LineStart.CR:
                            new_line_pos[self.order[0].value] = (
                                self.ranges[self.order[0].value].min()
                                - s_ru[self.order[0].value]
                            )
                            new_line_pos[self.order[1].value] = j
                            new_line_pos[self.order[2].value] = i

                        # TODO: add moving between lines to the time estimation
                        self.actions.append(ActionMoveTo(self.stages, new_line_pos))

                        # do the line
                        self.actions.append(
                            ActionFlyBy(
                                self.stages[self.order[0].value],
                                self.detector,
                                self.ranges[self.order[0].value],
                                self.order,
                                [j, i],
                                t_ru[self.order[0].value],
                                s_ru[self.order[0].value],
                                reverse=reverse,
                            )
                        )
                        self.ta += calc_movetime(
                            stage=self.stages[self.order[0].value],
                            dist=2 * s_ru
                            + self.ranges[self.order[0].value].max()
                            - self.ranges[self.order[0].value].min(),
                        )

        elif self.line_type == LineType.PTBYPT:
            self._build_line_scans(start_line_pos=start_pos)
            # # go to start position
            # self.actions.append(ActionMoveTo(self.stages, start_pos))
            # maxind = max(range(len(start_pos)), key=start_pos.__getitem__)
            # self.ta += calc_movetime(stage=self.stages[maxind], dist=max(start_pos))

            # # do a single line
            # reverse = False
            # self.actions.append(
            #     ActionPtByPt(
            #         self.stages[self.order[0].value],
            #         self.detector,
            #         self.ranges[self.order[0].value],
            #         self.order,
            #         start_pos[1:],
            #         reverse=reverse,
            #     )
            # )
            # pts_per_line = len(self.ranges[self.order[0].value]) - 1
            # dist_btw_meas = (
            #     self.ranges[self.order[0].value].max()
            #     - self.ranges[self.order[0].value].min()
            # )
            # self.ta += pts_per_line * calc_movetime(
            #     stage=self.stages[self.order[0].value], dist=dist_btw_meas
            # )
            # self.ta += self.detector.get_ta() * len(self.ranges[self.order[0].value])

            # # if order[1] or order[2] is bigger than 1,
            # # move correct iterator by one and do a for loop.
            # # Otherwise this is a single line scan and loop is not necessary.
            # if (
            #     len(self.ranges[self.order[1].value]) > 1
            #     or len(self.ranges[self.order[2].value]) > 1
            # ):
            #     # reduce range by one (one line already done)
            #     range_2 = self.ranges[self.order[2].value]
            #     range_1 = self.ranges[self.order[1].value]
            #     if len(range_1) > 1:
            #         range_1 = range_1[1:]
            #     else:
            #         range_2 = range_2[1:]
            #     # for loop: move to the beginning of the next line, do the line
            #     for i in range_2:
            #         for j in range_1:
            #             # go to a new line
            #             new_line_pos = [0.0] * 3
            #             if self.line_start == LineStart.SNAKE:
            #                 if reverse:
            #                     # stage is at min of order[0] range
            #                     new_line_pos[self.order[0].value] = self.ranges[
            #                         self.order[0].value
            #                     ].min()
            #                     pass
            #                 else:
            #                     # stage is at max of order[0] range
            #                     new_line_pos[self.order[0].value] = self.ranges[
            #                         self.order[0].value
            #                     ].max()

            #                 new_line_pos[self.order[1].value] = j
            #                 new_line_pos[self.order[2].value] = i

            #                 reverse = not reverse
            #             elif self.line_start == LineStart.CR:
            #                 new_line_pos[self.order[0].value] = self.ranges[
            #                     self.order[0].value
            #                 ].min()
            #                 new_line_pos[self.order[1].value] = j
            #                 new_line_pos[self.order[2].value] = i

            #             self.actions.append(ActionMoveTo(self.stages, new_line_pos))

            #             # do the line
            #             self.actions.append(
            #                 ActionPtByPt(
            #                     self.stages[self.order[0].value],
            #                     self.detector,
            #                     self.ranges[self.order[0].value],
            #                     self.order,
            #                     [j, i],
            #                     reverse=reverse,
            #                 )
            #             )
            #             pts_per_line = len(self.ranges[self.order[0].value]) - 1
            #             dist_btw_meas = (
            #                 self.ranges[self.order[0].value].max()
            #                 - self.ranges[self.order[0].value].min()
            #             )
            #             self.ta += pts_per_line * calc_movetime(
            #                 stage=self.stages[self.order[0].value], dist=dist_btw_meas
            #             )
            #             self.ta += self.detector.get_ta() * len(
            #                 self.ranges[self.order[0].value]
            #             )
            # else:
            #     raise NotImplemented("Single line scan not implemented yet")
        else:
            raise ValueError(
                f"Expected line_type of value {LineType.FLYBY} or {LineType.PTBYPT}. Got {self.line_type}"
            )

        if self.fin_home:
            # TODO: update accordingly if everything works fine
            self.actions.append(ActionMoveTo(self.stages, [0, 0, 0]))
            maxdists = [r.max() for r in self.ranges]
            maxind = max(range(len(maxdists)), key=maxdists.__getitem__)
            self.ta += calc_movetime(stage=self.stages[maxind], dist=max(maxdists))
        self.is_built = True

    def run(self):
        if not self.is_built:
            raise Exception("Scan routine not built.")
        # start the routine
        start_time = time()
        if self.use_tqdm:
            planned_actions = tqdm(self.actions, desc="Scanning", unit="action")
        else:
            planned_actions = self.actions
        for action in planned_actions:
            # add to history
            self.history.append(str(action))
            d = action.run()
            self.data = pd.concat([self.data, pd.DataFrame(data=d)], ignore_index=True)
        stop_time = time()
        self.ta_act = stop_time - start_time
        return self.data

    def _calculate_starting_position(
        self, other_axis_positions: Tuple[float], reverse: bool
    ):
        """`other_axis_position` in the same order as `self.order`"""
        j, i = other_axis_positions
        starting_position = [0.0] * 3
        if self.line_start == LineStart.SNAKE:
            if reverse:
                # stage is at min of order[0] range
                starting_position[self.order[0].value] = self.ranges[
                    self.order[0].value
                ].min()
                pass
            else:
                # stage is at max of order[0] range
                starting_position[self.order[0].value] = self.ranges[
                    self.order[0].value
                ].max()

            starting_position[self.order[1].value] = j
            starting_position[self.order[2].value] = i

            reverse = not reverse
        elif self.line_start == LineStart.CR:
            starting_position[self.order[0].value] = self.ranges[
                self.order[0].value
            ].min()
            starting_position[self.order[1].value] = j
            starting_position[self.order[2].value] = i
        return starting_position

    def _build_line_scans(self, start_line_pos: List[np.float64]):
        # if order[1] or order[2] is bigger than 1,
        # move correct iterator by one and do a for loop.
        # Otherwise this is a single line scan and loop is not necessary.
        reverse = False
        previous_position = [stage.status["position"] for stage in self.stages]
        if (
            len(self.ranges[self.order[1].value]) > 1
            or len(self.ranges[self.order[2].value]) > 1
        ):
            range_2 = self.ranges[self.order[2].value]
            range_1 = self.ranges[self.order[1].value]

            # move to the beginning of the next line, do the line
            for i in range_2:
                for j in range_1:
                    # go to the starting position of next line
                    starting_position_action = ActionMoveTo(self.stages, start_line_pos)
                    self.actions.append(starting_position_action)
                    self.ta += starting_position_action.get_ta(
                        prev_position=previous_position
                    )

                    # do the line
                    line_action = ActionPtByPt(
                        self.stages[self.order[0].value],
                        self.detector,
                        self.ranges[self.order[0].value],
                        self.order,
                        [j, i],
                        reverse=reverse,
                    )
                    self.actions.append(line_action)
                    self.ta += line_action.get_ta()

                    # calculate new starting position and set previous
                    start_line_pos = self._calculate_starting_position(
                        other_axis_positions=(j, i), reverse=reverse
                    )
                    previous_position = line_action.last_position

                    # reverse the line
                    reverse = not reverse

        else:
            raise NotImplementedError("Single line scan not implemented yet")


# LiveView
class LiveView:
    """Live reading from THz bolometer line and plotting.
    Implementation uses separate threads for detector control, plotting,
    and handling standard input to identify when to stop the program.
    """

    def __init__(self, detector: BoloLine, logger: logging.Logger = None) -> None:
        """Initialize all threads necessary for concurrent detector control,
        plotting and reading from standard input (to stop the execution).

        Args:
            detector (BoloLine): detector handle
            logger (logging.Logger, optional): logger handle. Defaults to None.
        """
        self.detector = detector
        self.measurements = Queue()
        self.shutdown_event = threading.Event()
        if logger is None:
            self._log = logging.getLogger(__name__)
        else:
            self._log = logger

        self.detector_thread = threading.Thread(
            target=self._detector_controller,
            args=(self.shutdown_event, self.measurements),
            name="detector_controller",
        )
        self.interrupt_thread = threading.Thread(
            target=self._interrupt_controller,
            args=(self.shutdown_event,),
            name="interrupt_controller",
        )
        threading.excepthook = self._interrupt_hook

    def start(self) -> None:
        """Start all threads and join them on the finish.

        The plotting controller needs to be in the main thread
        (otherwise Tk has some problems).
        """
        self.detector_thread.start()
        self.interrupt_thread.start()
        self._plot_controller(self.shutdown_event, self.measurements)
        self.detector_thread.join()
        self.interrupt_thread.join()

    def _interrupt_hook(self, args):
        self._log.error(
            f"Thread {args.thread.getName()} failed with exception {args.exc_value}"
        )
        self._log.error(f"Traceback{traceback.print_tb(args.exc_traceback)}")
        self._log.error("Shutting down")
        self.shutdown_event.set()

    def _detector_controller(self, shutdown_event: threading.Event, queue: Queue):
        while True:
            # sleep(2)
            # y = np.random.random([10,1])
            measurement = self.detector.measure()
            det_no_samp = len(measurement)
            det_freq = self.detector.get_freq() * 1000
            queue.put(
                {"data": measurement, "det_no_samp": det_no_samp, "det_freq": det_freq}
            )
            if shutdown_event.is_set():
                break
        print("Detector thread finished")

    def _interrupt_controller(self, shutdown_event: threading.Event) -> None:
        input("Press ENTER to close LiveView")
        shutdown_event.set()

    def _plot_controller(self, shutdown_event: threading.Event, queue: Queue):
        plt.set_loglevel("error")
        logging.getLogger("PIL").setLevel(logging.ERROR)
        plt.ion()
        y = np.random.random([10, 1])
        yx = y
        fft = y
        fftx = y
        plt.subplots(
            nrows=2,
            ncols=1,
        )
        while True:
            try:
                payload = self.measurements.get_nowait()
                y = payload["data"]
                det_no_samp = payload["det_no_samp"]
                det_freq = payload["det_freq"]
                dt = 1 / det_freq
                yx = np.arange(0, det_no_samp * dt, dt)
                fft = np.abs(np.fft.rfft(y))
                fftx = np.fft.rfftfreq(len(y), dt)
                # self._log.debug(f"step: {dt}")
            except Empty:
                pass

            # plot data

            plt.subplot(211)
            plt.plot(yx, y)
            plt.ylabel("Amplitude [V]")
            plt.xlabel("Time [s]")
            plt.ylim(bottom=0, top=3.3)

            # plot fft
            plt.subplot(212)
            plt.plot(fftx, fft)
            plt.ylabel("Amplitude [V]")
            plt.xlabel("Frequency [Hz]")
            plt.ylim(bottom=0.0, top=5.0)
            # plt.xlim(left=0.0, right=1050)
            plt.xlim(left=0.0)

            plt.draw()
            plt.pause(0.0001)
            plt.clf()
            if shutdown_event.is_set():
                plt.ioff()
                plt.close("all")
                break
        print("Plot thread finished")
