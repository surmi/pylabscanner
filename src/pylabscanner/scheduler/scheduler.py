from time import time
from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray
from tqdm import tqdm

from ..devices.devices import calc_startposmod
from ..devices.manager import DeviceManager
from .commands import (
    ActionFlyBy,
    ActionHome,
    ActionMoveTo,
    ActionPtByPt,
    LineStart,
    LineType,
    StageAxis,
    calc_movetime,
)


class ScanScheduler:
    """Scheduler that creates a plan of steps for scanning with stages.

    Call `make_schedule()` before running (`run()` method) the routine to build list
    of steps to perform.

    Changing any of the input parameters results in the plan and if the plan is
    already scheduled, the plan will be descheduled (will require `make_schedule()`).
    """

    # TODO: validate fly-by range has at least 2 points
    # TODO: validate fly-by range can fit in the stage range
    def __init__(
        self,
        manager: DeviceManager,
        ranges: dict[str, ndarray],
        line_type: LineType = LineType.FLYBY,
        line_start: LineStart = LineStart.SNAKE,
        fin_home: bool = True,
        use_tqdm: bool = True,
    ):
        self._manager = manager
        self._ranges = ranges
        # in case if range is passed backwards
        for k in self._ranges:
            self._ranges[k].sort()
        self.line_type = line_type
        self.line_start = line_start
        self.fin_home = fin_home
        self.use_tqdm = use_tqdm

        self._init_internal_params()

    def _init_internal_params(self):
        self.data = {"x": [], "y": [], "z": [], "MEASUREMENT": []}
        self.actions = []
        self.history = []
        self.is_built = False
        self.ta = 0.0  # acqusition time estimaiton (whole scan)
        self.ta_act = 0.0  # actual asqusition time

    @property
    def manager(self):
        return self._manager

    @manager.setter
    def manager(self, value: DeviceManager):
        self._manager = value
        if self.is_built:
            self._init_internal_params()

    @property
    def ranges(self):
        return self._ranges

    @ranges.setter
    def ranges(self, value: dict[str, ndarray]):
        self._ranges = value
        if self.is_built:
            self._init_internal_params()

    def fill_metadata(self, metadata_output: None | dict[str, Any] = None):
        if metadata_output is None:
            metadata_output = {}

        for axis in self.ranges:
            metadata_output[f"{axis} axis range [beg:end:no pts|pos]"] = self.ranges[
                axis
            ]
        metadata_output["scanning mode"] = self.line_type
        metadata_output["scanning line start"] = self.line_start

        return metadata_output

    def make_schedule(self):
        """Build the scan routine."""

        # FLYBY ---------------------------------------------------------------
        if self.line_type == LineType.FLYBY:
            raise NotImplementedError(
                "FlyBy scanning not implemented yet"
            )  # TODO: require update after new changes to pt-by-pt scan will be tested
            # vels = [
            #     steps2mm(stage.velparams["max_velocity"], stage.convunits["vel"])
            #     for stage in self.stages
            # ]
            # line scan always starts at min
            start_pos = {axis: self._ranges[axis].min() for axis in self._ranges}
            t_ru = []
            s_ru = []
            for s in self.stages:
                ss, tt = calc_startposmod(s)
                s_ru.append(ss)
                t_ru.append(tt)
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
                    self._ranges[self.order[0].value],
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
                + self._ranges[self.order[0].value].max()
                - self._ranges[self.order[0].value].min(),
            )

            # if order[1] or order[2] is bigger than 1,
            # move correct iterator by one and do a for loop.
            # Otherwise this is a single line scan and loop is not necessary.
            if (
                len(self._ranges[self.order[1].value]) > 1
                or len(self._ranges[self.order[2].value]) > 1
            ):
                # reduce range by one (one line already done)
                range_2 = self._ranges[self.order[2].value]
                range_1 = self._ranges[self.order[1].value]
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
                                    self._ranges[self.order[0].value].min()
                                    - s_ru[self.order[0].value]
                                )
                                pass
                            else:
                                # stage is at max of order[0] range
                                new_line_pos[self.order[0].value] = (
                                    self._ranges[self.order[0].value].max()
                                    + s_ru[self.order[0].value]
                                )

                            new_line_pos[self.order[1].value] = j
                            new_line_pos[self.order[2].value] = i

                            reverse = not reverse
                        elif self.line_start == LineStart.CR:
                            new_line_pos[self.order[0].value] = (
                                self._ranges[self.order[0].value].min()
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
                                self._ranges[self.order[0].value],
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
                            + self._ranges[self.order[0].value].max()
                            - self._ranges[self.order[0].value].min(),
                        )

        elif self.line_type == LineType.PTBYPT:
            self._build_line_scans()
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
                f"Expected line_type of value {LineType.FLYBY} or "
                f"{LineType.PTBYPT}. Got {self.line_type}"
            )

        if self.fin_home:
            home_action = ActionHome(self._manager)
            self.actions.append(home_action)
            self.ta += home_action.ta()
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
            d = action.run()
            if d is not None:
                for k in self.data:
                    self.data[k].extend(d[k])

            self.history.append(str(action))
        stop_time = time()
        self.data = pd.DataFrame(data=self.data)
        self.ta_act = stop_time - start_time
        return self.data

    def _calculate_starting_position(
        self, other_axis_positions: dict[str, float], scan_axis: str, reverse: bool
    ):
        """`other_axis_position` in the same order as `self.order`"""
        starting_position = {}
        if self.line_start == LineStart.SNAKE:
            if reverse:
                starting_position[scan_axis] = self._ranges[scan_axis][-1]
            else:
                starting_position[scan_axis] = self._ranges[scan_axis][0]

        elif self.line_start == LineStart.CR:
            starting_position[scan_axis] = self._ranges[scan_axis][0]

        for axis_name in other_axis_positions:
            starting_position[axis_name] = other_axis_positions[axis_name]

        return starting_position

    def _scan_dimensionality(self):
        """Identify dimensionality of the scan"""
        dimensionality = 0
        for axis_name in StageAxis.ordered_names():
            range = self.ranges.get(axis_name, np.zeros(shape=1))
            if len(range) > 1:
                dimensionality += 1
        return dimensionality

    def _axis_order(self):
        """Identify order of axis for definition of steps"""
        order = []
        rest = []
        for axis_name in StageAxis.ordered_names():
            range = self.ranges.get(axis_name, np.zeros(shape=1))
            if len(range) > 1:
                order.append(axis_name)
            else:
                rest.append(axis_name)
        for el in rest:
            order.append(el)
        return order

    def _build_line_scans(self):
        reverse = False
        previous_position = self._manager.home_position
        axis_order = self._axis_order()
        if self._scan_dimensionality() == 2:
            step_axis = axis_order[1]
            step_range = self._ranges[step_axis]
            scan_axis = axis_order[0]
            scan_range = self._ranges[scan_axis]
            static_axis = axis_order[2]

            # move to the beginning of the next line and then do the line
            for i in step_range:
                # calculate new starting position
                other_axis_position = {
                    static_axis: self._ranges[static_axis][0],
                    step_axis: i,
                }
                start_line_pos = self._calculate_starting_position(
                    other_axis_positions=other_axis_position,
                    scan_axis=scan_axis,
                    reverse=reverse,
                )

                # go to the starting position of next line
                starting_position_action = ActionMoveTo(
                    manager=self._manager, destination=start_line_pos
                )
                self.actions.append(starting_position_action)
                self.ta += starting_position_action.ta(prev_position=previous_position)

                # do the line
                if reverse:
                    action_range = np.flip(scan_range)
                else:
                    action_range = scan_range
                line_action = ActionPtByPt(
                    movement_axis=StageAxis[scan_axis],
                    measuring_range=action_range,
                    manager=self._manager,
                    starting_position=start_line_pos,
                )
                self.actions.append(line_action)
                self.ta += line_action.get_ta()

                # set previous position
                previous_position = line_action.last_position

                # reverse the line
                reverse = not reverse

        else:
            raise NotImplementedError("Single line and 3D scans not implemented yet")
