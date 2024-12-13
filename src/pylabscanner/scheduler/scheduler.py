from time import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from tqdm import tqdm

from ..devices.devices import LTS, Detector, Source, calc_startposmod
from ..devices.manager import DeviceManager
from .commands import (
    ActionFlyBy,
    ActionMoveTo,
    ActionPtByPt,
    LineStart,
    LineType,
    StageAxis,
    calc_movetime,
)


class ScanScheduler:
    """Scheduler for scanning with stages.

    Call `build()` before running (`run()` method) the routine to build list
    of steps to perform.
    """

    # TODO: validate fly-by range has at least 2 points
    # TODO: validate fly-by range can fit in the stage range
    def __init__(
        self,
        # stages: List[LTS],
        # detector: Detector,
        # source: Source,
        manager: DeviceManager,
        ranges: List[ndarray],
        order: tuple[StageAxis] = (StageAxis.X, StageAxis.Y, StageAxis.Z),
        line_type: LineType = LineType.FLYBY,
        line_start: LineStart = LineStart.SNAKE,
        fin_home: bool = True,
        use_tqdm: bool = True,
    ):
        # self.stages = stages
        # self.detector = detector
        # self.source = source
        self.manager = manager
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
                f"Expected line_type of value {LineType.FLYBY} or "
                f"{LineType.PTBYPT}. Got {self.line_type}"
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
                    self.ta += starting_position_action.ta(
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
