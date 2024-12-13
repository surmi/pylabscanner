import logging
import threading
import traceback
from asyncio import TaskGroup, run
from queue import Empty, Queue

import matplotlib.pyplot as plt
import numpy as np

from .devices import (
    BoloLine,
    BoloLineConfiguration,
    DetectorInitParams,
    LTSConfiguration,
    LTSStage,
    MockBoloLine,
    MockLTSStage,
    StageInitParams,
)


class DeviceManager:
    # TODO:acquisition time (How to get acquisition for specific device?)
    # TODO:measure series?
    def __init__(
        self,
        stage_init_params: dict[str, StageInitParams],
        detector_init_params: DetectorInitParams,
        stage_configurations: None | dict[str, LTSConfiguration] = None,
        detector_configuration: None | BoloLineConfiguration = None,
    ):
        self.stages: dict[str, LTSStage] = {}
        for label in stage_init_params:
            init_params: StageInitParams = stage_init_params[label]
            if init_params.is_mockup:
                self.stages[label] = MockLTSStage(
                    init_params.serial_number,
                    init_params.rev,
                    init_params.initialize,
                )
            else:
                self.stages[label] = LTSStage(
                    init_params.serial_number,
                    init_params.rev,
                    init_params.initialize,
                )
        if detector_init_params.is_mockup:
            self.detector = MockBoloLine(
                idProduct=detector_init_params.idProduct,
                idVendor=detector_init_params.idVendor,
                sensor=detector_init_params.sensor,
                samples=detector_init_params.samples,
                freq=detector_init_params.freq,
                initialize=detector_init_params.initialize,
            )
        else:
            self.detector = BoloLine(
                idProduct=detector_init_params.idProduct,
                idVendor=detector_init_params.idVendor,
                sensor=detector_init_params.sensor,
                samples=detector_init_params.samples,
                freq=detector_init_params.freq,
                initialize=detector_init_params.initialize,
            )
        self.configure(
            detector_configuration=detector_configuration,
            stage_configurations=stage_configurations,
        )

    def initialize(self):
        for label in self.stages:
            self.stages[label].initialize()
        self.detector.initialize()

    def configure(
        self,
        detector_configuration: None | BoloLineConfiguration = None,
        stage_configurations: None | dict[str, LTSConfiguration] = None,
    ):
        if detector_configuration is not None:
            self.detector.configure(configuration=detector_configuration)
        if stage_configurations is not None and stage_configurations:
            for label in stage_configurations:
                self.stages[label].configure(configuration=stage_configurations[label])

    @property
    def current_configuration(self):
        res = {"detector": self.detector.current_configuration}
        for label in self.stages:
            res[label] = self.stages[label].current_configuration
        return res

    @property
    def current_position(self):
        res = {}
        for axis_name in self.stages:
            res[axis_name] = self.stages[axis_name].current_position
        return res

    async def home_async(self, stage_label: str | list[str]):
        if isinstance(stage_label, str):
            if stage_label == "all":
                stage_label = ["x", "y", "z"]
            else:
                stage_label = [stage_label]
        async with TaskGroup() as tg:
            tasks = []
            for label in stage_label:
                stage = self.stages[label]
                tasks.append(
                    tg.create_task(
                        stage.home(), name=f"Homing stage with label {label}"
                    ),
                )

    def home(self, stage_label: str | list[str]):
        run(self.home_async(stage_label=stage_label))

    async def move_stage_async(self, stage_destination: dict[str, float]):
        async with TaskGroup() as tg:
            tasks = []
            for label in stage_destination:
                stage = self.stages[label]
                destination = stage_destination[label]
                tasks.append(
                    tg.create_task(
                        stage.go_to(destination=destination),
                        name=f"Moving stage labeled as {label} to position {destination}",
                    )
                )

    def move_stage(self, stage_destination: dict[str, float]):
        run(self.move_stage_async(stage_destination=stage_destination))

    def ta_move_stage(
        self,
        destination: dict[str, float] | float | None = None,
        previous_position: dict[str, float] | float | None = None,
        distance: dict[str, float] | float | None = None,
    ):
        """Maximum arrival time for selected stage(s).
        Provide `destination`, `destination`+``previous_position` or `distance`
        """

        if previous_position is None:
            previous_position = self.current_position
        elif isinstance(previous_position, float):
            previous_position = {"x": previous_position}
        if isinstance(destination, float):
            destination = {"x": destination}
        if isinstance(distance, float):
            distance = {"x": distance}
        if destination is None:
            if distance is not None:
                axis_to_iterate = distance.keys()
            else:
                raise ValueError(
                    "If `distance` is None then at least `destination` "
                    "needs to be not None."
                )
        else:
            axis_to_iterate = destination.keys()

        ta = 0
        # for axis_name in destination:
        for axis_name in axis_to_iterate:
            if distance is None:
                if destination is None and previous_position is None:
                    raise ValueError(
                        "If `distance` is None then at least "
                        "`destination` needs to be not None."
                    )
                axis_distance = abs(
                    destination[axis_name] - previous_position[axis_name]
                )
            else:
                axis_distance = distance[axis_name]
            ta = max(
                ta,
                self.stages[axis_name].calculate_arrival_time(distance=axis_distance),
            )
        return ta

    def ta_home(
        self,
        position_from: dict[str, float] | None = None,
        stage_label: str | list[str] = "all",
    ):
        if isinstance(stage_label, str):
            if stage_label == "all":
                stage_label = [label for label in self.stages]
            else:
                stage_label = [stage_label]
        ta = 0
        for axis_name in stage_label:
            if position_from is None:
                distance = self.stages[axis_name].current_position
            else:
                distance = position_from[axis_name]
            ta = max(
                ta,
                self.stages[axis_name].calculate_arrival_time(
                    distance=distance, use_homing_velocity=True
                ),
            )
        return ta

    @property
    def ta_measurement(self):
        return self.detector.acquisition_time

    def measure(self):
        return self.detector.measure()

    @property
    def samples_per_measurement(self):
        return self.detector._samples.nsamp

    def live_view(self):
        lv = LiveView(detector=self.detector)
        lv.start()

    # TODO:acquisition time (How to get acquisition for specific device?)
    # TODO:measure series?


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

    def _inRoutineterrupt_hook(self, args):
        self._log.error(
            f"Thread {args.thread.getName()} failed with exception " f"{args.exc_value}"
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
