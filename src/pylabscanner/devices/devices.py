import logging
import math
import threading
import traceback
from abc import ABC, abstractmethod
from asyncio import TaskGroup, wait_for
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from queue import Empty, Queue
from time import sleep
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports

from .LTS import LTS, LTSC
from .utility import mm2steps, steps2mm


class BoloMsgSensor(Enum):
    """Parts of message that indicate which sensor is being read."""

    FIRST = (1).to_bytes(1)  # first sensor
    SECOND = (2).to_bytes(1)  # second sensor
    THIRD = (3).to_bytes(1)  # third sensor
    FOURTH = (4).to_bytes(1)  # fourth sensor


class BoloMsgSamples(Enum):
    """Parts of message that indicate number of samples."""

    S100 = ((1).to_bytes(1), 100)  # 100 samples
    S200 = ((2).to_bytes(1), 200)  # 200 samples
    S500 = ((5).to_bytes(1), 500)  # 500 samples
    S1000 = ((10).to_bytes(1), 1000)  # 1000 samples
    S2000 = ((20).to_bytes(1), 2000)  # 2000 samples
    S5000 = ((50).to_bytes(1), 5000)  # 5000 samples

    def __init__(self, msg, nsamp) -> None:
        self.msg = msg  # part of the message
        self.nsamp = nsamp  # number of samples


class BoloMsgFreq(Enum):
    """Parts of message that indicate frequency."""

    F1 = ((1).to_bytes(1), 1)  # 1 kHz
    F2 = ((2).to_bytes(1), 2)  # 2 kHz
    F5 = ((5).to_bytes(1), 5)  # 5 kHz
    F10 = ((10).to_bytes(1), 10)  # 10 kHz
    F20 = ((20).to_bytes(1), 20)  # 20 kHz
    F40 = ((40).to_bytes(1), 40)  # 40 kHz

    def __init__(self, msg, freq) -> None:
        self.msg = msg  # part of the message
        self.freq = freq  # frequency in kHz


@dataclass
class BoloLineConfiguration:
    """Bolometer line configuration"""

    frequency: BoloMsgFreq
    sampling: BoloMsgSamples
    sensor_id: BoloMsgSensor

    def __eq__(self, other) -> bool:
        return all(
            [
                self.frequency == other.frequency,
                self.sampling == other.sampling,
                self.sensor_id == other.sensor_id,
            ]
        )


@dataclass
class LTSConfiguration:
    """LTS configuration"""

    velocity: float
    acceleration: float

    def __eq__(self, other) -> bool:
        return all(
            [self.acceleration == other.acceleration, self.velocity == other.velocity]
        )


class DeviceNotFoundError(Exception):
    def __init__(self, device_name=None, msg=None, *args: object) -> None:
        super().__init__(*args)
        self.device_name = device_name
        self.msg = msg

    def __str__(self):
        if self.msg is None:
            if self.device_name is None:
                return "Device not found."
            else:
                return f"{self.device_name} not found."
        else:
            return self.msg


class DeviceNotInitialized(RuntimeError):
    def __init__(self, device_name=None, msg=None, *args: object) -> None:
        super().__init__(*args)
        self.device_name = device_name
        self.msg = msg

    def __str__(self):
        if self.msg is None:
            if self.device_name is None:
                return "Requested operation require initialization."
            else:
                return f"{self.device_name} require initizliation before the operation."
        else:
            return self.msg


class Device(ABC):
    @staticmethod
    def check_initialized(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.is_initialized:
                raise DeviceNotInitialized(device_name=self.__str__)
            return func(self, *args, **kwargs)

        return wrapper

    @property
    @abstractmethod
    def is_initialized(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def device_name(self):
        raise NotImplementedError


class Detector(Device, ABC):
    """Abstract class for detectors."""

    @abstractmethod
    def initialize(self):
        """Initialize communication with the detector."""
        raise NotImplementedError

    @abstractmethod
    def configure(self, configuration):
        """Change configuration of the detector"""
        raise NotImplementedError

    @property
    @abstractmethod
    def current_configuration(self, configuration):
        """Current configuration of the detector"""
        raise NotImplementedError

    @abstractmethod
    def measure(self):
        """Measure single value."""
        raise NotImplementedError

    @property
    @abstractmethod
    def acquisition_time(self):
        """Time of acquisition in seconds."""
        raise NotImplementedError


class BoloLine(Detector):
    """Class for reading from line of bolometers by Luvitera.

    Tested for THz mini (line of 4 wideband bolometers).
    """

    # TODO: Add detection of detector disconnecting (timoeout?)

    def initialize(self):
        """Initialize communication with the detector."""
        restr = "(?i)" + self._hid
        try:
            self._port = next(list_ports.grep(restr)).name
        except StopIteration:
            raise DeviceNotFoundError(msg="Bolometer line not found.")
        self._dev = serial.Serial(
            port=self._port,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            timeout=0,  # NOTE: under test (previously 0.1); currently: non-blocking
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            write_timeout=0.1,
        )
        # first communication
        # self._makemsgparts(sensor=(0).to_bytes(1))
        self._dev.reset_input_buffer()
        self._dev.reset_output_buffer()
        frame = []
        try:
            self._write()
            sleep(self._read_delay)
            frame.append(self._read())
            self._write()
            sleep(self._read_delay)
            frame.append(self._read())
            self._write()
            sleep(self._read_delay)
            frame.append(self._read())
            self._write()
            sleep(self._read_delay)
        except serial.SerialTimeoutException as e:
            raise e
        frame.append(self._read())
        for ans in frame:
            if len(ans) > 0:
                self._idstr = ans[0:5]
                self._sn = int(self._idstr[0])
                self._prodyear = int(self._idstr[1]) + 2000
                senstype = int(self._idstr[2]) * 100 + int(self._idstr[3])
                if senstype == 0:
                    self._senstype = "WIDEBAND"
                else:
                    self._senstype = f"{senstype}"
                self._fw = int(self._idstr[4]) * 0.1
                break

        # finish setup
        self._is_initialized = True

    @Device.check_initialized
    def configure(self, configuration: BoloLineConfiguration):
        """Change configuration of the detector.
        Requires connection to be initialized first.
        """
        if (
            configuration.frequency is None
            or configuration.sampling is None
            or configuration.sensor_id is None
        ):
            raise ValueError("Configuration parameters can't be None")
        self._sensor = configuration.sensor_id
        self._freq = configuration.frequency
        self._samples = configuration.sampling
        self._makemsg()
        self._recalculate_ta()
        self._dev.reset_input_buffer()
        self._dev.reset_output_buffer()
        self._write()
        sleep(self._read_delay)
        sleep(self._ta)
        self._read()

    @property
    @Device.check_initialized
    def acquisition_time(self) -> float:
        """Total acqusition time (between write to the device and read from
        the device) in seconds"""
        return self._ta + self._read_delay

    @property
    def current_configuration(self):
        """Current configuration of the detector"""
        return BoloLineConfiguration(
            frequency=self._freq, sampling=self._samples, sensor_id=self._sensor
        )

    @property
    def sensor(self) -> str:
        """Number of the sensor to read from."""
        return self._sensor.name

    # def set_sensor(self, sensor: BoloMsgSensor) -> None:
    #     """Change sensor to read from. Recalculates internal parameters"""
    #     self._sensor = sensor
    #     self._makemsg()

    @property
    def samples(self) -> int:
        """Number of samples to register in single read."""
        return self._samples.nsamp

    # def set_samples(self, samples: BoloMsgSamples) -> None:
    #     """Change number of measured samples. Recalculates internal parameters"""
    #     self._samples = samples
    #     self._recalculate_ta()
    #     self._makemsg()

    @property
    def frequency(self) -> int:
        """Reading frequency in kHz."""
        return self._freq.freq

    # def set_freq(self, freq: BoloMsgFreq) -> None:
    #     """Change frequency of measuring samples. Recalculates internal parameters"""
    #     self._freq = freq
    #     self._recalculate_ta()
    #     self._makemsg()

    def _recalculate_ta(self) -> None:
        """Recalculate time of acquisition after changing number of samples or
        frequency."""
        self._ta = self._samples.nsamp / (self._freq.freq * 1000)

    def _write(self) -> None:
        """Write message to the detector."""
        try:
            self._dev.write(self._msg)
        except serial.SerialTimeoutException as e:
            raise e

    def _read(self) -> bytes:
        """Reads all bytes available in the buffor."""
        while True:
            if (
                self._dev.in_waiting != 0
                and self._dev.in_waiting >= self._samples.nsamp * 2
            ):
                # print(self._dev.in_waiting)
                break
            sleep(self._read_delay)

        return self._dev.read(self._dev.in_waiting)

    def _makemsg(self) -> None:
        """Updates massage send on every write to the detector."""
        self._msg = self._sensor.value + self._samples.msg + self._freq.msg

    def _makemsgparts(self, sensor=None, samples=None, frequency=None) -> None:
        """Set message send on every write to the detector based on parameters
        (if provided)."""
        sensor = self._sensor.value if sensor is None else sensor
        samples = self._samples.msg if samples is None else samples
        frequency = self._freq.msg if frequency is None else frequency
        self._msg = sensor + samples + frequency

    # def _cold_start(self) -> None:
    #     """To be used before first measurement after downtime of the device
    #     (device disconnected from power source)."""
    #     self._dev.reset_input_buffer()
    #     self._dev.reset_output_buffer()

    #     self._write()
    #     sleep(self._read_delay)
    #     sleep(self._ta)
    #     self._read()

    #     self._write()
    #     sleep(self._read_delay)
    #     sleep(self._ta)
    #     self._read()

    # def get_msg(self) -> bytes:
    #     """Message written to the detector."""
    #     return self._msg

    def pairwise(self, iterable) -> List[float]:
        """Return iterable with bytes converted to numbers."""
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        data = []
        for p in zip(a, a):
            data.append((p[1] + p[0] * 256) / 65536 * 3.3)
        return data

    def __init__(
        self,
        idVendor="03EB",
        idProduct="2404",
        sensor: BoloMsgSensor = BoloMsgSensor.FIRST,
        samples: BoloMsgSamples = BoloMsgSamples.S100,
        freq: BoloMsgFreq = BoloMsgFreq.F1,
        initialize: bool = False,
        # cold_start=False,
    ) -> None:
        self._hid = f"{idVendor}:{idProduct}"  # hardware id (vendor id : product id)
        self._read_delay = (
            0.001  # selected experimentally (longer wait time may be necessary)
        )
        self._write_delay = 0.001
        # self._samples = BoloMsgSamples.S100
        # self._freq = BoloMsgFreq.F1
        # self._ta = self._samples.nsamp / (self._freq.freq * 1000)
        self._sensor = sensor
        self._samples = samples
        self._freq = freq
        self._dev = None
        self._idstr = None
        self._sn = None
        self._prodyear = None
        self._senstype = None
        self._is_initialized = False
        self._makemsg()
        self._recalculate_ta()

        if initialize:
            self.initialize()

        # self._sensor = sensor
        # self._samples = samples
        # self._freq = freq
        # self._makemsg()
        # self._recalculate_ta()
        # if cold_start:
        # NOTE: single measurement right after connecting physically the
        # device and setting measurement parameters have some weird peak. Use
        # this method make read of it
        # self._cold_start()

    def description(self):
        """Detialed description of detector properties."""
        if any(
            (
                self._senstype is None,
                self._prodyear is None,
                self._sn is None,
                self._fw is None,
            )
        ):
            return self.__str__()
        return (
            f"Luvitera Mini THz Sensor, 4 pixel array with {self._senstype} "
            "sensor type.\n Sensor details:\n -year of production: "
            f"{self._prodyear}\n -S/N: {self._sn}\n -FW: {self._fw}"
        )

    def _trimans(self, ans) -> bytes:
        """Removes filler bytes (string that is spammed by the detector equal
        to `self._idstr`) from detector's answer."""
        divided = ans.split(self._idstr)
        res = None
        for b in divided:
            if len(b) > 0:
                res = b
                break
        return res

    @Device.check_initialized
    def measure(self) -> List[float]:
        """Perform single measurement. Requires additional write and read due
        to the fact how the detector works.

        Raises:
            serial_timeout_exception: exception raised on timeout while writing
            command to the detector

        Returns:
            List[float]: Measured data converted from list of bytes to floats.
        """
        self._dev.reset_input_buffer()
        self._dev.reset_output_buffer()

        try:
            self._write()
            sleep(self._write_delay)
            self._read()

            self._write()
            sleep(self._write_delay)
        except serial.SerialTimeoutException as serial_timeout_exception:
            raise serial_timeout_exception

        # data = self.pairwise(self._read())
        data = self.pairwise(self._trimans(self._read()))
        # all_frames.append(self._trimans(self._read()))

        return data

    @Device.check_initialized
    def live_view_read(self) -> List[float]:
        # WARNING: not tested yet
        self._dev.reset_input_buffer()
        self._dev.reset_output_buffer()

        try:
            self._write()
            sleep(self._write_delay)
            data = self.pairwise(self._trimans(self._read()))

        except serial.SerialTimeoutException as serial_timeout_exception:
            raise serial_timeout_exception

        return data

    @Device.check_initialized
    def measure_series(
        self, n: int, interval: float = None, start_delay: float = -1
    ) -> List[List[float]]:
        """Measures `n` times (+ additional write/read due to the device properties).

        Args:
            n (int): number of measurements to take. Has to be >= 1.
            interval (float, optional): Time between each measurement in
                seconds. Defaults to None.
            start_delay (float, optional): Time delaying the first measurement
                in seconds. Defaults to 0.

        Raises:
            ValueError: thrown when provided parameters are outside of limits.

        Returns:
            List[List[float]]: List of measurements.
        """
        # TODO: correct doc
        data = []

        if interval is None:
            interval = self._write_delay
        elif interval < self._write_delay:
            raise ValueError(
                "interval has to be greater or equal to the sum of _ta and _readDelay"
            )

        if n > 0 and start_delay >= 0:
            self._dev.reset_input_buffer()
            self._dev.reset_output_buffer()

            sleep(
                start_delay
            )  # don't think we need that high time precision, but FYI when
            # sleep is called with 0 it will release GIL (on Windows at least)

            try:
                self._write()
                sleep(interval)
                self._read()

                for _ in range(n - 1):
                    self._write()
                    sleep(interval)
                    data.append(self._read())

                # one more time just to read last data point
                self._write()
            except serial.SerialTimeoutException as serial_timeout_exception:
                raise serial_timeout_exception
            sleep(self._write_delay)
            data.append(self._read())
        else:
            raise ValueError(
                "n has to be greater than 0 and start_delay has to be greater "
                "than or equal to 0"
            )

        # retrieve decimal values from frames
        return [self.pairwise(self._trimans(i)) for i in data]

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def device_name(self) -> str:
        return "Luvitera Mini THz Sensor"


class MockBoloLine(BoloLine):
    """Mock for line of bolometers."""

    def __init__(
        self,
        idVendor="03EB",
        idProduct="2404",
        sensor: BoloMsgSensor = BoloMsgSensor.FIRST,
        samples: BoloMsgSamples = BoloMsgSamples.S100,
        freq: BoloMsgFreq = BoloMsgFreq.F1,
        initialize: bool = False,
        # cold_start=False,
    ) -> None:
        self._hid = f"{idVendor}:{idProduct}"  # hardware id (vendor id : product id)
        self._read_delay = (
            0.001  # selected experimentally (longer wait time may be necessary)
        )
        self._dev = None
        self._write_delay = 0.001
        self._sensor = sensor
        self._samples = samples
        self._freq = freq
        self._idstr = None
        self._sn = None
        self._prodyear = None
        self._senstype = None
        self._makemsg()
        self._recalculate_ta()
        self._is_initialized = False

        if initialize:
            self.initialize()

    def initialize(self):
        """Initialize the detector."""
        # self._idstr = "ids"
        self._sn = 0
        self._prodyear = 2024
        self._senstype = "WIDEBAND"
        self._is_initialized = True

    @Device.check_initialized
    def configure(self, configuration: BoloLineConfiguration):
        """Initialize the detector."""
        if (
            configuration.frequency is None
            or configuration.sampling is None
            or configuration.sensor_id is None
        ):
            raise ValueError("Configuration parameters can't be None")
        self._sensor = configuration.sensor_id
        self._freq = configuration.frequency
        self._samples = configuration.sampling
        self._makemsg()
        self._recalculate_ta()

    @Device.check_initialized
    def measure(self):
        """Measure single value."""
        # TODO: Mock data aquisition
        # TODO: Mock data parsing
        # data = self.pairwise(self._trimans(self._read()))
        data = np.random.uniform(high=3.3, size=self.samples).tolist()
        return data

    @property
    def device_name(self) -> str:
        return "Mock-up of Luvitera Mini THz Sensor"


class Source(ABC):
    """Abstract class for sources."""

    @abstractmethod
    def initialize(self):
        """Initialize the source"""
        raise NotImplementedError

    @abstractmethod
    def configure(self, configuration):
        """Configure the source"""
        raise NotImplementedError


class MotorizedStage(Device, ABC):
    """Abstract class for motorized stages."""

    @property
    @abstractmethod
    def current_position(self):
        """Current position of the platform."""
        raise NotImplementedError

    @abstractmethod
    def initialize(self):
        """Initialize the stage."""
        raise NotImplementedError

    @abstractmethod
    def configure(self, configuration):
        """Configure the source"""
        raise NotImplementedError

    @abstractmethod
    def calculate_arrival_time(self, distance: float):
        """Calculate time it takes for the platform to cover specified
        distance."""
        raise NotImplementedError

    @abstractmethod
    async def home(self):
        """Home the stage asynchronously."""
        raise NotImplementedError

    @abstractmethod
    async def go_to(self, destination: float):
        """Go to desired position asynchronously."""
        raise NotImplementedError


class LTSStage(MotorizedStage):
    """Abstraction for LTS stages"""

    # TODO: make docstrings
    # TODO: test with hardware
    # TODO: add logging?

    def __init__(self, serial_number: str, rev: str = "LTS", initialize: bool = False):
        self.serial_number = serial_number
        self.rev = rev
        if self.rev not in ("LTS", "LTSC"):
            raise ValueError('rev argument can only be "LTS" or "LTSC"')
        self._is_initialized = False
        if initialize:
            self.initialize()

    @property
    @Device.check_initialized
    def current_position(self):
        """Current position of the platform in mm."""
        return steps2mm(self.stage.status["position"], self.stage.convunits["pos"])

    def initialize(self):
        """Initialize communication with the stage."""
        try:
            if self.rev == "LTS":
                self.stage = LTS(serial_number=self.serial_number, home=False)
            elif self.rev == "LTSC":
                self.stage = LTSC(serial_number=self.serial_number, home=False)
            self._is_initialized = True
        except serial.SerialException:
            raise DeviceNotFoundError(
                device_name=self.device_name, msg="Stage not detected."
            )

    @Device.check_initialized
    def configure(self, configuration: LTSConfiguration):
        """Configure the stage"""
        # TODO: test with physical device
        # TODO: add configuration of jog, home, and move?
        if not self.is_requested_configuration_valid(configuration):
            raise ValueError("Wrong configuration value.")
        self.stage._loop.call_later(
            0.15,
            self.stage.set_velocity_params,
            mm2steps(LTSConfiguration.acceleration, self.stage.convunits["acc"]),
            mm2steps(LTSConfiguration.velocity, self.stage.convunits["vel"]),
        )
        sleep(0.2)

    @Device.check_initialized
    def calculate_arrival_time(self, distance: float):
        """Calculate time it takes for the platform to cover specified
        distance."""
        s_ru, t_ru = calc_startposmod(stage=self.stage)
        max_velocity = steps2mm(
            self.stage.velparams["max_velocity"], self.stage.convunits["vel"]
        )
        acceleration = steps2mm(
            self.stage.velparams["acceleration"], self.stage.convunits["acc"]
        )

        if distance <= 2 * s_ru:
            return 2 * np.sqrt(distance / acceleration)
        else:
            return 2 * t_ru * (distance - 2 * s_ru) / max_velocity

    @property
    def current_configuration(self):
        return LTSConfiguration(
            velocity=self.stage.velparams["max_velocity"],
            acceleration=self.stage.velparams["acceleration"],
        )

    @Device.check_initialized
    def is_request_position_valid(self, requested_position: float | int) -> bool:
        min_pos = self.stage.physicallimsmm["minpos"]
        max_pos = self.stage.physicallimsmm["maxpos"]
        return requested_position >= min_pos and requested_position <= max_pos

    @Device.check_initialized
    def is_requested_configuration_valid(
        self, requested_configuration: LTSConfiguration
    ) -> bool:
        max_vel = self.stage.physicallimsmm["maxdrivevel"]
        max_acc = self.stage.physicallimsmm["maxacc"]
        return (
            requested_configuration.acceleration <= max_acc
            and requested_configuration.acceleration > 0.0
            and requested_configuration.velocity <= max_vel
            and requested_configuration.velocity > 0.0
        )

    @Device.check_initialized
    async def home(self):
        """Home the stage asynchronously."""
        self.stage.aso_home(waitfinished=True)

    @Device.check_initialized
    async def go_to(self, destination: float):
        """Go to desired position asynchronously."""
        if not self.is_request_position_valid(destination):
            raise ValueError("Invalid destination value.")
        self.stage.aso_move_absolute(
            position=mm2steps(destination, self.stage.convunits["pos"])
        )

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def device_name(self) -> str:
        if self.rev:
            return f"Thorlabs {self.rev} stage"
        return "Thorlabs stage"


def calc_startposmod(stage: LTS) -> tuple[float, float]:
    """Calculate modification of position due to the stage needing to ramp up
    to constant velocity.

    Args:
        stage (APTDevice_Motor): stage object from which the velocity
            parameters are taken.

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


class MockLTSStage(LTSStage):
    """Mock-up class for motorized stages."""

    # TODO: add temporal simulation of movement?
    class MockStage:
        def __init__(self):
            self.velparams = {"max_velocity": 20.0, "acceleration": 20.0}
            self.physicallimsmm = {
                "minpos": 0.0,
                "maxpos": 300.0,
                "maxvel": 50.0,
                "maxdrivevel": 40,
                "maxacc": 50,
            }

    def __init__(self, serial_number: str, rev: str = "LTS", initialize: bool = False):
        self._current_position = 0.0
        self.serial_number = serial_number
        self.rev = rev
        self._is_initialized = False
        # self.velparams = {"max_velocity": 20.0, "acceleration": 20.0}
        # self.physiclalimsmm = {
        #     "minpos": 0.0,
        #     "maxpos": 300.0,
        #     "maxvel": 50.0,
        #     "maxdrivevel": 40,
        #     "maxacc": 50,
        # }
        # TODO: jog params
        # TODO: move params
        # TODO: homing params
        if initialize:
            self.initialize()

    @property
    @Device.check_initialized
    def current_position(self):
        """Current position of the platform."""
        return self._current_position

    def initialize(self):
        """Initialize the stage."""
        self.stage = MockLTSStage.MockStage()
        self._is_initialized = True

    @Device.check_initialized
    def configure(self, configuration: LTSConfiguration):
        """Configure the stage"""
        if not self.is_requested_configuration_valid(configuration):
            raise ValueError("Wrong configuration value.")
        self.stage.velparams["acceleration"] = configuration.acceleration
        self.stage.velparams["max_velocity"] = configuration.velocity

    @Device.check_initialized
    def calculate_arrival_time(self, distance: float):
        """Calculate time it takes for the platform to cover specified
        distance."""
        max_velocity = 20.0
        acceleration = 20.0
        t_ru = max_velocity / acceleration
        s_ru = 1 / 2 * float(str(acceleration)) * float(str(t_ru)) ** 2
        s_ru = math.ceil(s_ru)

        if distance <= 2 * s_ru:
            return 2 * np.sqrt(distance / acceleration)
        else:
            return 2 * t_ru * (distance - 2 * s_ru) / max_velocity

    @property
    def current_configuration(self):
        return LTSConfiguration(
            velocity=self.stage.velparams["max_velocity"],
            acceleration=self.stage.velparams["acceleration"],
        )

    @Device.check_initialized
    async def home(self):
        """Home the stage asynchronously."""
        self._current_position = 0.0

    @Device.check_initialized
    async def go_to(self, destination: float):
        """Go to desired position asynchronously."""
        if not self.is_request_position_valid(destination):
            raise ValueError("Invalid destination value.")
        self._current_position = destination

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def device_name(self) -> str:
        if self.rev:
            return f"Thorlabs {self.rev} stage"
        return "Thorlabs stage"


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


@dataclass
class StageInitParams:
    serial_number: str
    rev: str = "LTS"
    initialize: bool = False
    is_mockup: bool = False


@dataclass
class DetectorInitParams:
    idVendor: str = "03EB"
    idProduct: str = "2404"
    sensor: BoloMsgSensor = BoloMsgSensor.FIRST
    samples: BoloMsgSamples = BoloMsgSamples.S100
    freq: BoloMsgFreq = BoloMsgFreq.F1
    initialize: bool = False
    is_mockup: bool = False


class DeviceManager:
    def __init__(
        self,
        stage_init_params: dict[str, StageInitParams],
        detector_init_params: DetectorInitParams,
        stage_configurations: None | dict[str, LTSConfiguration],
        detector_configuration: None | BoloLineConfiguration,
    ):
        self.stages: dict[str, LTSStage]
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
        res = {"detector": self.detector.current_configuration()}
        for label in self.stages:
            res[label] = self.stages[label].current_configuration()
        return res

    def home(self, stage_label: str | list[str]):
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
                        wait_for(stage.home()), name=f"Homing stage with label {label}"
                    ),
                )

    # move_axes (speficified by label)
    def move_stage(self, stage_destination: dict[str, float]):
        async with TaskGroup() as tg:
            tasks = []
            for label in stage_destination:
                stage = self.stages[label]
                destination = stage_destination[label]
                tasks.append(
                    tg.create_task(
                        wait_for(stage.go_to(destination=destination)),
                        name=f"Moving stage labeled as {label} to position {destination}",
                    )
                )

    # measure
    def measure(self):
        return self.detector.measure()

    # live view
    def live_view(self):
        lv = LiveView(detector=self.detector)
        lv.start()

    # TODO:acquisition time (How to get acquisition for specific device?)
    # TODO:measure series?
