import math
from abc import ABC, abstractmethod
from enum import Enum
from time import sleep
from typing import List

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


class Detector(ABC):
    """Abstract class for detectors."""

    @abstractmethod
    def measure(self):
        """Measure single value."""
        raise NotImplementedError

    @abstractmethod
    def measure_series(self):
        """Make series of measurements."""
        raise NotImplementedError

    @abstractmethod
    def get_ta(self):
        """Time of acquisition in seconds."""
        raise NotImplementedError


class BoloLine(Detector):
    """Class for reading from line of bolometers by Luvitera.

    Tested for THz mini (line of 4 wideband bolometers).
    """

    def get_ta(self) -> float:
        """Total acqusition time (between write to the device and read from
        the device) in seconds"""
        return self._ta + self._read_delay

    def get_sensor(self) -> str:
        """Number of the sensor to read from."""
        return self._sensor.name

    def set_sensor(self, sensor: BoloMsgSensor) -> None:
        """Change sensor to read from. Recalculates internal parameters"""
        self._sensor = sensor
        self._makemsg()

    def get_samples(self) -> int:
        """Number of samples to register in single read."""
        return self._samples.nsamp

    def set_samples(self, samples: BoloMsgSamples) -> None:
        """Change number of measured samples. Recalculates internal parameters"""
        self._samples = samples
        self._recalculate_ta()
        self._makemsg()

    def get_freq(self) -> int:
        """Reading frequency in kHz."""
        return self._freq.freq

    def set_freq(self, freq: BoloMsgFreq) -> None:
        """Change frequency of measuring samples. Recalculates internal parameters"""
        self._freq = freq
        self._recalculate_ta()
        self._makemsg()

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

    def _cold_start(self) -> None:
        """To be used before first measurement after downtime of the device
        (device disconnected from power source)."""
        self._dev.reset_input_buffer()
        self._dev.reset_output_buffer()

        self._write()
        sleep(self._read_delay)
        sleep(self._ta)
        self._read()

        self._write()
        sleep(self._read_delay)
        sleep(self._ta)
        self._read()

    def get_msg(self) -> bytes:
        """Message written to the detector."""
        return self._msg

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
        cold_start=False,
    ) -> None:
        self._hid = f"{idVendor}:{idProduct}"  # hardware id (vendor id : product id)
        restr = "(?i)" + self._hid
        try:
            self._port = next(list_ports.grep(restr)).name
        except StopIteration:
            raise DeviceNotFoundError(msg="Bolometer line not found.")
        self._read_delay = (
            0.001  # selected experimentally (longer wait time may be necessary)
        )
        self._write_delay = 0.001
        self._dev = serial.Serial(
            port=self._port,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            timeout=0,  # NOTE: under test (previously 0.1); currently: non-blocking
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            write_timeout=0.1,
        )

        self._samples = BoloMsgSamples.S100
        self._freq = BoloMsgFreq.F1
        self._ta = self._samples.nsamp / (self._freq.freq * 1000)

        # first communication
        self._makemsgparts(sensor=(0).to_bytes(1))
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
        self._initialized = True
        self._sensor = sensor
        self._samples = samples
        self._freq = freq
        self._makemsg()
        self._recalculate_ta()
        if cold_start:
            # NOTE: single measurement right after connecting physically the
            # device and setting measurement parameters have some weird peak. Use
            # this method make read of it
            self._cold_start()

    def __str__(self):
        return (
            f"Luvitera Mini THz Sensor, 4 pixel array with {self._senstype} "
            f"sensor type."
        )

    def __repr__(self):
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


class Source(ABC):
    """Abstract class for sources."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError


class MotorizedStage(ABC):
    """Abstract class for motorized stages."""

    @abstractmethod
    def get_current_position(self):
        """Current position of the platform."""
        raise NotImplementedError

    @abstractmethod
    def initialize(self):
        """Initialize the stage."""
        raise NotImplementedError

    @abstractmethod
    def get_arrival_time(self, distance: float):
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
        if initialize:
            self.initialize()

    def get_current_position(self):
        """Current position of the platform in mm."""
        return steps2mm(self.stage.status["position"], self.stage.convunits["pos"])

    def initialize(self):
        """Initialize the stage."""
        if self.rev == "LTS":
            self.stage = LTS(serial_number=self.serial_number, home=False)
        elif self.rev == "LTSC":
            self.stage = LTSC(serial_number=self.serial_number, home=False)

    def get_arrival_time(self, distance: float):
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

    async def home(self):
        """Home the stage asynchronously."""
        self.stage.aso_home(waitfinished=True)

    async def go_to(self, destination: float):
        """Go to desired position asynchronously."""
        self.stage.aso_move_absolute(
            position=mm2steps(destination, self.stage.convunits["pos"])
        )


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


class MockStage(MotorizedStage):
    """Abstract class for motorized stages."""

    # TODO: add temporal simulation of movement?

    def __init__(self):
        self.current_position = 0.0

    @abstractmethod
    def get_current_position(self):
        """Current position of the platform."""
        return self.current_position

    @abstractmethod
    def initialize(self):
        """Initialize the stage."""
        pass

    @abstractmethod
    def get_arrival_time(self, distance: float):
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

    @abstractmethod
    async def home(self):
        """Home the stage asynchronously."""
        self.current_position = 0.0

    @abstractmethod
    async def go_to(self, destination: float):
        """Go to desired position asynchronously."""
        self.current_position = destination
