import asyncio
from time import sleep

from thorlabs_apt_device import aptdevice_motor
from thorlabs_apt_device import protocol as apt
from thorlabs_apt_device.enums import EndPoint
from thorlabs_apt_device.protocol.functions import _pack

from .utility import mm2steps, steps2mm


class LTS(aptdevice_motor.APTDevice_Motor):
    def __init__(
        self,
        serial_port=None,
        vid=None,
        pid=None,
        manufacturer=None,
        product=None,
        serial_number="45",
        location=None,
        home=True,
        invert_direction_logic=True,
        swap_limit_switches=True,
        initDefault=True,
        bays=(EndPoint.USB,),
    ):
        super().__init__(
            serial_port=serial_port,
            vid=vid,
            pid=pid,
            manufacturer=manufacturer,
            product=product,
            serial_number=serial_number,
            location=location,
            home=False,
            invert_direction_logic=invert_direction_logic,
            swap_limit_switches=swap_limit_switches,
            status_updates="auto",
            controller=EndPoint.USB,
            bays=bays,
            channels=(1,),
        )
        # Wait for pooling (from `aptdevice.__init__()`) to initialize
        sleep(0.2)

        self.req_info()

        # no bays and channels
        bay_i = 0
        channel_i = 0
        self.bays = (EndPoint.USB,)
        self.channels = (1,)

        # for identification
        self.serial_number = serial_number

        self.status = self.status_[0][0]
        """Alias to first bay/channel of :data:`APTDevice_Motor.status_`."""
        # add status 'move_completed'
        self.status.update({"move_completed": True})

        self.velparams = self.velparams_[0][0]
        """Alias to first bay/channel of :data:`APTDevice_Motor.velparams_`"""

        self.genmoveparams = self.genmoveparams_[0][0]
        """Alias to first bay/channel of :data:`APTDevice_Motor.genmoveparams_`"""

        self.jogparams = self.jogparams_[0][0]
        """Alias to first bay/channel of :data:`APTDevice_Motor.jogparams_`"""

        self.physiclalimsmm = {
            "minpos": 0.0,
            "maxpos": 300.0,
            "maxvel": 50.0,
            "maxdrivevel": 40,
            "maxacc": 50,
        }
        """
        Dictionary with physical limits of the stage in metric units.
        """

        self.convunits = {"pos": 409600, "vel": 21987328, "acc": 4506}
        """
        Convertions ratios for position, velocity and acceleration.
        """

        self.physiclalimsmu = {
            "minpos": mm2steps(self.physiclalimsmm["minpos"], self.convunits["pos"]),
            "maxpos": mm2steps(self.physiclalimsmm["maxpos"], self.convunits["pos"]),
            "maxvel": mm2steps(self.physiclalimsmm["maxvel"], self.convunits["vel"]),
            "maxdrivevel": mm2steps(
                self.physiclalimsmm["maxdrivevel"], self.convunits["vel"]
            ),
            "maxacc": mm2steps(self.physiclalimsmm["maxacc"], self.convunits["acc"]),
        }

        # Enable the stage
        if not self.status["channel_enabled"]:
            self.set_enabled(True)
            sleep(0.3)  # wait for enable
        self.req_info()
        sleep(0.2)

        # Initialize with default parameters (based on values in Kinesis 1.14.37)
        if initDefault:
            # homing params; velocity is guessed
            self._loop.call_later(
                0.15,
                self.set_home_params,
                mm2steps(5, self.convunits["vel"]),
                mm2steps(0.5, self.convunits["pos"]),
            )

            # move params
            self._loop.call_later(
                0.15, self.set_move_params, mm2steps(0.05, self.convunits["pos"])
            )

            # jog params
            self._loop.call_later(
                0.15,
                self.set_jog_params,
                mm2steps(5.0, self.convunits["pos"]),
                mm2steps(10.0, self.convunits["acc"]),
                mm2steps(10.0, self.convunits["vel"]),
            )

            # velocity params
            self._loop.call_later(
                0.15,
                self.set_velocity_params,
                mm2steps(20.0, self.convunits["acc"]),
                mm2steps(20.0, self.convunits["vel"]),
            )
            sleep(0.2)

        # Home the device
        if home:
            self._loop.call_later(0.15, self.home, bay_i, channel_i)
            sleep(0.2)

    def __str__(self) -> str:
        return f"LTS ({self.serial_number})"

    def _process_message(self, m) -> None:
        super()._process_message(m)
        if m.msg == "mot_move_completed":
            self.status["move_completed"] = True

    def req_info(self) -> None:
        """Schedule `hw_req_info`."""
        source = EndPoint.HOST
        dest = EndPoint.BAY0

        self._log.debug("Requesting info.")
        self._loop.call_soon_threadsafe(
            self._write, apt.hw_req_info(source=source, dest=dest)
        )

    async def aso_close_wait(self, wait=1) -> None:
        """Schedule `close()` method on the device.

        Args:
            wait (int, optional): time to wait after closing (in seconds).
                Defaults to 1.
        """
        self._loop.call_soon_threadsafe(self.close)
        await asyncio.sleep(wait)

    async def aso_wait_for_homed(self, interval=0.1) -> None:
        """Wait for the stage to be homed.
        Recomended to be used within `asyncio.wait_for` to add timeout.

        Args:
            interval (float, optional): interval in which to test wheather
                stage is homed. Defaults to 0.1.
        """
        while not self.status["homed"]:
            await asyncio.sleep(interval)

    async def aso_wait_for_move_completed(self, interval=0.1) -> None:
        """Wait for the motion to finish.
        Recomended to be used within `asyncio.wait_for` to add timeout.

        Args:
            interval (float, optional): interval in which to test
                wheather motion is complited. Defaults to 0.1.
        """
        while not self.status["move_completed"]:
            await asyncio.sleep(interval)

    async def aso_home(self, waitfinished=True) -> None:
        """Home a stage with built in waiting for the action to finish.
        Recomended to be used within `asyncio.wait_for` with timeout.

        Args:
            waitfinished (bool, optional): if `True` waits for the motion to
                finish. Defaults to `True`.
        """
        # Default for LTS
        bay = 0
        channel = 0

        super().home(bay, channel)
        call_delay = 0.15
        # timerHandle = self._loop.call_later(
        #     call_delay, self.homed
        # )  # note: different loop
        await asyncio.sleep(call_delay + 0.1)

        interval = 0.1
        # Wait for the stage to start moving
        while not (self.status["moving_forward"] or self.status["moving_reverse"]):
            await asyncio.sleep(interval)

        if self.status["homed"]:
            # Wait for the movement to finish
            if waitfinished:
                while self.status["moving_forward"] or self.status["moving_reverse"]:
                    await asyncio.sleep(interval)
        else:
            # wait for status 'homed'
            if waitfinished:
                while not self.status["homed"]:
                    await asyncio.sleep(interval)

    async def aso_move_absolute(
        self, position: int | float, waitfinished: bool = True
    ) -> None:
        """Move a stage (absolute move command) with built in waiting for the
        action to finish.

        Recomended to be used within `asyncio.wait_for` with
        timeout.

        Args:
            position (int | float): absolute position to move to.
            waitfinished (bool, optional): if `True` waits for the motion to
                finish. Defaults to `True`.

        Raises:
            ValueError: if the required position is out of physical limits of
                the stage.
        """
        # Default for LTS
        bay = 0
        channel = 0

        if (
            position < self.physiclalimsmu["minpos"]
            or position > self.physiclalimsmu["maxpos"]
        ):
            raise ValueError(
                "Move absolute: required position out of physical limits of the stage."
            )

        super().move_absolute(position=position, now=True, bay=bay, channel=channel)
        self.status["move_completed"] = False

        interval = 0.1

        # wait for the movement to start
        while not (self.status["moving_forward"] or self.status["moving_reverse"]):
            await asyncio.sleep(interval)

        # wait for motion to finish
        if waitfinished:
            while not self.status["move_completed"]:
                await asyncio.sleep(interval)

        # Also viable option
        # while self.status['moving_forward'] or self.status['moving_reverse']:
        #     await asyncio.sleep(interval)

    def sync_home(self) -> None:
        """Homes the stage without waiting for the action to finish."""
        # Default for LTS
        bay = 0
        channel = 0

        super().home(bay, channel)
        self._loop.call_later(0.25, self.homed())  # note: different loop
        sleep(0.5)

        interval = 0.1

        # Wait for the stage to start moving
        while not (self.status["moving_forward"] or self.status["moving_reverse"]):
            sleep(interval)

    def homed(self) -> None:
        """
        Require stage to respond with "homed" message after finishing homing.
        """
        # Default for LTS
        bay = 0
        channel = 0

        self._log.debug(
            f"Homed (homing with confirmation) [bay={self.bays[bay]:#x}, "
            f"channel={self.channels[channel]}]."
        )
        self._loop.call_soon_threadsafe(
            self._write,
            mot_move_homed(
                source=EndPoint.HOST,
                dest=self.bays[bay],
                chan_ident=self.channels[channel],
            ),
        )

    def move_absolute(self, position: int | float) -> None:
        """Synchronous method for moving a stage (absolute move command)
        without waiting for the action to finish.

        Args:
            position (int | float): absolute position to move to.

        Raises:
            ValueError: if the required position is out of physical limits of
                the stage.
        """
        # Default for LTS
        bay = 0
        channel = 0

        if (
            position < self.physiclalimsmu["minpos"]
            or position > self.physiclalimsmu["maxpos"]
        ):
            raise ValueError(
                "Move absolute: required position out of physical limits of"
                " the stage."
            )

        super().move_absolute(position=position, now=True, bay=bay, channel=channel)
        self.status["move_completed"] = False

        interval = 0.1

        # wait for the movement to start
        while not (self.status["moving_forward"] or self.status["moving_reverse"]):
            sleep(interval)


class LTSC(LTS):
    def __init__(
        self,
        serial_port=None,
        vid=None,
        pid=None,
        manufacturer=None,
        product=None,
        serial_number="45",
        location=None,
        home=True,
        invert_direction_logic=True,
        swap_limit_switches=True,
        initDefault=True,
    ):
        super().__init__(
            serial_port=serial_port,
            vid=vid,
            pid=pid,
            manufacturer=manufacturer,
            product=product,
            serial_number=serial_number,
            location=location,
            home=False,
            invert_direction_logic=invert_direction_logic,
            swap_limit_switches=swap_limit_switches,
            bays=(EndPoint.BAY0,),
        )
        # Currently LTSxC (replacement for older LTSx) reports at `EndPoint.BAY0`
        # (LTSx reported on `EndPoint.USB`).

    def __str__(self) -> str:
        return f"LTSC ({self.serial_number})"

    async def aso_home(self, waitfinished=True) -> None:
        """Asynchronous method for homing a stage with built in waiting for the
        action to finish.

        Recomended to be used within `asyncio.wait_for` with timeout.

        Args:
            waitfinished (bool, optional): if `True` waits for the motion to
                finish. Defaults to `True`.
        """
        # Default for LTS
        bay = 0
        channel = 0

        super().home(bay, channel)
        self._loop.call_later(0.25, self.homed)  # note: different loop
        sleep(0.5)

        interval = 0.1

        # Wait for the stage to start moving
        while not self.status["homing"]:
            await asyncio.sleep(interval)

        if self.status["homed"]:
            # Wait for the movement to finish
            if waitfinished:
                while self.status["homing"]:
                    await asyncio.sleep(interval)
        else:
            # wait for status 'homed'
            if waitfinished:
                while not self.status["homed"]:
                    await asyncio.sleep(interval)


def mot_move_homed(dest: int, source: int, chan_ident: int) -> bytes:
    """Generate `mot_move_homed` message. Arguments values according to
    "Thorlabs Motion Controllers Host-Controller Communications Protocol",
    p34-36, issue 37, 22 May 2023.

    Args:
        dest (int): destination of message.
        source (int): source of message.
        chan_ident (int): channel. Applicable if slot/bay unit. For single
            channel devices check the protocol.

    Returns:
        bytes: prepared message.
    """
    return _pack(0x0444, dest, source, param1=chan_ident)


def mot_req_statusbits(dest: int, source: int, chan_ident: int) -> bytes:
    """Generate `mot_req_statusbits` message. Arguments values according to
    "Thorlabs Motion Controllers Host-Controller Communications Protocol",
    p34-36, issue 37, 22 May 2023.

    Args:
        dest (int): destination of message.
        source (int): source of message.
        chan_ident (int): channel. Applicable if slot/bay unit. For single
            channel devices check the protocol.

    Returns:
        bytes: _description_
    """
    return _pack(0x0429, dest, source, param1=chan_ident)


async def aso_close(devs: LTS | list[LTS]) -> None:
    if type(devs) is not list:
        devs = [devs]

    tasks = []
    for d in devs:
        tasks.append(
            asyncio.create_task(d.asoclosewait(), name=f"close_{d.serial_number}")
        )

    await asyncio.wait({*tasks}, return_when=asyncio.ALL_COMPLETED)


async def aso_home_devs(
    devs: LTS | list[LTS], timeout: int | float = 61, waitfinished: bool = True
) -> None:
    """Asynchronously begin `home` operation on all provided stages and wait
    for finish. All calls are done with timeout of 61 seconds.

    Args:
        devs (LTS | list[LTS]): single instance or list of instances of LTS
            objects.
        timeout (int | float, optional): timeout for movement to finish in
            seconds. Recommended value ca. 60 (assuming LTS300 and homing speed
            5mm/s it should cover the whole range). Defaults to 61.
        waitfinished (bool, optional): wait for the action to finish. Defaults
            to True.
    """
    if type(devs) is not list:
        devs = [devs]

    async with asyncio.TaskGroup() as tg:
        try:
            tasks = []
            for stage in devs:
                tasks.append(
                    tg.create_task(
                        asyncio.wait_for(stage.aso_home(waitfinished), timeout),
                        name=f"Homing stage with s/n {stage.serial_number}",
                    )
                )
        except TimeoutError:
            devs[0]._log.warn("Timout during homing")


async def aso_move_devs(
    devs: LTS | list[LTS],
    pos: int | float | list[int] | list[float],
    timeout: int | float = 61,
    waitfinished: bool = True,
) -> None:
    """Asynchronously begin `absolute move` operation on all provided stages and
    wait for finish. All calls are done with timeout of 61 seconds.

    Recommended for moving into specific position (not for scanning while
    moving).

    Args:
        devs (LTS | list[LTS]): single instance or list of instances of LTS
            objects.
        pos (int | float): requested position in microsteps.
        timeout (int | float, optional): timeout for movement to finish in
            seconds. Recommended value ca. 60 (assuming LTS300 and homing speed
            5mm/s it should cover the whole range). Defaults to 61.
        waitfinished (bool, optional): wait for the action to finish. Defaults
            to True.
    """
    if type(devs) is not list:
        devs = [devs]

    # if multiple devices but only one distance -> assume all go to the same pos
    if type(pos) is not list:
        pos = [pos] * len(devs)

    async with asyncio.TaskGroup() as tg:
        try:
            tasks = []
            for stage, position_to_go in zip(devs, pos):
                if position_to_go != stage.status["position"]:
                    tasks.append(
                        tg.create_task(
                            asyncio.wait_for(
                                stage.aso_move_absolute(position_to_go, waitfinished),
                                timeout=timeout,
                            ),
                            name=f"Moving stage {stage.serial_number} to "
                            f"position "
                            f"{steps2mm(position_to_go, stage.convunits["pos"])}",
                        )
                    )
                else:
                    stage._log.debug("Stage already in position.")
        except TimeoutError:
            devs[0]._log.warn("Timeout during moving")
