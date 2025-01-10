from threading import Event, Thread
from time import sleep

import pytest

from pylabscanner.devices.manager import DeviceManager, LiveView


@pytest.mark.detector
@pytest.mark.stage
@pytest.mark.device
class TestLiveView:
    def test_initialization(self, default_manager: DeviceManager):
        lv = LiveView(manager=default_manager)
        assert lv.measurements.empty()
        assert not lv.shutdown_event.is_set()
        assert not lv.detector_thread.is_alive()
        assert not lv.interrupt_thread.is_alive()

    def _interrupt_after_time(self, time: int, event: Event):
        """`time` - time in seconds

        `event` - event to fire after the time runs"""
        print("interrupt start")
        sleep(time)
        event.set()

    def test_gathers_data(self, default_manager: DeviceManager):
        lv = LiveView(manager=default_manager)
        lv.detector_thread.start()
        timeout_thread = Thread(
            target=self._interrupt_after_time,
            args=(1, lv.shutdown_event),
            name="Test timeout thread",
        )
        timeout_thread.start()
        lv.detector_thread.join()
        timeout_thread.join()
        assert not lv.measurements.empty()
        queue_data = lv.measurements.get()
        data = queue_data["data"]
        det_no_samp = queue_data["det_no_samp"]
        det_freq = queue_data["det_freq"]
        assert det_no_samp == default_manager.detector.samples
        assert det_freq == default_manager.detector.frequency == det_freq
        assert len(data) == default_manager.detector.samples
