import pytest

from pylabscanner.devices import (
    BoloLine,
    BoloLineConfiguration,
    BoloMsgFreq,
    BoloMsgSamples,
    BoloMsgSensor,
    DeviceNotFoundError,
    DeviceNotInitialized,
    MockBoloLine,
)

# TODO: how to do prerequesits? E.g. the device needs to be connected for some
# tests on actual class, the device needs to be initialized
# TODO: how to switch between mock and actual class

# LTS
# ======> test for mock and for actual class
# Assert initialization required
# initialize -> the device is initialized | the device is not initialized
# configure -> TODO changing max_velocity and max_acceleration
# home -> homing is done within specified amount of time (timeout)
# go_to -> correct distance is reached within specified amount of time (timeout)
# get_current_position -> correct variable (value and type) is returned
# get_arrival_time -> correct variable (value and type) is returned
# ======> test for actual class only (may require monkeypatch)
# Returns right exception on device not connected


@pytest.mark.detector
@pytest.mark.device
class TestDetector:
    def _initialize_bolometer(self, mock_bolometer: bool, **args):
        if mock_bolometer:
            return MockBoloLine(**args)
        else:
            return BoloLine(**args)

    def test_not_initialized_error(self, mock_devices: bool):
        detector = self._initialize_bolometer(mock_bolometer=mock_devices)
        with pytest.raises(DeviceNotInitialized):
            configuration = BoloLineConfiguration(
                frequency=BoloMsgFreq.F1,
                sampling=BoloMsgSamples.S100,
                sensor_id=BoloMsgSensor.FIRST,
            )
            detector.configure(configuration=configuration)
        with pytest.raises(DeviceNotInitialized):
            detector.measure()
        with pytest.raises(DeviceNotInitialized):
            detector.get_ta()

    def test_detector_initializes(self, mock_devices: bool):
        detector = self._initialize_bolometer(mock_bolometer=mock_devices)
        assert not detector.is_initialized
        detector.initialize()
        assert detector.is_initialized

    def test_detector_initializes_on_startup(self, mock_devices: bool):
        detector = self._initialize_bolometer(
            mock_bolometer=mock_devices, initialize=True
        )
        assert detector.is_initialized

    def test_detector_configures(self, mock_devices: bool):
        detector = self._initialize_bolometer(
            mock_bolometer=mock_devices, initialize=True
        )
        configuration = BoloLineConfiguration(
            frequency=BoloMsgFreq.F1,
            sampling=BoloMsgSamples.S100,
            sensor_id=BoloMsgSensor.FIRST,
        )
        detector.configure(configuration=configuration)
        assert detector.get_current_configuration() == configuration

    def test_detector_measures(self, mock_devices: bool):
        detector = self._initialize_bolometer(
            mock_bolometer=mock_devices, initialize=True
        )
        no_samples = detector.get_current_configuration().sampling.nsamp
        data = detector.measure()
        assert len(data) == no_samples

    def test_detector_device_not_found_error(self):
        detector = self._initialize_bolometer(
            mock_bolometer=False, idProduct="0000", idVendor="0000"
        )
        with pytest.raises(DeviceNotFoundError):
            detector.initialize()
