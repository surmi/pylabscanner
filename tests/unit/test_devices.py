import asyncio

import pytest

from pylabscanner.devices import (
    BoloLine,
    BoloLineConfiguration,
    BoloMsgFreq,
    BoloMsgSamples,
    BoloMsgSensor,
    DetectorInitParams,
    DeviceManager,
    DeviceNotFoundError,
    DeviceNotInitialized,
    LTSConfiguration,
    LTSStage,
    MockBoloLine,
    MockLTSStage,
    StageInitParams,
)


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
            detector.acquisition_time

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
        assert detector.current_configuration == configuration

    def test_detector_measures(self, mock_devices: bool):
        detector = self._initialize_bolometer(
            mock_bolometer=mock_devices, initialize=True
        )
        no_samples = detector.current_configuration.sampling.nsamp
        data = detector.measure()
        assert len(data) == no_samples

    def test_detector_device_not_found_error(self):
        detector = self._initialize_bolometer(
            mock_bolometer=False, idProduct="0000", idVendor="0000"
        )
        with pytest.raises(DeviceNotFoundError):
            detector.initialize()


@pytest.mark.stage
@pytest.mark.device
class TestStage:
    def _initialize_stage(self, mock_stage: bool, **args):
        if mock_stage:
            return MockLTSStage(**args)
        else:
            return LTSStage(**args)

    def _get_serial_number(self, mock_stage: bool) -> str:
        if mock_stage:
            return "0000000"
        # TODO: add reading from configuration or env. var.

    def test_not_initialized_error(self, mock_devices):
        stage = self._initialize_stage(
            mock_stage=mock_devices, serial_number=self._get_serial_number(mock_devices)
        )
        with pytest.raises(DeviceNotInitialized):
            stage.current_position
        with pytest.raises(DeviceNotInitialized):
            configuration = LTSConfiguration(velocity=20, acceleration=20)
            stage.configure(configuration=configuration)
        with pytest.raises(DeviceNotInitialized):
            distance = 100
            stage.calculate_arrival_time(distance=distance)
        with pytest.raises(DeviceNotInitialized):
            stage.home()
        with pytest.raises(DeviceNotInitialized):
            distance = 150
            stage.go_to(destination=distance)

    def test_stage_initializes(self, mock_devices: bool):
        stage = self._initialize_stage(
            mock_stage=mock_devices, serial_number=self._get_serial_number(mock_devices)
        )
        assert not stage.is_initialized
        stage.initialize()
        assert stage.is_initialized

    def test_stage_initializes_on_startup(self, mock_devices: bool):
        stage = self._initialize_stage(
            mock_stage=mock_devices,
            serial_number=self._get_serial_number(mock_devices),
            initialize=True,
        )
        assert stage.is_initialized

    def test_stage_configures(self, mock_devices: bool):
        stage = self._initialize_stage(
            mock_stage=mock_devices,
            serial_number=self._get_serial_number(mock_devices),
            initialize=True,
        )
        configuration = LTSConfiguration(acceleration=30, velocity=30)
        stage.configure(configuration=configuration)
        assert stage.current_configuration == configuration

    def test_stage_go_to(self, mock_devices: bool):
        stage = self._initialize_stage(
            mock_stage=mock_devices,
            serial_number=self._get_serial_number(mock_devices),
            initialize=True,
        )
        destination = 100.0 if stage.current_position != 100.0 else 50.0
        asyncio.run(stage.go_to(destination=destination))
        assert stage.current_position == destination

    def test_stage_home(self, mock_devices: bool):
        stage = self._initialize_stage(
            mock_stage=mock_devices,
            serial_number=self._get_serial_number(mock_devices),
            initialize=True,
        )
        if stage.current_position == 0.0:
            asyncio.run(stage.go_to(destination=50.0))
        asyncio.run(stage.home())
        assert stage.current_position == 0.0

    def test_stage_device_not_found_error(self):
        stage = self._initialize_stage(
            mock_stage=False,
            serial_number=self._get_serial_number(False),
        )
        with pytest.raises(DeviceNotFoundError):
            stage.initialize()

    def test_stage_validate_distance(self, mock_devices: bool):
        stage = self._initialize_stage(
            mock_stage=mock_devices,
            serial_number=self._get_serial_number(mock_devices),
            initialize=True,
        )
        destination = -100
        with pytest.raises(ValueError):
            asyncio.run(stage.go_to(destination=destination))
        destination = 400
        with pytest.raises(ValueError):
            asyncio.run(stage.go_to(destination=destination))

    def test_stage_validate_configuration(self, mock_devices: bool):
        stage = self._initialize_stage(
            mock_stage=mock_devices,
            serial_number=self._get_serial_number(mock_devices),
            initialize=True,
        )
        configuration = LTSConfiguration(acceleration=60, velocity=60)
        with pytest.raises(ValueError):
            stage.configure(configuration=configuration)
        configuration = LTSConfiguration(acceleration=-20, velocity=-20)
        with pytest.raises(ValueError):
            stage.configure(configuration=configuration)


@pytest.mark.detector
@pytest.mark.stage
@pytest.mark.device
class TestDeviceManager:
    def _define_init_params(self, use_mockups: bool):
        return (
            DetectorInitParams(is_mockup=use_mockups),
            {
                "x": StageInitParams(serial_number="123", is_mockup=use_mockups),
                "y": StageInitParams(serial_number="123", is_mockup=use_mockups),
                "z": StageInitParams(serial_number="123", is_mockup=use_mockups),
            },
        )

    def _init_manager(self, mock_devices: bool):
        detector_params, stage_params = self._define_init_params(mock_devices)
        manager = DeviceManager(
            stage_init_params=stage_params,
            detector_init_params=detector_params,
        )
        manager.initialize()
        return manager

    def test_manager_initializes(self, mock_devices: bool):
        self._init_manager(mock_devices=mock_devices)

    def test_manager_configures(self, mock_devices: bool):
        manager = self._init_manager(mock_devices=mock_devices)
        detector_configuration = BoloLineConfiguration(
            frequency=BoloMsgFreq.F5,
            sampling=BoloMsgSamples.S500,
            sensor_id=BoloMsgSensor.FOURTH,
        )
        lts_configuration = LTSConfiguration(velocity=10, acceleration=20)
        manager.configure(
            detector_configuration=detector_configuration,
            stage_configurations={"x": lts_configuration},
        )
        current_configuration = manager.current_configuration

        assert current_configuration["detector"] == detector_configuration
        assert current_configuration["x"] == lts_configuration

    def test_manager_moves_stages(self, mock_devices: bool):
        manager = self._init_manager(mock_devices=mock_devices)
        assert manager.stages["x"].current_position == 0.0
        destination = 100.0
        manager.move_stage(stage_destination={"x": destination})
        assert manager.stages["x"].current_position == destination

    def test_manager_homes_stages(self, mock_devices: bool):
        manager = self._init_manager(mock_devices=mock_devices)
        destination = 100.0
        manager.move_stage(stage_destination={"x": destination})
        assert manager.stages["x"].current_position == destination
        manager.home(stage_label="all")
        assert manager.stages["x"].current_position == 0.0

    def test_manager_measures(self, mock_devices: bool):
        manager = self._init_manager(mock_devices=mock_devices)
        detector = manager.detector
        no_samples = detector.current_configuration.sampling.nsamp
        data = detector.measure()
        assert len(data) == no_samples
