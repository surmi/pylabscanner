import pytest
from fixtures.setup import mock_measurement_data, mock_metadata  # noqa


def pytest_addoption(parser):
    parser.addoption(
        "--use-mocks",
        action="store_true",
        help="Use mocks devices instead of actual devices.",
    )


@pytest.fixture
def mock_devices(request):
    return request.config.getoption("--use-mocks")
