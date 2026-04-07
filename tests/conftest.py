"""Pytest configuration and shared fixtures."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--server", default="http://localhost:8000", help="Server URL to test against"
    )


@pytest.fixture
def server_url(request):
    """Server URL fixture, configurable via --server flag."""
    return request.config.getoption("--server")
