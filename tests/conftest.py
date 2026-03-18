import os
import pytest

from core.driver.appium_adapter import AppiumAdapter
from core.utils.config_loader import load_yaml


@pytest.fixture(scope="session")
def device_config():
    config_path = os.path.join("configs", "device.yaml")
    return load_yaml(config_path)


@pytest.fixture(scope="function")
def appium_adapter(device_config):
    server_url = device_config["appium"]["server_url"]

    capabilities = {}
    capabilities.update(device_config["device"])
    capabilities.update(device_config["app"])

    implicit_wait = device_config.get("runtime", {}).get("implicit_wait", 10)
    evidence_dir = device_config.get("runtime", {}).get("evidence_dir", "evidence")

    adapter = AppiumAdapter(
        server_url=server_url,
        capabilities=capabilities,
        implicit_wait=implicit_wait,
        evidence_dir=evidence_dir,
    )

    adapter.start()
    yield adapter
    adapter.quit()