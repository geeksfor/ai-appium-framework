import os
import pytest

from core.driver.appium_adapter import AppiumAdapter
from core.utils.config_loader import load_yaml
from core.report.evidence import EvidenceManager
from core.report.step_logger import StepLogger
from core.report.step_runner import StepRunner
from core.executor.executor import Executor

@pytest.fixture(scope="function")
def executor(appium_adapter, step_runner):
    return Executor(adapter=appium_adapter, step_runner=step_runner)

@pytest.fixture(scope="session")
def evidence_manager(device_config):
    base_dir = device_config.get("runtime", {}).get("evidence_dir", "evidence")
    return EvidenceManager(base_dir=base_dir)

@pytest.fixture(scope="function")
def step_runner(evidence_manager, appium_adapter):
    step_logger = StepLogger()
    return StepRunner(
        evidence=evidence_manager,
        step_logger=step_logger,
        driver_getter=lambda: appium_adapter.driver
    )

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