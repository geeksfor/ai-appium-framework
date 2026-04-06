import os
import pytest

from core.driver.appium_adapter import AppiumAdapter
from core.utils.config_loader import load_yaml
from core.report.evidence import EvidenceManager
from core.report.step_logger import StepLogger
from core.report.step_runner import StepRunner
from core.executor.executor import Executor
from core.perception.ocr import QwenOCRBoxesProvider
from core.perception.perception import Perception
from core.heal.heal_policy import HealPolicy, PolicyRunnerHealAIProvider
from core.policy.policy_runner import PolicyRunner
from core.policy.qwen_client import QwenClient


@pytest.fixture(scope="session")
def perception():
    # 有 key 就优先启用 bbox OCR，没有就返回 None（后面 StepRunner 自动兜底）
    if os.getenv("DASHSCOPE_API_KEY"):
        ocr = QwenOCRBoxesProvider(
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.getenv("DASHSCOPE_BBOX_MODEL", "qwen-vl-ocr-latest"),
        )
        return Perception(ocr_provider=ocr)
    return None


@pytest.fixture(scope="session")
def evidence_manager(device_config):
    base_dir = device_config.get("runtime", {}).get("evidence_dir", "evidence")
    return EvidenceManager(base_dir=base_dir)


@pytest.fixture(scope="session")
def policy_runner(evidence_manager):
    if not os.getenv("DASHSCOPE_API_KEY"):
        return None

    try:
        qwen = QwenClient(
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus"),
        )
        return PolicyRunner(qwen=qwen, evidence_manager=evidence_manager)
    except Exception:
        return None


@pytest.fixture(scope="function")
def step_runner(evidence_manager, appium_adapter, perception):
    step_logger = StepLogger()
    return StepRunner(
        evidence=evidence_manager,
        step_logger=step_logger,
        driver_getter=lambda: appium_adapter.driver,
        perception=perception,
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


@pytest.fixture(scope="function")
def executor(appium_adapter, step_runner, perception, policy_runner):
    ai_provider = PolicyRunnerHealAIProvider(policy_runner) if policy_runner else None
    click_healer = HealPolicy(
        locator_store_path="core/heal/locator_store.yaml",
        ai_provider=ai_provider,
        accept_threshold=0.72,
    )

    return Executor(
        adapter=appium_adapter,
        step_runner=step_runner,
        perception=perception,
        click_healer=click_healer,
    )
