import os
import pytest

from core.driver.appium_adapter import AppiumAdapter
from core.executor.executor import Executor
from core.heal.heal_policy import HealPolicy, PolicyRunnerHealAIProvider
from core.perception.ocr import build_ocr_provider_from_env
from core.perception.perception import Perception
from core.policy.policy_runner import PolicyRunner
from core.policy.qwen_client import QwenClient
from core.report.evidence import EvidenceManager
from core.report.step_logger import StepLogger
from core.report.step_runner import StepRunner
from core.runtime.app_session import AppSession
from core.runtime.project_loader import load_project_profile
from core.utils.config_loader import load_yaml


@pytest.fixture(scope="session")
def project_id():
    return os.getenv("TEST_PROJECT", "wechat")


@pytest.fixture(scope="session")
def project_profile(project_id):
    return load_project_profile(project_id)


@pytest.fixture(scope="session")
def perception():
    ocr = build_ocr_provider_from_env()
    if ocr is None:
        return None
    print(f"[OCR INIT] provider={ocr.__class__.__name__} env={os.getenv('OCR_PROVIDER')}")
    return Perception(ocr_provider=ocr)


@pytest.fixture(scope="session")
def device_config():
    config_path = os.path.join("configs", "device.yaml")
    return load_yaml(config_path)


@pytest.fixture(scope="session")
def evidence_manager(device_config):
    base_dir = device_config.get("runtime", {}).get("evidence_dir", "evidence")
    return EvidenceManager(base_dir=base_dir)


@pytest.fixture(scope="session")
def policy_runner(evidence_manager):
    print("[DEBUG] DASHSCOPE_API_KEY exists =", bool(os.getenv("DASHSCOPE_API_KEY")))
    print("[DEBUG] DASHSCOPE_BASE_URL =", os.getenv("DASHSCOPE_BASE_URL"))
    print("[DEBUG] DASHSCOPE_MODEL =", os.getenv("DASHSCOPE_MODEL"))
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("[DEBUG] policy_runner skipped: DASHSCOPE_API_KEY missing")
        return None
    try:
        qwen = QwenClient(
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus"),
        )
        runner = PolicyRunner(qwen=qwen, evidence_manager=evidence_manager)
        print("[DEBUG] policy_runner created:", runner)
        return runner
        # return PolicyRunner(qwen=qwen, evidence_manager=evidence_manager)
    except Exception:
        print("[DEBUG] policy_runner init failed:", repr(e))
        return None


@pytest.fixture(scope="function")
def appium_adapter(device_config, project_profile):
    server_url = device_config["appium"]["server_url"]
    capabilities = {}
    capabilities.update(device_config["device"])
    capabilities.update(device_config.get("app", {}))
    if not capabilities.get("appPackage") and project_profile.app_package:
        capabilities["appPackage"] = project_profile.app_package
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
def step_runner(evidence_manager, appium_adapter, perception):
    step_logger = StepLogger()
    return StepRunner(
        evidence=evidence_manager,
        step_logger=step_logger,
        driver_getter=lambda: appium_adapter.driver,
        perception=perception,
    )


@pytest.fixture(scope="function")
def executor(appium_adapter, step_runner, perception, policy_runner):
    print("[DEBUG] executor sees DASHSCOPE_API_KEY =", bool(os.getenv("DASHSCOPE_API_KEY")))
    print("[DEBUG] executor policy_runner =", policy_runner)

    ai_provider = PolicyRunnerHealAIProvider(policy_runner) if policy_runner is not None else None
    print("[DEBUG] executor ai_provider =", ai_provider)
    
    click_healer = HealPolicy(
        locator_store_path="core/heal/locator_store.yaml",
        ai_provider=ai_provider,
        accept_threshold=0.70,
    )
    return Executor(adapter=appium_adapter, step_runner=step_runner, perception=perception, click_healer=click_healer)


@pytest.fixture(scope="function")
def app(appium_adapter, step_runner, executor, perception, policy_runner, project_profile):
    return AppSession(
        adapter=appium_adapter,
        step_runner=step_runner,
        executor=executor,
        perception=perception,
        policy_runner=policy_runner,
        project_profile=project_profile,
    )
