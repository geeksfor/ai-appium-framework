from appium import webdriver
from appium.options.android import UiAutomator2Options


def create_android_driver(server_url: str, capabilities: dict):
    options = UiAutomator2Options().load_capabilities(capabilities)
    try:
        from appium.webdriver.client_config import AppiumClientConfig

        client_config = AppiumClientConfig(remote_server_addr=server_url)
        try:
            client_config.timeout = 120
        except Exception:
            pass
        return webdriver.Remote(server_url, options=options, client_config=client_config)
    except Exception:
        return webdriver.Remote(server_url, options=options)
