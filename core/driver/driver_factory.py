from appium import webdriver
from appium.options.android import UiAutomator2Options


def create_android_driver(server_url: str, capabilities: dict):
    options = UiAutomator2Options().load_capabilities(capabilities)
    driver = webdriver.Remote(server_url, options=options)
    return driver