import pytest


# @pytest.mark.skip(reason="示例用法：需要真机/Appium/项目页面配合，不在离线单测里执行")
def test_login_success_demo(app):
    app.launch()
    app.expect_state("Login")
    app.tap_semantic("login_button")
    app.expect_state("Home")
