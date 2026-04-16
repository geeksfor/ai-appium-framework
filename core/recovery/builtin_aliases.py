from __future__ import annotations

from typing import Dict, List


# 这里只保留“语义别名”，不再携带 preferred_regions。
# 区域优先级统一交给 core.recovery.action_semantics.default_regions_for_action(...)
BUILTIN_ALIASES: Dict[str, List[str]] = {
    "close": ["关闭", "跳过", "点击跳过", "我知道了", "知道了", "稍后", "取消"],
    "back": ["返回", "上一步", "返回首页"],
    "login_button": ["登录", "去登录", "立即登录"],
    "confirm": ["确定", "确认", "完成", "继续", "下一步", "提交"],
    "username": ["用户名", "账号", "手机号"],
    "password": ["密码", "请输入6~20位密码", "请输入密码"],
    "save": ["保存", "提交保存", "保存并继续"],
    "add": ["新增", "添加", "新建", "创建"],
    "edit": ["编辑", "修改", "更新"],
    "delete": ["删除", "移除", "删掉"],
    "search": ["搜索", "查找", "检索"],
}


def get_builtin_aliases(logical_name: str) -> List[str]:
    key = str(logical_name or "").strip()
    if not key:
        return []
    return list(BUILTIN_ALIASES.get(key, []))
