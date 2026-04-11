# Day 13：AI 决策 + OCR/CV 落地点击（终极优化版）

## 目标
把旧链路“AI 直接输出 bbox/坐标点击”升级成：

- AI：负责页面理解、目标决策、候选文案/区域提示
- OCR：负责文本控件定位
- CV 模板：负责固定 icon（关闭 X、返回箭头等）
- 执行层：负责点击、落盘调试图、失败兜底

## 关键改动

### 1. CLICK DSL 支持语义点击
`core/executor/action_schema.py`

新增：
- `target_type`
- `text_candidates`
- `region_hint / region_hints`
- `template_id / template_paths`
- `verify_texts`

这样 AI 不必硬产坐标，可以输出：

```json
{
  "type": "CLICK",
  "target_type": "text",
  "target": "登录",
  "text_candidates": ["登录", "立即登录"],
  "region_hint": "bottom_primary_area"
}
```

### 2. Perception 落盘图片尺寸
`core/perception/perception.py`

新增：
- `image_width / image_height`
- `screen_width / screen_height`
- `ocr_mode`

这样后续 bbox 归一化更稳，也方便排查 screenshot/window size 差异。

### 3. 新增本地 OCR Provider
`core/perception/ocr.py`

新增：
- `RapidOCRBoxesProvider`
- `EasyOCRBoxesProvider`
- `build_ocr_provider_from_env()`

建议：
- 默认优先本地 OCR（RapidOCR / EasyOCR）
- 没有本地 OCR 再回退到 Qwen bbox

### 4. ClickResolver 升级为混合定位器
`core/recovery/click_resolver.py`

能力：
- OCR 候选打分
- 框合理性过滤
- 根据区域提示修正点击点
- 支持模板匹配
- AI 只给语义候选时，重新交给 OCR/CV 落地
- 仅最后兜底才接受 AI 直接给坐标

### 5. 新增模板匹配与调试图
- `core/recovery/template_matcher.py`
- `core/recovery/click_debugger.py`

输出：
- `ocr_boxes_debug.png`
- `artifacts/heal/*/click_debug.png`

调试时一眼能看出：
- OCR 框对不对
- 最终点击点在哪里
- 模板是否命中

### 6. HealPolicy 改为调用 ClickResolver
`core/heal/heal_policy.py`

保留原有 `heal_click()` 接口，方便兼容现有 Executor。
但内部不再“盲信 AI 坐标”，而是统一走：
- selector / target -> semantic target
- OCR / CV -> click point
- save_path -> 候选 json + debug 图

### 7. Policy Prompt 改为优先语义点击
`assets/prompts/policy_v1.txt`

现在的 prompt 会明确要求：
- 优先输出 `target / text_candidates / region_hint`
- 只有极其确定时才输出 `x_pct / y_pct`

## 推荐环境变量

```bash
export OCR_PROVIDER=rapidocr   # 或 easyocr / qwen
export DASHSCOPE_API_KEY=xxx   # 仅 qwen 模式需要
```

## 模板目录

默认模板目录：

```text
assets/templates/
```

可放：
- `assets/templates/close_x.png`
- `assets/templates/popup_close/*.png`
- `assets/templates/back_arrow.png`

## 建议落地顺序
1. 先装本地 OCR（RapidOCR 优先）
2. 保留现有 StateMachine / PolicyRunner / PopupHandlers
3. 让 AI 只出 semantic CLICK
4. 用 `ocr_boxes_debug.png` 和 `click_debug.png` 复盘失败点击
5. 只为高频 icon 维护少量模板

## 结论
这版框架的主路线已经从：

- **AI 直接定位**

升级成：

- **AI 决策 + OCR/CV 定位 + 可视化调试 + 点击兜底**

更适合 H5 pageSource 很难拿到的黑盒场景。
