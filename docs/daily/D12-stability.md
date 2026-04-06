# Day 12：回放与稳定性（可复现）

## 目标

让同一条用例可以重复回放，至少支持 10 次批量执行，并在失败时能快速定位原因，而不是只看一条失败日志。

本次实现包含三部分：

1. 单次回放：根据历史 run 的 `actions.json` 重新执行。
2. 失败分类：把失败归到固定类别，减少人工排查成本。
3. 批量稳定性：同一条动作计划连续执行 N 次，输出稳定性报告。

---

## 已落地能力

### 1）单次回放

入口：`core/report/replay.py`

支持：

- 读取某个 run 下的 `actions.json`
- 读取最近一条带 `actions.json` 的 run
- 用 `Executor` 重放动作
- 输出 `replay_summary.json`

### 2）失败分类

当前固定分类：

- 定位失败
- 弹窗未覆盖
- 加载超时
- AI 输出非法
- 未知失败

分类依据来自：

- `meta.json`
- `ocr.txt`
- `perception.json`
- `policy/policy_parsed.json`

失败时会额外输出：

- `failure_classification.json`

### 3）批量回放稳定性

新增批量能力：

- `replay_plan_batch(...)`
- `replay_run_batch(...)`
- `replay_latest_run_batch(...)`

默认推荐 10 次连续回放。

每次回放都会生成一个独立 run，最后再汇总成一个 batch 报告目录：

```text
evidence/_batches/<batch_id>/
```

输出：

- `source_actions.json`
- `stability_report.json`
- `stability_report.md`

---

## 稳定性报告字段

`stability_report.json` 里核心字段：

- `total_runs`
- `passed_runs`
- `failed_runs`
- `success_rate`
- `unstable`
- `category_counts`
- `avg_executed`
- `min_executed`
- `max_executed`
- `items[]`

说明：

- `unstable=true` 表示同一条用例既有成功也有失败，属于典型不稳定用例
- `category_counts` 用于看主要失败类型是否集中
- `executed` 用于判断失败是否总发生在相同步骤附近

---

## 推荐使用方式

### 单次回放

```bash
python tools/replay_plan.py --latest
```

### 指定 run 回放

```bash
python tools/replay_plan.py --source-run-id 20260327_120000_ab12cd
```

### 10 次批量回放

```bash
python tools/replay_plan.py --latest --times 10
```

### 指定间隔和首错即停

```bash
python tools/replay_plan.py --latest --times 10 --sleep-s 1 --stop-on-first-failure
```

---

## 失败分类与改进策略

### 一、定位失败

表现：

- CLICK 失败
- 没找到合适 bbox
- 自愈后仍失败
- 文本断言未命中

改进策略：

- 扩充 locator_store 同义词
- 优先使用 bbox OCR 而不是纯文本 OCR
- 为关键按钮补 `target / selector / logical_name`
- 把 click recovery 从“tap 抛异常”扩展到“点击后无进展”也触发

### 二、弹窗未覆盖

表现：

- OCR 中存在明显权限/引导/授权文案
- `perception.overlay_suspected=true`
- `meta.extra.no_progress=true`

改进策略：

- 补充 `PopupHandlers` 规则
- 给 `PopupRuleHandler` 注入 `policy_runner`
- 增加常见弹窗模板词和区域提示
- 对登录/权限相关弹窗做单独分类处理

### 三、加载超时

表现：

- `TimeoutError`
- 错误信息出现 loading / timeout / wait timeout

改进策略：

- 增加 WAIT 前后的就绪判断
- 为目标页面加入“可见文本到达”断言
- 对弱网环境单独设置更宽松的等待策略

### 四、AI 输出非法

表现：

- policy JSON 解析失败
- schema 校验失败
- 输出字段不合法

改进策略：

- 强化 prompt 中的输出约束
- 所有 AI 输出先过 parser / schema 再执行
- 保留 raw output 和 parsed output 便于复盘
- 对常见错误输出加兜底修复逻辑

---

## 当前边界

当前批量回放已经可以用于“同一条用例跑 10 次并统计失败类型”，但还没做：

- 按页面状态聚合失败
- 按步骤名聚合失败热区
- 自动输出“最可能根因”排名
- 点击后无进展自动二次恢复

这些可以放到后续增强版继续做。
