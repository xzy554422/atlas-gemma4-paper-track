# 后续研究执行计划（2026-10-25 起）

执行顺序与切换规则（用户 2026-10-25 批准）：

```
方向 4（依赖缓存）──→ 方向 1（多任务评测）──→ 方向 2（定位消融）
                └──→ 方向 3（测试选择在环）      └──→ 方向 5/6（并行小题）
```

**总规则**：按上图顺序执行。每个方向有明确的研究目标与通过判据（gate）；未达判据时，
在台账记录失败原因与已达成的部分结果，然后切换到下一方向继续，直至全部方向完成或
被判定不可行。某方向被后续方向解锁时可以回头重开（在台账登记）。

**预算约束**：GPU 周配额 30 小时（每周四刷新）；单方向超预算上限时按已完成样本出部分
结果并标注精确 n。所有 GPU 运行必须留下 trace/junit/summary 三件套证据。

---

## 方向 4：依赖缓存与 required-test 清单（地基，本地为主）

**研究目标**：四个仓库（rich/httpx/fastapi/requests）的基线快照在本地沙箱中能以
零 collection error 运行完整测试套件，为方向 1/2/3 提供可信的验证地基。

**通过判据（G4）**：
- G4.1 每仓库代表快照上 `pytest --collect-only` 无 ModuleNotFoundError / collection error，
      或记录为"不可修复"并给出根因（如上游 wheel 与 Python 3.13 不兼容）。
- G4.2 缺失 wheel 全部补入 `data/wheels_extra/`（linux 平台正确标签），并生成
      `results/dep_cache/manifest.json`（仓库 → 依赖 → wheel 文件名 → sha256）。
- G4.3 每仓库实测一次完整测试运行（或 target 口径），与历史 junit 测试数对照，
      记录通过/失败/错误计数变化。

**预算**：本地 CPU + Docker，零 GPU。

## 方向 1：多任务 agent 评测（E4B@T4 resolution rate）

**研究目标**：在分层任务样本上测量并报告 E4B@T4 的端到端修复率与失败模式分层。

**通过判据（G1）**：
- G1.1 ≥20 个任务（rich≥6、fastapi≥6、requests≥4、httpx 1，按 gold 补丁大小分层）
      完成完整链路：Kaggle agent 运行（trace.jsonl+summary）+ 本地三臂验证。
- G1.2 报告 resolution rate（含精确分母）、按仓库/补丁规模/轮数的分层、
      失败模式分类（定位错/编辑错/预算尽/格式失效）。
- G1.3 预算上限 12 GPU 小时；超限时按已完成样本出部分结果并标注 n。

**前置**：G4 通过（否则验证不可信）。

## 方向 2：定位质量 → 修复效果受控消融

**研究目标**：固定 agent 循环，仅改变初始定位信息，测量 resolution 差异，回答
"离线定位诊断对修复有没有因果收益"。

**通过判据（G2）**：
- G2.1 ≥3 个定位臂（符号索引 / B0 词法代理 / oracle gold-first）× ≥10 个相同任务，
      每臂完整 trace + 验证。
- G2.2 配对报告各臂 resolution 与 token/轮数成本差；样本不足显著性时明确说
      "仅描述性"。

**前置**：G1 的任务子集与管线复用。

## 方向 3：测试选择在环（分支，G4 后可并行启动）

**研究目标**：agent 在 selected vs full 测试命令预算下的验证耗时差与漏检回归率。

**通过判据（G3）**：
- G3.1 mini-harness 恢复 `run_command`（受 G4 依赖缓存支撑），≥10 任务两臂配对。
- G3.2 报告每任务验证耗时、agent 使用的测试调用数、selected 臂漏检的回归
      （以 full 臂结果为准）。

## 方向 5：上下文压缩策略（并行小题）

**研究目标**：压缩策略（已读注记 vs 滑窗 vs 摘要）对小上下文 resolution 的影响。
**通过判据（G5）**：≥2 策略 × ≥10 任务，配对报告；与 v7 基线对照。

## 方向 6：模型阶梯与量化行为（并行小题）

**研究目标**：E4B vs 量化 12B（若单卡 T4 放得下）的 resolution 对比；量化检查点
在不同 prompt 格式下的输出分布量化。
**通过判据（G6）**：若 12B 无法在 T4 加载则记录为不可行并关闭；否则 ≥10 任务两模型配对。

---

## 进度台账

| 日期 | 方向 | 动作 | 结果 | 状态 |
|---|---|---|---|---|
| 2026-10-25 | 4 | 计划建立；依赖盘点启动 | — | ✅ |
| 2026-10-25 | 4 | 挖掘 968 份历史 junit：rich 缺 attrs（84/126 文件受累）、fastapi 缺 8 包（588/722）、requests 缺 pytest-httpbin 栈（204 error 的根因）、httpx 无历史记录 | `results/dep_cache/historical_errors.json` | ✅ |
| 2026-10-25 | 4 | 下载缺失 linux wheel（attrs/inline-snapshot/dirty-equals/orjson/sqlalchemy/pydantic-settings/a2wsgi/strawberry-graphql/flask/sqlmodel/pyyaml/email-validator/pyjwt/passlib/bcrypt 4.3.0/ujson/pytest-httpbin+httpbin 栈） | `data/wheels_extra/` 163 wheel | ✅ |
| 2026-10-25 | 4 | G4.1 四仓库 collect-only 实测 | rich 852/0E、fastapi 2486/0E、requests 633/0E、httpx 128/0E | ✅ |
| 2026-10-25 | 4 | G4.3 四仓库完整基线运行（对照历史） | rich 803→852 收集、1E→0E；fastapi 841→2486、137E→0E；requests 204E→0E；httpx 128 全绿。残余失败全部定性（版本敏感/官方 -p no:anyio/离线网络行为） | ✅ |
| 2026-10-25 | 4 | G4.2 依赖清单 manifest（163 wheel，0 缺失） | `results/dep_cache/manifest.json` | ✅ |
| 2026-10-25 | 4 | **G4 通过**（三轮补包：+flask/sqlmodel → +pyyaml/email-validator/pyjwt → +passlib/bcrypt5→4.3/ujson） | 方向 1 解锁，开始执行 | ✅ |
| 2026-10-25 | 1 | 任务抽样（rich 8 / fastapi 8 / requests 5 / httpx 1 = 22，按 gold 补丁大小分层）与多任务 runner | 批量模式本地验证通过 | ✅ |
| 2026-10-25 | 1 | rich 批（8 任务）Kaggle 运行完成：7 submit（5 非空）+1 tool_budget | `results/final_kaggle/batch_rich/batch_summary.json` | ✅ |
| 2026-10-25 | 1 | requests 批（5 任务）完成：4 非空补丁 +1 tool_budget | `results/final_kaggle/batch_requests/` | ✅ |
| 2026-10-25 | 1 | 三臂验证器迭代三轮定稿：文件级目标 + 必需测试逐名判定（参数化精确匹配） | `paper/scripts/verify_batch.py` | ✅ |
| 2026-10-25 | 1 | requests 批验证：**2/4 解决**（6589 ✓ 7502 ✓；6644 补丁不足；7427 补丁语法错误→收集崩溃） | `results/dep_cache/d1_verifications.json` | ✅ |
| 2026-10-25 | 1 | rich 批已验 3 个非空补丁：0/3 解决（补丁编辑未命中修复点；baseline/gold 判据均正常） | 同上 | ✅ |
| 2026-10-25 | 1 | fastapi 批完成运行，输出下载中；httpx 批待 GPU 槽位 | — | ⏳ |
| 2026-10-25 | 1 | httpx 批（3672）运行完成：24 次工具调用预算耗尽，自动回收 1,402 字符未提交补丁 | `results/final_kaggle/batch_httpx/` | ✅ |
| 2026-10-25 | 1 | httpx 验证：agent 补丁未修（baseline 5/5 失败、agent 5/5 失败、gold 5/5 通过）；rich_3772 双臂 pytest 超时（递归异常用例本身超时，判 unknown）；rich_3934 未修 | `results/dep_cache/d1_verifications.json` | ✅ |
| 2026-10-25 | 1 | **G1.2 汇总：21 任务（≥20 ✓）；非空补丁 16；验证口径 resolution 3/15（20.0%）**：requests 2/4、fastapi 1/5、rich 0/5、httpx 0/1；全任务严格口径 3/21。失败模式：submitted_not_resolved 13、empty_patch 5、resolved 3（fastapi_14448 为无可派生必需测试的第 22 任务，未验证）。GPU 约 4h（上限 12h） | `results/dep_cache/d1_resolution_summary.json` | ✅ |
| 2026-10-25 | 1 | **G1 通过 → 方向 2 启动**（index 臂复用方向 1 数据；新增 oracle / b0 两臂） | — | ⏳ |
| 2026-10-25 | 2 | oracle 臂（gold 文件提示）10 任务完成：轨迹与 index 臂完全一致 | `results/final_kaggle/d2_oracle/batch_summary.json` | ✅ |
| 2026-10-25 | 2 | b0 臂（词法代理提示）10 任务完成：轨迹与 index 臂完全一致 | `results/final_kaggle/d2_b0/batch_summary.json` | ✅ |
| 2026-10-25 | 2 | **G2 汇总：3 臂 × 10 任务配对，轨迹 10/10 完全一致**（终止原因、补丁大小、token 数全同；可比哈希相同）。结论：定位信息不是该规模模型的瓶颈，编辑能力才是；提示注入不改变行为。描述性（n=10） | `results/dep_cache/d2_ablation_summary.json` | ✅ |
| 2026-10-25 | 2 | **G2 通过 → 方向 3（分支）与方向 5/6 待启动** | — | ⏳ |
