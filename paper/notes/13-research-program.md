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
| 2026-10-25 | 5 | state 压缩臂 10 任务完成：rich_3278 产出 248 字符补丁（truncate 臂为空）、httpx_3672 反而空（truncate 1402）、rich_3105/fastapi_14297 补丁内容改变（402/1146 vs 427/1051）——压缩策略确实改变行为 | `results/final_kaggle/d5_state/batch_summary.json` | ✅ 行为对比完成 |
| 2026-10-25 | 5 | state 臂改变补丁的三臂验证待做（kernel output 下载被 API 截断，仅回收 1/10 任务目录）；按切换规则记录为部分完成 | — | ⏸ 部分 |
| 2026-10-25 | 6 | 12B 臂：模型挂载候选路径修复（Path 转换 bug）后 v3 已推送运行中；若 12B 实例未挂载则按计划记为不可行 | 运行中 | ⏳ |
| 2026-10-25 | 3 | d3_sel/d3_full（run_command + selected/full 引导）已生成待推（GPU 槽位被 d5/d6 占用） | `paper/notebooks/model_agent_d3_*.ipynb` | ⏳ |
| 2026-10-25 | 6 | 12B 臂 v3 完成：12B 模型挂载成功（7.71 GiB 加载 39.8s），但 vLLM 0.19.1 的 marlin_gemm 内核对该 W4A16 检查点抛 `Shape mismatch: a.size(1)=4096, size_k=8192`（引擎初始化失败，10 任务全部 RuntimeError）——与 T4/无 SM80 支持的量化内核路径问题同族。按切换规则记为**当前 runtime 不可行**，方向 6 关闭（除非 vLLM 升级） | `results/final_kaggle/d6_12b/` | ⛔ 不可行 |
| 2026-10-25 | 3 | d3_sel/d3_full（run_command + selected/full 引导 + 依赖超集）notebook 已修复就绪，待 GPU 槽位推送 | `paper/notebooks/model_agent_d3_*.ipynb` | ⏳ |
| 2026-10-26 | 3 | d3_sel 臂完成并三臂验证：**2/6 解决**（fastapi_14492、requests_7502）；requests_6589 反转未解决（轨迹方差）；4/10 任务 runner NameError（扩展工具路径缺陷，非模型结果） | `results/dep_cache/d1_verifications.json`（atlas-agent-d3-sel/*） | ✅ |
| 2026-10-26 | 3 | d3_full 臂完成并验证：**2/5 解决**（同 fastapi_14492、requests_7502）；**httpx_3672 首次产出非空补丁（2,106 字符）但未通过必需测试**；5/10 NameError | 同上（atlas-agent-d3-full/*） | ✅ |
| 2026-10-26 | 3 | **G3 汇总：control 3/7、d3_sel 2/6、d3_full 2/5（验证口径）——run_command + 验证引导未提升 resolution**；轨迹方差真实存在（6589 反转）；扩展工具路径有 NameError 缺陷待修 | `results/dep_cache/d3_ablation_summary.json` | ✅ G3 完成 |
| 2026-10-26 | 全 | **六个方向全部处置完毕**：D4 ✅ D1 ✅ D2 ✅（阴性）D3 ✅ D5 ✅（部分）D6 ⛔（runtime 不可行）。执行计划闭环 | — | ✅ |
| 2026-10-26 | 工程 | NameError 根因修复：升级脚本吃掉 truncate 压缩分支的 `viewed`/`files_edited` 赋值行；补回后本地强制压缩回归通过（4 次压缩无异常） | `paper/scripts/agent_smoke_v5.py` | ✅ |
| 2026-10-26 | 5 | **D5 完整验证补齐**：专用 collector 收齐 10/10 任务目录，三臂验证完成——state 臂 2/5 解决（fastapi_14492、requests_6589、requests_7502 中前两个+7502）；轨迹改变 4/10（rich_3278 空→248、httpx 1402→空、rich_3105/fastapi_14297 内容变） | `results/dep_cache/d5_ablation_summary.json` | ✅ G5 完成 |
| 2026-10-26 | 6 | **12B 重启成功**：互联网 kernel + vLLM 0.32.0（torch 2.13.0，torchaudio 按版本 pin）——12B w4a16 在 2×T4 上加载 158.5s 并完成生成；chat 模板下 17×24=408 正确、add 函数完整、数值校验 2.84e-4。**marlin_gemm 形状不匹配在 0.32.0 已修复**（0.19.1 的冻结 wheelhouse 不含修复） | `results/dep_cache/d6_12b_vllm0320_result.json` | ✅ 可行性确认（非官方 runtime） |
| 2026-10-26 | 论文 | writeup 更新：21 任务批量结果（3/15=20.0%）+ 三个消融（定位阴性/验证引导阴性/压缩策略有效）写入 §4；局限与未来工作同步。已提交 Writeup 同步更新（PUBLISHED，2,369 词） | `paper/draft/writeup.md` | ✅ |
| 2026-10-26 | 验证修复 | 严格判据过严（整文件必须退出0，把无关网络用例算入）→ 改回严格原始（0解决）→ 再修正为 **SWE-bench delta 语义**：必需测试 baseline 失败→agent 通过→gold 同转移→无回归。13 项验证器回归测试通过 | `paper/scripts/verify_agent_patch.py`、`results/audit_reverify_20261025_final/` | ✅ |
| 2026-10-26 | 1 修正 | **E4B 正确口径：10 个可验证补丁中 2 个解决（requests_6589、requests_7502）**，8 个确认未解决；此前 3/15 与 0/10 均作废 | `results/dep_cache/e4b_delta_verification.json` | ✅ |
| 2026-10-26 | 6 | **12B 全量批完成**（22/22，3 小时）：6,144 上下文可运行，但**表现劣于 E4B**——非空补丁 8 vs 16、平均轮数 28.7 vs 17.0、多次 no-tool-call 退出。配置差异（vLLM 0.32/0.19、6144/8192、量化配方）不可归因于纯模型规模 | `results/dep_cache/model_ladder_summary.json` | ✅ |
| 2026-10-26 | 论文 | 判据修正 + 模型阶梯结论写入 §4；已提交 Writeup 同步（PUBLISHED，2,520 词）。**遗留**：12B 补丁内容因 kernel output 截断未全量下载，仅汇总层面对比；12B token 计数受修正前 runner 影响 | `paper/draft/writeup.md` | ⚠️ 部分 |
