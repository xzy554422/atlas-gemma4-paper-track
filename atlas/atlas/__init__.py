"""Atlas — 面向单机预算的图引导 SWE 智能体验证框架。

四个模块，各自对应论文的一节：

  graphio.py  官方 graph/embedding 格式读写（含 L2 归一化）
  impact.py   H1 静态测试影响分析（反向 k-hop）
  locate.py   H2 三档注入策略（B0 / B1 / B2）
  emit.py     生成 agent.yaml / sub_agents / skills（受官方硬限制约束）

定位与度量口径见 README.md；数据集见 datasets/swe_atlas_129.jsonl。
"""

__version__ = "0.1.0"

__all__ = ["graphio", "impact", "locate", "emit"]
