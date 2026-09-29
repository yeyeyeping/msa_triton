# 文档索引

文档按用途分为两组：`knowledge/` 解释算子是什么、接口如何约定、训练如何
工作；`debugging/` 保存如何排查、如何复测以及每个版本的实际结果。

## 知识与设计说明

目录：[knowledge/](knowledge/)。了解实现时，建议先读训练流程，再读接口方案。

| 文档 | 内容 |
|---|---|
| [MSA 训练与 KL 流程](knowledge/msa_training_flow.md) | 三阶段数据流、Indexer 训练／冻结行为、KL 与梯度边界 |
| [接口与实施方案](knowledge/implementation_plan.md) | TND 接口、eager 适配、算子拆分、前反向与集成边界 |
| [vLLM Ascend k2q 参考与适配](knowledge/vllm_ascend_k2q.md) | 稀疏边反转、CSR 布局、推理参考与本地训练实现的差异 |

[eager 迁移来源](../eager/PROVENANCE.md) 与代码一起保留在 `eager/`，记录
锁定的 Transformers 版本、原生行为和适配差异。

## 调试、审查与报告

目录：[debugging/](debugging/)。理解问题的完整来龙去脉先读复盘，确认实际
通过范围再读验证记录。历史报告按对应提交解释，不作为新的执行指令。

| 文档 | 内容 |
|---|---|
| [完整调试复盘](debugging/debugging_retrospective.md) | 从接口约定、数值反例到 NPU attention 修复通过的证据链与失败尝试 |
| [验证方法与执行记录](debugging/validation.md) | FP64 门禁、CPU／NPU 的验证边界、分阶段测试统计 |
| [必要修复记录](debugging/correctness_fixes.md) | score 胜者精度、地址宽度、按需反向及 attention 累加修复 |
| [原始代码审查](debugging/code_review.md) | 原始优化建议与风险分析，作为历史材料保留 |
| [审查复核与反例](debugging/code_review_response.md) | 对建议的逐项复核、数值反例、采纳或拒绝的依据 |
| [NPU 首轮失败](debugging/npu_validation_20260928.md) | 首轮 90 passed／48 failed 的分解与第一轮兼容性候选 |
| [NPU 候选复测失败](debugging/npu_retest_20260929.md) | 旧候选复测、环境信息与取证过程 |
| [attention 编译取证](debugging/npu_attention_compile_probe.md) | 同 IR 重放、PlanMemory pass、六项结构消融及判断边界 |
| [attention 48 项复测协议与结果](debugging/npu_attention_streaming_retest.md) | 逐 query 补偿实现的实机验证、命令与短回传格式 |
| [历史 NPU 执行代理任务](debugging/npu_glm_execution.md) | 旧候选的分阶段执行、哈希核对、失败采集与报告模板 |

后续新增接口、数学和架构说明放入 `knowledge/`；实验步骤、执行协议、审查
和结果记录放入 `debugging/`，并在这里补充入口。文档之间使用相对链接。
