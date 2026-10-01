# WolfGym 代码与复现入口

当前研究入口是组合任务闭环监管。数值协议见
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，状态见
[EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md)。旧静态任务和单战术接口不作为
另一套当前实验配置。

## 模块边界

| 层 | 主要路径 | 责任 |
| --- | --- | --- |
| 模拟器 | `src/wolfgym/core/` | 账户、交易、社会传播、角色、组合攻击、状态与随机流 |
| 公开观察 | `src/wolfgym/observations/` | 从已实现状态派生只读公开字段、episode 内身份和序列化 |
| 动作执行 | `src/wolfgym/regulation/` | 定义和执行控制；组合入口只开放资产 WAIT/FREEZE 子集 |
| 组合 Gym | `src/wolfgym/env/combination.py`、`combination_contract.py` | 人口/资源/源码锁定，三世界、四动作、预算、30 日、奖励与评价隔离 |
| 划分与计划 | `src/wolfgym/benchmark/combination.py`、`combination_experiments.py`、`combination_extensions.py` | 固定社会身份、两条件完整清单、联合预检与独立扩展计划 |
| 统一输入/规则 | `src/wolfgym/evaluation/combination_public.py` | `public-structured-v1`、五帧历史、21 基线实例、报警/目标选择 |
| 决策适配 | `src/wolfgym/evaluation/combination_decision.py` | RR/RL/LR/LL、严格动作 JSON、局部事务状态、错误 WAIT |
| 可信宿主 | `combination_host.py`、`combination_model_host.py` | 仿真、公开请求、三世界原始数据、回执、恢复与动作重放 |
| 费用持久化 | `src/wolfgym/evaluation/combination_requests.py` | SQLite 请求账本、尝试/成功缓存、额度预留与未知费用 |
| 呈现 | `combination_presentation.py`、`combination_presented_host.py` | 一致公开换名/换序及物理动作反映射 |
| 计分与分析 | `combination_task.py`、`counterfactual_welfare.py`、`combination_results.py` | H/G/事件、完整结果校验、社会内配对和区间 |
| 学习 | `src/wolfgym/env/combination_learning.py`、`evaluation/combination_bc.py` | 1895 维同信息编码、真实 episode sampler、训练策略 |
| 有界 Qwen 批次 | `src/wolfgym/evaluation/combination_qwen_*.py` | 已冻结请求设置、输入说明与针对性恢复；不提供通用自动重试 |

模拟器不调用评价模型生成标签。ObservationBuilder 只读已实现公开状态，不重新生成
文本。可信宿主能访问隐藏世界和结果；模型请求函数只能获得白名单公开请求字典，
不能获得环境对象、文件工具、root seed、条件标签或私有评价 API。
这是研究 API 的信息隔离，不宣称能够沙箱化任意 Python 代码。

## Gym 入口

使用 `episode_plan(split)` 选择题目，`make_combination_env(episode)` 创建世界。
显式 Gym 注册函数 `register_combination_gym()` 返回 `WolfGym-Combination-v0`。
安装项目及 `rl` 可选依赖后使用；普通模块导入不应自动启动训练或模型请求。

`reset()` 返回 `(observation, info)`，`step(action)` 返回
`(observation, reward, terminated, truncated, info)`。公开 info 包含同一结构化观测、
三资产动作目标和合法 mask。默认 reward=0。`training_reward=True` 仅用于 train
及退役工程探针，终值返回有符号 ΔH；validation/test 拒绝训练奖励。

运行合同绑定执行文件指纹，构造/reset 在分配市场前拒绝源码和资源漂移。
`close()` 解除三世界、历史和大对象引用；可重复 close，再 reset 重建同一题目。
外部提前 cutoff 不可提交完整成绩。

性能实现 `combination_fast.py`、`combination_learning_fast.py` 和观察快速路径用于
等价执行，不能改变公开视图、日时序、RNG 或损害定义。优化需要同工程轨迹的
账户、金融功能、公开输入/动作与评分等价验证，不能只比较最终汇总均值。

## 常用脚本职责

以下是代码导航，不是启动授权；已完成的阶段不应重复执行。

| 入口 | 用途 |
| --- | --- |
| `scripts/plan_combination_benchmark.py` | 仅生成固定划分元数据清单 |
| `scripts/plan_combination_experiments.py` | 联合配置与预检，不等于放行付费/test |
| `scripts/plan_combination_extensions.py` | 独立扩展合同和种子审查 |
| `scripts/run_combination_baselines.py` / `report_combination_baselines.py` | 强基线执行 / 独立原始账户与公开动作审计 |
| `scripts/train_combination_policy.py` / `report_combination_learning.py` | 三 seed PPO / 全量验证审计 |
| `scripts/run_combination_bc_pilot.py`、`collect_combination_bc_round2.py`、`train_combination_bc_round2.py` | 已收口的两轮 BC 与数据聚合比较 |
| `scripts/run_combination_bc_final.py` / `report_combination_bc_final.py` | 固定最终 BC 三 seed / 验证审计 |
| `scripts/run_combination_qwen_dev32.py` / `report_combination_qwen_dev32.py` | 当前暂停的 32 社会 Qwen 开发批次 / 全网格审计 |
| `scripts/probe_combination_qwen_network.py` | 无认证网络检查；GET 正常不证明付费 POST 稳定 |

历史恢复脚本各自对应已记录的故障和原 manifest，不能作为通用“换目录继续跑”
入口，也不能恢复时自动放宽预算/次数。当前 pending 请求和恢复约束见状态文档。

联合规划器用 `additional_diagnostic` 记录显式决定：草案 `None` 为未决并阻止冻结，
最终必须选择 `none` 或 `rename_reorder`/`liquidity`/`benign_spread` 中一项。
核心清单为四主 LL 加 M1/M3 各 RL、LR、当前帧 LL，共 10 单元；单项诊断再加两个，
共 12。仅选流动性或良性传播时要求对应独立扩展合同/清单；未选扩展不能成为
预检的必填项。旧 16 单元/179,712 请求 inventory 是历史全量设想，当前草案
尚未决定附加项。结构齐全仍不能代替供应商证据、费用约束和最终执行冻结。

## 源码、配置与本地产物

- `src/`、`scripts/`、`tests/`、运行时 JSON、依赖说明和这五份文档是代码仓库内容。
- `data/` 保存本地运行清单、原始账户、请求账本、响应/费用、模型和分析回执；
  大型产物和可能包含服务信息的响应不随源码默认上传。
- `build/`、`dist/`、`*.egg-info`、`__pycache__` 和测试缓存是可再生构建产物。
- 运行时机器版本与文件名保留以维持可执行合同，文档只维护当前稳定入口。

历史清单绑定文件路径及字节。删除旧文档、移动脚本或格式化绑定源码可能影响
历史恢复；先保留本地来源归档，保持旧 manifest 和账本原样。新运行应使用新清单
和当前协议，不伪造旧指纹。恢复旧运行需要审查兼容性和 pending 用量，不能通过
删除校验继续执行。

本地 `data/cleanup_snapshot_20261001.tar.gz` 保留本次整理前的完整代码与文档，
不随源码默认上传，可用于核对历史 manifest 所需路径和字节。受运行合同保护的
`src/wolfgym/evaluation/switching_probe.py` 仍保留历史文档路径及原字节；其旧 probe
不是当前研究入口，历史复现应结合该归档审查依赖，不能将当前协议冒充旧预声明。

## 验证口径

代码行为修改应运行受影响的定向测试；协议/文档更新检查机器数值、当前机器
暂停状态、内部链接和源码入口。测试范围、执行与未执行项分别报告，不把历史
验收数拼接成一次全仓库通过，也不为仓库整理重新生成实验或调用付费模型。

可在本地使用 `PYTHONPATH=src python -m pytest tests -q` 验证现有测试。
部分测试需要 `rl`/`llm` 等可选依赖或本地研究产物；缺少产物不是完整实验已验收
的证据。科研审计还须检查完整网格、原始账户与动作，不能仅测试摘要 JSON。
