# WolfGym 公开观测与语言契约

当前主评测输入为 `public-structured-v1`，包含结构化公开证据及选定消息的按资产
汇总，不含消息正文。底层公开消息接口保留，用于审计和未来独立表示实验；
不能把“有公开文本”写成已证实语言语义必不可少。
主实验设定见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。

## 主输入

[combination_public.py](../src/wolfgym/evaluation/combination_public.py) 白名单生成公开帧：

| 内容 | 固定规则 |
| --- | --- |
| 时间/权限 | day 0–29、society_size、information_level、remaining_budget、四动作 mask、三个公开 action_targets |
| 市场 | 7 个原公开数值，不提供资产价格、sector、市值或基本面 |
| 社会 | 6 个原公开传播/集中度数值 |
| 资产 | 全部三资产：公开 ID 和 5 个原公开数值 |
| 实体 | 当期至多 5 个公开实体：ID 和 5 个数值，身份按 ID 对齐 |
| 消息汇总 | 对原选定公开列表按资产 count、audience_reach 合计、present；不增加列表外消息 |
| 历史 | 当前及此前至多四帧，缺失前缀显式 mask；另保留所有已执行 FREEZE 的日期/目标/费用 |

源公开值可往返序列化，不额外应用旧张量的统一 ±10 截断；原 schema 个别字段
本来就有定义内 clip，不能把两者混淆。学习策略内部单位缩放、log 压缩与 float32
属于算法实现，必须记录。原检测器与目标规则可从主输入复原。

每 episode、normal/attack 条件和模型实例分别建新会话。供应商只接收当前明确
窗口及动作记录，不保留隐藏会话记忆或上一轮解释。当前帧消融仅删此前四帧，
日期、预算、字段和完整干预记录相同；当前帧的三日统计仍在。

## 字段语义与校准说明

当前 Qwen 说明读取 [combination_input_dictionary_v2.txt](../src/wolfgym/evaluation/combination_input_dictionary_v2.txt)。
字典解释单位、统计窗口、选择方式和动作实际权限，不能提供推荐动作、报警阈值、
攻击日历或任务答案。关键区别：

- return 为比例，0.01 表示 1%；volatility_3d 是短窗口总体标准差，不是年化。
- volume_zscore 是短窗口统计，源字段 clip 到 [-10,10]，不是校准尾概率。
- market liquidity 是相对 spread 代理，不是现金或可成交库存。
- 实体是按公开活动分数选出的极值，非随机五人；排名变化不代表身份相同。
  activity_change 是相对历史活动的比例，不是 z-score。
- 消息列表最多取近三日的八条，先最近日，再较高已实现 reach，最后公开 ID 排序。
  汇总 reach 不去重接收者，可以超过 1；零 reach 消息和没有消息用 present 区分。
  count/reach 都不是全市场消息总量。
- 极端活动、传播或实体统计本身不等于实质损害或最优干预标签。

校准 LL 的开发结果没有改善总保护；说明的工程正确性与策略效果分别验收，
不根据弱成绩继续调提示。状态见 [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md)。

## 底层公共消息生命周期

1. 规则或可选 LLM agent 发出带结构化内容和 public text 的 Message。
2. 规则消息若无正文，确定性 renderer 仅依据公开内容生成普通语言。
3. 社会环境记录已实现正文和 audience size，状态快照深拷贝日志。
4. ObservationBuilder 读取已实现窗口，使用 episode 内 sender/asset 身份，确定性选择消息。
5. fork 后历史原样回放，不再调用模型生成或释义。

`regulatory-observation-v2` 不含 messages；底层
`regulatory-observation-v3` 由 `ObservationConfig(include_messages=True)` 选用。
这两个 schema 是序列化边界，不是当前主评测的两个竞争版本。
主结构化适配器从 v3 的选定消息提取汇总，再去掉正文、kind 和逐条发送者链接。
旧 Diagnostic/Natural 记录不重写或重新贴当前任务标签。

底层每条公开消息只包含 day、episode 内 sender/asset ID、普通消息类型
post/reshare/challenge、public text、已实现 audience_reach/society_size。
正文中的原始模拟器 ID 替换为公开 ID。普通警告或“failure”等单词不构成自动泄露；
检查依据结构化来源和禁止元数据，不能为了防泄露删除合法观点。

## 信息和反事实边界

无论结构化或文本，策略均不能读取隐藏角色/恶意标记、真实目标、机制标签、
基本面、reference 账户、未来状态、损害判据结果或 oracle 标签。
公开 ID 仅用于引用，不编码 seed、条件或角色。ObservationBuilder 只读，
无 API 调用；实时、未缓存模型调用不能插入匹配标签的反事实续跑。

社会传播随机数由 continuation seed、稳定消息身份、接收者和事件类型派生。
过滤或衰减一条消息不改变无关消息的随机抽样；真实干预造成的行为变化仍保留。
标签和损害由模拟器/可信宿主产生，不由 LLM 评论或解释生成。

若以后比较结构化输入、额外文本及等语义 canonical rendering，必须固定同一
状态和划分，先检查文本是否补充了结构化未覆盖的公开变量。输入消融不改变市场
动态；不能把正文打乱成与数值矛盾的内容当成干净消融，也不创建新语言动态
来追求语言贡献。当前首版没有完成这类语言实验。
