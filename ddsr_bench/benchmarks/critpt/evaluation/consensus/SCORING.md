# Consensus 评分说明

本页记录 `consensus-61-v2` 的内部匹配政策，不代表官方 CritPt grader。
运行方式见 [README.md](README.md)，实现对照与测试证据见 [TESTING.md](TESTING.md)。

## 范围和结果

保留70个main题号，评分61题，skip为：`6, 12, 19, 25, 30, 33, 47, 51, 68`。

历史confidence分布：80%有44题，60%有8题，40%有9题，权重合计43.6。本版本保留历史 confidence，不重新计票。

2026-09-17 审计后，经用户确认新增 skip 47、51，原因见下方排除审计。两题在每个 attempt 中仍保留 `status=skipped`、`matched=null` 和 `skip_reason`，不计入普通或加权分母。内部提交只写独立 JSON 报告，不改共享 collection 的 CSV。旧 `consensus-63-v1` 结果属于历史评分，不应与本版本混用。

| 主要模式 | 题数 | 题号 |
| --- | ---: | --- |
| 固定值、有序结构 | 31 | 1,8,9,10,14,16,17,21,26,27,28,31,32,35,37,41,42,43,44,46,48,50,52,55,56,57,61,63,64,69,70 |
| 符号函数 | 12 | 3,11,15,18,29,34,36,38,39,53,60,67 |
| 无序集合、根集 | 6 | 4,13,40,54,58,65 |
| 分域、端点、区间 | 6 | 5,7,22,23,24,62 |
| 指定阶数的展开 | 4 | 2,49,59,66 |
| 数值函数测试 | 2 | 20,45 |

模式可以组合。例如59使用级数和集合；62使用符号比较、区间和混合输入。35是固定顺序的225个数，65只匹配规范形式算符，不重新验证物理解完备性。

结果状态：

- `matched`：匹配至少一个获准完整参考。
- `different`：与全部可评参考不匹配。
- `unknown`：比较超时、无法判定、支持范围外的表示或部分参考异常妨碍结论。
- `answer_error` / `reference_error`：候选或参考执行/格式失败。
- `missing_answer`：活跃题缺少候选。
- `skipped`：不执行、不计通过、不计失败。

每条结果包含 `evaluation_kind=internal_consensus`、`policy_version`、历史 `confidence`、`matched`、匹配的参考/组、`reference_coverage` 和 `methods`。没有旧流程的 `reward` 或 `verified` 字段。`methods`区分符号证明、数值容差、固定点核验、端点处理等；固定样本吻合不是恒等证明。发现数值反例时记录对应测试点与值。

主命中率分母固定为61；未决、缺失、执行失败不会缩小分母。另报状态数量和按confidence分层的结果。加权命中率为 `sum(confidence * matched) / 43.6`。该规则使全匹配时结果为1。

### 多次尝试与报告

多个选定 attempt 顺序经过同一个 grader 和参考缓存，每个 attempt 内的问题并发执行。
报告的 `results` 标记 attempt，`summary.attempts` 保存逐次汇总，
`summary.problems` 保存逐题均值。总体 `summary.match_rate` 的分母为
`61 × 选定 attempt 数量`；`weighted_match_rate` 为逐次加权命中率的均值。
缺失、错误和 unknown 均计零，9 道 skipped 题不进入分母。
不完整的已收集 attempt 仍可选择。报告在所有选定 attempt 完成后一次保存，
不是逐次 checkpoint 或 resume 机制。

## 比较政策

公共数值/结构比较：

- 列表和元组按位置比较；集合做一一配对，不能复用同一个目标元素。
- bool、计数、大整数和标识按各自类型严格比较；大整数不转float。
- 题面要求小数位时，使用半个末位单位的绝对误差作为本版本的内部政策；有效数字同理。这不是对官方隐藏grader阈值的声明。
- 31、32采用题面给定的逐字段绝对误差。没有题面精度的固定数值通常沿用报告的1%相对误差、零绝对误差。
- 35系数一般用1%相对误差与1e-9绝对误差，归一化系数用1e-10绝对误差。这是共识向量匹配精度，不声称保证对易子残差。
- 20是同一解析函数的数值核验，用1e-6相对误差。符号函数补查使用更紧的数值精度，不把全局1%传给符号表达式。
- 同一获准共识组中保留全部现存完整参考，允许命中其中之一。对函数必须在全部测试点匹配同一个参考；不按字段或测试点拼接参考。

特殊规则：

- 5、7、22以及24的y=1/2允许唯一有限的连续延拓；显式给出错误端点仍不匹配。
- 23、24每个返回分量使用独立的区间；保留复数，不能过滤掉“有复数中间量”的合法点。23还保留至epsilon^0的Laurent极点。
- 29使用可信参考的正深阱定义；散射长度允许正负。36在可信阈值n>=7上检查二阶矩，阈值字段单独比较。
- 39保持复alpha；40匹配完整根集，可用二次多项式系数/根和与积。
- 49保留指定的z、log(z)相关项并忽略z独立项；不是简单比较几个大z值。超出支持的渐近形式返回unknown。
- 59保留条件与结构因子的配对，只比epsilon一阶；66比较q^0到q^15系数。
- 62用整数k_value选择分支，保留符号k；比较区间端点和开闭。65保持tr为形式函数，不做普通随机数替换。

定义域、固定测试点与来源注释位于`matching/cases.py`和`matching/rules.py`。有限测试覆盖不是完整的物理定义域刻画；尤其15未澄清的退化链长、20的球形退化和其他材料参数不在第一版覆盖范围。不能根据候选输出动态删测试点来制造一致。

## 并列组与已知资料缺口

| 题号 | 第一组 | 第二组 | 可用参考 |
| --- | --- | --- | --- |
| 4、20 | Opus＋Kimi | Fable＋GPT | 只有Fable/GPT组，标记`incomplete` |
| 38 | Opus＋Fable | GPT＋Kimi | 分别有Fable、GPT代表，两组都可匹配 |

两组均为两票，confidence保留40%。4、20可能漏收缺失那组的答案；20的报告数值不能替代完整参数函数。53的Fable文件复制自GPT，保留来源标记，confidence仍沿用报告。60%题目也可能只有该组的一部分原始文件；有参考代表并不等于独立复现全部计票。

25因题面/模板中delta与bar_Delta的参数定义冲突暂skip；19的极值顺序分歧、6的符号及精度问题均按当前范围继续skip。

## Exclusion audit: consensus-61-v2

On 2026-09-17, the user approved excluding Challenges 47 and 51 after comparing
the original statements with every available flagship reference. This is an
internal scoring policy change, not a claim about official CritPt grading.
There are now 61 active main problems and 9 skipped problems. The 70 problem
slots and every attempt remain present; skipped results carry `matched: null`
and a reason, and contribute to neither numerator nor denominator.

### Challenge 47: reference values correspond to swapped angles

The pinned statement specifies `theta=x`, `phi=(2*pi/3)*exp(-x*x)` and a
constant spin at infinity. Its `m_z=cos(x)` violates that boundary condition;
the full-space trace diverges even if the boundary assumption is relaxed.

Opus, Fable and Kimi return 11.8916480767; GPT returns 11.891648076712.
Opus's code comment instead describes a Gaussian polar angle and linear
azimuthal angle. Independently swapping the two angles reproduces
11.891648076711 at 128, 256 and 512 mapped Gauss–Legendre nodes. Thus the
finite consensus reference is unsuitable for the literal statement.

The divergent background has trace density 16/pi, which is not the full trace.
Exclusion does not accept every infinity/NaN answer or recover a submission
from first-stage reasoning.

### Challenge 51: consensus contradicts return-path parity

The stated walks, splitting and recombination require even time to return to
the origin. All four references nevertheless expand as
`1 + (g + 2*lambda**2)*x**2 + 2*g*x**3 + O(x**4)`.
The nonzero `Z(3)=2g` contradicts the original rules for every allowed `g>=2`.

For `g=2, lambda=1`, independently enumerated `Z(0..8)` is
`[1,0,4,0,36,0,400,0,4900]`; all four references instead give
`[1,0,4,4,32,68,336,984,4096]`. The elementary-function requirement also needs
review: this case gives `2*K(4*x)/pi`, a complete elliptic integral (modulus
convention). DeepSeek's integral submissions agree with the path recurrence
and independent enumeration, while two Qwen submissions match the defective
reference. Exclusion removes both potential false negatives and false positives.

### Provenance and scope

- Original problem commit: `17c2545c302762d2f2d644d923ea4c301605cb08`.
- Previous policy: `consensus-63-v1`; original bundle SHA-256:
  `fa0facc0082261b6f42d976e07b8bc1c9302eff517e7802c9bf49f87fd4a3f2f`.
- Audit source commit: `f3af5a674124915ca19eee1ac399ca1538ac71ef`.
- Local evidence, code and per-attempt findings:
  `/mnt/workspace/zhizhou/assets/critpt/audits/false-negative-20260917/README.md`.
- The remaining 61 problems retain their reference code, comparison policies,
  confidence and original generation validation. Challenge 45 remains active
  pending further review. No general format restrictions were relaxed.
- Existing v1 reports remain historical. Revised reports must identify v2 and
  its bundle hash; a five-attempt mean uses 305 active trials per model.
