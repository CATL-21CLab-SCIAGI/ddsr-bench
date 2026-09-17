# consensus-63-v1 验证记录

日期：2026-09-16。验证环境：Python 3.12.14、SymPy 1.14.0、NumPy 2.5.3、SciPy 1.18.1。

| 检查 | 结果 |
| --- | --- |
| 完整仓库测试 `python -m pytest -q` | 153 项通过，包含共识模块的 49 项测试 |
| 新模块及测试的 Ruff、Black 检查 | 通过 |
| 参考包构建 | 70 个题号，63 题启用，7 题 skip，212 份参考记录 |
| 全部参考回放 | 212 条 matched，7 条 skipped |
| 以每题首份参考作为候选的 CLI 评分 | 63 条 matched，7 条 skipped；普通及加权命中率均为 1 |
| 置信度统计 | 46 题 0.8、8 题 0.6、9 题 0.4；权重合计 45.2 |
| 资料缺口标记 | #4、#20 的 `reference_coverage` 为 `incomplete` |

使用的参考包 SHA-256（已原样纳入`data/consensus-63-v1.json`）：

`fa0facc0082261b6f42d976e07b8bc1c9302eff517e7802c9bf49f87fd4a3f2f`

回放文件为 `.consensus/replay-final.json`，CLI 评分文件为 `.consensus/score-final.json`。这些运行结果留在 Git 忽略的本地目录；参考包随代码提交并默认加载，不再依赖原始答案目录。

回放使用已审查代码和显式 `--trusted-local`。它验证执行、输出传递和比较链路；由于参考自身也在可匹配列表中，不能据此证明物理正确性或不同参考之间的一致性。上述命中率是回归检查结果，不是新模型的评测成绩。

## 非共识答案对照

另将 63 道启用题中现存的全部 6 份非最大共识组答案作为候选，与同一参考包比较。没有修改 grader、参考包或 confidence；结果为 5 条 `different`、1 条 `matched`，没有执行错误或未决结果。

| 题号 | 候选模型 | 结果 | 检查结论 |
| --- | --- | --- | --- |
| 11 | GPT | different | 第二个 beta 函数的系数不同，固定域测试点给出数值反例 |
| 17 | Fable | different | 标量符号相反 |
| 28 | Fable | different | 三个指数中的后两项不同 |
| 41 | Fable | different | 数值差异超过题面有效数字对应的容差 |
| 45 | GPT | matched | 9 个正质量输入全部吻合；数学条件在正质量范围内等价 |
| 56 | Fable | different | 两个数值中的第一项通过精度检查，第二项不同 |

45 的通过符合此前题面审查结论：设两个正质量比为 `a,b`，GPT 条件中的乘积为 `(a-b)*(1/a-1/b)=-(a-b)^2/(a*b)`，严格小于零等价于 `a != b`，与参考的交叉乘积不等条件相同。因此原报告的分组不能直接当作负例标签。该题历史 confidence 仍保持 0.4，没有重新计票。

原始结果、源文件哈希、每个输入的候选/参考输出以及比较依据保存在本地 `.consensus/nonconsensus-evaluation.json`，不随代码提交。这次实测说明这 5 份答案与当前参考存在可识别差异，不提供官方正确性结论。

## 仓库与安装包分发

参考包为430,880字节（约421 KiB），使用普通Git文件管理，内容与上述已验证包逐字节相同。`score`和`replay`默认定位包内数据，不依赖工作目录；显式`--bundle`仍可覆盖默认值。

已从仅含Git跟踪文件的临时源码目录构建wheel，验证wheel内的参考包大小和SHA-256，再安装到独立临时目录。从空工作目录运行已安装模块，不传`--bundle`，完整回放得到212条matched和7条skipped。该结果保存在本地`.consensus/packaged-reference-replay.json`。

`.dockerignore`在代码白名单之后排除`ddsr_bench/benchmarks/critpt/evaluation/consensus/data/`，参考包仅供宿主评测端使用。当前主机没有 Docker。容器命令参数有测试覆盖，但尚未构建镜像并做容器端到端回放；正式运行未审查候选前仍需完成这一检查。现有 `critpt_eval/grading`、官方提交和训练导出流程未改动。

## 原生 rollout 输入与逐题容错

将每题首份参考组织为`Challenge_N_main__attempt-0/artifacts/answer.py`，通过真实CLI选择attempt 0评分，结果为63条matched、7条skipped；整个summary及bundle哈希与原平铺目录评分一致。结果保存在本地`.consensus/native-reference-score.json`。

另用模拟模型响应调用原有`run_trial`生成真实目录，再经consensus CLI评分。已验证：第一步空正文且`finish_reason=length`时仍调用第二步；第二步产出合法参考则正常匹配，第一步的截断信息保留在结果中；最终空正文造成的上游错误按题记录，不中断其他题，评分分母仍为63。此测试固定现有two-step行为，没有改动推理流程或增加重试。

新增回归覆盖11种坏输入（损坏JSON、错误结构/ID、缺失或null代码、空代码、未闭合围栏、非法编码等）、逐题失败后继续评分、attempt明确选择及不跨attempt补题、空目录/错误路径拒绝、混合job拒绝、缺失产物与生成错误区分、可选元数据损坏的诊断，以及skip不执行。比较器、参考包、confidence和生成代码保持不变。

## Linux 隔离后端

2026-09-17 在当前 DSW 上验证：Ubuntu 24.04、Python 3.12.3、Bubblewrap 0.9.0、libseccomp 2.5.5、NumPy 1.26.4、SciPy 1.17.1、SymPy 1.14.0、mpmath 1.3.0。主机没有可用 Docker daemon；本节使用显式 `--backend linux`，不使用 `--trusted-local`。

| 检查 | 结果 |
| --- | --- |
| 完整仓库测试 | 181 项通过，其中新增 Linux 后端测试 16 项 |
| consensus 代码及测试的 Ruff、Black 检查 | 通过 |
| 全部参考在 Linux 隔离后端回放，4 路并发、60 秒超时 | 212 条 matched、7 条 skipped，healthy=true |
| 绕过 AST 校验器直接测试 OS 限制 | 宿主测试文件、参考包、凭据环境变量不可见；根目录不可写；fork、exec、socket、unshare、亲和性修改及发送信号被拒绝 |
| 资源及生命周期 | 2 GiB 分配失败、硬限制不能上调；单文件 8 MB、临时目录总量 64 MiB 限制生效；墙钟超时和中断清理通过，临时文件不跨 worker 保留 |
| 隔离不可用 | 缺少 Bubblewrap 或预检失败时报错，不自动降级 |

完整回放产物：`.consensus/linux-reference-replay-20260917.json`，已被 Git 忽略。报告包含沙箱政策 `consensus-linux-v1` 及实际依赖版本；参考包 SHA-256 与上文一致。这是参考回归检查，没有执行 Qwen/DeepSeek 候选的正式评分，也没有修改格式失败处理。

Linux 后端共享宿主内核，使用 namespace、seccomp 和硬资源限制；不提供虚拟机隔离或 cgroup 总内存配额。测试证明列出的限制在上述环境生效，不代表完整的安全审计。Docker 路径的端到端验证仍是独立待办。

## 与生成阶段统一校验规则

2026-09-17 将原 `grading/validation.py` 的实现提取到 `critpt_eval/code_validation.py`，生成入口和 consensus 共用该实现。经逐字节比较，新文件与初始提交 `d83b0d9` 的原校验器完全相同；源码 SHA-256 为 `1b388bfd2107206393c9c121b8dcfb25065921880f80fa7b9c52a129e4bb5a50`，评分报告新增 `code_validation_sha256` 记录。

全仓库测试 194 项通过，Ruff、Black 检查通过。新增检查覆盖共享实现身份、嵌套辅助函数、lambda、try/except、raise、with、delete、yield、危险属性和名称被 worker 拒绝，以及全部 212 份参考通过原校验器。代码提取也统一为生成阶段约定，因此缺少结束围栏但代码完整的输入不再单凭围栏不完整而拒绝。

在 Linux 沙箱中重新完整回放，结果仍为 212 条 matched、7 条 skipped；产物为 `.consensus/linux-shared-validation-replay-20260917.json`。比较器、参考包、confidence、题目范围及生成阶段规则均未改变。
