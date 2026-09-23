# Consensus 测试与对照记录

本页区分可运行的检查与已完成的历史记录；测试通过不证明参考答案的物理正确性。
使用方式见 [README.md](README.md)，比较政策和排除依据见 [SCORING.md](SCORING.md)。

## 运行检查

以下命令在仓库根目录运行，不调用模型或官方提交 API：

```bash
python -m pytest -q
python -m ruff check ddsr_bench/benchmarks/critpt/evaluation/consensus tests/benchmarks/critpt/consensus
python -m black --target-version py312 --check ddsr_bench/benchmarks/critpt/evaluation/consensus tests/benchmarks/critpt/consensus
```

测试覆盖极小数和大整数、非贪心集合配对、复根、复共轭、明确错误端点、级数阶数、区间开闭及符号k、跨参考拼接拒绝、非有限参考、执行超时、固定分母、数据包完整性、独立模块边界等。

普通测试使用合成参考；设置 `DDSR_CRITPT_REFERENCES=/path/to/reference.json`
后启用两项真实参考检查。未设置时跳过；设置后文件缺失则失败。
Linux/Bubblewrap 测试在不兼容的主机上跳过。单元测试不等于 Docker 实机对照；
完整参考回放使用 [README.md](README.md#linux-诊断后端无需-docker-daemon) 中的
`replay --all-references`，并显式选择所需后端。参考数据始终留在仓库外。

## 当前检查状态（2026-09-23）

最新收尾检查为 438 passed、14 个 Linux-only skipped；Black、Ruff、diff、shell
语法、69 个文档链接和源码空目录检查通过。干净 wheel 含 152 项，独立安装后的
CLI、四个 benchmark 导入及必填配置检查通过，无私有参考或生成产物。
通过安装后的 CLI，从无 summary 的临时 job 提交 DeepSeek 题 1、2 的 attempt
0、1，真实 Harbor/Docker 执行的四条结果与已有对照一致（different、matched、
different、answer_error），参考执行 8 次、缓存命中 8 次，固定分母为 122。
这是四条答案的集成检查，不是重新执行全部 700 条历史对照。
审计镜像复用原对照的依赖层，清空旧 Python 包后复制当前源码；107 个 Python
文件逐字节一致，无旧模块。标准 Dockerfile 的离线构建因依赖层缓存未命中失败，
本次未验证联网安装新依赖。原 687 份答案与参考文件哈希未变，无遗留 worker。
检查产物位于仓库外临时目录 `ddsr-release-check.flbekm`，未发起模型或 AA 请求。

此前多 attempt 内部提交更新后为 434 passed、14 个 Linux-only skipped。新测试通过
共享 submit 入口选择多个 attempt，检查一次 runtime 初始化、参考缓存跨 attempt
命中、答案与比较不复用，以及缺失/错误/unknown 的固定分母和提交前全量检查。
另将已有 700 条 Docker 结果注入更新后的调度路径，验证两模型各五个 attempt 的
逐 attempt 汇总、逐题均值及整体均值一致：Qwen 58/305，DeepSeek 92/305。
这是调度和聚合的回归检查，不是重新执行 700 条 Docker 任务。报告键沿用 DDSR，
最终一次写入，不恢复原 CLI 的逐题落盘与进度文件。

此前收尾修复为 431 passed、14 个 Linux-only skipped，包含通过环境变量指定的
真实参考检查。新增检查覆盖重排/并发官方提交、请求或保存失败后的待处理标记、
不完整 attempt 的收集和内部固定分母，以及必填参考路径和 `--references` 入口。
配置恢复 `references: ???`；新增测试确认其阻止缺少路径的内部提交，但不阻止
官方提交。未配置真实参考文件时为 429 passed、16 skipped。
Black、Ruff、diff、shell 语法与 21 份 Markdown 的 67 个本地链接检查通过，
源码目录无空文件夹。干净 wheel 含 152 项，代码与工作区一致，未包含私有参考、
本地配置、旧模块或生成产物；独立安装后的四个 benchmark 注册项、命令入口和
必填配置检查通过。
没有新官方请求，也未重跑下方 700 条 Docker 对照；共识比较和执行政策未改动。

提交 `2f623ac` 前重新检查：418 passed、14 个 Linux-only skipped；
Black、Ruff、diff、22 份 Markdown 的 61 个本地链接和空目录检查通过。
干净源码构建的 wheel 与当前包代码一致，不含私有参考和生成产物。
以下测试数量属于各记录时点，不应当作当前测试总数。

## 2026-09-22 Docker 对照与缓存检查

原版 `a52ca182742555cc3e0c79d5fbe4f50ddc058ca7` 与整理后的 Harbor worker
版本完成两种模型、各五个 attempt 的对照：700 条评分记录及十份汇总一致。
对照统一 candidate/answer 状态和诊断措辞；不比较已停止重建的历史生成元数据
及输入路径。匹配结果、失败详情、参考覆盖、固定分母和置信度加权分数均保留。
共享 collection 已通过 `error` 字段保留生成失败详情。

- Qwen matched：`[13, 14, 9, 9, 13]`，合计 58/305。
- DeepSeek matched：`[20, 20, 19, 15, 18]`，合计 92/305。
- 每个 attempt 保留九个 skipped；题 4、20 的参考覆盖不足状态不变。

上述对照未启用跨 attempt 磁盘缓存。最新代码启用共享缓存后，也完成全部十个
attempt 的 700 条评分对照，逐题结果和汇总均与未缓存版本一致。后八个 attempt
以两个内部提交并发运行，每个提交仍为四题并发，共享同一私有缓存目录。
单独执行的冷/热缓存及跨模型计时如下：

| 运行 | 用时 | 参考缓存命中 | 参考执行 |
| --- | --- | --- | --- |
| DeepSeek attempt 0，冷缓存 | 140.46 秒 | 20 | 180 |
| 同一 attempt，热缓存 | 57.09 秒 | 200 | 0 |
| Qwen attempt 0，复用缓存 | 54.73 秒 | 192 | 4 |

冷缓存内的命中来自相同参考的去重。热运行约快 59%；答案和比较始终重新执行，
并非复用不可信容器。计时使用四题并发，每 worker 1 CPU、1 GiB、60 秒超时；
只是本机单次观察，不代表其他机器或资源配置的性能。

两版镜像基于相同依赖层；参考 SHA-256 为
`6b9edc5057a92148701bed69afa3b4fb121da89223c6978dc01ddca7005a44bc`。
当前审计镜像 ID 为
`sha256:c77be81b39a4acb7d93a91df2a6b9113b4077bdedf27d0f364f62c7aaf17bd74`；
worker、序列化、匹配、采样和 AST 校验代码与工作区逐字节一致。
687 个已有答案文件及参考文件哈希未变；没有新模型调用或官方提交。

实机隔离检查覆盖四个并发 worker、网络禁用、只读根文件系统、非 root 用户、
权限与资源限制、无宿主目录挂载、参考路径及宿主环境变量不可见、worker 间文件
隔离，以及正常结束、超时和取消后的清理。参考缓存文件权限为 0600。
全部审计结束后未遗留 consensus 容器或网络；审计镜像中未发现参考答案文件。
worker 使用 `network_mode: none`，不启动重复的动态防火墙 sidecar；Docker
`init` 转发退出信号并回收子进程，超时限制未缩短。

提供本地真实参考文件后，全仓库为 442 passed、14 skipped；剩余跳过项只需要
Linux/Bubblewrap。默认外部路径不可用时为 440 passed、16 skipped。
Black、Ruff、diff、22 份 Markdown 的 61 个本地链接及空目录检查通过。
干净 wheel 构建与独立安装导入通过，不包含旧模块、私有参考或生成产物。
原始审计记录保存在仓库外的 `ddsr-consensus-audit.ibSGK6` 临时目录：
`optimized/comparison.json` 记录原版对照，`cache-check/comparison.json` 记录
冷/热计时，`cache-check/full-comparison.json` 记录全部缓存对照。

### 2026-09-22 未使用适配器清理

已移除未接入内部提交的 Harbor consensus verifier、其逐题参考读取 helper
和专属测试。内部提交仍直接调用同一 grader；上述 Docker 评分路径没有改变。
普通 CritPt reference verifier 与 `check_resume` 均保留。
删除 24 个专属测试后，全仓库为 418 passed、14 个 Linux-only skipped；
Black、Ruff 和 diff 检查通过。本次未重新运行上述 700 条 Docker 对照。

## Harbor 环境接入与模块整理

这是完整 Docker 对照之前的记录：当时完成模块移动，但尚未进行新的
Docker 或 Linux 实机验证。后续完整对照见上方 2026-09-22 记录。

Docker worker 已通过 Harbor environment 管理，复用标准 CritPt 镜像。
本地 Docker 合成测试已验证执行、比较、超时及清理；未据此声称完成整个
参考包的 Docker 回放。当时历史 CLI 移入 `legacy/`；该目录及 score/batch 命令后来已删除，
目前只保留 build/replay 参考诊断。macOS 上的 Linux 隔离检查仍明确 skipped。

## 2026-09-18 Linux 回放（consensus-61-v2）

70 个题号中 61 题计分、9 题 skipped，启用参考 204 份，权重合计 43.6。
47、51 的排除依据见 [评分说明](SCORING.md)。

2026-09-18 验证环境：Ubuntu 24.04、Python 3.12.3、Bubblewrap 0.9.0、
SymPy 1.14.0、NumPy 1.26.4、SciPy 1.17.1、mpmath 1.3.0。

| 检查 | 结果 |
| --- | --- |
| 全仓库测试 | 323 passed |
| 迁移前 main 的原始测试原样运行 | 212 passed；覆盖旧版客户端接口和四个 benchmark |
| 无外部参考资产时的全仓库测试 | 321 passed、2 skipped；仅跳过真实参考包检查 |
| Black、Ruff、git diff --check | 通过 |
| 外部参考包的完整 Linux 回放 | 204 matched、9 skipped，healthy=true |
| 干净源码构建 wheel，独立安装目录与空工作目录检查 | 不含参考答案、凭据或本地配置；可读取外部默认资产 |
| Linux 隔离测试 | 宿主文件、参考包、凭据不可见；进程、网络、挂载及资源限制生效；超时与中断后清理 |

参考包位于 `/mnt/workspace/zhizhou/assets/critpt/references/consensus-61-v2.json`，
SHA-256 为 `6b9edc5057a92148701bed69afa3b4fb121da89223c6978dc01ddca7005a44bc`。
从仓库移出时逐字节校验，评分规则和内容不变。以上为当时的资产位置；
当前回放必须通过 `--references` 指定路径，评分报告保留实际哈希。
真实数据测试通过 `DDSR_CRITPT_REFERENCES` 指定路径并检查上述固定哈希。

回放及打包检查记录位于 `/mnt/workspace/zhizhou/assets/critpt/audits/` 下的
`external-reference-20260918-replay.json`、`external-reference-20260918-wheel.json`。

生成与评分仍共享原始校验器，保留辅助函数、lambda、try 等禁令。测试覆盖空正文仍执行
二阶段、错误产物按题记录、attempt 不混用、skipped 不执行且不进入分母，以及逐题及时保存。

## 历史集成检查

完整迁移经过 700 条历史 trial 对照，身份、状态、匹配结果、错误、元数据和答案哈希一致。
原 `consensus-63-v1` 为 212 份参考、7 个 skip；按 v2 排除 47、51 后，每模型
350 条记录中 305 条计分、45 条 skipped。Qwen3.8 为 58/305，DeepSeek 为 92/305。
来源版本与贡献者记录见 [UPSTREAM.md](../../../../../UPSTREAM.md#critpt-consensus-and-long-context-generation)。

### 2026-09-17 初始集成（consensus-63-v1）

- 全仓库 316 tests passed；Black、Ruff 和 diff 检查通过。
- Linux 完整参考回放为 212 matched、7 skipped；700 条历史 trial 的身份、状态、
  匹配、错误详情、元数据及答案哈希均与原报告一致。Qwen 为 60/315，DeepSeek
  为 92/315；分别保留 8、5 个生成格式错误，以及 DeepSeek 的 3 个 unknown。
- 共享 collection/trajectory 读取全部 700 条记录；两份映射后的 job 均可恢复，
  未新增模型请求。全部阶段 seed 与原 base-42 派生一致。
- 真实 Qwen vLLM 请求验证 xhigh、seed 42 和原生 262144 上下文预算；PAI 请求
  验证两种 Aliyun payload profile、max reasoning 和 seed 42。一个 CritPt
  静态两阶段测试刻意限制每阶段 2048 tokens，两次均因长度结束，记录格式失败。
  另一次真实 API、模拟环境上传的两阶段 Harbor adapter 测试生成了有效代码；
  这不是完整 Harbor 容器 trial。该 DSW 环境没有 Docker daemon。
- 当时 wheel 尚包含参考数据；2026-09-18 已将其移出，当前包不分发参考答案。
  四个 solver 构建上下文通过 Docker SDK 排除规则检查。

这些历史 job 已包含迁移前选定的 41 次格式重试，不能描述成未经重试的模型评测。
原始 trial、替换前备份、launch 快照和 grading 报告保留在仓库外的
`/mnt/workspace/zhizhou/assets/critpt/runs/`；源码 Git bundle、工作区存档和哈希
清单在同级 `migration-20260917/`。私有存档不分发；文档整理不会修改这些资产。

### 2026-09-22 提交接口检查（中间版本）

接入共享 collection 之前，internal submit 与原 CLI 的全部 700 条完整报告一致，
包括当时仍重建的生成元数据；该阶段为 322 passed、16 skipped。
随后共享 `action=collect` → internal submit 路径检查两种模型的 attempt 0，
140 条判定及两份汇总一致（Qwen 13/61、DeepSeek 20/61），为 329 passed、16 skipped。
输入来源及错误字段当时有差异；最新完整 Docker 对照及明确保留的字段见本页上方，
不能将中间版本的「完整报告一致」解读成当前所有历史字段仍然存在。

曾用六份非最大共识组答案检查比较器：11、17、28、41、56 判为 different，45 判为 matched。
45 的不同表达式在正质量参数范围内等价，原报告分组不能直接作为负例标签。
历史 confidence 保持不变，不据此重新计票。

这些结果验证执行和匹配链路，不证明参考答案的物理正确性或有限采样下的恒等关系。
Linux 后端共享宿主内核，不提供虚拟机隔离或 cgroup 总内存配额。
上述历史 DSW 验证环境无 Docker daemon；该次记录不包含 Docker 端到端回放
或完整 Harbor 容器 trial。后续 Docker worker 对照见 2026-09-22 记录；
内部提交不创建 Harbor agent trial。
