# CritPt 内部共识评分

`ddsr_bench.benchmarks.critpt.evaluation.consensus` 对候选答案做内部共识参考匹配，作为 `action=submit benchmark.submission.backend=internal` 的评分实现。参考包和执行设置位于 `benchmark.submission.internal`，命令示例见 [CritPt 指南](../../README.md#internal-submission)。评分不调用官方 API，不修改 trial reward 或训练导出结果。当前版本为 `consensus-61-v2`。

运行时不使用 LLM judge。参考匹配不等于官方正确性标签；历史 confidence 也不是校准后的正确概率。

- [SCORING.md](SCORING.md)：评分范围、结果含义、比较规则及参考排除依据。
- [TESTING.md](TESTING.md)：检查命令、已完成的对照测试及其局限。
- 贡献者来源与基线版本见 [UPSTREAM.md](../../../../../UPSTREAM.md#critpt-consensus-and-long-context-generation)。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `__init__.py` | 固定评分政策版本 |
| `__main__.py` | `python -m ...consensus` 命令入口 |
| `references.py` | 构建和验证外部参考包、校验来源哈希及模板参数列表 |
| `../../submission/internal.py` | 读取共享 collector 指定的 Python 答案并调度评分 |
| `grader.py` | 匹配完整参考、缓存成功的参考执行结果、汇总固定分母成绩 |
| `matching/compare.py` | 数值、符号、集合、区间和级数比较 |
| `matching/cases.py` | 执行参数和确定性比较采样；不是数据增强 |
| `matching/rules.py` | 符号假设、边界规则、排除项和精度 |
| `execution/runtime.py` | 正常提交仅用 Docker，固定镜像 ID 并记录运行来源 |
| `execution/docker.py` | 默认 Docker worker 的 Harbor 生命周期与资源限制 |
| `execution/linux.py` | Linux 沙箱与子进程监督，供参考回放使用 |
| `execution/worker.py` | 隔离进程内执行代码或比较结果 |
| `execution/serialization.py` | 有界、带类型的 JSON 编解码；不能替换为普通 JSON 或 pickle |
| `cli.py` | 参考构建与回放命令，见下方 Reference maintenance |

## 结果概览

当前政策 `consensus-61-v2` 保留 70 个 main 题号，61 题计分、9 题 skipped。
主命中率使用固定分母 61；缺失、未决和执行失败不会缩小分母。
结果不写回 trial reward，也不代表官方正确率。状态、加权分数及逐题规则见
[评分说明](SCORING.md)。

## 安装与外部参考包

以下命令在仓库根目录执行。安装方式沿用仓库的Python 3.12环境：

```bash
python -m pip install -e '.[dev]'
python -m ddsr_bench.benchmarks.critpt.evaluation.consensus --help
# 同一入口也注册为 ddsr-critpt-consensus
```

参考文件存放在仓库外，不随 Git 或 Python 安装包分发。正常提交必须配置
`benchmark.submission.internal.references`；诊断命令必须提供
`--references /path/to/reference.json`。Python 代码不预设机器路径。
文件缺失时直接报错，不生成成绩。加载时检查政策版本、模板和参考代码哈希，
评分报告记录整个文件的实际哈希；真实数据测试另核对下面的固定 SHA-256。

包大小为420,292字节（约410 KiB），包含70个题号、61道启用题的204份参考代码、模板、confidence、共识组和来源哈希。SHA-256 为 `6b9edc5057a92148701bed69afa3b4fb121da89223c6978dc01ddca7005a44bc`。204份记录包含重复来源，不能视为204份独立证据。正常评测只需要此包，无需原始模型JSON、外部清单或题面目录。

参考包仅供评测端使用，不放入 solver 工作目录、镜像构建上下文或安装包。候选执行请求只携带当前候选、模板和输入。

普通功能测试使用合成参考数据。设置 `DDSR_CRITPT_REFERENCES=/path/to/reference.json`
可启用真实数据检查；未设置时明确标记 skipped，已设置但文件不存在则失败。
完整参考验证仍使用下面的 `replay --all-references` 命令。

## 评分

使用 [CritPt 指南](../../README.md#internal-submission) 中的 `action=submit`。
提交从共享 collection 中选择 attempt，评分器直接读取保存的 Python 答案，
通过 Harbor environment 管理隔离 Docker worker。答案执行不挂载参考文件或缓存，
也不请求模型。参考缓存设置见下节。
修改 worker 后需重建标准镜像：

```bash
docker build -f docker/critpt/Dockerfile -t ddsr-bench-critpt:latest .
```

### Reference caching

`benchmark.submission.internal.cache_dir` 默认为
`${paths.output}/cache/critpt-consensus`，通常是 `outputs/cache/critpt-consensus`。
缓存只保存成功的参考执行输出；答案和比较始终重新执行，不复用容器。
设置为 `null` 只关闭磁盘缓存，同一次提交仍在内存中复用参考结果。
缓存应保持私有，并放在生成 job 目录之外。

新目录从 cold cache 开始；保留目录可在后续 attempt 或模型 job 中获得 warm cache。
这是缓存内容的状态，不是独立开关。复用要求参考代码、模板、输入、政策、
Docker image ID、timeout、CPU 和内存限制一致。设置变化、缓存缺失或损坏时
重新执行；失败的参考不缓存。冷启动也可能因相同参考获得内存缓存命中。
报告通过 `reference_cache.hits` 和 `reference_cache.executions` 记录参考执行情况。
更换缓存目录不会绕过已有提交报告的防覆盖检查。

## Reference maintenance

以下工具供参考文件维护者使用，正常提交不需要运行：

- `build`：从已审查的原始材料构建私有参考文件。
- `replay`：把参考代码作为待评分答案送入同一评分器，检查执行、序列化和匹配流程。

回放不评估模型输出，也不证明参考答案的物理正确性或所有参考之间的一致性。
验证记录见 [TESTING.md](TESTING.md)。

### 从原始材料重建（可选）

仅更新或审计参考包时需要已审查的 `consensus_manifest.json`、原始模型答案JSON，以及对应官方题面JSON。清单中的 `official_problem_path` 和各样本 `path` 相对于 `--source-root`：

```bash
python -m ddsr_bench.benchmarks.critpt.evaluation.consensus build \
  --manifest /path/to/consensus_manifest.json \
  --source-root /path/to/source-assets \
  --output .consensus/reference.json
```

构建器只收录报告中最大共识组的现存代码，校验原始文件SHA-256和模板签名，并保存全部70个槽位。61题之外的参考不执行。

`.consensus/`被Git忽略，用于本地评分结果、诊断文件或临时重建的包。输出文件使用排他创建，不覆盖已有结果。

清单的关键字段是：`problem_id`、`official_problem_path`、`reported_max_agreement`、`confidence`、`reported_groups`；每个最大组标记`is_maximal_group`，包含`models`和`samples`，样本保留`reported_model`、`path`、`sha256`及来源说明。构建只接收完整70题清单。

### 参考回放

默认使用 Harbor 管理的 Docker worker：

```bash
ddsr-critpt-consensus replay --references /path/to/reference.json \
  --all-references --output .consensus/reference-replay.json
```

默认 `replay` 每题选择第一份可用参考；`--all-references` 覆盖全部 204 份参考，
另保留 9 个 skip。`--problems 3,24,62` 可运行明确标为 partial 的局部检查。
Linux 和可信宿主执行仅用于这些维护工具，不用于正常内部提交。

### Linux 诊断后端（无需 Docker daemon）

Linux 后端仅用于参考回放诊断；下面的 replay 命令可显式选择 `--backend linux`，复用同一个 worker、校验器和比较器。默认后端为 Harbor 管理的 Docker；诊断后端不会自动启用。

需要宿主支持 user、mount、PID、IPC 和 network namespaces，并安装 Bubblewrap、libseccomp 和 `ldd`。Debian/Ubuntu 的系统依赖可用 `apt-get install bubblewrap libseccomp2 libc-bin` 安装；Python 依赖沿用上面的项目安装。当前适配器支持使用共享库的常规 Linux CPython 安装，已在 Ubuntu 24.04、Python 3.12 上验证；其他发行版或自定义 Python 安装必须通过启动预检。

```bash
python -m ddsr_bench.benchmarks.critpt.evaluation.consensus replay \
  --references /path/to/reference.json \
  --backend linux --all-references --jobs 4 --timeout 60 \
  --output .consensus/linux-reference-replay.json
```

每次执行都创建新的沙箱，候选、参考和比较请求分别运行：

- 只读挂载 Python 标准库、系统共享库、NumPy/SciPy/SymPy/mpmath 及必要 worker 模块，不挂载整个仓库、参考数据目录或整个 site-packages。宿主环境变量不传入，`.env` 和 API 凭据不可见。
- 隔离网络、用户和进程空间，worker 使用 UID/GID 65534、无 capabilities 和 `no_new_privs`；不挂载 `/proc`、`/sys` 或宿主工作目录。
- 在读取请求前设置不可上调的资源上限并加载 seccomp，禁止创建进程/线程、执行其他程序、创建 socket、切换 namespace 和挂载。数值库按单线程运行。
- 每个 worker 的虚拟地址空间上限为 1 GiB，绑定一个可用 CPU；CPU 时间上限为 `int(timeout)+2` 秒（硬上限再加 1 秒），父进程另按 `--timeout` 限制墙钟时间。文件上限 8 MB、文件描述符上限 64，独立 `/tmp` 总量上限 64 MiB，根文件系统只读。
- 超时或父进程中断会终止 worker；退出后临时文件随沙箱释放。启动前在真实沙箱内验证依赖、非特权状态和 seccomp；隔离不可用时直接报错，不退回宿主执行。

报告的 `provenance.linux_sandbox` 记录隔离政策版本、Bubblewrap、Python 和依赖版本及资源上限。此后端使用宿主已安装的数值库；复现时应保留相同依赖版本，它不提供 Docker image ID 的环境固定能力。

这是共享宿主内核的进程沙箱，不能当作虚拟机边界；1 GiB 是 `RLIMIT_AS` 虚拟地址空间限制，并非 cgroup 的总内存配额。Bubblewrap 的安全性取决于调用方设置的隔离参数，seccomp 也需要配合 namespace 和文件系统限制，见 [Bubblewrap 安全说明](https://github.com/containers/bubblewrap#sandbox-security)和 [Linux seccomp 文档](https://docs.kernel.org/userspace-api/seccomp_filter.html)。

### 可信代码的本地回放

若只回放**已经检查并信任的代码**，可以使用`--trusted-local`。它在本机子进程执行，**不是安全沙箱**，不应对未审查的模型代码使用。默认不会因Docker不可用而自动退回本机。

```bash
python -m ddsr_bench.benchmarks.critpt.evaluation.consensus replay \
  --references /path/to/reference.json \
  --output .consensus/reference-replay.json \
  --all-references --trusted-local --jobs 4
```
