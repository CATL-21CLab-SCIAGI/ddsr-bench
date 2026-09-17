# 独立的 CritPt 内部共识评测

`ddsr_bench.benchmarks.critpt.evaluation.consensus` 对候选答案做内部共识参考匹配。它使用独立入口、比较器、数据包和结果格式，只共享 `ddsr_bench.grading.validation` 的结构校验规则，不调用现有 reward grader，也不接入官方提交、Harbor任务编译、旧汇总或训练导出流程。当前版本为 `consensus-61-v2`。

运行时不使用LLM judge。参考匹配不等于官方正确性标签；历史confidence也不是校准后的正确概率。

## 范围和结果

保留70个main题号，评分61题，skip为：`6, 12, 19, 25, 30, 33, 47, 51, 68`。

历史confidence分布：80%有44题，60%有8题，40%有9题，权重合计43.6。本版本保留历史 confidence，不重新计票。

2026-09-17 审计后，经用户确认新增 skip 47、51，原因见 [排除记录](EXCLUSIONS.md)。两题在每个 attempt 中仍保留 `status=skipped`、`matched=null` 和 `skip_reason`，不计入普通或加权分母；CSV 的 score 留空。旧 `consensus-63-v1` 结果属于历史评分，不应与本版本混用。

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
- `candidate_error` / `reference_error`：候选或参考执行/格式失败。
- `missing_candidate`：活跃题缺少候选。
- `skipped`：不执行、不计通过、不计失败。

每条结果包含 `evaluation_kind=internal_consensus`、`policy_version`、历史 `confidence`、`matched`、匹配的参考/组、`reference_coverage` 和 `methods`。没有旧流程的 `reward` 或 `verified` 字段。`methods`区分符号证明、数值容差、固定点核验、端点处理等；固定样本吻合不是恒等证明。发现数值反例时记录对应测试点与值。

主命中率分母固定为61；未决、缺失、执行失败不会缩小分母。另报状态数量和按confidence分层的结果。加权命中率为 `sum(confidence * matched) / 43.6`。该规则使全匹配时结果为1。

## 安装与内置参考包

以下命令在仓库根目录执行。安装方式沿用仓库的Python 3.12环境：

```bash
python -m pip install -e '.[dev]'
python -m ddsr_bench.benchmarks.critpt.evaluation.consensus --help
# 同一入口也注册为 ddsr-critpt-consensus
```

仓库已包含 [`data/consensus-61-v2.json`](data/consensus-61-v2.json)，大小为420,292字节（约410 KiB）。它随Git和Python安装包一起分发。`score`和`replay`默认使用此文件，路径不依赖当前工作目录；也可用`--bundle /path/to/custom.json`显式指定其他包。正常评测不再需要原始模型JSON、外部清单或题面目录。

包内保存70个题号、61道启用题的204份参考代码、模板、confidence、共识组和来源哈希。其SHA-256为`6b9edc5057a92148701bed69afa3b4fb121da89223c6978dc01ddca7005a44bc`；47、51 的历史参考保存在 Git 的 v1 版本和审计记录中。204份记录包含重复来源，不能视为204份独立证据。

参考包仅供评测端使用。各 `docker/*/Dockerfile.dockerignore`明确排除`ddsr_bench/benchmarks/critpt/evaluation/consensus/data/`，因此标准solver/worker镜像只复制执行代码，不包含参考包。含标答的完整仓库或安装包不作为solver的可见工作目录。候选执行请求仍只携带当前候选、模板和输入。

### 从原始材料重建（可选）

仅更新或审计参考包时需要已审查的 `consensus_manifest.json`、原始模型答案JSON，以及对应官方题面JSON。清单中的 `official_problem_path` 和各样本 `path` 相对于 `--source-root`；本工作区可执行：

```bash
python -m ddsr_bench.benchmarks.critpt.evaluation.consensus build \
  --manifest ../internal-eval-assessment/consensus_manifest.json \
  --source-root .. \
  --output .consensus/bundle.json
```

构建器只收录报告中最大共识组的现存代码，校验原始文件SHA-256和模板签名，并保存全部70个槽位。61题之外的参考不执行。

`.consensus/`继续被Git忽略，用于本地评分结果、诊断文件或临时重建的包。输出文件使用排他创建，不覆盖已有结果。

清单的关键字段是：`problem_id`、`official_problem_path`、`reported_max_agreement`、`confidence`、`reported_groups`；每个最大组标记`is_maximal_group`，包含`models`和`samples`，样本保留`reported_model`、`path`、`sha256`及来源说明。构建只接收完整70题清单。

## 评分

默认使用Docker，将模型代码放在独立容器里执行。可以复用仓库现有Dockerfile构建包含新增模块的镜像：

```bash
docker build -f docker/critpt/Dockerfile -t ddsr-critpt-consensus:local .
python -m ddsr_bench.benchmarks.critpt.evaluation.consensus score \
  --candidates /path/to/candidate-answers \
  --output .consensus/model-score.json \
  --image ddsr-critpt-consensus:local \
  --jobs 4 --timeout 60
```

候选目录支持以下文件布局；同一题不能同时提供多个文件：

- `Challenge_1_main.py`
- `Challenge_1_main.json`，含`generated_code`，可带匹配的`problem_id`
- `Challenge_1_main/answer.py`，也可位于更深的运行目录中

也支持`ddsr-solve --config`的原生输出：`Challenge_1_main__attempt-0/artifacts/answer.py`。直接传入一个job目录即可：

```bash
ddsr-critpt-consensus score \
  --candidates outputs/static/critpt-official \
  --attempt 0 \
  --output .consensus/rollout-attempt-0.json \
  --image ddsr-critpt-consensus:local
```

若目录中只有一个attempt，会自动选择；多个attempt必须显式提供`--attempt`。每次评分只使用选定的attempt，缺题不会从其他attempt补取。平铺答案和原生trial不能混用，也不能把多个job目录合并成一次评分。每次结果仍保留70题槽位，分母为61道启用题。

候选JSON损坏、`generated_code`缺失/为null/不是字符串、空代码、无法提取代码或文件读取失败，均记录为该题的`candidate_error`，`error.stage=input`；其他题继续评分。原生trial没有答案产物但已有生成/格式校验错误时，保留上游错误，记为`candidate_error`、`error.stage=generation`；没有失败记录也没有答案时为`missing_candidate`。这两种情况均不通过，且不缩小分母。

候选和参考均使用 `ddsr_bench/grading/validation.py` 中与生成阶段共享的校验器。该实现原样来自初始提交 `d83b0d9` 的 `grading/validation.py`。要求只定义一个顶层 `answer()`，包括嵌套辅助函数在内的额外函数、`lambda`、`try/except` 等仍禁止；导入、签名、危险名称和属性也沿用原规则。代码提取同样沿用生成约定：优先第一个 Python 围栏块，缺少结束围栏本身不会导致拒绝，提取后的代码仍须校验。报告记录共享校验器源码 SHA-256。

路径不存在、没有识别到任何答案或trial、选错attempt、重复题号等属于整次调用的配置错误，命令返回非零并且不生成成绩，避免把路径错误误报为全零成绩。

原生输出中已有的`finish_reason`、`stop_reason`、token用量和正文是否为空，会保存在每题结果的`generation.responses`中；可选元数据文件读取失败会记录在`metadata_errors`。不从response重新生成答案，不按`finish_reason=length`直接改变分数，不修改one-step/two-step流程，也不增加重试或续写。尤其第一步空正文仍沿用现有two-step行为；最终已有合法答案时继续正常比较。API异常发生前未被原生成器保存的信息，评分器无法补回。

候选与每份参考分别执行，候选执行容器不挂载参考包、宿主目录或凭据。容器无网络、只读文件系统、非root、禁用额外能力，限制CPU、内存、进程数和运行时间。镜像在一次运行开始时解析为本地image ID，`--pull=never`，结果记录镜像ID及worker实际Python/科学计算库版本。

执行结果通过有界的类型化JSON传递；不使用pickle，也不把候选输出字符串交给`eval`/字符串`sympify`。比较也在独立受限worker中执行。

### Linux 隔离后端（无需 Docker daemon）

Linux 主机可以显式选择 `--backend linux`。实现仅位于 `consensus/linux.py`，复用同一个 worker、校验器和比较器，不引入通用任务框架，也不改生成或评分政策。默认后端仍为 Docker。

需要宿主支持 user、mount、PID、IPC 和 network namespaces，并安装 Bubblewrap、libseccomp 和 `ldd`。Debian/Ubuntu 的系统依赖可用 `apt-get install bubblewrap libseccomp2 libc-bin` 安装；Python 依赖沿用上面的项目安装。当前适配器支持使用共享库的常规 Linux CPython 安装，已在 Ubuntu 24.04、Python 3.12 上验证；其他发行版或自定义 Python 安装必须通过启动预检。

```bash
python -m ddsr_bench.benchmarks.critpt.evaluation.consensus replay \
  --backend linux --all-references --jobs 4 --timeout 60 \
  --output .consensus/linux-reference-replay.json

python -m ddsr_bench.benchmarks.critpt.evaluation.consensus score \
  --backend linux --candidates /path/to/native-job --attempt 0 \
  --jobs 4 --timeout 60 --output .consensus/linux-attempt-0.json
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
  --output .consensus/reference-replay.json \
  --all-references --trusted-local --jobs 4
```

默认replay每题选第一份可用参考；`--all-references`覆盖所有204份获准参考记录，另保留9个skip。`--problems 3,24,62`可做明确标为partial的诊断回放。回放成功说明参考可执行、序列化和匹配规则可用，不验证其物理正确性，也不证明每组参考在所有参数上彼此相等。

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

定义域、固定测试点与来源注释位于`fixtures.py`和`policy.py`。有限测试覆盖不是完整的物理定义域刻画；尤其15未澄清的退化链长、20的球形退化和其他材料参数不在第一版覆盖范围。不能根据候选输出动态删测试点来制造一致。

## 并列组与已知资料缺口

| 题号 | 第一组 | 第二组 | 可用参考 |
| --- | --- | --- | --- |
| 4、20 | Opus＋Kimi | Fable＋GPT | 只有Fable/GPT组，标记`incomplete` |
| 38 | Opus＋Fable | GPT＋Kimi | 分别有Fable、GPT代表，两组都可匹配 |

两组均为两票，confidence保留40%。4、20可能漏收缺失那组的答案；20的报告数值不能替代完整参数函数。53的Fable文件复制自GPT，保留来源标记，confidence仍沿用报告。60%题目也可能只有该组的一部分原始文件；有参考代表并不等于独立复现全部计票。

25因题面/模板中delta与bar_Delta的参数定义冲突暂skip；19的极值顺序分歧、6的符号及精度问题均按当前范围继续skip。

## 验证与边界

本次环境、测试数量和真实材料回放结果见[验证记录](VALIDATION.md)。

```bash
python -m pytest -q
python -m ruff check ddsr_bench/benchmarks/critpt/evaluation/consensus tests/benchmarks/critpt/consensus
python -m black --target-version py312 --check ddsr_bench/benchmarks/critpt/evaluation/consensus tests/benchmarks/critpt/consensus
```

测试覆盖极小数和大整数、非贪心集合配对、复根、复共轭、明确错误端点、级数阶数、区间开闭及符号k、跨参考拼接拒绝、非有限参考、执行超时、固定分母、数据包完整性、独立模块边界等。

本工作区已做真实材料的可信本地回放，以及 Linux 隔离后端的完整参考回放和隔离测试。当前主机没有可用的 Docker daemon，Docker 路径尚未端到端运行；选择 Docker 时仍应先构建镜像并运行同一回放。模块暂未接入 Harbor/旧 grading。

## DDSR 多次 attempt 批量评分

```bash
ddsr-critpt-consensus batch \
  --candidates /path/to/one/static/job \
  --output /path/to/new/grading-directory \
  --backend linux --jobs 4 --timeout 60 \
  --model-label qwen
```

`batch` 自动读取这个任务现有的所有 attempt。每个 trial 完成就原子保存
`trials/Challenge_N_main__attempt-A.json`，每轮还会保存完整的 `attempt-A.json`。
`summary.json` 记录逐题、逐 attempt 的 `status`、`matched` 及平均值；
`attempt-results.csv` 方便汇总，`progress.json` 表示完成进度。

平均值为 `matched / (61 × attempt 数)`。生成格式错误、缺失回答和 `unknown`
保留在分母中；9 个 skip 不进入分母。`unknown` 仍保留 `matched: null`，不会
改写成已经证实错误。参考执行错误、候选执行错误、符号归一化问题均保留原始
诊断。命令拒绝覆盖已有输出目录；异常退出时已落盘的 trial 仍保留。

DDSR 中共享校验器直接使用 `ddsr_bench/grading/validation.py`，其内容与原始
校验器逐字节相同。consensus 只共享结构校验规则，不调用现有 reward grader。
参考包跟随评测安装包分发，但每个 solver Docker 构建上下文都排除该包；Linux
候选进程只挂载白名单执行模块，无法读取参考包、`.env` 或整个仓库。
