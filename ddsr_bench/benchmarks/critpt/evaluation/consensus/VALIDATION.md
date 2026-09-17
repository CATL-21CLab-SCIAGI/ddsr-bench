# Consensus 验证记录

## 当前版本：consensus-61-v2

70 个题号中 61 题计分、9 题 skipped，启用参考 204 份，权重合计 43.6。
47、51 的排除依据见 [EXCLUSIONS.md](EXCLUSIONS.md)。

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
从仓库移出时逐字节校验，评分规则和内容不变。默认资产缺失或哈希不符时报错；
其他位置通过 `--bundle` 指定，评分报告保留实际哈希。

回放及打包检查记录位于 `/mnt/workspace/zhizhou/assets/critpt/audits/` 下的
`external-reference-20260918-replay.json`、`external-reference-20260918-wheel.json`。

生成与评分仍共享原始校验器，保留辅助函数、lambda、try 等禁令。测试覆盖空正文仍执行
二阶段、错误产物按题记录、attempt 不混用、skipped 不执行且不进入分母，以及逐题及时保存。

## 历史迁移验证

完整迁移经过 700 条历史 trial 对照，身份、状态、匹配结果、错误、元数据和答案哈希一致。
原 `consensus-63-v1` 为 212 份参考、7 个 skip；按 v2 排除 47、51 后，每模型
350 条记录中 305 条计分、45 条 skipped。Qwen3.8 为 58/305，DeepSeek 为 92/305。
迁移与真实请求的验证范围见 [MIGRATION.md](../../MIGRATION.md)。

曾用六份非最大共识组答案检查比较器：11、17、28、41、56 判为 different，45 判为 matched。
45 的不同表达式在正质量参数范围内等价，原报告分组不能直接作为负例标签。
历史 confidence 保持不变，不据此重新计票。

这些结果验证执行和匹配链路，不证明参考答案的物理正确性或有限采样下的恒等关系。
Linux 后端共享宿主内核，不提供虚拟机隔离或 cgroup 总内存配额。
当前 DSW 无 Docker daemon；Docker 端到端回放和完整 Harbor 容器 trial 尚未验证。
