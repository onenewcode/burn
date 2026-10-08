# 代码审查发现的问题

审查基线：`20426b9cd5fb547e247b05575cf771fd37676726`（版本 `0.22.0`），日期：2026-10-08。

本次整理了 8 个问题，每个问题独立成文，包含触发条件、实际与预期行为、源码位置、复现命令、影响和修复建议。没有修改库实现，也没有向远端提交 issue。

| 编号 | 建议优先级 | 问题 | 验证方式 |
| --- | --- | --- | --- |
| 001 | P2 | [WER 在预测与目标长度不同时漏算或越界](001-wer-sequence-length.md) | Flex 实际调用 |
| 002 | P1 | [BCE 概率输入为 0/1 时产生 NaN 梯度](002-bce-endpoint-gradient.md) | Flex + Autodiff 反向传播 |
| 003 | P1 | [零范数导致已满足 margin 的三元组仍产生 NaN 梯度](003-zero-norm-gradient.md) | Flex + Autodiff 反向传播 |
| 004 | P2 | [Gaussian NLL 的方差截断错误地阻断梯度](004-gaussian-nll-variance-gradient.md) | 实际梯度与解析公式对照 |
| 005 | P2 | [单线程 DataLoader 遇到坏样本后无限重复错误](005-dataloader-error-progress.md) | 自定义 Dataset 实际读取 |
| 006 | P2 | [多线程 DataLoader 交错迭代器可能互相阻塞](006-dataloader-overlapping-iterators.md) | 阻塞与释放对照实验 |
| 007 | P2 | [STFT 默认配置无法通过自身参数校验](007-stft-invalid-default.md) | 默认配置失败、合法配置成功 |
| 008 | P1 | [CPU NMS 整体读取部分初始化的布尔数组](008-nms-uninitialized-mask.md) | 源码、依赖核对与抽取操作的 Miri 验证 |

P1 表示可能破坏训练数值或违反内存有效性规则，应优先处理；P2 表示在明确条件下产生错误结果、崩溃或阻塞。这是本次审查的建议优先级。

## 审查与验证范围

检查了数据集变换、数据加载器、训练指标、学习率调度器、若干损失函数和向量范数，以及 STFT/ISTFT 和 CPU NMS 实现。按根因合并同类表现，例如零范数与三元组损失的 NaN 梯度合并为 003，没有重复计数。

验证环境：macOS aarch64，`rustc 1.97.0-nightly (f964de49b 2026-05-07)`；数值复现使用 Flex 的 F32 和 Autodiff。未验证 GPU 后端，不声称本次已穷尽全仓库缺陷。008 的 Miri 程序只复现对应内存操作，不是完整 NMS 调用；004 的兼容性期望通过公式分析得到，本轮没有运行 PyTorch。

已完成的检查：

- `cargo test -p burn-dataset --lib --offline --target-dir /tmp/burn-issue-audit-target`：44 个既有测试通过。
- [复现工程](repro/Cargo.toml) 的 001–007 全部编译并运行，输出与各文档一致。断言用于确认当前缺陷仍存在，退出码 0 不代表问题已修复。
- 008 由 Miri 报告 `constructing invalid value of type [bool; 16]`，退出码 1，符合复现预期。

## 运行复现

从仓库根目录执行，例如：

```sh
cargo run --manifest-path docs/issues/repro/Cargo.toml --features runtime --bin issue-001 --target-dir /tmp/burn-issue-audit-target
```

将编号换为 `issue-002` 至 `issue-007` 即可运行对应程序。001 和 007 会捕获预期 panic，因此终端出现 panic 文本后仍正常退出。006 检测阻塞后释放第一个迭代器，并验证第二个恢复，不会故意永久挂起。

008 需要 nightly Miri，不要启用 `runtime` feature：

```sh
cargo +nightly miri run --manifest-path docs/issues/repro/Cargo.toml --bin issue-008 --target-dir /tmp/burn-issue-audit-miri-target
```

复现工程是独立 workspace，依赖指向当前仓库源码，生成的 `Cargo.lock` 和默认 `target` 目录已忽略。上述结果对应本次审查基线；修复后应把确认缺陷的断言改成各文档的验收条件。
