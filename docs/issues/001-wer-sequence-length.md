# [P2] WER 在预测与目标序列长度不同时漏算错误或越界

基线：`20426b9cd5fb547e247b05575cf771fd37676726`。状态：已在 Flex 上实际复现。

## 问题与影响

`WordErrorRate::update` 使用目标序列长度切分预测张量。预测更长时，多余 token 被忽略或跨行错配；预测更短时，切片越界 panic。语音识别、文本生成中的插入和删除本应是 WER 正常统计的错误，当前实现可能虚报低错误率或中断评估。

触发条件：合法的二维整数张量，batch size 相同，但两侧序列长度不同。不需要 padding 或非法 token。

## 复现与结果

完整程序：[repro/src/001.rs](repro/src/001.rs)。从仓库根目录运行：

```sh
cargo run --manifest-path docs/issues/repro/Cargo.toml --features runtime --bin issue-001 --target-dir /tmp/burn-issue-audit-target
```

| 预测 | 目标 | 预期 | 实际 |
| --- | --- | --- | --- |
| `[1, 2, 3]` | `[1, 2]` | 一次插入 / 两个目标词 = 50% | 0% |
| `[1]` | `[1, 2]` | 一次删除 / 两个目标词 = 50% | 越界 panic |

捕获的错误：`range end index 2 out of range for slice of length 1`。

## 源码证据

- [wer.rs:63](../../crates/burn-train/src/metric/wer.rs#L63) 只获取 `targets.dims()`。
- [wer.rs:75](../../crates/burn-train/src/metric/wer.rs#L75) 用同一个 `seq_len` 计算起止位置，同时切分 `outputs_data` 和 `targets_data`。
- [cer.rs](../../crates/burn-train/src/metric/cer.rs) 中同类指标已分别读取两侧长度并校验 batch size，可作为参考。

单行较长预测已能证明漏算；多行情况下，预测行起点还会因步长错误而偏移。

## 修复方向与验收

分别读取两侧长度，校验 batch size 相等，每侧按自己的步长切片，再计算编辑距离。覆盖较长/较短预测、多行、带 padding、batch size 不匹配和等长输入。上表两组输入都应返回 50%，不再 panic。
