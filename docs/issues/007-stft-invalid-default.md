# [P2] StftOptions 默认 n_fft=400 无法通过 STFT/ISTFT 自身校验

基线：`20426b9cd5fb547e247b05575cf771fd37676726`。状态：默认 STFT 已实测 panic，n_fft=512 对照成功。

## 问题与影响

`StftOptions::default()` 返回 n_fft=400，但 stft 和 istft 都要求 n_fft 为 2 的幂。默认配置对于所有输入都会在公共参数校验处失败，默认音频特征提取流程无法开始计算。

该问题发生在后端计算前，不是信号太短或未开启 FFT 后端导致。

## 复现与结果

完整程序：[repro/src/007.rs](repro/src/007.rs)。从仓库根目录运行：

```sh
cargo run --manifest-path docs/issues/repro/Cargo.toml --features runtime --bin issue-007 --target-dir /tmp/burn-issue-audit-target
```

核心输入为 `[1, 1024]` 的零张量，不指定 window，使用 `StftOptions::default()`。

```text
stft: n_fft must be a power of two, got 400.
default n_fft=400, panicked=true
n_fft=512 control: shape=[1, 9, 257, 2]
```

预期：默认参数满足当前实现约束，合法长度输入能够完成变换。

## 源码证据

- [stft.rs:46](../../crates/burn-signal/src/functions/stft.rs#L46) 拒绝非 2 的幂。
- [stft.rs:73](../../crates/burn-signal/src/functions/stft.rs#L73) 的 Default 却调用 `Self::new(400)`。
- [stft.rs:88](../../crates/burn-signal/src/functions/stft.rs#L88) 在计算前执行校验；istft 也先调用相同的 assert_valid。

## 修复方向与验收

将默认 FFT 长度改为支持的值，或真正提供默认 400 所需的非 2 的幂变换支持。不能只删除校验而继续使用不支持该长度的计算路径。

增加不覆盖任何字段的默认 STFT 测试，以及同配置下 STFT/ISTFT 往返测试。
