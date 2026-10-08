# [P1] BCE 概率输入为 0 或 1 时，有限损失仍产生 NaN 梯度

基线：`20426b9cd5fb547e247b05575cf771fd37676726`。状态：已在 Flex + Autodiff、F32 上实际复现。

## 问题与影响

BCE 默认的概率模式支持输入 0 和 1，已有测试也检查这些输入的前向结果。当前实现截断对数值以避免前向无穷，但反向仍经过 `log(0)` 或 `log1p(-1)`，产生 NaN 梯度。

这会让完全正确的预测污染模型梯度。Sigmoid 的浮点饱和输出也可能触发，不需要范围外概率。

## 复现与结果

完整程序：[repro/src/002.rs](repro/src/002.rs)。从仓库根目录运行：

```sh
cargo run --manifest-path docs/issues/repro/Cargo.toml --features runtime --bin issue-002 --target-dir /tmp/burn-issue-audit-target
```

概率张量 `[0.0, 1.0]` 开启梯度，目标为 `[0, 1]`，使用默认配置并调用 `backward()`。实际输出：

```text
probability input: loss=0, gradient=[NaN, NaN]
logits control: gradient=[0.0, -2e-44]
```

预期：合法边界概率应产生有限梯度，并采用明确的稳定化约定。这里不限定唯一的精确端点导数，NaN 已足以证明训练数值有问题。

## 源码证据

- [binary_cross_entropy.rs:117](../../crates/burn-nn/src/loss/binary_cross_entropy.rs#L117) 在 `log1p` / `log` 之后才调用 `clamp_min(-100.0)`。
- [Autodiff tensor.rs:3150](../../crates/burn-autodiff/src/ops/tensor.rs#L3150) 的 log 梯度使用输入倒数。
- [Autodiff tensor.rs:3188](../../crates/burn-autodiff/src/ops/tensor.rs#L3188) 的 log1p 梯度使用 `1 / (input + 1)`。

端点处局部导数无穷大，即使上游梯度为零，仍得到 `0 * inf = NaN`。现有端点前向测试无法发现该问题。

## 修复方向与验收

实现稳定的概率形式前后向计算，或在对数前构造安全操作数，并明确端点梯度约定。保留前向兼容性，不能只屏蔽最终 loss 的 NaN。

覆盖正确/错误的 0/1 概率、近端点概率、Sigmoid 饱和、权重和 smoothing 下的反向结果。临时可在模型提供原始 logits 时使用 `.with_logits(true)`；复现程序已验证 `[-100, 100]` 对照的梯度有限。
