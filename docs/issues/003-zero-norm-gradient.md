# [P1] 零向量 L2 范数产生 NaN 梯度，并污染已满足 margin 的三元组损失

基线：`20426b9cd5fb547e247b05575cf771fd37676726`。状态：已在 Flex + Autodiff、F32 上实际复现。

## 问题与影响

`l2_norm` 组合平方、求和与开方，未处理零范数的反向传播。零向量梯度为 NaN。`TripletMarginLoss` 通过 `lp_norm(..., 2, ...)` 复用此实现，因此 anchor 与 positive 相等时，即使已满足 margin、loss 为 0，仍可产生 NaN 梯度。

重复样本、相同 embedding 或零输出都可能遇到该情况，进而污染优化器状态和参数。

## 复现与结果

完整程序：[repro/src/003.rs](repro/src/003.rs)。从仓库根目录运行：

```sh
cargo run --manifest-path docs/issues/repro/Cargo.toml --features runtime --bin issue-003 --target-dir /tmp/burn-issue-audit-target
```

程序先计算零向量 L2 梯度，再计算三元组：anchor `[0, 0]`（需要梯度），positive `[0, 0]`，negative `[2, 0]`，默认 margin=1。

```text
L2 at zero: gradient=[NaN, NaN]
inactive triplet: loss=0, gradient=[NaN, NaN] (expected [0, 0])
```

预期：该三元组的 anchor 梯度应为 `[0, 0]`。单独范数在零点需要选择次梯度；三元组例子更明确：margin 表达式为 `0 - 2 + 1 = -1`，anchor 的一个邻域内 loss 恒为 0。

## 源码证据

- [vector_norm.rs:430](../../crates/burn-linalg/src/functions/vector_norm.rs#L430)：`x.square().sum_dims(dims).sqrt()`。
- [Autodiff tensor.rs:3329](../../crates/burn-autodiff/src/ops/tensor.rs#L3329)：sqrt 的反向通过 `input^(-0.5) / 2` 计算，零输入产生无穷。
- [triplet_margin.rs:104](../../crates/burn-nn/src/loss/triplet_margin.rs#L104)：使用 Lp 范数计算距离，再截断 margin 表达式。

零的平方梯度或被截断的上游梯度不能消除无穷局部导数。范数与三元组表现属于同一根因，合并为一个 issue。

## 修复方向与验收

为 L2 范数提供零点安全的反向，保持精确的前向零值，并选择有限次梯度。不要只在最终结果上 mask NaN。

覆盖零范数、混合零/非零批次、相等 anchor/positive、loss 为零的三元组；上述三元组梯度必须为零，非零输入的数值保持正确。
