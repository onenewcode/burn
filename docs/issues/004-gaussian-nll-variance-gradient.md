# [P2] Gaussian NLL 的方差稳定化截断阻断梯度，与声明的 PyTorch 行为不符

基线：`20426b9cd5fb547e247b05575cf771fd37676726`。状态：Burn 梯度已实测，期望值通过解析公式对照；本轮未运行 PyTorch。

## 问题与影响

`GaussianNLLLoss` 声明跟随 PyTorch 实现，但用普通可微的 `clamp_min(eps)` 截断方差。合法正方差小于 eps 时，方差梯度被置零。

PyTorch 的稳定化约定是前向使用截断值，截断本身不阻断 autograd。当前实现使方差过小的样本失去来自该损失的方差学习信号；前向数值仍正常。

## 复现与结果

完整程序：[repro/src/004.rs](repro/src/004.rs)。从仓库根目录运行：

```sh
cargo run --manifest-path docs/issues/repro/Cargo.toml --features runtime --bin issue-004 --target-dir /tmp/burn-issue-audit-target
```

默认 eps=`1e-6`，均值 `[0]`，目标 `[1]`，需要梯度的正方差 `[1e-8]`。实际输出：

```text
variance=1e-8: gradient=0, PyTorch-compatible expectation=-499999500000
variance=0.5 control: gradient=-1 (expected -1)
```

预期：令 `v=max(var, eps)`，按照截断保留梯度的兼容语义，导数应为：

```text
dL/dvar = 0.5 * (1/v - (input-target)^2/v^2)
        = -499999500000
```

F32 可以有舍入误差，但不应变成零。阈值以上方差 `0.5` 的对照梯度正确。

## 源码证据

- [gaussian_nll.rs](../../crates/burn-nn/src/loss/gaussian_nll.rs) 的类型文档声明跟随 PyTorch。
- [gaussian_nll.rs:110](../../crates/burn-nn/src/loss/gaussian_nll.rs#L110) 直接把 `var.clamp_min(self.eps)` 纳入计算图。
- [PyTorch GaussianNLLLoss 文档](https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html) 的 Note 说明 clamping 被 autograd 忽略，不影响相对于 var 的梯度。

## 修复方向与验收

采用前向截断、反向保留原方差梯度的实现，例如 detach 截断偏移，或使用自定义反向。不能 detach 整个方差张量。

覆盖 `0 < var < eps`、`var == eps`、`var > eps`，同时检查方差和均值梯度。保留现有前向结果，新增阈值以下的反向兼容性测试。
