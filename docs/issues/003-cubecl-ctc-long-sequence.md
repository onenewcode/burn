# CubeCL CTC 在长序列上把可达路径误判为不可达

- **严重级别：P1（训练结果错误）**
- **影响范围：** CubeCL 后端（包括 `burn-cpu`）的 CTC loss
- **代码位置：** `crates/burn-cubecl/src/kernel/ctc.rs:80`

## 问题

CTC alpha 递推使用 `-6e4` 作为有限的负无穷，并把所有小于 `-1e4` 的值直接当成“不可达”：

```rust
if mx < unreachable_threshold {
    mx
}
```

这是一个与序列长度和概率有关的数值截断，而不是仅针对真正的 sentinel。合法路径的 log 概率在长序列上也会低于 `-1e4`。一旦最终 alpha 被误判为不可达，`finalize_nll` 会走合成无穷值的分支：

```rust
if mx < unreachable_threshold {
    (F::new(1000.0) * F::cast_from(target_len as u32)).exp()
}
```

当 `target_len == 0` 时，这个表达式反而是 `exp(0) == 1.0`，而不是正确的有限损失或不可达的 `+inf`。

## 可观察影响

以最简单的合法 CTC 输入为例：每一帧有两个类别，blank 概率为 `0.5`，目标为空，输入长度为 15,000。唯一合法路径的概率是 `0.5^15000`，所以正确损失为：

```text
-15000 * ln(0.5) = 10397.2077...
```

当前 CubeCL 后端返回 `1.0`。这会让长序列训练的 loss 和梯度完全错误；对非空目标，类似误判会返回 `+inf`，随后 `zero_infinity` 可能把样本梯度清零。

该问题不依赖 GPU，使用 CPU CubeCL runtime 即可触发。

## 最小复现

创建临时工程 `/tmp/burn-ctc-repro`：

`Cargo.toml`：

```toml
[package]
name = "burn-ctc-repro"
version = "0.1.0"
edition = "2021"

[dependencies]
burn-core = {
    path = "/Volumes/HIKSEMI/code/burn/crates/burn-core",
    default-features = false,
    features = ["std", "cpu"],
}
```

`src/main.rs`：

```rust
use burn_core::tensor::module::ctc_loss;
use burn_core::tensor::{Device, Tensor, TensorData};

fn main() {
    let t = 15_000usize;
    let device = Device::default(); // Cube(Cpu(CpuDevice))

    // C=2, 每帧 blank 概率为 0.5，目标为空，因此只有全 blank 路径。
    let log_probs: Tensor<3> = Tensor::full([t, 1, 2], (0.5f32).ln(), &device);
    let targets: Tensor<2, burn_core::tensor::Int> =
        Tensor::from_data(TensorData::from([[0i64]]), &device);
    let input_lengths: Tensor<1, burn_core::tensor::Int> =
        Tensor::from_data(TensorData::from([t as i64]), &device);
    let target_lengths: Tensor<1, burn_core::tensor::Int> =
        Tensor::from_data(TensorData::from([0i64]), &device);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0);
    println!("{:?}", loss.into_data().iter::<f32>().collect::<Vec<_>>());
}
```

运行：

```sh
env -u CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER \
  cargo run --manifest-path /tmp/burn-ctc-repro/Cargo.toml
```

当前代码输出：

```text
[1.0]
```

而正确结果应接近：

```text
[10397.2077]
```

把 `t` 改为 `3` 时当前实现输出约 `2.0794415`，说明错误由 `-1e4` 截断阈值触发，而不是复现工程中的概率或 shape 错误。

## 修复方向

不要用固定的 `-1e4` 将合法的低概率 alpha 当成不可达。应使用不会与合法 log 概率冲突的表示（例如后端可表达的 `-inf`），或携带显式可达标记；如果必须使用有限 sentinel，则应在最终结果中区分“sentinel 状态”和真实的小概率值，并为 `target_len == 0` 保留正确的 finite loss 路径。

同样地，将复现中的 `targets` 改为 `[[1i64]]`、`target_lengths` 改为 `[1i64]`（仍然是可达的单标签目标），当前实现输出 `[inf]`，而正确损失应为有限值。这说明问题并非空目标的特殊路径，而是阈值把合法 alpha 状态整体标成了不可达。
