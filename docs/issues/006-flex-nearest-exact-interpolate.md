# P2：Flex 后端未实现公开的 `NearestExact` 插值模式

## 影响

`InterpolateMode` 公开提供了 `NearestExact`，CubeCL 后端也实现了该模式，但启用 Flex 后端时
任何合法的 `burn::tensor::module::interpolate` 调用都会直接 panic：
`interpolate: unsupported mode NearestExact / dtype F32`。因此同一模型在 CPU Flex 设备上无法运行，
模型配置只要使用该模式就会在推理/训练的前向阶段失败。

## 根因

`crates/burn-flex/src/ops/module.rs:740-798` 的 `ModuleOps::interpolate` 只匹配
`InterpolateMode::Nearest`、`Bilinear`、`Bicubic` 和 `Lanczos3`，没有 `NearestExact` 分支，最终落入
`crates/burn-flex/src/ops/module.rs:794-797` 的通用 panic。反向分支同样没有
`NearestExact`（从 `crates/burn-flex/src/ops/module.rs:801` 开始）。

该模式不是非法参数：`crates/burn-std/src/ops.rs:414-436` 将其作为公开枚举值，CubeCL
映射（`crates/burn-cubecl/src/kernel/interpolate/base.rs:157-158`）明确将它转为
`NearestMode::Exact`。Exact 的坐标规则与 Floor 不同（按半像素中心取整），不能简单地把它当作
`Nearest` 静默替换。

## 复现

创建最小工程 `/tmp/burn-nearest`：

```toml
[package]
name = "burn-nearest"
version = "0.1.0"
edition = "2021"

[dependencies]
burn = { path = "/Volumes/HIKSEMI/code/burn/crates/burn", default-features = false,
          features = ["std", "flex"] }
```

`src/main.rs`：

```rust
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Device, Tensor};

fn main() {
    let device = Device::flex();
    let input = Tensor::<4>::from_floats([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]], &device);
    let _ = interpolate(
        input,
        InterpolateOptions::new(InterpolateMode::NearestExact).with_output_size([1, 5]),
    );
}
```

运行：

```sh
cargo run --manifest-path /tmp/burn-nearest/Cargo.toml
```

Flex 后端稳定输出：

```text
thread 'main' panicked at .../burn-flex/src/ops/module.rs:...
interpolate: unsupported mode NearestExact / dtype F32
```

同样的输入在当前 CubeCL CPU 后端使用 `NearestExact` 可正常得到
`[3.0, 3.0, 4.0, 5.0, 5.0]`，说明该枚举值及其语义是受支持的，而非应被调用方过滤的非法模式。

## 修复方向

为 Flex 增加 Exact 最近邻的前向/反向坐标映射，并在 `ModuleOps` 两个分支中注册；或者在后端
能力声明中明确拒绝该模式并提供可查询的能力检查，避免公开 API 在运行时无条件 panic。前者可
复用现有最近邻实现，仅将索引映射改为 `((2 * out + 1) * in) / (2 * out)` 的 Exact 规则。

