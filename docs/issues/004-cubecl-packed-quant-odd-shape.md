# CubeCL 拒绝规范允许的非整除 `PackedU32` 量化张量

- **严重级别：P2（合法模型无法加载）**
- **影响范围：** CubeCL 后端（包括 `burn-cpu`）从 `TensorData` 加载 `PackedU32` 量化张量
- **代码位置：** `crates/burn-cubecl/src/ops/qtensor.rs:110`

## 问题

`burn-std` 的量化数据格式明确允许 packed 轴不是 packing factor 的整数倍：每一行会向上取整并用零补齐。`QuantizedBytes` 和 `quantized_data_len` 都按这个规则生成数据，例如 Q4 的形状 `[1, 3]` 需要一整个 32-bit word。

但 CubeCL 创建量化 tensor 时要求最后一维必须整除 `scheme.num_quants()`：

```rust
QuantStore::PackedU32(_) => {
    if !shape_last.is_multiple_of(num_quants) {
        panic!("Can't store in u32")
    }
    shape_value[rank - 1] = shape_last.div_ceil(num_quants);
    size_of::<u32>()
}
```

因此同一个公开的 `TensorData` 表示，在 Flex/序列化路径合法，却在 CubeCL 后端加载时直接 panic。真实模型中常见的最后一维（例如词表、通道数、投影维度）不一定是 8/16/32 的倍数，这会使权重无法迁移到 CubeCL 设备。

## 最小复现

创建临时工程 `/tmp/burn-ctc-repro`（也可复用该工程）：

`Cargo.toml`：

```toml
[package]
name = "burn-quant-shape-repro"
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
use burn_core::tensor::quantization::{QuantScheme, QuantStore, QuantValue};
use burn_core::tensor::{Device, Tensor, TensorData};

fn main() {
    let device = Device::default(); // Cube(Cpu(CpuDevice))
    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q4S)
        .with_store(QuantStore::PackedU32(0));

    // Q4 每个 u32 可放 8 个值；形状 [1, 3] 按格式应补齐到一个 word。
    let data = TensorData::quantized(vec![0i8, 1, 2], [1, 3], scheme, &[1.0], None);
    println!("dtype={:?}, bytes={}", data.dtype(), data.as_bytes().len());

    let tensor: Tensor<2> = Tensor::from_data(data, &device);
    println!("{:?}", tensor.dequantize().into_data());
}
```

运行：

```sh
env -u CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER \
  cargo run --manifest-path /tmp/burn-quant-shape-repro/Cargo.toml
```

`TensorData::quantized` 成功创建数据（Q4 `[1, 3]` 的 packed 数据长度为 8 字节，包含一个 word 和 scale），但 CubeCL 后端随后稳定 panic：

```text
thread 'main' panicked at crates/burn-cubecl/src/ops/qtensor.rs:113:
Can't store in u32
```

## 修复方向

CubeCL 应按 `QuantizedBytes` 的布局接受非整除 packed 轴：将逻辑形状转换为 `div_ceil(num_quants)` 的存储形状，并确保 kernel 对每行 padding 位不读取为有效元素。至少应在 `supports_dtype`/构造入口明确拒绝该形状并返回可诊断错误，而不是让合法的公共 `TensorData` 在后端加载时无条件 panic。
