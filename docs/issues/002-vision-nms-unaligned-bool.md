# CPU NMS 将 `Vec<bool>` 未对齐重解释为 SIMD 整数，触发未定义行为

- **严重级别：P1（内存安全）**
- **影响范围：** `burn-vision` CPU NMS（Flex 后端）
- **代码位置：** `crates/burn-vision/src/backends/cpu/nms.rs:162`

## 问题

NMS 用 `Vec<bool>` 保存抑制状态。为了快速判断一整个 SIMD 块是否已经被抑制，代码把 `bool` 指针直接转换成 `*const u32`、`*const u64` 或 `*const u128`：

```rust
4 => *(suppressed.as_ptr().add(i) as *const u32) == 0x01010101,
8 => *(suppressed.as_ptr().add(i) as *const u64) == 0x0101010101010101,
16 => *(suppressed.as_ptr().add(i) as *const u128)
    == 0x01010101010101010101010101010101,
```

`Vec<bool>` 的元素对齐只有 1 字节，不能保证其首地址满足 4/8/16 字节对齐。即使 `i` 是 SIMD lane 的倍数，基础指针仍然可能未对齐。通过这些指针读取会构造未对齐引用，属于 Rust 未定义行为。

## 影响

这是公共 `Tensor::nms` API 在普通安全输入上的内存安全问题。不同分配器、优化级别或架构下可能表现为崩溃、错误读取抑制状态或内存破坏。问题在使用 4-lane SIMD 的路径上即可触发，不需要大型输入。

## 最小复现

创建临时工程 `/tmp/burn-nms-repro`：

`Cargo.toml`：

```toml
[package]
name = "burn-nms-repro"
version = "0.1.0"
edition = "2021"

[dependencies]
burn-vision = {
    path = "/Volumes/HIKSEMI/code/burn/crates/burn-vision",
    default-features = false,
    features = ["flex"],
}
burn-core = {
    path = "/Volumes/HIKSEMI/code/burn/crates/burn-core",
    default-features = false,
    features = ["std", "flex"],
}
```

`src/main.rs`：

```rust
use burn_core::tensor::{Device, Tensor};
use burn_vision::{Nms, NmsOptions};

fn main() {
    let device = Device::default();
    let boxes: Tensor<2> = Tensor::from_data(
        [[0.0f32, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]],
        &device,
    );
    let scores: Tensor<1> = Tensor::from_data([0.9f32, 0.8], &device);
    let kept = boxes.nms(scores, NmsOptions::default());
    println!("{:?}", kept.into_data().try_into_vec::<i32>().unwrap());
}
```

普通运行通常得到正确的 NMS 结果：

```text
[0]
```

使用 nightly Miri 运行：

```sh
env -u CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER \
  cargo +nightly miri run --manifest-path /tmp/burn-nms-repro/Cargo.toml
```

当前代码稳定报告：

```text
Undefined Behavior: accessing memory based on pointer with alignment 1,
but alignment 4 is required
--> crates/burn-vision/src/backends/cpu/nms.rs:162:22
```

## 修复方向

使用按 SIMD 宽度对齐的整数/字节缓冲区保存抑制标志（例如 `AVec<u32, ...>`），或使用显式逐元素检查，避免将 `Vec<bool>` 指针重解释为更高对齐类型。修复后应保留普通运行测试，并加入 Miri 覆盖。
