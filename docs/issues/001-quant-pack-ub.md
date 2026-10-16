# 量化打包函数把 `Vec<i8>` 重解释为 `Vec<u32>`，触发未定义行为

- **严重级别：P1（内存安全）**
- **影响范围：** `burn-std` 的公开函数 `burn_std::tensor::quantization::pack_i8s_to_u32s`
- **代码位置：** `crates/burn-std/src/tensor/quantization/base.rs:478-513`

## 问题

`pack_i8s_to_u32s` 在小端目标上把 `Vec<i8>` 的指针直接转换成 `*mut u32`，再用
`Vec::from_raw_parts` 构造 `Vec<u32>`：

```rust
let mut values = core::mem::ManuallyDrop::new(values);
let ptr = values.as_mut_ptr() as *mut u32;
unsafe { Vec::from_raw_parts(ptr, len, capacity) }
```

`Vec<i8>` 的分配只保证 1 字节对齐，而 `Vec<u32>` 要求指针至少 4 字节对齐。因此即使调用者只传入普通的、安全构造的 `Vec<i8>`，返回的 `Vec<u32>` 也可能从未对齐的地址创建。随后任何读取返回值（例如 `Debug`、索引或切片）都会构造未对齐的 `&[u32]`，这是 Rust 未定义行为。

此外，`capacity` 使用 `values.capacity() / 4`。当输入长度不是 4 的倍数时，补零会按 `Vec` 的增长策略产生不是 4 的倍数的字节容量（例如输入长度 5 通常得到容量 10），重建后的 `Vec<u32>` 只声明 2 个元素的容量（8 字节），但原分配实际是 10 字节；释放时传给分配器的布局也与原始布局不一致，违反 `Vec::from_raw_parts` 的安全前提。

## 影响

这是公开安全 API 在合法输入上的内存安全问题，不需要调用者编写 `unsafe` 代码即可触发。结果可能表现为：

- 在 Miri 中立即报告未对齐引用；
- 在某些分配器或优化级别下崩溃、错误释放或内存破坏；
- 量化数据打包路径得到不可靠结果。

当前测试 `should_pack_i8s_to_u32_padded` 本身就使用了长度为 5 的输入，但只检查数值结果，未覆盖上述内存模型违规。

## 最小复现

创建临时工程 `/tmp/burn-pack-repro`：

`Cargo.toml`：

```toml
[package]
name = "burn-pack-repro"
version = "0.1.0"
edition = "2021"

[dependencies]
burn-std = { path = "/Volumes/HIKSEMI/code/burn/crates/burn-std", default-features = false, features = ["std"] }
```

`src/main.rs`：

```rust
fn main() {
    // 输入长度不是 4 的倍数，函数会在内部补零并重解释分配。
    let packed = burn_std::tensor::quantization::pack_i8s_to_u32s(vec![
        -128, 2, -3, 127, 55,
    ]);
    println!("{packed:?}");
}
```

普通运行通常打印数值，但这不能证明代码符合 Rust 内存模型：

```text
$ cargo run --manifest-path /tmp/burn-pack-repro/Cargo.toml
[2147287680, 55]
```

使用当前工具链的 Miri 运行（需要 nightly 和 `miri` 组件）可以稳定复现：

```sh
env -u CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER \
  cargo +nightly miri run --manifest-path /tmp/burn-pack-repro/Cargo.toml
```

Miri 报告：

```text
Undefined Behavior: constructing invalid value of type &[u32]:
encountered an unaligned reference (required 4 byte alignment but found 1)
```

## 修复方向

不要重解释原始 `Vec<i8>` 的分配。应分配正确对齐的 `Vec<u32>`，逐组按字节组合（或先复制到对齐的 `u32` 缓冲区），并让返回向量的容量与实际分配严格匹配；同时为非 4 倍数长度和 Miri 增加回归测试。
