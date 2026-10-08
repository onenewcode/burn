# [P1] CPU NMS 在 4/8 lane SIMD 路径中整体读取未完全初始化的 bool 数组

基线：`20426b9cd5fb547e247b05575cf771fd37676726`。状态：已核对源码与依赖写入范围，抽取相同内存操作后经 Miri 验证；未对完整 NMS 运行 Miri。

## 问题与影响

CPU NMS 创建 `MaybeUninit<[bool; 16]>`，只写入实际 lane 数量的 bool，便对整个数组调用 assume_init。

F32 SIMD 宽度为 4（如 NEON/SSE）或 8（如 AVX2）时，尾部仍未初始化。构造整个 `[bool; 16]` 已违反 Rust 值有效性规则，后续只访问前 lanes 项并不能使其合法。

普通 NMS 数据路径即可触发未定义行为，可能导致优化相关的不可预测结果；本次没有证据声称存在特定崩溃或可利用方式。16 lane 的写入覆盖完整数组，不属于本报告触发范围。

## 源码证据与可达条件

- [cpu/nms.rs:156](../../crates/burn-vision/src/backends/cpu/nms.rs#L156) 分配未初始化数组。
- [cpu/nms.rs:199](../../crates/burn-vision/src/backends/cpu/nms.rs#L199) 写入 mask，下一行 assume_init 整个数组。
- 已核对本次依赖 macerator 0.5.1：`Scalar::mask_store_as_bool` 契约写入 lanes 个 bool；aarch64 的 `mask_store_as_bool_32` 只写 `128/32 = 4` 个元素。

例如两个合法且互不重叠的框通过分数过滤，使用默认 NMS 配置（不限输出数量），就会进入 `!all_suppressed` 分支，在上述 SIMD 宽度下执行部分写入后的整体构造。

## 最小内存复现

完整程序：[repro/src/008.rs](repro/src/008.rs)。它抽取分配、相同覆盖范围的写入与 assume_init，不加载 Burn 或调用 SIMD。安装 nightly Miri 后，从仓库根目录运行：

```sh
cargo +nightly miri run --manifest-path docs/issues/repro/Cargo.toml --bin issue-008 --target-dir /tmp/burn-issue-audit-miri-target
```

实际退出码为 1：

```text
error: Undefined Behavior: constructing invalid value of type [bool; 16]:
at [4], encountered uninitialized memory, but expected a boolean
let mask_buf = unsafe { mask_buf.assume_init() };
```

预期：所有被构造为 bool 的元素均已初始化，内存有效性检查通过。普通原生执行没有报错不代表没有 UB。

## 修复方向与验收

可使用完整初始化的 `[false; 16]`，让 mask store 覆盖前 lanes 项；或只读取未初始化存储中已写入的前缀，避免整体构造数组。

检查 4、8、16 lane，覆盖保留两个框、发生抑制、无抑制和输出数量限制。修复后的等价内存操作应通过 Miri，NMS 结果与标量参考一致。
