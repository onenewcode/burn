# `pack_i8s_to_u32s` 将 1 字节对齐的 `Vec<i8>` 重解释为 `Vec<u32>`，造成未定义行为

## 严重级别

**P1（库健全性问题）**

这是一个公开的安全函数，但接受合法的 `Vec<i8>` 输入即可构造出违反 `Vec<u32>` 对齐要求的值。即使多数默认分配器恰好返回更高对齐地址，库仍不能依赖未声明的分配器行为；自定义分配器、替代运行时或严格对齐架构上会触发未定义行为。

## 问题描述

`crates/burn-std/src/tensor/quantization/base.rs` 中的公共函数在小端平台上执行以下转换：

```rust
// crates/burn-std/src/tensor/quantization/base.rs:478-516
pub fn pack_i8s_to_u32s(values: Vec<i8>) -> Vec<u32> {
    #[cfg(target_endian = "little")]
    {
        let mut values = values;
        let remainder = values.len() % 4;
        if remainder != 0 {
            values.extend(core::iter::repeat_n(0, 4 - remainder));
        }

        let len = values.len() / 4;
        let capacity = values.capacity() / 4;

        // Pre-forget the old vec and re-interpret as u32
        let mut values = core::mem::ManuallyDrop::new(values);
        let ptr = values.as_mut_ptr() as *mut u32;

        unsafe { Vec::from_raw_parts(ptr, len, capacity) }
    }
}
```

问题在于：

- `Vec<i8>` 的元素类型 `i8` 只要求 1 字节对齐；
- `Vec<u32>` 的元素类型 `u32` 要求 4 字节对齐；
- `values.as_mut_ptr() as *mut u32` 不检查、不调整、不重新分配，只是把可能 1 字节对齐的指针当作 `u32` 指针；
- `Vec::from_raw_parts(ptr, len, capacity)` 要求 `ptr` 满足 `u32` 的对齐要求；
- 返回后的 `Vec<u32>` 在 slice/deref/比较/读取时形成未对齐的 `&[u32]`。

因此，对任意合法 `Vec<i8>` 调用这个安全函数都可能立即违反 Rust 的别名/对齐规则。

## 最小复现

将以下内容保存为 `crates/burn-std/tests/pack_alignment_repro.rs`：

```rust
use std::alloc::{GlobalAlloc, Layout, System};

struct OneByteAlignedAllocations;

unsafe impl GlobalAlloc for OneByteAlignedAllocations {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() == 1 {
            unsafe {
                let adjusted = Layout::from_size_align(layout.size() + 1, 1).unwrap();
                System.alloc(adjusted).add(1)
            }
        } else {
            unsafe { System.alloc(layout) }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.align() == 1 {
            unsafe {
                let adjusted = Layout::from_size_align(layout.size() + 1, 1).unwrap();
                System.dealloc(ptr.sub(1), adjusted);
            }
        } else {
            unsafe { System.dealloc(ptr, layout) }
        }
    }
}

#[global_allocator]
static ALLOCATOR: OneByteAlignedAllocations = OneByteAlignedAllocations;

#[test]
fn pack_i8s_reinterprets_an_i8_allocation_as_u32() {
    let packed = burn_std::tensor::quantization::pack_i8s_to_u32s(vec![-128, 2, -3, 127]);
    assert_eq!(packed, vec![2147287680]);
}
```

该自定义分配器对 `Layout::align() == 1` 的分配返回只满足 1 字节对齐的指针，这是 `Vec<i8>` 允许的最小保证。

运行 Miri：

```sh
cargo miri test -p burn-std --test pack_alignment_repro
```

## 实测输出

在当前 checkout（commit `88b8dd55e`）上执行：

```text
running 1 test
test pack_i8s_reinterprets_an_i8_allocation_as_u32 ... error:
Undefined Behavior: constructing invalid value of type &[u32]:
encountered an unaligned reference (required 4 byte alignment but found 1)

--> alloc/src/vec/mod.rs:1840:13
    &*core::intrinsics::aggregate_raw_ptr::<*const [u32], _, _>(
        self.as_ptr(), self.len
    )

stack backtrace:
    std::vec::Vec::<u32>::as_slice
    <std::vec::Vec<u32> as std::ops::Deref>::deref
    std::vec::partial_eq::<impl PartialEq for Vec<u32>>::eq
    pack_i8s_reinterprets_an_i8_allocation_as_u32

error: aborting due to 1 previous error
```

Miri 明确指出：函数返回的 `Vec<u32>` 在比较时形成了 1 字节对齐的 `&[u32]`。

## 预期行为

安全函数对任意合法 `Vec<i8>` 输入都不得产生未定义行为。`pack_i8s_to_u32s` 应构造真正满足 `u32` 对齐的输出缓冲，而不是重解释原有 `i8` 分配。

## 建议修复

删除指针重解释，改为安全打包：

```rust
#[cfg(target_endian = "little")]
pub fn pack_i8s_to_u32s(values: Vec<i8>) -> Vec<u32> {
    values
        .chunks(4)
        .map(|chunk| {
            let mut word = 0_u32;
            for (byte_index, value) in chunk.iter().enumerate() {
                word |= (*value as u32 & 0xff) << (byte_index * 8);
            }
            word
        })
        .collect()
}
```

注意保留现有语义：

- 空输入返回空 `Vec<u32>`；
- 长度不是 4 的倍数时，末尾用 0 填充；
- 大端实现继续使用安全循环；
- 保留现有 `should_pack_i8s_to_u32*` 测试；
- 增加上述 Miri 回归测试，防止未来重新引入 unsafe 转换。

