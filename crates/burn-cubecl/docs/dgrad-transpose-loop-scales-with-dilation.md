# 直接转置卷积的循环长了 `dilation` 倍

|          |                                                                          |
| -------- | ------------------------------------------------------------------------ |
| 状态     | open                                                                     |
| 影响     | `dilation = 4` 时，kernel 58% 的时间花在不做任何算术的循环迭代上          |
| 何时触发 | autotune 关闭时（此时 `ConvTranspose2dStrategy` 默认就是 `Direct`）、调用方显式要 `Direct`、或 `col2im` 拒绝该形状 |
| 文件     | `src/kernel/conv/conv_transpose2d/transpose_direct.rs`                   |
| 复现     | 同一文件，模块 `dilation_loop_length`                                     |
| 难度     | 改循环：低；做到文件里自己那条 TODO 要求的事：中                          |

> **范围。** 这是 kernel 的缺陷，不是一个开了 tuning 的构建实际会跑的东西。
> autotune 打开时，`conv_transpose2d` 的 tuner 在每个 dilation 上都会选 `col2im`，
> 根本走不到这个 kernel——见 [这不是什么](#这不是什么)。它在 tuner 不在场时才要紧：
> 不带 `autotune` feature 的构建里 `ConvTranspose2dStrategy` 默认是 `Direct`，而且
> 任何 `col2im` 接不下的形状也会落到这个 kernel 上。

## 那个循环

`conv_transpose2d_direct_kernel` 遍历候选输入位置，保留能被某个 kernel tap 命中的：

```rust
let kms_h = (kernel_h * args.dilation_0) as i32 - stride_0_i;
let y_start = ((out_y + args.padding_0) as i32 - kms_h) / stride_0_i;
let y_end   = clamp(kms_h + y_start + 1, 0, input.shape(2) as i32) as usize;

for in_y in y_start..y_end {
    let numerator_tmp = in_y * args.conv_stride_0;
    let numerator_h = numerator_h_base - numerator_tmp;

    if numerator_h_base >= numerator_tmp && numerator_h.is_multiple_of(args.dilation_0) {
        let kernel_y = numerator_h / args.dilation_0;
        ...                      // 乘加只在这里面
    }
}
```

跨度是 `kms_h + 1` = `kernel * dilation - stride + 1` 次迭代，其中只有 `kernel` 次
做乘加。其余的迭代执行一次运行时取模——除数是 kernel 参数，编译期折不掉——然后
直接落空。在 `kernel = 5, dilation = 4, stride = 1` 下，是 20 次迭代干 5 个 tap
的活。

## 症状

一次受控扫描：同样的张量、同样的输出形状、同样多的乘加，dilation 的代价用 padding
补掉，于是唯一变化的东西就是循环的长度。

| dilation | padding | 循环迭代 | 其中有用 | `conv_transpose2d_direct` | ms / 迭代 | `conv_transpose2d_col2im` |
| -------: | ------: | -------: | -------: | ------------------------: | --------: | ------------------------: |
|        1 |      10 |        5 |        5 |                 104.38 ms |     20.88 |                   4.70 ms |
|        2 |      12 |       10 |        5 |                 141.12 ms |     14.11 |                   5.01 ms |
|        4 |      16 |       20 |        5 |                 246.83 ms |     12.34 |                   5.60 ms |

`[128, 128, 176, 1] -> [128, 128, 160, 1]`，`kernel = 5`，单位 stride——一个 dilated
TCN 中最宽那层残差卷积的数据梯度。用 `cargo test` 测得，见
[如何测量](#如何测量)。

把它当一条直线读，而不是一个比值。用首末两行拟合 `ms = a + b * iterations`，得到约
56 ms 固定开销、每次迭代约 9.5 ms，据此预测中间那行误差在 8% 以内。所以：

- 这个 kernel **主要受循环长度支配**，而不是受算术支配。三行的算术量完全相同。
- `dilation = 4` 比 `dilation = 1` 多出的 15 次迭代花掉 **142 ms，占 kernel 的
  58%**——而它们一次乘法都没做。
- `dilation 4 / dilation 1` 是 2.36×，而不是纯循环受限会给出的 4×，因为存在固定
  开销。浪费无论如何都是真的；只是比值不是衡量它的正确方式。

最后一列是同一个梯度走这个目录 tuner 里的另一个候选，它**对 dilation 是平的**
（4.70 / 5.01 / 5.60 ms）。所以「带 dilation 的梯度」本身并不贵，而且这个 kernel
的天花板并不远：`dilation = 4` 时它是 `col2im` 的 44×，`dilation = 1` 时是 22×。

## 这不是什么

它不是一个开了 tuning 的构建实际会跑的东西。在每个 dilation 上测量，tuner 的选择
从时钟而不是日志读出：

| dilation | `direct` | `col2im` | `conv_transpose2d_autotune` | 收敛到   |
| -------: | -------: | -------: | --------------------------: | -------- |
|        1 | 104.20 ms |  4.68 ms |                     4.64 ms | `col2im` |
|        2 | 141.26 ms |  5.04 ms |                     5.08 ms | `col2im` |
|        4 | 247.34 ms |  5.54 ms |                     5.59 ms | `col2im` |

tuner 每次都选对了，而走到这个目录的数据梯度 `conv_data_backward_fallback` 收敛在
**5.56 ms**，不是 kernel 的 247 ms。

有两种读数会告诉你相反的结论，都不要信：

- **热身太短。** 梯度路径会走到 autotune，而 autotune 是异步收敛的；只热两次量到
  248 ms，热二十次量到 5.56 ms。前一个数字是飞行中的 tuner，不是梯度。
- **冷缓存。** 同一个机制的另一端：清掉持久缓存，最初几次调用跑的是临时候选。

所以不要把这件事写成「数据梯度慢了 20 倍」。事实是：一个开销为替代方案 44× 的
kernel，蹲在一个会避开它的 tuner 后面——直到 tuner 不在。

## 如何测量

```bash
cargo test -p burn-cubecl --release --features metal \
    dilation_loop_length -- --ignored --nocapture --test-threads=1
```

- `--release`，否则量出来的是宿主而不是 kernel。
- `--ignored`，测试被标成 ignored 是因为它需要真实设备，而且要跑几秒。
- `--nocapture`，打印出来的表格就是结果；断言只回答「问题还在不在」。
- `--test-threads=1`，两个测试共用一块 GPU 就是在互相测量。

把 `--features metal` 换成 `cuda`、`vulkan` 或 `wgpu` 可以读另一个后端。

**缺陷还在时这个测试是通过的。** 它失败的时候请读表格：那意味着多出来的迭代不再花
它们原来花的时间，也就是本文想要的结果。

测试刻意做的几件事：

- **直接调用 `conv_transpose2d_direct`。** 经 `conv_data_backward_fallback` 走过去
  会经过 `conv_transpose2d` 自己的 tuner，它在其中一些 dilation 上选 `col2im`、另一些
  上选这个 kernel——于是端到端扫描量的是「选择」，而不只是 kernel。
- **用 padding 补掉 dilation 的代价。** `padding = (176 - 160 + dilation * 4) / 2`
  是前向卷积输出尺寸公式的反解，所以三行都从同样的输入产出 `[128, 128, 160, 1]`。
  padding 只改变循环**从哪里开始**，从不改变它跑多长。
- **给浪费定价，而不是给比值定价。** `(dilation 4 的 ms - dilation 1 的 ms) / dilation 4 的 ms`
  就是 kernel 花在无所事事的迭代上的份额，它不依赖任何关于固定开销的模型。
- **先跑二十次不计时，再计时五次。** `col2im` 内部会走到 `matmul` 的 autotune，而
  tuning 是异步收敛的——收敛前一次调用跑的是 tuner 临时挑的候选。热身太短量到的是
  这个瞬态：只热两次的话，任何走到 autotune 的路径都可能差一个数量级。
- **把每次结果从计时闭包里返回**，这样它排进队列的东西不会被当作死代码消掉。

## 修复方案

### 遍历 tap，而不是候选位置

在 `stride = 1` 下——dilated TCN 全程都是这个——映射可以直接反解。输出位置 `out_y`
只会被满足 `out_y + padding - k * dilation == in_y` 的那个 tap `k` 命中，所以循环
应该跑 `k in 0..kernel`，由它算出 `in_y`，再做边界检查：

```rust
for k in 0..kernel_h {
    let num = numerator_h_base as i32 - (k * args.dilation_0) as i32;
    if num >= 0 && (num as usize).is_multiple_of(args.conv_stride_0) {
        let in_y = num as usize / args.conv_stride_0;
        if in_y < input.shape(2) { ... }
    }
}
```

循环长度变成 `kernel`，与 dilation 无关，而取模的除数变成 stride——常见情形可以用
`#[comptime] stride_is_one: bool` 或 `ComptimeOption` 在编译期特化掉。

预期：`dilation = 1` 不变，`dilation = 2` 和 `dilation = 4` 降到大致 `dilation = 1`
的开销。按上面的数字，`dilation = 4` 从 247 ms 降到约 105 ms。

`in_x` 那层循环同样处理；它在宽度轴上结构完全相同。

### 然后去做文件里自己那条 TODO 要求的事

即使修好，这个 kernel 在 `dilation = 1` 时仍是 `col2im` 的 22×。更大的收益是
`backward_data/fallback.rs` 已经写出来的那一条：

```rust
// We don't yet have NHWC kernels for conv_transpose so need to do this.
// Should eventually use NHWC kernels instead
let out_grad = permute_nhwc_to_nchw(out_grad);
let weights = permute_nhwc_to_nchw(weights);
```

一个 NHWC 的转置卷积能省掉这趟往返，也能不再把一个 1D 问题 reshape 成退化的
`width = 1` 2D 问题。再往远一点，数据梯度可以像权重梯度那样走到通用的
`im2col`/`col2im` + GEMM 路径——它目前的候选集里只有 `dgrad_im2col_1x1`，那只服务
逐点卷积。

### 如何验证修好了

跑上面的测量。目标是 **`direct ms` 那一列变平**：三个 dilation 在噪声范围内相等，
因为它们做的算术完全相同。此时 `ms / iteration` 那一列应当**随 dilation 上升**，
这正是「循环不再走它用不上的位置」的特征。
