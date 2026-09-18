# Metal 上八个卷积候选里有六个拒绝

|          |                                                                  |
| -------- | ---------------------------------------------------------------- |
| 状态     | open                                                             |
| 影响     | Apple GPU 上只剩一个加速候选存活，autotune 几乎没得选              |
| 文件     | cubek 的卷积/matmul tile 选择；`cubecl-cpp` 的 Metal dialect 能力声明 |
| 复现     | `src/kernel/conv/forward/implicit_gemm/launch.rs`，模块 `candidate_survival` |
| 难度     | tile 目录：中；async barrier：高                                  |

## 症状

一个普通的稠密卷积——`kernel = 5`、`dilation = 4`、`groups = 1`、单位 stride、
128 通道、每个维度都是 8 的倍数——在 `metal`（cubecl/wgpu → MSL）上逐个交给每个
前向候选：

```text
conv_direct          ok
conv_im2col_1x1      declined: Unknown
simple_sync_cmma     ok
simple_sync_mma      declined: No tile size is available for the problem.
simple_async_cmma    declined: "Async barrier instructions are not available on the current device"
simple_async_mma     declined: No tile size is available for the problem.
simple_tma_cmma      declined: "Async barrier instructions are not available on the current device"
simple_tma_mma       declined: No tile size is available for the problem.
-> 2 of 8 survived
```

两个存活者里，`conv_direct` 是朴素 kernel——每个输出元素一个线程，循环
`kernel * channels`——而 `conv_im2col_1x1` 对任何非 1x1 的 kernel 按定义就会拒绝。
所以**真正存活的加速候选恰好只有一个**：`simple_sync_cmma`。

用 `cargo test` 测得，见 [如何测量](#如何测量)。`forward/tune.rs` 里那五个 depthwise
tunable 不计入：它们按设计会拒绝任何不是「每通道一个 filter」的形状，稠密形状从来
就没有过它们。

## 两种失败，两个不同的答案

六个拒绝干净地分成两类，把它们混在一起是要避免的错误：

**三个说「No tile size is available for the problem」**——`simple_sync_mma`、
`simple_async_mma`、`simple_tma_mma`。所有 `mma` 变体，且仅有它们。

这一类值得追，因为障碍不在设备：

- `cubecl-cpp` 的 Metal dialect（`src/metal/mma.rs`）在 **8x8** 矩阵上发射
  `simdgroup_load`、`simdgroup_multiply_accumulate` 和 `simdgroup_store`。硬件有矩阵
  单元，代码生成器也能用它。文件里自己写着：as of Metal 3.2 只支持 8x8x8 的
  fragment，别的尺寸直接 panic。
- `simple_sync_cmma` 用的就是同一批 `simdgroup_matrix` 操作，而且**能跑**，在这台
  设备上是 1.68 TFLOP/s，对照同 FLOP 下通用 `matmul` 的 2.02 TFLOP/s。

所以 Metal 上的矩阵路径是通的。「No tile size is available」是关于**选择器目录**的
陈述，不是关于设备的：`Mma` 这条 tile-kind 路径大概在提供 16x16 或 16x8——常见的
NVIDIA 形状——而 Metal 一个都不匹配。

**两个说「Async barrier instructions are not available on the current device」**
——`simple_async_cmma` 和 `simple_tma_cmma`。加上上面的 `mma` 变体，就是所有 `async`
和 `tma` 候选。

这一类是真正的能力缺口。Metal 没有等价的异步 barrier 语义，`cubecl-cpp` 的 Metal
dialect 也没有声称有。让它们复活意味着为 Metal 写一条同步的双缓冲路径，那是另一件
更大的工程。

## 为什么它没看起来那么要紧

唯一存活的加速候选是*好的*：`simple_sync_cmma` 跑这一层用 2.20 ms，对照同 FLOP 的
`matmul` 的 1.83 ms。Metal 并不缺卷积性能。

它缺的是**余量**。只有一个加速候选时：

- 任何被这个候选拒绝的形状都会落到 `conv_direct`，速率大约是八分之一。
- autotune 只有一个真正的决定要做，而在这个后端上它目前做错了——对这个形状它选了
  `conv_direct`。见 `autotune-picks-the-slower-convolution.md`，那才是今天真正在
  花钱的缺陷。

先修选择；这件事拓宽的是「选择在多大的集合上做」。

补一句判断依据：`mma` 和 `cmma` 是同一套 implicit-GEMM 算法的两条 tile-kind 路径，
驱动的是同一批 `simdgroup_matrix` 指令，所以「补上 8x8 目录」换来的余量是**推测性
的**——它大概会接受和 `cmma` 差不多的形状集合。这也是本条排在选择缺陷后面的原因。

## 如何测量

```bash
cargo test -p burn-cubecl --release --features metal \
    candidate_survival -- --ignored --nocapture --test-threads=1
```

- `--nocapture`，打印出来的表格就是结果。
- `--ignored`，测试被标成 ignored 是因为它需要真实设备。
- `--release` 只是为了构建快；这里没有计时。每个候选被调用一次、记录其结果，完全
  就是 autotune 会看到的东西。

**这个测试与设备相关。** 缺陷存在时它通过，所以在一块加速候选都能编译的 GPU 上
——较新的 NVIDIA 卡——它*应该*失败。那不是回归，那说明问题在那里不存在。先读表格，
再看断言。

它断言两件事：至少五个候选拒绝，以及至少一个是因为缺 tile size 而拒绝。第二条才是
可行动的部分，单独检查是为了：即使 async 变体永远不可用，tile 目录被修好这件事也
看得见。

顺带一句，为什么这件事非得用测试读、不能用 autotune 日志读：在日志里，所有在 setup
阶段拒绝的候选都写成 `An unknown error happened. / The profiled window resolved no
device timing`，真实原因被吞掉了。

## 修复方案

### tile 目录

1. **找到拒绝是在哪里抛出的。** `No tile size is available for the problem` 来自
   cubek 的 `matmul/src/definition/error.rs`。往回找到为
   `AcceleratedTileKind::Mma` 枚举 tile 尺寸的那段代码，把它考虑过哪些尺寸、每个
   为什么被否掉打出来。
2. **和 Metal 声明的对照。** `cubecl-cpp/src/metal/mma.rs` 只支持 8x8，别的没有。
   如果选择器要求 16x16 或 16x8，那么不管设备报告什么，在 Apple 上都匹配不到任何
   东西，而把 8x8 加进目录就是全部的修复。
3. **拿 `cmma` 校对。** `simple_sync_cmma` 已经在成功驱动同一批 `simdgroup_matrix`
   操作，所以它的配置就是「Metal 接受什么」的一个可用参照。

### async 与 TMA 变体

把它们在 Metal 上声明为不支持，并从该后端的候选集里去掉，而不是每遇到一个新形状就
编译一遍、失败一遍。现在每一个不同的卷积形状都要为四个跑不了的候选付编译时间，并
往 autotune 日志里写四行错误让人去读。

如果哪天为 Metal 写了同步的双缓冲路径，那时它们再回来。

### 如何验证修好了

跑上面的测量。目标是**至少一个 `mma` 变体能 setup**，也就是第二条断言翻转。然后
去测它：能编译的候选还不等于快的候选，而在这个后端上 autotune 自己的 benchmark
不能用来告诉你哪个快——用带同步的挂钟，就像
`src/kernel/conv/forward/tune.rs` 里的 `picks_the_slower` 那样。
