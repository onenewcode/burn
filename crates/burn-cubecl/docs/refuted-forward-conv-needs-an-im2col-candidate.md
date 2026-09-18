# 已推翻：前向卷积需要一个 im2col + GEMM 候选

|          |                                                          |
| -------- | -------------------------------------------------------- |
| 状态     | **不是问题——不要实现它**                                  |
| 为何保留 | 这个推理很有说服力，源码支持它，端到端的数字看起来也在确认它 |
| 真正原因 | `autotune-picks-the-slower-convolution.md`               |

## 假设

Apple GPU 上一个稠密卷积的速率，大约是本 crate 自己的 `matmul` 在同 FLOP 下的
八分之一：

| 路径                              | 中位数   | 速率         |
| --------------------------------- | -------: | -----------: |
| 卷积，实际执行的那条               | 14.71 ms | 0.25 TFLOP/s |
| `matmul`，同样 3,690,987,520 FLOP |  1.83 ms | 2.02 TFLOP/s |

而 `src/kernel/conv/forward/tune.rs` 里的候选集看起来正好能解释这件事。对一个
`groups = 1`、`kernel = 5` 的卷积，五个 depthwise tunable 按设计拒绝，
`conv_im2col_1x1` 也拒绝——`check_pointwise_strided` 要求每个 kernel 维度都是 1、
没有 padding、没有 dilation。所以这个集合里根本没有「针对一般 kernel 尺寸的
im2col + GEMM」候选。

同时 `src/kernel/conv/im2col.rs` 里有一个**通用**的 `im2col()`——任意 kernel 尺寸、
任意 padding、任意 dilation——而且它已经在隔壁一个调用方那里投入使用：同一文件里的
`wgrad_im2col()` 把权重梯度降解成 `im2col` + `matmul`，由 `backward_weight/tune.rs`
注册。同一个文件、同一个函数；权重梯度调它，前向不调。

结论自己就写出来了：加一个通用的前向 `conv_im2col`，注册上，卷积就走上矩阵路径。

## 它为什么是错的

**前向候选集里已经有一个跑在 GEMM 速率上的候选，而且它能用。** 在同一层上直接测量：

| 候选               | 中位数   | 速率         |
| ------------------ | -------: | -----------: |
| `conv_direct`      | 14.69 ms | 0.25 TFLOP/s |
| `simple_sync_cmma` |  2.20 ms | 1.68 TFLOP/s |
| `matmul`，同 FLOP  |  1.83 ms | 2.02 TFLOP/s |

`simple_sync_cmma`——通过 tiled matmul 组件做的 implicit GEMM——距通用矩阵路径不到
20%。没有什么是 im2col 候选能补上的：这个卷积**今天就能**用 2.2 ms 跑完，用的是
集合里已有的代码，不需要物化任何列矩阵。

14.71 ms 不是候选集的错。它是 autotune 选出来的：它把 `conv_direct` 记成
1.985 ms、把 `simple_sync_cmma` 记成 15.571 ms——大致是两个候选各自被记成了对方的
开销——然后照此选择。见 `autotune-picks-the-slower-convolution.md`。

加一个 `conv_im2col` 等于为了达到集合已经不需要它就能达到的速率，去物化一个
57.7 MB 的列矩阵——而且由于 tuner 的数字就是那个样子，新候选还得去打败一个被低报的
`conv_direct` 才能被选上。它可以更快，然后照样输。

## 那个陷阱

支持这个假设的证据来自权重梯度那一侧的对比，那里 im2col 路径和直接路径都已经实现：

| 路径                            | 中位数（热 2 次） | 中位数（热 20 次） |
| ------------------------------- | ----------------: | -----------------: |
| `wgrad_im2col`                  |           4.07 ms |            4.08 ms |
| `conv_weight_backward_fallback` |          43.17 ms |            3.81 ms |

那 10.6× 的优势完全是热身不足造成的假象。`conv_weight_backward_fallback` 会调
`conv_forward_nhwc`，后者会 autotune，而 tuning 是异步收敛的——最初若干次调用里
fallback 跑的是临时候选。收敛之后，它并不比 im2col 路径慢。

所以：**任何会走到 autotune 的东西，比较之前先热到内部 tuner 收敛。** 这里二十次
够了，两次不够，而两次给出的数字指向了错误的修复方向。

## 从这件事里该带走什么

- Metal 上的卷积候选集是够用的。坏的是那个*选择*，而那在隔壁一份文档里。
- 一条需要物化自己输入的路径，值得在「别的路径都到不了矩阵路径」时加进来。这里有
  路径到得了。
- 不要在 autotune 缓存冷的情况下比较两条路径。看起来更快的那条，可能只是内部 tuner
  先收敛完的那条。
