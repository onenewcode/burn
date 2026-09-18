# autotune 选中更慢的那个卷积，因为它给候选计时计错了

|          |                                                              |
| -------- | ------------------------------------------------------------ |
| 状态     | open                                                         |
| 影响     | 一个普通的稠密卷积比候选集里**已经存在**的候选慢 **7.4×**     |
| 根因     | `cubecl-runtime` 的 autotune 计时；burn 侧看到的是它给出的选择 |
| 复现     | `src/kernel/conv/forward/tune.rs`，模块 `picks_the_slower`     |
| 难度     | 中——计时在上游，但 burn 侧可以拒绝一个它没理由相信的测量      |

## 症状

一个普通的稠密卷积，`[128, 128, 160] -> [128, 128, 176]`，`kernel = 5`、
`dilation = 4`、`padding = 16`、`groups = 1`、单位 stride。两个存活的候选各自
直接测量，旁边是 tuner 收敛后的选择：

| 路径                                | 中位数   | 速率         |
| ----------------------------------- | -------: | -----------: |
| `conv_direct`                       | 16.20 ms | 0.23 TFLOP/s |
| `simple_sync_cmma`                  |  2.20 ms | 1.68 TFLOP/s |
| **`conv_autotune`，收敛后**         | **16.32 ms** | **0.23 TFLOP/s** |
| `matmul`，同样 3,690,987,520 FLOP   |  2.01 ms | 1.84 TFLOP/s |

tuner 落在 `conv_direct` 上——每个输出元素一个线程、循环 `kernel * channels`——
白扔掉 7.4×。`simple_sync_cmma` 在同样的算术量下距通用矩阵路径不到 20%，也就是说
这个卷积有一个完全够用的候选，只是没被选上。

用 `cargo test` 测得，见 [如何测量](#如何测量)。

## 它为什么会选它

打开 `[autotune.logger] level = "full"`，同一个形状下 tuner 自己的记录是：

```text
Fastest result conv_direct-ConvAutotuneKey { kernel_size: [5], dilation: [4], ... }
Autotune[0] name conv_direct      => mean 7.157ms, median 8.988ms, variance 34.287µs, min 1.838ms, max 15.951292ms
Autotune[7] name simple_sync_cmma => mean 15.816ms, median 15.874ms, variance 17ns,    min 15.596ms, max 15.951292ms
```

把记录下来的数字和实测的放在一起：

| 候选               | tuner 记录 | 实际跑多久 |
| ------------------ | ---------: | ---------: |
| `conv_direct`      | 中位数 8.99 ms（min 1.84） | 16.20 ms |
| `simple_sync_cmma` |   15.87 ms |    2.20 ms |

**每个候选被记成了大致对方的开销。** tuner 不是噪声大，而是系统性地把工作归到了
错误的窗口里，于是它做出的决定恰好是反的。

两条线索说明这是「窗口张冠李戴」，不是测量抖动：

- **两个候选报出了完全相同的 `max`：15.951292ms，精确到纳秒。** 两个不同的窗口
  解析出同一个值，不可能是巧合——它们读的是同一对时间戳。
- **两者的分布形状截然相反。** `conv_direct` 是 0 号候选，每轮第一个被 launch；
  它的分布是 `min 1.838 / median 8.988 / max 15.951`，即大部分窗口抓到的是
  `simple_sync_cmma` 量级的工作，少数抓到了一整个 `conv_direct`。
  `simple_sync_cmma` 是 7 号；它和 0 号之间的候选（五个 depthwise 与
  `conv_im2col_1x1`）全部拒绝、什么也没 launch，所以在它之前最后一个真正排入队列
  的就是 `conv_direct`。它的分布紧紧地钉在 15.5～15.9 ms，`variance` 只有 17 ns
  ——就是 `conv_direct` 的开销，而且每一次都是。

换个跑次，`conv_direct` 记录的中位数会在 2 ms 和 9 ms 之间漂（取决于有几个窗口
抓到了自己的工作），而 `simple_sync_cmma` 始终稳定在 15.5 ms 上下。**稳定地读到
别人的耗时，比读不准更糟**：它每次都会输掉本该赢的比较。

### 两个候选的位置

`max` 完全相同这一点，把嫌疑指向设备计时（timestamp query）的时间戳配对，而不只是
tuner 的循环：

1. **`cubecl-wgpu` 的 query set 配对**（`src/compute/timings.rs`）。窗口打开时
   `start_profile` 只登记 token，真正的 `start` 要等到**下一个 compute pass 被创建**
   时由 `init_query_set` 赋值；窗口关闭时 `stop_profile_setup` 写的是
   `end = self.current`——也就是「最近一次创建的 query set」。一个候选在 setup 阶段
   就拒绝、没有 launch 任何 pass，`self.current` 就仍然指着上一个候选的 query set。
   这正好是「一个窗口解析出另一个窗口的时间戳」的形状，也解释了为什么两个分布共享
   同一个 `max`。
2. **tuner 的采样循环**（`cubecl-runtime/src/tune/tune_benchmark.rs`、`tuner.rs`）。
   `sample_once` 把每个候选包在 `client.profile` 里，round-robin 地交错采样，中间
   不排空队列。

两者都还是假设，不是结论。要分辨，直接复现归属错误即可：交替 profile 两个开销差
很多的 kernel、中间不排空，看每个窗口报的是不是自己那一个。上面的分布预测它报的是
前一个。

### 什么**不是**根因

`client.profile` 这个 API 单独调用时是准的。围绕这个卷积取十个窗口、按
`resolve_bench` 的方式解析，得到 `min 15.811 / median 15.836 ms`，而带显式同步的
挂钟是 `min 15.933 / median 16.217 ms`——一致到 1% 以内，且没有任何一个窗口解析成
`NotMeasured`。

所以单个窗口的原语本身没问题；出问题的是**窗口在背靠背、不排空的使用模式下的配对**。
这也是本文不提「去修 `client.profile`」的原因。

### 顺带一提：日志不会告诉你候选为什么拒绝

同一份日志里，所有在 setup 阶段就拒绝的候选都长这样：

```text
simple_sync_mma: An unknown error happened.
The profiled window resolved no device timing
```

真实原因（缺 tile size、缺 async barrier）被吞掉了。想知道原因得看
`accelerated-conv-candidates-decline-on-metal.md` 里那个测试，而不是日志。

## 如何测量

```bash
cargo test -p burn-cubecl --release --features metal \
    picks_the_slower -- --ignored --nocapture --test-threads=1
```

- `--release`，否则量出来的是宿主而不是 kernel。
- `--ignored`，测试被标成 ignored 是因为它需要真实设备。
- `--nocapture`，打印出来的表格就是结果；断言只回答「问题还在不在」。
- `--test-threads=1`，两个测试共用一块 GPU 就是在互相测量。

把 `--features metal` 换成 `cuda`、`vulkan` 或 `wgpu` 可以读另一个后端。

**缺陷还在时这个测试是通过的。** 它失败的时候请读表格：那意味着 tuner 落在了快
的候选上，也就是本文想要的结果。

测试刻意做的几件事：

- **从时钟而不是日志读取决定。** tuner 收敛到哪个候选，体现在一次 tuned 调用的开销
  上。不解析日志、不需要配置文件、什么都不用准备。
- **先跑二十次不计时，再计时五次。** tuning 是异步收敛的；在收敛之前，一次 tuned
  调用跑的是 tuner 临时挑的候选。只热两次量到的是这个瞬态，而不是决定。`matmul`
  同理，它内部也会 autotune。
- **把每次结果从计时闭包里返回**，这样它排进队列的东西不会被当作死代码消掉。

### 看到记录下来的数字

在二进制运行的目录下放一个 `cubecl.toml`（`cargo test -p burn-cubecl` 的工作目录
是 crate 根，即 `crates/burn-cubecl/`），并关掉持久缓存，让 tuning 真的发生：

```toml
[autotune]
disable_cache = true

[autotune.logger]
level = "full"
file = "/tmp/autotune.log"

[compilation.logger]
level = "disabled"

[profiling.logger]
level = "disabled"
```

```bash
# 跑任何会卷积的东西，然后：
grep -E 'Fastest result|^Autotune\[' /tmp/autotune.log
```

不要在缓存命中的情况下看这份日志：结果直接从缓存取时，日志里只有一行
`validate checksum`，没有任何候选的耗时——那一轮根本没有 tuning 发生。反过来，也不要
在缓存冷的情况下取**时间测量**：tuning 收敛之前，一次计时调用跑的是临时候选，读数
既不是 tuner 的选择也不是最好的选择。

## 修复方案

测量是 `cubecl-runtime` 的，所以修复在上游。burn 能做的是：不再对一个它有理由怀疑
的测量采取行动。

### 上游：让每个窗口归属于它自己的 launch

1. **先直接复现归属错误。** 交替 profile 两个开销差很多的 kernel、中间不排空，检查
   每个窗口报的是不是自己那一个。上面的分布预测它报的是前一个。
2. **让窗口等它所包住的工作。** 不管用什么机制，一个窗口在它内部排入的工作被打上
   时间戳之前不能解析完成。单独使用时 `client.profile` 与挂钟一致，说明原语做得到；
   被绕开的地方在交错采样这条路径上。
3. **别让 `self.current` 充当 `end`。** 一个什么都没 launch 的窗口应当解析为
   「未测量」，而不是去读上一个候选的 query set。

### 上游：拒绝一个自相矛盾的测量

一个跑在毫秒量级的 kernel，`max` 是 `min` 的 8.7×，这不叫测量过了，这叫被一个移动
的窗口采样过了。`BenchmarkComputations` 本来就带着分布，所以 tuner 可以拒绝在它上面
做决定：重新采样，或者退回到带显式同步的挂钟计时。**一个略微偏大但一致的数字，比一个
偶尔小 8 倍的数字值钱**，因为后者会输掉每一场它本该赢的比较。

只要 tuner 还在坏数据上做选择，**每一个新加进来的卷积候选都得去打败一个被低报的
`conv_direct`**——一个更快的 kernel 可以被加进去，然后看起来还是输了。这就是为什么
卷积路径上的其它事情都该排在这件事后面。

### burn 侧：把决定记在能被检查的地方

`conv_autotune` 的选择目前只有在进程启动前就打开 autotune logger 才看得见。如果能在
运行时读到「收敛的 key 旁边是哪个候选」，像本文这样的测试就可以断言**选了哪个候选**，
而不是从耗时反推。

### 如何验证修好了

跑上面的测量。目标是 **`conv_autotune (settled)` 那一行落到 `simple_sync_cmma` 旁边**
——在这台设备上是约 2.2 ms 而不是 16 ms。一个正确的 tuner 不可能比它最好的候选更好，
所以那一行就是全部的验收标准。

然后用日志核对记录下来的数字：`conv_direct` 应当读到约 16 ms，`simple_sync_cmma`
约 2 ms，各自分布收紧，且两者的 `max` 不再相同。
