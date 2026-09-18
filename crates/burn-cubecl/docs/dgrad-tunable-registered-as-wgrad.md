# 数据梯度的 fallback 被注册成了 `wgrad_fallback`

|          |                                                        |
| -------- | ------------------------------------------------------ |
| 状态     | open                                                   |
| 影响     | 运行时无影响；autotune 日志把一个 dgrad 的决定归给了 wgrad |
| 文件     | `src/kernel/conv/backward_data/tune.rs`                |
| 复现     | 同一文件，模块 `dgrad_tunable_name`                     |
| 难度     | 一个字符串加一个函数名                                  |

## 问题

`backward_data/tune.rs` 是**数据**梯度的 tuner。它的第一个候选：

```rust
let tunables = TUNER.init(&tune_id, || {
    TunableSet::new(create_key::<N>, create_wgrad_input::<N>)   // <- wgrad
        .with(Tunable::new(
            "wgrad_fallback",                                   // <- wgrad
            |(out_grad, weights, input_shape, options)| {
                conv_data_backward_fallback::<N>(out_grad, weights, input_shape, options)
            },                                                  // <- 跑的是 dgrad
        ))
```

这个 tunable 以权重梯度命名，跑的是数据梯度。构造它输入的那个函数也一样。

## 为什么值得修

autotune logger 会打印 tunable 的名字。给一个 dilated 网络的反向过程调 tuning，
得到的是这样：

```text
Fastest result wgrad_fallback-ConvAutotuneKey { kernel_size: [5], ... dilation: [4], ... }
 - Tuning: wgrad_fallback (compilation & bench: 5.304845084s)
 - Tuning: dgrad_im2col_1x1 ...
```

赢家是 `wgrad_fallback`，它下面一个候选却是 `dgrad_*`。读的人要么以为自己在看一个
权重梯度的决定——然后跑去 `backward_weight/`，那里确实有一个真的 `wgrad_fallback`
——要么以为两个 tuner 的结果被交错在一起了。

**这个名字不只是写错了，它是被占用的。** `backward_weight/tune.rs` 为
`conv_weight_backward_fallback` 注册了 `wgrad_fallback`。于是两个 tuner 会为完全
不同的 kernel 打印同一个获胜名字，日志根本说不出是哪一个赢了。

这件事在运行时不花一分钱，在「读日志查反向为什么慢」的人身上花掉全部。

## 修复

在 `src/kernel/conv/backward_data/tune.rs` 里：

- `"wgrad_fallback"` → `"dgrad_fallback"`
- `create_wgrad_input` → `create_dgrad_input`

两个都要改，不是改一个：下面的测试会检查它们保持一致，因为只改一半等于把同样的
困惑挪到另一个地方。

改名的影响范围已经确认过：`create_wgrad_input` 声明为 `pub fn`，但
`crates/burn-cubecl/src/kernel/conv/mod.rs` 里 `backward_data` 是私有模块，所以它在
crate 外不可达；全仓库对它的唯一引用就是上面那行 `TunableSet::new`。

tunable 名字会进 autotune 缓存的 checksum，所以改名会让已有的持久缓存条目失效，
触发一轮重新 tuning。这是一次性的，而且是正确行为——缓存的键基于一组候选，而这组
候选的身份变了。

## 如何测量

没什么可测的。这个缺陷是一个字符串，它的代价由读日志的人支付。测试不需要设备也不
花时间：

```bash
cargo test -p burn-cubecl --features metal dgrad_tunable_name -- --nocapture
```

`--nocapture` 会打印候选清单和这次撞名，那是值得看的部分：

```text
=== what this tuner calls its own candidates ===
  "wgrad_fallback",
  "dgrad_im2col_1x1",
  "simple_sync_cmma",
  ...
  -> the first of these runs conv_data_backward_fallback

=== the collision ===
  backward_data/tune.rs    "wgrad_fallback" -> conv_data_backward_fallback
  backward_weight/tune.rs  "wgrad_fallback" -> conv_weight_backward_fallback
  -> a log naming the winner cannot say which tuner won
```

**缺陷还在时这两个测试是通过的。** 改名落地后它们会失败，那时这份文档就可以退休。

它们读的是文件自己的源码文本，这一点需要明说：tunable 的名字没有别的途径拿得到。
`TuneFn::name` 对 `cubecl-runtime` 私有，`TunableSet` 不暴露访问器，而这个集合是在
`dgrad_autotune` 内部就地构造的、不返回。替代方案——用
`[autotune.logger] level = "full"` 跑一遍 tuner 再 grep 日志文件——为了断言一个
字符串，需要一块设备、一个配置文件和一次清空的缓存。

只匹配本文件测试模块之前的部分，所以测试找到的是注册处，永远不会是它自己关于注册的
那段文字。

### 如何验证修好了

```bash
cargo test -p burn-cubecl --features metal dgrad_tunable_name
```

两个测试都应当失败，各自点名这份文档。然后删掉那个模块和这个文件。

想看日志变对，在工作目录放一个 `cubecl.toml`：

```toml
[autotune]
disable_cache = true

[autotune.logger]
level = "full"
file = "/tmp/autotune.log"
```

```bash
# 跑任何一个卷积反向过程，然后：
grep 'Fastest result' /tmp/autotune.log
```

一个数据梯度的决定应当点名 `dgrad_fallback`。
