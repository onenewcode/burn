# [P2] 多线程 DataLoader 交错使用两个迭代器时可能永久阻塞

基线：`20426b9cd5fb547e247b05575cf771fd37676726`。状态：已验证第二个迭代器阻塞、释放第一个后恢复；顺序复用正常。

## 问题与影响

同一 MultiThreadDataLoader 的所有 iter 调用共享持久 worker。每个 worker 完整处理一个迭代命令后才接收下一个命令，每个迭代器的输出队列容量为 100。

保留未耗尽的第一个迭代器，再读取第二个时，如果前者通道填满，worker 阻塞在前者的 send，无法处理后者。同线程调用者若等第二个返回才继续第一个，就无法自行解除等待。

公开的 `iter(&self)` 允许这种共享借用，没有单活动迭代器限制或显式错误提示。训练中额外取样、预览或共享同一个 loader 时可能触发。

## 复现与结果

完整且可自行退出的程序：[repro/src/006.rs](repro/src/006.rs)。从仓库根目录运行：

```sh
cargo run --manifest-path docs/issues/repro/Cargo.toml --features runtime --bin issue-006 --target-dir /tmp/burn-issue-audit-target
```

200 个整数样本，batch size 1，1 个 worker。读取第一个迭代器的一个 batch 并保留它，再读取第二个。

```text
second iterator blocked while first is retained: true
after dropping first: Some(Ok([0]))
sequential reuse control: 200 batches
```

程序在另一线程等第二个迭代器，确认 1 秒内无结果后 drop 第一个，第二个恢复。另一个直接同线程调用 second.next 的初始实验在 5 秒观察期内未返回，已终止该复现进程。永久阻塞的判断同时依据下述通道依赖，不只依据计时。

预期：受支持的两个迭代器能够独立前进；如果实现限制单活动迭代器，应明确限制并在冲突时立即反馈，而不是无期限阻塞。

## 源码证据

- [multithread.rs:13](../../crates/burn-core/src/data/dataloader/multithread.rs#L13)：输出队列容量 100。
- [multithread.rs:198](../../crates/burn-core/src/data/dataloader/multithread.rs#L198)：worker 顺序接收命令，在内部循环阻塞发送所有 batch。
- [multithread.rs:243](../../crates/burn-core/src/data/dataloader/multithread.rs#L243)：每次 iter 将命令发往同一组 worker。

等待关系：调用者等第二个结果；worker 等第一个通道腾出空间；第一个通道需要调用者读取或释放。增大队列只会移动触发阈值。

## 修复方向与验收

让活动迭代器独立推进，或明确限制单活动迭代器并立即报告冲突；保留持久 worker 原有的资源复用目的。覆盖超过队列容量、同线程交错、跨线程共享、提前 drop 和顺序遍历，避免以无期限阻塞的方式处理调用冲突。
