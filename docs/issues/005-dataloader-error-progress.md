# [P2] 单线程 DataLoader 遇到永久读取错误后无限重复同一错误

基线：`20426b9cd5fb547e247b05575cf771fd37676726`。状态：已通过自定义 Dataset 实际复现。

## 问题与影响

`BatchDataloaderIterator::next()` 只有在 `get_many()` 成功后才推进索引。永久读取失败时，每次 next 都访问同一批索引。

公开接口以 Result 返回读取错误，调用者用 `filter_map(Result::ok)` 或 `continue` 跳过坏样本时会陷入无限循环，后续正常数据不可达。首次遇错就退出的调用者不受该循环问题影响。

## 复现与结果

完整程序：[repro/src/005.rs](repro/src/005.rs)。从仓库根目录运行：

```sh
cargo run --manifest-path docs/issues/repro/Cargo.toml --features runtime --bin issue-005 --target-dir /tmp/burn-issue-audit-target
```

Dataset 长度为 2，`get(0)` 永久返回 I/O 错误，`get(1)` 返回 42。使用 batch size 1 和默认单线程模式，连续调用三次 next：

```text
get(0)
next: Some(Err(... "bad sample" ...))
get(0)
next: Some(Err(... "bad sample" ...))
get(0)
next: Some(Err(... "bad sample" ...))
items_processed=0
```

预期：返回错误后按明确策略继续后续数据，或者结束迭代。未经配置的永久重试不应成为默认行为。

## 源码证据

- [batch.rs:176](../../crates/burn-core/src/data/dataloader/batch.rs#L176) 的 Err 分支直接返回。
- [batch.rs:180](../../crates/burn-core/src/data/dataloader/batch.rs#L180) 的 `current_index += chunk_size` 因此未执行。
- [dataset/iterator.rs](../../crates/burn-dataset/src/dataset/iterator.rs) 的基础 Dataset 迭代器在读取前推进索引，错误后仍可继续。

## 修复方向与验收

明确并统一错误后的策略。支持跳过失败批次时，也应推进失败索引并说明整批处理方式；采用 fail-fast 时，应返回一次错误后结束。避免无次数限制的隐式重试。

覆盖失败位于首批、中间批、末批，连续失败与自定义 batch strategy，确认遍历有限且进度不会永久停在同一点。
