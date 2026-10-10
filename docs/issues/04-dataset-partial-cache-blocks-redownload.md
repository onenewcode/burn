# 部分写入的数据集缓存会被误判为完整缓存，阻止自动修复

## 严重级别

**P2**

下载或解包中断后，后续运行不会重新下载，而是永久使用不完整缓存，直到用户手动删除缓存。该问题影响数据可用性，并容易被误解为数据集解析错误。

## 问题描述

内置数据集下载逻辑使用“目标文件/目录是否存在”作为完整缓存标志，但下载和解包直接写入最终路径，没有临时文件、原子替换或完成标记。

AG News 的逻辑如下：

```rust
// crates/burn-dataset/src/nlp/ag_news.rs:94-108
let cache_dir = dirs::cache_dir()
    .expect("Could not get cache directory")
    .join("burn-dataset");
let agnews_dir = cache_dir.join("ag_news_csv");

if !agnews_dir.exists() {
    let bytes = downloader::download_file_as_bytes(url, filename);
    let gz_buffer = GzDecoder::new(&bytes[..]);
    let mut archive = Archive::new(gz_buffer);
    archive.unpack(cache_dir).unwrap();
}
```

如果网络中断、进程被杀或解包中途失败，`ag_news_csv` 目录可能已经被创建但缺少 `train.csv` / `test.csv`。下一次调用 `AgNewsDataset::new()` 时：

1. `agnews_dir.exists()` 为 true；
2. 下载分支被跳过；
3. `train()` / `test()` 尝试读取不存在的 CSV 并 panic。

CIFAR 使用相同的目录存在性判断和解包方式。MNIST 则按最终文件路径写入，若文件写入中途失败，后续 `file_name.exists()` 也会跳过下载。

## 最小复现

将以下内容保存为 `crates/burn-dataset/tests/partial_ag_news_cache_repro.rs`：

```rust
use std::fs;

#[test]
fn partial_ag_news_cache_is_never_repaired() {
    let home = tempfile::tempdir().unwrap();
    let cache = home.path().join("Library").join("Caches");
    let partial = cache.join("burn-dataset").join("ag_news_csv");
    fs::create_dir_all(&partial).unwrap();

    let old_home = std::env::var("HOME").unwrap();
    unsafe {
        std::env::set_var("HOME", home.path());
    }

    let dataset = burn_dataset::nlp::AgNewsDataset::new();

    unsafe {
        std::env::set_var("HOME", old_home);
    }

    let panic = std::panic::catch_unwind(|| dataset.train());
    let error = if let Err(error) = panic {
        error
    } else {
        panic!("partial cache should make train panic");
    };

    let message = error
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| error.downcast_ref::<&str>().map(|message| message.to_string()))
        .unwrap_or_default();
    assert!(message.contains("Failed to parse CSV file"));
}
```

运行：

```sh
cargo test -p burn-dataset --features builtin-sources \
  --test partial_ag_news_cache_repro -- --nocapture
```

## 实测输出

在当前 checkout（commit `88b8dd55e`）上执行：

```text
running 1 test

thread 'partial_ag_news_cache_is_never_repaired' panicked at
crates/burn-dataset/src/nlp/ag_news.rs:124:48:
Failed to parse CSV file: Custom { kind: Other, error: Error(Io(Os {
  code: 2,
  kind: NotFound,
  message: "No such file or directory"
})) }

test partial_ag_news_cache_is_never_repaired ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

注意：该测试没有网络访问，也没有提供任何完整数据集文件，仅凭一个空目录就让 `AgNewsDataset::new()` 跳过了下载逻辑。

## 预期行为

部分缓存应在下一次调用时被清理并重新下载，或至少返回明确错误，指导用户修复缓存。不应把它当作完整缓存继续使用。

## 建议修复

- 将下载写入临时文件，解包写入临时目录。
- 校验解包结果包含必需文件、文件大小/记录数和非空内容。
- 全部校验通过后，将临时目录/文件原子重命名到最终缓存路径。
- 失败时清理临时路径，不影响已有完整缓存。
- 对 MNIST 的单个文件也使用“临时文件 + 原子重命名 + 完整性校验”。
- 为 AG News、CIFAR 和 MNIST 添加部分缓存回归测试。

