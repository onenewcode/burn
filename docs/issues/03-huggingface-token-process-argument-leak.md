# HuggingFace token 作为子进程命令行参数传递，可被本机用户读取

## 严重级别

**P2**

在多用户机器、容器共享 PID namespace、CI 或带进程审计的环境中，token 会暴露给能够读取进程列表的其他主体；若相关审计日志记录完整 argv，泄露还会持久化。

## 问题描述

`HuggingfaceDatasetLoader::with_huggingface_token` 将 token 保存到结构体中。加载数据集时，`import` 函数把 token 作为 Python 子进程的命令行参数传递：

```rust
// crates/burn-dataset/src/source/huggingface/downloader.rs:239-242
if let Some(huggingface_token) = huggingface_token {
    command.arg("--token");
    command.arg(huggingface_token);
}
```

该进程随后执行：

```text
python3 importer.py ... --token <HF_TOKEN>
```

在 Unix-like 系统上，任何有权限读取 `/proc/<pid>/cmdline` 或调用 `ps` 的本机用户通常都能看到完整 argv。HuggingFace token 属于访问凭证，不应出现在进程命令行中。

## 最小复现

将以下内容保存为 `crates/burn-dataset/tests/huggingface_token_repro.rs`：

```rust
use std::fs;
use std::os::unix::fs::PermissionsExt;

#[test]
fn huggingface_token_is_passed_as_process_argument() {
    let home = tempfile::tempdir().unwrap();
    let bin = home.path().join("bin");
    let output = home.path().join("argv.txt");
    fs::create_dir(&bin).unwrap();

    let python = bin.join("python3");
    let script = format!(
        "#!/bin/sh\n\
         if [ \"$1\" = \"--version\" ]; then \
             echo 'Python 3.11.0'; \
         else \
             printf '%s\\n' \"$@\" > {:?}; \
         fi\n",
        output
    );
    fs::write(&python, script).unwrap();
    fs::set_permissions(&python, fs::Permissions::from_mode(0o755)).unwrap();

    let old_home = std::env::var("HOME").unwrap();
    let old_path = std::env::var("PATH").unwrap();
    unsafe {
        std::env::set_var("HOME", home.path());
        std::env::set_var("PATH", format!("{}:{}", bin.display(), old_path));
    }

    let result = burn_dataset::HuggingfaceDatasetLoader::new("dataset")
        .with_base_dir(home.path().to_str().unwrap())
        .with_huggingface_token("hf_secret_TOKEN_123")
        .with_use_python_venv(false)
        .db_file();

    unsafe {
        std::env::set_var("HOME", old_home);
        std::env::set_var("PATH", old_path);
    }

    result.unwrap();
    let argv = fs::read_to_string(&output).unwrap();
    println!("recorded argv: {argv}");
    assert!(argv.lines().any(|argument| argument == "hf_secret_TOKEN_123"));
}
```

运行：

```sh
cargo test -p burn-dataset --features sqlite \
  --test huggingface_token_repro -- --nocapture
```

## 实测输出

在当前 checkout（commit `88b8dd55e`）上执行：

```text
running 1 test
recorded argv: /var/folders/.../importer.py
--name
dataset
--file
/var/folders/.../dataset.db
--token
hf_secret_TOKEN_123

test huggingface_token_is_passed_as_process_argument ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

复现中的假 Python 解释器记录了实际 argv，证明 token 会原样出现在子进程命令行。

## 预期行为

认证 token 不应出现在任何子进程命令行、日志或可被 `ps` 读取的位置。

## 建议修复

使用标准输入、文件描述符或环境变量传递 token：

- **首选 stdin**：父进程向 Python importer 写入 token，importer 从 stdin 读取；
- **文件描述符**：传入 `/dev/fd/N`，并确保该 fd 不可被无关进程继承；
- **环境变量**：设置 `HF_TOKEN`，并使用 `Command::env`；仍需注意子进程及其后代不要转储环境；
- Python 端删除 `--token` 参数，避免 argparse 帮助信息或错误报告继续暴露该路径。

同时为该问题添加回归测试：假 Python 记录 argv，断言 secret 不在其中，并通过受控 stdin/env 验证子进程能收到 token。

