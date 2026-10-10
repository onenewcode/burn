# 通用网络下载器不校验 HTTP 状态，缺失 `Content-Length` 时直接 panic

## 严重级别

**P2**

## 问题描述

`burn-std::network::downloader::download_file_as_bytes` 是多个数据集和预训练权重的公共下载入口，但实现存在两个核心缺陷：

```rust
// crates/burn-std/src/network.rs:20-53
#[tokio::main(flavor = "current_thread")]
pub async fn download_file_as_bytes(url: &str, message: &str) -> Vec<u8> {
    let mut response = Client::new().get(url).send().await.unwrap();
    let total_size = response.content_length().unwrap();

    // ...

    let mut bytes: Vec<u8> = Vec::with_capacity(total_size as usize);
    while let Some(chunk) = response.chunk().await.unwrap() {
        let num_bytes = bytes.write(&chunk).unwrap();
        // ...
    }

    bytes
}
```

问题：

1. **没有检查 `response.status()`**：`404`、`403`、`500` 等错误响应的 body 会被当作文件内容返回。调用方随后可能把 HTML/JSON 错误页写入缓存或交给 gzip/PyTorch 解析器，产生误导性失败。
2. **`content_length().unwrap()`**：HTTP chunked transfer、某些 CDN、代理和动态响应可能不提供 `Content-Length`。这是合法响应，但当前实现直接 panic。
3. **没有下载大小上限**：`Content-Length` 直接转换为 `Vec::with_capacity`，恶意或异常服务器可诱导大额内存分配。

该函数被 MNIST、CIFAR、AG News、LPIPS/FID/AFINE 等预训练权重下载路径使用，因此影响的是真实公共下载路径，而不是孤立测试代码。

## 最小复现

在 `crates/burn-std/tests/network_downloader_repro.rs` 中加入：

```rust
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

fn spawn_response(response: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();

    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0_u8; 1024];
        let _ = stream.read(&mut request).unwrap();
        stream.write_all(response.as_bytes()).unwrap();
    });

    format!("http://{address}/file")
}

#[test]
fn chunked_response_without_content_length_panics() {
    let url = spawn_response(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n",
    );

    let result = std::panic::catch_unwind(|| {
        burn_std::network::downloader::download_file_as_bytes(&url, "repro");
    });

    assert!(result.is_err(), "expected downloader to panic");
}

#[test]
fn http_error_body_is_returned_as_successful_download() {
    let url = spawn_response(
        "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found",
    );

    let bytes = burn_std::network::downloader::download_file_as_bytes(&url, "repro");
    assert_eq!(bytes, b"not found".as_slice());
}
```

运行：

```sh
cargo test -p burn-std --features network --test network_downloader_repro -- --nocapture
```

## 实测输出

在当前 checkout（commit `88b8dd55e`）上执行：

```text
running 2 tests

thread 'chunked_response_without_content_length_panics' panicked at
crates/burn-std/src/network.rs:24:52:
called `Option::unwrap()` on a `None` value

test http_error_body_is_returned_as_successful_download ... ok
test chunked_response_without_content_length_panics ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

说明：

- 第一个测试捕获到了真实 panic：合法 chunked 响应缺少 `Content-Length`。
- 第二个测试通过表示 `404 Not Found` body 被当作成功下载数据返回。

## 预期行为

- 非 2xx 响应应返回错误，错误信息包含 HTTP 状态码和 URL。
- 缺失 `Content-Length` 的 2xx 响应应继续流式读取，或返回明确错误；不应 panic。
- 应设置最大下载大小，避免异常 `Content-Length` 造成无上限内存分配。
- 网络请求、流读取和写入错误应作为 `Result` 返回给调用方。

## 建议修复

将公共 API 改为返回 `Result<Vec<u8>, DownloadError>`：

```rust
pub async fn download_file_as_bytes(
    url: &str,
    message: &str,
) -> Result<Vec<u8>, DownloadError>
```

具体处理：

1. `response.error_for_status()`；
2. `content_length()` 为 `None` 时使用动态进度或未知总量进度；
3. 对 `content_length` 和实际读取字节数设置上限；
4. 逐块读取并累积，不因单次 chunk 失败直接 `unwrap`；
5. 为缺失长度、HTTP 错误、超过大小上限、IO 错误分别建立错误类型。

调用方（数据集和权重加载器）也应改为在下载失败时不写入缓存文件，避免半成品文件被后续运行误判为已缓存。

