# 远端提前断开后 `ExternalCommService::close` 会 panic

## 严重级别

**P2（常见网络断开触发关闭流程崩溃，并中断其余通道清理）**

## 问题描述

`ExternalCommService::download_tensor` 会把打开的 WebSocket 数据通道缓存在 `self.channels` 中：

```rust
// crates/burn-communication/src/external_comm.rs:177-199
async fn get_data_stream(
    &self,
    address: Address,
) -> Arc<Mutex<<P::Client as ProtocolClient>::Channel>> {
    let mut streams = self.channels.lock().await;
    match streams.get(&address) {
        Some(stream) => stream.clone(),
        None => {
            let stream = P::Client::connect(address.clone(), "data").await?;
            let stream = Arc::new(Mutex::new(stream));
            streams.insert(address.clone(), stream.clone());
            stream
        }
    }
}
```

如果远端在请求后关闭连接，`download_tensor` 只返回 `None`，不会从缓存移除这个已经结束的通道。

随后 `close()` 会遍历所有缓存通道并逐个调用 `close`，但对每个失败都直接 `expect`：

```rust
// crates/burn-communication/src/external_comm.rs:123-139
pub async fn close(&self) {
    let mut streams = self.channels.lock().await;
    for (_, stream) in streams.drain() {
        let mut stream = stream.lock().await;

        stream
            .close()
            .await
            .expect("Failed to close WebSocket stream");
    }
}
```

因此，一个已经完成关闭握手/已经收到 Close 的通道在执行 `stream.close()` 时会得到 tungstenite 的 `SendAfterClosing` 协议错误，进而 panic。由于循环没有错误隔离，排在它之后的其它通道也不会被清理。

这不是异常网络条件：对端崩溃、服务重启、任务完成或正常断开都会把缓存通道置为结束状态。之后调用公共的 `close()` 就可能触发 panic。

## 最小复现

将以下内容保存为 `crates/burn-remote/tests/external_comm_close_repro.rs`：

```rust
#![cfg(feature = "websocket")]

use std::sync::Arc;

use burn_communication::external_comm::ExternalCommService;
use burn_communication::websocket::{WebSocket, WsServer, WsServerChannel};
use burn_communication::{Address, CommunicationChannel, ProtocolServer};
use burn_flex::Flex;
use tokio_util::sync::CancellationToken;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn close_panics_when_cached_data_stream_has_ended() {
    let cancel = CancellationToken::new();
    let service = Arc::new(ExternalCommService::<Flex, WebSocket>::new(cancel.clone()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address: Address = format!("ws://{}", listener.local_addr().unwrap())
        .as_str()
        .into();

    let server = WsServer::new(0).route(
        "/data",
        |mut channel: WsServerChannel| async move {
            let _ = channel.recv().await;
            let _ = channel.close().await;
        },
    );
    let shutdown_token = cancel.clone();
    tokio::spawn(async move {
        let shutdown = async move {
            shutdown_token.cancelled().await;
        };
        server.serve_on(listener, shutdown).await.unwrap();
    });

    let downloader = service.clone();
    tokio::spawn(async move {
        let _ = downloader.download_tensor(address, 123_u64.into()).await;
    })
    .await
    .unwrap();

    let result = tokio::spawn(async move { service.close().await }).await;
    assert!(result.is_err(), "close unexpectedly succeeded");
    cancel.cancel();
}
```

运行：

```sh
cargo test -p burn-remote --test external_comm_close_repro --features websocket -- --nocapture
```

## 实测输出

在当前 checkout（commit `88b8dd55e`）上执行：

```text
running 1 test

thread 'tokio-rt-worker' panicked at crates/burn-communication/src/external_comm.rs:133:18:
Failed to close WebSocket stream: Tungstenite(Protocol(SendAfterClosing))
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
test close_panics_when_cached_data_stream_has_ended ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

测试断言 `service.close()` 所在任务 panic。正常远端关闭后调用 `close()` 本应完成清理而不是崩溃。

## 预期行为

- 对已经结束的通道调用 `close` 应视为幂等成功，或至少被记录并跳过；
- 一个通道关闭失败不应阻止其它通道继续清理；
- `close()` 不应因常见对端断开而 panic；
- `download_tensor` 发现连接结束后应移除失效缓存，或后续 `close()` 能安全处理该状态。

## 建议修复

把关闭结果从 `expect` 改为容错处理：

```rust
for (_, stream) in streams.drain() {
    let mut stream = stream.lock().await;
    if let Err(err) = stream.close().await {
        log::warn!("Failed to close WebSocket stream: {err:?}");
    }
}
```

如果需要保留错误信息，可把 `close` 改为 `async fn close(&self) -> Result<(), Error>`，并使用 `try_join_all` / loop 收集每个通道的结果；但当前 API 无返回值时至少不能 panic。

同时在 `download_tensor` 中处理 `Ok(None)` / receive error，移除该地址对应的失效通道，避免后续请求复用已关闭连接。

补充回归测试：

- 远端关闭后调用 `close()` 不 panic；
- 多个缓存通道中第一个已关闭时，其余通道仍被关闭；
- 失效通道不会在下一次 `download_tensor` 中被复用。
