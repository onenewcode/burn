# WebSocket tensor 数据服务的坏请求可导致 handler 永久阻塞或 panic

## 严重级别

**P2（远程/损坏客户端可造成数据服务 handler 挂起或崩溃，影响服务可用性）**

## 问题描述

`ExternalCommService::handle_data_channel` 直接反序列化网络消息，并对所有错误调用 `expect` 或 `panic`：

```rust
// crates/burn-communication/src/external_comm.rs:236-255
while !self.cancel_token.is_cancelled() {
    match channel.recv().await {
        Ok(message) => {
            if let Some(msg) = message {
                let bytes = msg.data;
                let msg: ExternalCommMessage = rmp_serde::from_slice(&bytes)
                    .expect("Can deserialize messages from the websocket.");
                let ExternalCommMessage::TensorRequest(transfer_id) = msg else {
                    panic!("Received a message that wasn't a tensor request! {msg:?}");
                };

                let bytes = self.get_exposed_tensor_bytes(transfer_id).await.unwrap();
                channel.send(Message::new(bytes)).await.unwrap();
            }
            // ...
        }
        Err(err) => panic!("Failed to receive message from websocket: {err:?}"),
    };
}
```

这里有三个独立故障点：

- 任意非法 MessagePack 或非法枚举值会使 handler panic；
- 不存在的 `transfer_id` 进入 `get_exposed_tensor_bytes` 的无限循环；
- 该循环内部不检查 `cancel_token`，因此取消服务后已阻塞的 handler 仍无法退出。

`get_exposed_tensor_bytes` 的等待逻辑：

```rust
// crates/burn-communication/src/external_comm.rs:203-226
loop {
    {
        let mut exposed_tensors = self.exposed_tensors.lock().await;
        if let Some(mut exposed_state) = exposed_tensors.remove(&transfer_id) {
            // ...
        }
    }
    // No matching tensor, wait for a new one to come in.
    self.new_tensor_notify.notified().await;
}
```

与之相比，Iroh 传输为等待 transfer 设置了超时，并在超时后返回错误；WebSocket 数据服务没有任何等待上限。

该问题不等同于 `/data` 路由缺失鉴权：即使路由未来增加鉴权，已授权客户端的 bug、协议不兼容或连接损坏仍会触发同样的 handler 挂起或 panic。当前无鉴权路由会放大攻击面。

## 最小复现

将以下内容保存为 `crates/burn-remote/tests/data_handler_fault_repro.rs`：

```rust
#![cfg(feature = "websocket")]

use std::sync::Arc;
use std::time::Duration;

use burn_communication::external_comm::{
    ExternalCommServer, ExternalCommService, TensorTransferId,
};
use burn_communication::websocket::{WebSocket, WsClient, WsServer};
use burn_communication::{Address, CommunicationChannel, Message, ProtocolClient};
use burn_flex::Flex;
use burn_tensor::TensorData;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

#[derive(Deserialize, Serialize)]
enum WireMessage {
    TensorRequest(TensorTransferId),
    Tensor(TensorData),
}

async fn start_data_server() -> (CancellationToken, Address) {
    let cancel = CancellationToken::new();
    let service = Arc::new(ExternalCommService::<Flex, WebSocket>::new(cancel.clone()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address: Address = format!("ws://{}", listener.local_addr().unwrap())
        .as_str()
        .into();
    let server = WsServer::new(0).route_external_comm(service);

    let shutdown_token = cancel.clone();
    tokio::spawn(async move {
        let shutdown = async move {
            shutdown_token.cancelled().await;
        };
        server.serve_on(listener, shutdown).await.unwrap();
    });

    (cancel, address)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_transfer_id_blocks_even_after_cancellation() {
    let (cancel, address) = start_data_server().await;
    let mut channel = WsClient::connect(address, "data").await.unwrap();
    let request = rmp_serde::to_vec(&WireMessage::TensorRequest(999_u64.into())).unwrap();
    channel
        .send(Message::new(request.into()))
        .await
        .unwrap();

    let before_cancel = timeout(Duration::from_millis(500), channel.recv()).await;
    assert!(before_cancel.is_err(), "unknown transfer ID unexpectedly responded");

    cancel.cancel();
    let after_cancel = timeout(Duration::from_millis(500), channel.recv()).await;
    assert!(
        after_cancel.is_err(),
        "handler exited after cancellation instead of remaining blocked"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_binary_message_panics_handler() {
    let (cancel, address) = start_data_server().await;
    let mut channel = WsClient::connect(address, "data").await.unwrap();
    channel
        .send(Message::new(bytes::Bytes::from_static(b"not messagepack")))
        .await
        .unwrap();

    let result = timeout(Duration::from_millis(500), channel.recv()).await;
    assert!(
        matches!(result, Err(_) | Ok(Err(_)) | Ok(Ok(None))),
        "connection remained alive after invalid message"
    );

    cancel.cancel();
}
```

运行：

```sh
cargo test -p burn-remote --test data_handler_fault_repro --features websocket -- --nocapture
```

## 实测输出

在当前 checkout（commit `88b8dd55e`）上执行：

```text
running 2 tests

thread 'tokio-rt-worker' panicked at crates/burn-communication/src/external_comm.rs:242:30:
Can deserialize messages from the websocket.: Syntax("invalid value: integer `110`, expected variant index 0 <= i < 2")
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
test invalid_binary_message_panics_handler ... ok
test unknown_transfer_id_blocks_even_after_cancellation ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.01s
```

两个测试的通过含义：

- 非法二进制消息触发服务端 handler panic；
- 未知 transfer ID 在 500ms 内无响应；
- 随后调用 `cancel.cancel()`，连接仍无响应，说明阻塞中的 handler 没有响应取消令牌。

## 预期行为

数据服务应把对端输入视为不可靠输入：

- MessagePack 反序列化失败应关闭该连接或返回协议错误，不应 panic；
- 非请求类型消息应返回错误或关闭连接；
- 未知 transfer ID 应在固定超时内失败并返回错误/关闭连接；
- 取消令牌应能终止等待中的 handler；
- 单个连接的协议错误不应影响其他连接和监听服务。

## 建议修复

将 `handle_data_channel` 改为错误驱动：

```rust
match rmp_serde::from_slice::<ExternalCommMessage>(&msg.data) {
    Ok(ExternalCommMessage::TensorRequest(id)) => { /* ... */ }
    Ok(_) => {
        log::warn!("unexpected /data message");
        break;
    }
    Err(err) => {
        log::warn!("invalid /data message: {err:?}");
        break;
    }
}
```

为 `get_exposed_tensor_bytes` 增加与 Iroh 一致的等待超时，并在每次等待前和超时后检查 `cancel_token`。可抽象为：

```rust
tokio::select! {
    _ = self.cancel_token.cancelled() => return None,
    _ = tokio::time::sleep(TRANSFER_WAIT_TIMEOUT) => return None,
    notified = self.new_tensor_notify.notified() => { notified; }
}
```

修复后补充上述两个回归测试，并验证：

- 非法消息不产生 panic 输出；
- 未知 ID 在超时内让客户端收到失败或连接关闭；
- 取消服务后 handler 在合理时间内退出；
- 其他 `/data` 连接仍可继续完成合法下载。
