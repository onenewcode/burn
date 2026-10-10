# `max_downloads = 0` 被 WebSocket 数据服务解释为无限次下载

## 严重级别

**P2（访问次数限制被静默绕过，可导致敏感 tensor 反复暴露）**

## 问题描述

`ExternalCommService::expose_data` 是公共 API，其 `max_downloads` 参数的语义是“该 tensor 允许被下载多少次”。实现没有拒绝 `0`：

```rust
// crates/burn-communication/src/external_comm.rs:101-118
pub async fn expose_data(
    &self,
    tensor_data: TensorData,
    max_downloads: u32,
    transfer_id: TensorTransferId,
) {
    // ...
    exposed_tensors.insert(
        transfer_id,
        TensorExposeState {
            bytes,
            max_downloads,
            cur_download_count: 0,
        },
    );
}
```

下载计数逻辑只在计数恰好等于上限时删除 tensor：

```rust
// crates/burn-communication/src/external_comm.rs:212-220
if let Some(mut exposed_state) = exposed_tensors.remove(&transfer_id) {
    exposed_state.cur_download_count += 1;
    let bytes = if exposed_state.cur_download_count == exposed_state.max_downloads {
        exposed_state.bytes
    } else {
        let bytes = exposed_state.bytes.clone();
        exposed_tensors.insert(transfer_id, exposed_state);
        bytes
    };
    return Some(bytes);
}
```

当 `max_downloads == 0` 时：

- 第 1 次下载后 `cur_download_count == 1`；
- `1 != 0`，tensor 被重新插入；
- 之后每次下载计数为 `2, 3, 4, ...`，永远不会等于 `0`。

因此，调用方明确表达“不允许下载”的输入被解释为“无限次下载”，并且每次都会返回同一份数据。这不是返回 `None` 或构造错误，而是安全边界的反向静默失效。

## 最小复现

将以下内容保存为 `crates/burn-remote/tests/zero_max_downloads_repro.rs`：

```rust
#![cfg(feature = "websocket")]

use std::sync::Arc;

use burn_communication::external_comm::{
    ExternalCommServer, ExternalCommService, TensorTransferId,
};
use burn_communication::websocket::{WebSocket, WsClient, WsServer};
use burn_communication::{Address, CommunicationChannel, Message, ProtocolClient};
use burn_flex::Flex;
use burn_tensor::TensorData;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

#[derive(Deserialize, Serialize)]
enum WireMessage {
    TensorRequest(TensorTransferId),
    Tensor(TensorData),
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zero_max_downloads_allows_unlimited_requests() {
    let cancel = CancellationToken::new();
    let service = Arc::new(ExternalCommService::<Flex, WebSocket>::new(cancel.clone()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address: Address = format!("ws://{}", listener.local_addr().unwrap())
        .as_str()
        .into();
    let server = WsServer::new(0).route_external_comm(service.clone());

    let shutdown_token = cancel.clone();
    tokio::spawn(async move {
        let shutdown = async move {
            shutdown_token.cancelled().await;
        };
        server.serve_on(listener, shutdown).await.unwrap();
    });

    service
        .expose_data(TensorData::from([13.0_f32, 37.0]), 0, 123_u64.into())
        .await;

    let mut channel = WsClient::connect(address, "data").await.unwrap();
    for _ in 0..3 {
        let request =
            rmp_serde::to_vec(&WireMessage::TensorRequest(123_u64.into())).unwrap();
        channel.send(Message::new(request.into())).await.unwrap();
        let response = channel.recv().await.unwrap().unwrap();
        let message: WireMessage = rmp_serde::from_slice(&response.data).unwrap();

        match message {
            WireMessage::Tensor(data) => {
                assert_eq!(data, TensorData::from([13.0_f32, 37.0]));
            }
            WireMessage::TensorRequest(_) => panic!("server returned the request"),
        }
    }

    cancel.cancel();
}
```

运行：

```sh
cargo test -p burn-remote --test zero_max_downloads_repro --features websocket -- --nocapture
```

## 实测输出

在当前 checkout（commit `88b8dd55e`）上执行：

```text
running 1 test
test zero_max_downloads_allows_unlimited_requests ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

测试连续请求 3 次均拿到同一 tensor。对 `max_downloads = 0` 来说，任何一次成功下载都是违规；3 次成功证明该值被解释为无限次下载。

## 预期行为

`max_downloads = 0` 不应使 tensor 可被下载。至少应有以下一种明确行为：

- `expose_data` 返回错误，说明下载次数必须大于 `0`；
- 构造时校验并拒绝 `0`；
- 下载请求立即返回失败，且不暴露数据。

同时，正数上限应精确允许 `max_downloads` 次：第 1 到 `max_downloads - 1` 次复制返回，第 `max_downloads` 次返回后删除，第 `max_downloads + 1` 次失败。

## 建议修复

首选在公共 API 入口校验：

```rust
if max_downloads == 0 {
    // 返回错误，或按现有 API 风格 debug_assert!/panic 于构造期。
}
```

计数判断也应避免依赖 `cur == max`：

```rust
let should_remove = exposed_state.cur_download_count >= exposed_state.max_downloads;
```

并补充以下回归测试：

- `max_downloads = 0` 不能返回数据；
- `max_downloads = 1` 只允许一次；
- `max_downloads = 2` 允许两次，第三次失败；
- 与 Iroh 传输保持相同语义。
