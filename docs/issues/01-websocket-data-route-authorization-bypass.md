# WebSocket `/data` 路由绕过会话鉴权，tensor 传输 capability 被截断为 64 位

## 严重级别

**P2（若服务暴露在不受信任网络或传输 ID 泄露，可升级为 P1）**

## 问题描述

远程 WebSocket 服务端有两条路由：

- `/session`：每个客户端都必须经过 `authorizer.authorize(...)`，并提交 `Credential`；
- `/data`：直接调用 `ExternalCommService::handle_data_channel`，没有任何会话鉴权、来源绑定或握手认证。

相关代码：

```rust
// crates/burn-remote/src/transport/websocket/server.rs:93-119
WsServer::new(0)
    .route_with_max_message_size(
        "/session",
        MAX_FRAME_SIZE,
        move |channel: WsServerChannel| {
            // ...
            authorizer.authorize(AuthorizationRequest {
                client,
                device_index: init.device_index,
                credential: &Credential::from(init.authorization.as_slice()),
            })
            // ...
        },
    )
    .route_external_comm(external)
```

`route_external_comm` 的实现只是把 `/data` 直接交给 tensor 数据服务：

```rust
// crates/burn-communication/src/external_comm.rs:71-74
fn route_external_comm(self, state: Arc<ExternalCommService<B, P>>) -> Self {
    self.route("/data", async move |stream: S::Channel| {
        state.handle_data_channel(stream).await;
    })
}
```

同时，WebSocket 传输把 256 位的 `TransferCapability` 截断成 64 位作为 `/data` 的 bearer key：

```rust
// crates/burn-remote/src/shared/task.rs:58-65
pub(crate) fn websocket_id(self) -> u64 {
    u64::from_le_bytes(self.0[..8].try_into().expect("..."))
}
```

该 64 位 ID 没有绑定目标节点。任何能获得该 ID 的网络端点都可以直接连接 `/data` 并下载 tensor。相比之下，Iroh 路径使用完整 capability，并把 capability 绑定到目标 peer。

更严重的是，默认 WebSocket transport 绑定 `0.0.0.0`：

```rust
// crates/burn-remote/src/transport/websocket/server.rs:51-58
TcpListener::bind(("0.0.0.0", port)).await
```

因此，只要 64 位 transfer ID 通过日志、调试信息、未加密流量或其他途径泄露，局域网/可达网络中的任意客户端都能绕过 `/session` 的 token 鉴权直接读取 tensor 数据。

## 最小复现

在 `crates/burn-remote/tests/data_route_repro.rs` 中加入以下测试：

```rust
#![cfg(feature = "websocket")]

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
async fn websocket_data_route_has_no_authentication() {
    let cancel = CancellationToken::new();
    let service = ExternalCommService::<Flex, WebSocket>::new(cancel.clone());
    let service = std::sync::Arc::new(service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address: Address = format!("ws://{}", listener.local_addr().unwrap()).as_str().into();
    let server = WsServer::new(0).route_external_comm(service.clone());

    tokio::spawn(async move {
        let shutdown = async move {
            cancel.cancelled().await;
        };
        server.serve_on(listener, shutdown).await.unwrap();
    });

    service
        .expose_data(TensorData::from([13.0_f32, 37.0]), 1, 123_u64.into())
        .await;

    // 注意：这里没有 session token，也没有 authorizer。
    let mut channel = WsClient::connect(address, "data").await.unwrap();
    let request = rmp_serde::to_vec(&WireMessage::TensorRequest(123_u64.into())).unwrap();
    channel.send(Message::new(request.into())).await.unwrap();

    let response = channel.recv().await.unwrap().unwrap();
    let message: WireMessage = rmp_serde::from_slice(&response.data).unwrap();

    match message {
        WireMessage::Tensor(data) => assert_eq!(data, TensorData::from([13.0_f32, 37.0])),
        WireMessage::TensorRequest(_) => panic!("server returned the request"),
    }
}
```

运行：

```sh
cargo test -p burn-remote --test data_route_repro --features websocket -- --nocapture
```

## 实测输出

在当前 checkout（commit `88b8dd55e`）上执行：

```text
running 1 test
test websocket_data_route_has_no_authentication ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

该测试通过即表示漏洞存在：客户端没有提供任何 session credential，仍能凭 64 位 transfer ID 从 `/data` 拿到完整 tensor 数据。

## 预期行为

`/data` 至少应满足以下任一安全模型：

- 使用与 `/session` 相同的认证机制；
- 将 transfer capability 绑定到目标 peer，并拒绝其他来源；
- 使用完整 256 位 capability，而不是截断为 64 位；
- 明确限制 `/data` 只监听回环地址或私有受信接口。

## 建议修复

- 为 `/data` 增加握手认证，校验目标 peer 身份和 transfer capability。
- 不要使用 `TransferCapability::websocket_id()` 截断 256 位 capability；至少使用完整 capability 作为查找键。
- 在 `ExternalCommService` 的 exposed tensor state 中记录允许的目标地址/peer，并在下载请求中校验来源。
- 将默认 WebSocket 监听地址改为可配置，并在文档中明确 `0.0.0.0` + 未加密 `/data` 的风险。
- 增加本 issue 中的回归测试，并改成“无认证连接必须被拒绝”的断言。

