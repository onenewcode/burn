# P1：CPU 连通分量在空图像上发生未定义行为

## 影响

`Tensor<2, Bool>::connected_components`（`Connectivity::Four` 或 `Eight`）对合法的零元素输入
（例如形状 `[0, 3]`）会进入 CPU 连通分量实现的“第一行”扫描。实现无条件地对空输入指针做生成代码访问，导致进程直接崩溃并触发越界指针解引用。任何将空图像作为动态批次/裁剪结果传入视觉流水线的程序都可能触发该问题。

## 根因

`crates/burn-vision/src/backends/cpu/connected_components.rs:66-74` 将输入交给两个扫描器之一。
四连通实现 `crates/burn-vision/src/backends/cpu/connected_components/spaghetti_4c/mod.rs:54-69`
无条件执行第一行扫描；八连通实现也在 `crates/burn-vision/src/backends/cpu/connected_components/spaghetti/mod.rs:77-96`
无条件执行前两行扫描：

```rust
let img = img.as_ptr();
let mut img_labels = vec![0.elem(); h * w];
// h == 0 时仍然执行
let img_row00 = img.add(0);
let img_labels_row00 = img_labels.as_mut_ptr().add(0);
include!("Spaghetti4C_first_line_forest_code.rs");
```

当 `h == 0` 时，输入没有任何元素，生成的扫描代码仍会读取 `img_row00`/`img_row01`，并可能写入空的
`img_labels`；这违反了两个模块对手工 `unsafe` 访问“始终有界”的安全前提。

## 复现

在当前工作树创建最小工程 `/tmp/burn-cc-empty`：

```toml
[package]
name = "burn-cc-empty"
version = "0.1.0"
edition = "2021"

[dependencies]
burn = { path = "/Volumes/HIKSEMI/code/burn/crates/burn", default-features = false,
          features = ["std", "flex", "vision"] }
```

`src/main.rs`：

```rust
use burn::tensor::{Bool, Device, Tensor, TensorData};
use burn::vision::{ConnectedComponents, Connectivity};

fn main() {
    let device = Device::flex();
    let input: Tensor<2, Bool> =
        Tensor::from_data(TensorData::new(Vec::<bool>::new(), [0, 3]), &device);
    let output = input.connected_components(Connectivity::Four);
    println!("{:?}", output.into_data());
}
```

运行：

```sh
cargo run --manifest-path /tmp/burn-cc-empty/Cargo.toml
```

实际结果为进程退出码 `139`（SIGSEGV），没有返回一个形状为 `[0, 3]` 的空标签张量。将输入
形状改为 `[1, 0]` 则不会经过该无条件第一行循环，能够正常返回空结果，进一步说明触发条件是
`h == 0` 的路径。把复现中的 `Connectivity::Four` 改为 `Connectivity::Eight` 同样会崩溃。

## 修复方向

在进入任一扫描器前对 `h == 0 || w == 0` 直接返回空标签数组（并为统计路径构造只含背景项的
一致统计结果），或让生成的第一行扫描仅在 `h > 0 && w > 0` 时执行。所有后续 `unsafe` 指针
访问都应在该前置条件下进行。

