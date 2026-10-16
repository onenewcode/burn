# 当前代码问题清单

以下问题均基于当前工作树代码独立复现，复现命令和实际输出写在各自文档中：

1. [量化打包函数的 `Vec<i8>` 未定义行为](001-quant-pack-ub.md)（P1，内存安全）
2. [CPU NMS 的未对齐 SIMD 读取](002-vision-nms-unaligned-bool.md)（P1，内存安全）
3. [CubeCL CTC 长序列误判可达路径](003-cubecl-ctc-long-sequence.md)（P1，训练结果错误）
4. [CubeCL 拒绝非整除 `PackedU32` 量化形状](004-cubecl-packed-quant-odd-shape.md)（P2，合法模型无法加载）
5. [CPU 连通分量在空图像上发生未定义行为](005-vision-connected-components-empty-ub.md)（P1，内存安全）
6. [Flex 后端未实现公开的 `NearestExact` 插值模式](006-flex-nearest-exact-interpolate.md)（P2，合法调用稳定 panic）
