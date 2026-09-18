use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::ops::ConvOptions;
use cubecl::{
    ir::ElemType,
    tune::{LocalTuner, Tunable, TunableSet, anchor, local_tuner},
};
use cubek::convolution::{AcceleratedTileKind, DepthwiseStrategy, DepthwiseTiling};

use crate::{
    CubeAutotuneKey, CubeTuneId,
    kernel::conv::{
        ConvAutotuneKey, conv_direct, conv_im2col_1x1, forward::depthwise::conv_depthwise,
        forward::implicit_gemm::*,
    },
    tensor::CubeTensor,
};

/// The tilings the depthwise routine is offered under, beside the one it picks for itself.
///
/// Chosen greedily against a sweep of the whole grid over EfficientNet-B4's depthwise layers:
/// these four are what closes the gap between the routine's own rule (16.6 ms of depthwise
/// convolution at batch 4) and picking the best tiling per shape (15.7 ms). Adding more moves it
/// by under 1%, and every one of them costs a dense convolution a setup call that declines.
const DEPTHWISE_8X4_LINED: DepthwiseStrategy = DepthwiseStrategy::Fixed(DepthwiseTiling {
    rows: 8,
    cols: 4,
    chans: 1,
    lines: 2,
});
const DEPTHWISE_2X4_SCALAR: DepthwiseStrategy = DepthwiseStrategy::Fixed(DepthwiseTiling {
    rows: 2,
    cols: 4,
    chans: 1,
    lines: 1,
});
const DEPTHWISE_8X2_LINED: DepthwiseStrategy = DepthwiseStrategy::Fixed(DepthwiseTiling {
    rows: 8,
    cols: 2,
    chans: 1,
    lines: 2,
});
const DEPTHWISE_4X2_SCALAR: DepthwiseStrategy = DepthwiseStrategy::Fixed(DepthwiseTiling {
    rows: 4,
    cols: 2,
    chans: 1,
    lines: 1,
});

/// Executes autotune on convolution operations
pub fn conv_autotune<const N: usize>(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
) -> CubeTensor {
    let client = input.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tune_id = CubeTuneId::new(&input.client, &input.device);
    let tunables = TUNER.init(&tune_id, || {
        TunableSet::new(create_key::<N>, create_conv_input::<N>)
            .with(Tunable::new(
                "conv_direct",
                |(input, weight, bias, options)| conv_direct::<N>(input, weight, bias, options),
            ))
            // Declines with `NotDepthwise` on anything that is not one filter per channel, so
            // each of these costs a dense shape nothing but the setup call that rejects it.
            //
            // Several tilings rather than one because they are not close: over EfficientNet-B4's
            // depthwise layers, the best tile per shape beats the best single tile by 8%, and
            // which one wins swings with the window's depth and the block's width in a way the
            // shape does not predict. The routine's own default is the fallback when no tuning
            // has run; these are what let a run that does tune land on the right one.
            .with(Tunable::new(
                "conv_depthwise",
                |(input, weight, bias, options)| {
                    conv_depthwise::<N>(input, weight, bias, options, DepthwiseStrategy::Routine)
                },
            ))
            .with(Tunable::new(
                "conv_depthwise_8x4_lined",
                |(input, weight, bias, options)| {
                    conv_depthwise::<N>(input, weight, bias, options, DEPTHWISE_8X4_LINED)
                },
            ))
            .with(Tunable::new(
                "conv_depthwise_2x4_scalar",
                |(input, weight, bias, options)| {
                    conv_depthwise::<N>(input, weight, bias, options, DEPTHWISE_2X4_SCALAR)
                },
            ))
            .with(Tunable::new(
                "conv_depthwise_8x2_lined",
                |(input, weight, bias, options)| {
                    conv_depthwise::<N>(input, weight, bias, options, DEPTHWISE_8X2_LINED)
                },
            ))
            .with(Tunable::new(
                "conv_depthwise_4x2_scalar",
                |(input, weight, bias, options)| {
                    conv_depthwise::<N>(input, weight, bias, options, DEPTHWISE_4X2_SCALAR)
                },
            ))
            .with(Tunable::new(
                "conv_im2col_1x1",
                |(input, weight, bias, options)| conv_im2col_1x1::<N>(input, weight, bias, options),
            ))
            .with(Tunable::new(
                "simple_sync_cmma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_sync(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_sync_mma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_sync(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_async_cmma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_async(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_async_mma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_async(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_tma_cmma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_tma(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_tma_mma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_tma(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
    });

    TUNER.execute(&tune_id, &client, tunables, (input, weight, bias, options))
}

pub fn create_conv_input<const N: usize>(
    _key: &CubeAutotuneKey,
    (input, weights, bias, options): &(CubeTensor, CubeTensor, Option<CubeTensor>, ConvOptions<N>),
) -> (CubeTensor, CubeTensor, Option<CubeTensor>, ConvOptions<N>) {
    (
        input.clone(),
        weights.clone(),
        bias.clone(),
        options.clone(),
    )
}

fn create_key<const N: usize>(
    (input, weights, bias, options): &(CubeTensor, CubeTensor, Option<CubeTensor>, ConvOptions<N>),
) -> CubeAutotuneKey {
    let dtype = input.dtype;
    let rank = input.meta.shape().num_dims();
    let dim_c = rank - 1;

    let batch_size = input.meta.shape()[0];
    let in_channels = input.meta.shape()[dim_c];
    let out_channels = weights.meta.shape()[0];

    let kernel_size = weights.meta.shape()[1..dim_c].to_vec();
    let in_shape = input.meta.shape()[1..dim_c]
        .iter()
        .map(|shape| anchor(*shape, None, None, None))
        .collect();

    let ConvOptions {
        stride,
        padding,
        dilation,
        groups,
    } = options.clone();

    let lhs_stride_align = if input.meta.strides()[dim_c] == 1 {
        stride_align(input.meta.strides(), dtype_to_storage_type(input.dtype))
    } else {
        0
    };
    let lhs_shape_align = pow2_factor(in_channels).min(lhs_stride_align);
    let rhs_stride_align = if weights.meta.strides()[dim_c] == 1 {
        stride_align(weights.meta.strides(), dtype_to_storage_type(weights.dtype))
    } else {
        0
    };
    let rhs_shape_align = pow2_factor(in_channels).min(rhs_stride_align);

    CubeAutotuneKey::Conv(ConvAutotuneKey::new(
        kernel_size,
        stride.to_vec(),
        padding.to_vec(),
        dilation.to_vec(),
        groups,
        in_channels,
        out_channels,
        in_shape,
        batch_size,
        bias.is_some(),
        dtype,
        lhs_shape_align,
        lhs_stride_align,
        rhs_shape_align,
        rhs_stride_align,
    ))
}

/// Maximum factor relevant for strides. Currently set to 2^10 because that's 128-byte swizzle's
/// repeat number, so it's the largest align that can have performance impacts.
const MAX_STRIDE_FACTOR: u32 = 10;

/// Defines the non-contiguous stride alignment in terms of powers of two
fn stride_align(strides: &[usize], elem: ElemType) -> u8 {
    let max = MAX_STRIDE_FACTOR;
    let dim_c = strides.len() - 1;
    let factor = strides[..dim_c]
        .iter()
        .map(|it| (*it * elem.size_bits()) / 8)
        .map(|it| it.trailing_zeros())
        .min()
        .unwrap_or(max);
    factor.min(max) as u8
}

/// Defines the potential vectorization.
fn pow2_factor(axis: usize) -> u8 {
    axis.trailing_zeros().min(4) as u8
}

/// The tuner above picks the slowest surviving candidate for an ordinary dense
/// convolution, because the benchmark it decides on reports each candidate at
/// roughly the other's cost.
///
/// Write-up, evidence and fix plan:
/// `crates/burn-cubecl/docs/autotune-picks-the-slower-convolution.md`.
///
/// ```bash
/// cargo test -p burn-cubecl --release --features metal \
///     picks_the_slower -- --ignored --nocapture --test-threads=1
/// ```
///
/// **This test passes while the defect is present.** A failure because the
/// tuner landed on the fast candidate is the result the document asks for.
#[cfg(test)]
mod picks_the_slower {
    use std::time::Instant;

    use burn_backend::{DType, ops::ConvOptions};
    use burn_std::Shape;
    use cubecl::client::Client;
    use cubek::convolution::AcceleratedTileKind;

    use super::{conv_autotune, conv_gemm_simple_sync};
    use crate::{
        CubeDevice,
        kernel::{
            conv::conv_direct,
            matmul::{MatmulStrategy, matmul},
        },
        ops::numeric::full,
        tensor::CubeTensor,
    };

    /// The widest residual convolution of a dilated TCN, where this was found.
    /// An ordinary dense layer: `groups = 1`, unit stride, power-of-two
    /// channels, every dimension a multiple of 8.
    const BATCH: usize = 128;
    const CHANNELS: usize = 128;
    const KERNEL: usize = 5;
    const DILATION: usize = 4;
    const PADDING: usize = 16;
    const LENGTH_IN: usize = 160;
    const LENGTH_OUT: usize = LENGTH_IN + 2 * PADDING - DILATION * (KERNEL - 1);
    /// `[BATCH * LENGTH_OUT, REDUCTION] x [REDUCTION, CHANNELS]` is the GEMM
    /// this convolution reduces to, and it carries the same FLOP count — the
    /// rate to read the candidates against.
    const REDUCTION: usize = CHANNELS * KERNEL;
    const FLOPS: u64 = (2 * BATCH * LENGTH_OUT * CHANNELS * CHANNELS * KERNEL) as u64;

    fn options() -> ConvOptions<1> {
        ConvOptions::new([1], [PADDING], [DILATION], 1)
    }

    fn filled(device: &CubeDevice, shape: impl Into<Shape>) -> CubeTensor {
        full::<f32>(shape.into(), device, 0.031_25)
    }

    /// Median of five timed runs after twenty untimed ones, with the queue
    /// drained inside each window.
    ///
    /// Twenty, not two: one of the rows below is the tuner itself, and tuning
    /// resolves asynchronously — until it settles, a tuned call runs whichever
    /// candidate the tuner has provisionally picked. `matmul` reaches autotune
    /// internally for the same reason. A short warm-up measures the transient
    /// rather than the decision.
    ///
    /// `op` hands its result back so that nothing it queued is eliminated as
    /// dead code.
    fn median_ms(client: &Client, mut op: impl FnMut() -> CubeTensor) -> f64 {
        let mut sink = None;
        let mut timed = Vec::new();
        for run in 0..25 {
            let start = Instant::now();
            sink = Some(op());
            futures_lite::future::block_on(client.sync()).expect("the device drains");
            if run >= 20 {
                timed.push(start.elapsed());
            }
        }
        drop(sink);

        timed.sort_unstable();
        timed[timed.len() / 2].as_secs_f64() * 1e3
    }

    fn row(label: &str, ms: f64) {
        println!(
            "  {label:<26} {ms:>9.3} ms   {:>5.2} TFLOP/s",
            FLOPS as f64 / (ms / 1e3) / 1e12
        );
    }

    /// The two candidates that survive here, measured directly, and the tuner's
    /// own settled choice next to them.
    ///
    /// No autotune log is parsed: which candidate the tuner settled on is
    /// visible in what a tuned call costs. A tuner deciding on sound numbers
    /// puts the last row beside the fastest of the rows above it.
    #[test]
    #[ignore = "measures a device; run explicitly, see the module docs"]
    fn the_tuner_settles_on_the_slower_of_two_surviving_candidates() {
        let device = CubeDevice::default();
        let client = device.client();

        let input = filled(&device, [BATCH, LENGTH_IN, CHANNELS]);
        let weight = filled(&device, [CHANNELS, KERNEL, CHANNELS]);
        let bias = filled(&device, [CHANNELS]);

        let direct = median_ms(&client, || {
            conv_direct::<1>(input.clone(), weight.clone(), Some(bias.clone()), options())
                .expect("conv_direct declines nothing")
        });
        let accelerated = median_ms(&client, || {
            conv_gemm_simple_sync::<1>(
                input.clone(),
                weight.clone(),
                Some(bias.clone()),
                options(),
                AcceleratedTileKind::Cmma,
            )
            .expect("this test needs an accelerated candidate; see the document")
        });
        let tuned = median_ms(&client, || {
            conv_autotune::<1>(input.clone(), weight.clone(), Some(bias.clone()), options())
        });

        // The same arithmetic on the general matrix path, as the rate this
        // device actually reaches.
        let lhs = filled(&device, [BATCH * LENGTH_OUT, REDUCTION]);
        let rhs = filled(&device, [REDUCTION, CHANNELS]);
        let gemm = median_ms(&client, || {
            matmul(
                lhs.clone(),
                rhs.clone(),
                None,
                MatmulStrategy::default(),
                DType::F32,
            )
            .expect("the GEMM has a candidate")
        });

        println!("\n=== what the forward tuner settles on ===");
        println!("  {device:?}, batch {BATCH}, {FLOPS} FLOP each");
        row("conv_direct", direct);
        row("simple_sync_cmma", accelerated);
        row("conv_autotune (settled)", tuned);
        row("matmul, same FLOP", gemm);
        println!(
            "  slower / faster candidate  {:.1}x,  tuner is on the {} one",
            direct.max(accelerated) / direct.min(accelerated),
            if (tuned - direct).abs() < (tuned - accelerated).abs() {
                "conv_direct"
            } else {
                "simple_sync_cmma"
            },
        );

        let (fast, slow) = (direct.min(accelerated), direct.max(accelerated));
        assert!(
            slow / fast >= 2.0,
            "the two candidates are within {:.1}x of each other, so this shape \
             no longer shows the tuner a choice worth getting right; pick \
             another, or update \
             crates/burn-cubecl/docs/autotune-picks-the-slower-convolution.md",
            slow / fast,
        );
        assert!(
            tuned > (fast + slow) / 2.0,
            "a tuned convolution costs {tuned:.3} ms, nearer the {fast:.3} ms \
             candidate than the {slow:.3} ms one, so the tuner is choosing \
             correctly now; retire \
             crates/burn-cubecl/docs/autotune-picks-the-slower-convolution.md",
        );
    }
}
