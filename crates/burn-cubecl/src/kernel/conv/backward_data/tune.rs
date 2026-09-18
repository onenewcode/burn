use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::ops::ConvOptions;
use burn_std::Shape;
use cubecl::{
    ir::ElemType,
    tune::{LocalTuner, Tunable, TunableSet, anchor, local_tuner},
};
use cubek::convolution::AcceleratedTileKind;

use crate::{
    CubeAutotuneKey, CubeTuneId,
    kernel::conv::{
        ConvAutotuneKey,
        backward_data::{fallback::conv_data_backward_fallback, implicit_gemm::*},
        im2col::dgrad_im2col_1x1,
    },
    tensor::CubeTensor,
};

/// Executes autotune on conv2d operations
pub fn dgrad_autotune<const N: usize>(
    out_grad: CubeTensor,
    weights: CubeTensor,
    input_shape: Shape,
    options: ConvOptions<N>,
) -> CubeTensor {
    let client = out_grad.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    // Note: TMA isn't currently implemented properly, and will always error.
    // It's kept here so it gets automatically enabled as soon as cubek updates.
    // No CMMA for TMA because swizzling will be mandatory for good performance on dgrad.
    let tune_id = CubeTuneId::new(&out_grad.client, &out_grad.device);
    let tunables = TUNER.init(&tune_id, || {
        TunableSet::new(create_key::<N>, create_wgrad_input::<N>)
            .with(Tunable::new(
                "wgrad_fallback",
                |(out_grad, weights, input_shape, options)| {
                    conv_data_backward_fallback::<N>(out_grad, weights, input_shape, options)
                },
            ))
            // Declines every shape but the pointwise one. It earns its place
            // because a device with no accelerated matmul for the dtype
            // declines every candidate below, leaving the fallback unopposed.
            .with(Tunable::new(
                "dgrad_im2col_1x1",
                |(out_grad, weights, input_shape, options)| {
                    dgrad_im2col_1x1::<N>(out_grad, weights, input_shape, options)
                },
            ))
            .with(Tunable::new(
                "simple_sync_cmma",
                |(input, grad, shape, options)| {
                    dgrad_gemm_simple_sync(input, grad, shape, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_sync_mma",
                |(input, grad, shape, options)| {
                    dgrad_gemm_simple_sync(input, grad, shape, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_async_cmma",
                |(input, grad, shape, options)| {
                    dgrad_gemm_simple_async(input, grad, shape, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_async_mma",
                |(input, grad, shape, options)| {
                    dgrad_gemm_simple_async(input, grad, shape, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_tma_mma",
                |(input, grad, shape, options)| {
                    dgrad_gemm_simple_tma(input, grad, shape, options, AcceleratedTileKind::Mma)
                },
            ))
    });

    TUNER.execute(
        &tune_id,
        &client,
        tunables,
        (out_grad, weights, input_shape, options),
    )
}

pub fn create_wgrad_input<const N: usize>(
    _key: &CubeAutotuneKey,
    (out_grad, weights, input_shape, options): &(CubeTensor, CubeTensor, Shape, ConvOptions<N>),
) -> (CubeTensor, CubeTensor, Shape, ConvOptions<N>) {
    (
        out_grad.clone(),
        weights.clone(),
        input_shape.clone(),
        options.clone(),
    )
}

fn create_key<const N: usize>(
    (out_grad, weights, input_shape, options): &(CubeTensor, CubeTensor, Shape, ConvOptions<N>),
) -> CubeAutotuneKey {
    let dtype = out_grad.dtype;
    let rank = out_grad.meta.num_dims();
    let dim_c = rank - 1;

    let batch_size = out_grad.meta.shape()[0];
    let in_channels = input_shape[dim_c];
    let out_channels = out_grad.meta.shape()[dim_c];

    let kernel_size = weights.meta.shape()[1..dim_c].to_vec();
    let in_shape = input_shape[1..dim_c]
        .iter()
        .map(|shape| anchor(*shape, None, None, None))
        .collect();

    let ConvOptions {
        stride,
        padding,
        dilation,
        groups,
    } = options.clone();

    let lhs_stride_align = if out_grad.meta.strides()[dim_c] == 1 {
        stride_align(
            out_grad.meta.strides(),
            dtype_to_storage_type(out_grad.dtype),
        )
    } else {
        0
    };
    let lhs_shape_align = pow2_factor(out_channels).min(lhs_stride_align);
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
        false,
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

/// The data-gradient fallback above is registered under a weight-gradient name,
/// and `backward_weight/tune.rs` registers the same name for real.
///
/// Write-up and fix plan:
/// `crates/burn-cubecl/docs/dgrad-tunable-registered-as-wgrad.md`.
///
/// ```bash
/// cargo test -p burn-cubecl --features metal dgrad_tunable_name -- --nocapture
/// ```
///
/// No device and no timing: the defect is a string, and what it costs is paid by
/// whoever reads an autotune log. So this runs in a plain `cargo test`.
///
/// **These tests pass while the defect is present.** They fail once the name is
/// corrected, which is when the document can be retired.
///
/// # Why they read source text
///
/// A tunable's name is what the autotune logger prints, and nothing exposes it:
/// `TuneFn::name` is private to `cubecl-runtime`, `TunableSet` has no accessor
/// for it, and the set is built inline inside [`dgrad_autotune`] rather than
/// returned. Reading the registration is the only way to observe the thing that
/// is wrong. Only the part of this file above this module is matched, so what is
/// found is the registration and never this module's own text about it.
#[cfg(test)]
mod dgrad_tunable_name {
    /// This file's source, up to the attribute that opens this module.
    fn registration() -> &'static str {
        include_str!("tune.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("a split yields a first part")
    }

    /// The tunable that runs `conv_data_backward_fallback` is called
    /// `wgrad_fallback`, and the function building its inputs is called
    /// `create_wgrad_input`.
    #[test]
    fn the_data_gradient_fallback_is_registered_under_a_wgrad_name() {
        let source = registration();

        assert!(
            source.contains("conv_data_backward_fallback::<N>"),
            "this test is reading the wrong file: no data-gradient fallback is \
             registered in it",
        );
        assert!(
            source.contains(r#""wgrad_fallback""#) && !source.contains(r#""dgrad_fallback""#),
            "the data gradient's fallback is no longer registered under a \
             weight-gradient name; retire \
             crates/burn-cubecl/docs/dgrad-tunable-registered-as-wgrad.md",
        );
        assert!(
            source.contains("create_wgrad_input"),
            "the input builder was renamed but the tunable was not, or the other \
             way round; both halves are in one document: \
             crates/burn-cubecl/docs/dgrad-tunable-registered-as-wgrad.md",
        );

        println!("\n=== what this tuner calls its own candidates ===");
        for line in source.lines() {
            let line = line.trim();
            if line.starts_with('"') && line.ends_with("\",") {
                println!("  {line}");
            }
        }
        println!("  -> the first of these runs conv_data_backward_fallback");
    }

    /// The name is not merely wrong, it is taken: `backward_weight/tune.rs`
    /// registers `wgrad_fallback` for its own fallback. Two tuners therefore
    /// report the same winning name for entirely different kernels, which is
    /// what makes a log misleading rather than just mislabelled.
    #[test]
    fn the_weight_gradient_tuner_registers_the_same_name_for_real() {
        let wgrad = include_str!("../backward_weight/tune.rs");
        assert!(
            wgrad.contains(r#""wgrad_fallback""#)
                && wgrad.contains("conv_weight_backward_fallback"),
            "the weight-gradient tuner no longer registers a `wgrad_fallback`, \
             so the collision is gone; update \
             crates/burn-cubecl/docs/dgrad-tunable-registered-as-wgrad.md",
        );

        println!("\n=== the collision ===");
        println!("  backward_data/tune.rs    \"wgrad_fallback\" -> conv_data_backward_fallback");
        println!("  backward_weight/tune.rs  \"wgrad_fallback\" -> conv_weight_backward_fallback");
        println!("  -> a log naming the winner cannot say which tuner won");
    }
}
