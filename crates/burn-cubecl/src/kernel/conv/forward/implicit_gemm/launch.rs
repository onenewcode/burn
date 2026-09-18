use crate::{ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::ops::{ConvOptions, conv::calculate_conv_output_sizes};
use cubek::{
    convolution::{
        AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy,
        components::ConvSetupError, launch_ref,
    },
    matmul::definition::{MatmulElems, MatmulGlobalElems},
    std::InputBinding,
};

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components. Uses [`CmmaLargeMAlgorithm`] for the stage size
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_gemm_simple_sync<const N: usize>(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor, ConvSetupError> {
    let algorithm = match tile_kind {
        AcceleratedTileKind::Cmma => ConvAlgorithm::SimpleSyncCyclic,
        AcceleratedTileKind::Mma => ConvAlgorithm::SimpleSyncStrided,
    };
    launch_convolution_forward::<N>(
        &Strategy::Inferred {
            algorithm,
            tile_kind,
        },
        input,
        weight,
        bias,
        options,
    )
}

pub fn conv_gemm_simple_async<const N: usize>(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor, ConvSetupError> {
    let algorithm = match tile_kind {
        AcceleratedTileKind::Cmma => ConvAlgorithm::SimpleAsyncCyclic,
        AcceleratedTileKind::Mma => ConvAlgorithm::SimpleAsyncStrided,
    };
    launch_convolution_forward::<N>(
        &Strategy::Inferred {
            algorithm,
            tile_kind,
        },
        input,
        weight,
        bias,
        options,
    )
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components. Uses [`CmmaLargeMAlgorithm`] for the stage size
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_gemm_simple_tma<const N: usize>(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor, ConvSetupError> {
    launch_convolution_forward::<N>(
        &Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleAsyncTma,
            tile_kind,
        },
        input,
        weight,
        bias,
        options,
    )
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components, using the specified algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn launch_convolution_forward<const N: usize>(
    strategy: &Strategy,
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let out_dtype = input.dtype;
    let rank = input.meta.shape().num_dims();
    let batch_size = input.meta.shape()[0];
    let dim_c = rank - 1;
    let shape = &input.meta.shape()[1..dim_c];

    let out_channels = weight.meta.shape()[0];
    let weight_shape = &weight.meta.shape()[1..dim_c];

    let mut out_shape = calculate_conv_output_sizes(
        weight_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        shape,
    );

    out_shape.insert(0, batch_size);
    out_shape.push(out_channels);

    let out = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        out_shape.into(),
        out_dtype,
    );

    let bias = bias.map(|bias| {
        let dtype = bias.dtype;
        InputBinding::Normal(bias.binding(), dtype_to_storage_type(dtype))
    });

    let client = input.client.clone();
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: dtype_to_storage_type(input.dtype),
        rhs: dtype_to_storage_type(weight.dtype),
        out: dtype_to_storage_type(out_dtype),
    });
    let input_dtype = input.dtype;
    let weight_dtype = weight.dtype;
    let input = InputBinding::new(input.binding(), dtype_to_storage_type(input_dtype));
    let weight = InputBinding::new(weight.binding(), dtype_to_storage_type(weight_dtype));

    launch_ref::<N>(
        strategy,
        &client,
        ConvolutionInputs::Forward {
            input,
            weight,
            bias,
            out: out.clone().binding(),
        },
        ConvolutionArgs {
            stride: options.stride,
            padding: options.padding_begin(),
            dilation: options.dilation,
        },
        dtypes,
    )?;

    Ok(out)
}

/// Most of the accelerated candidates above decline on an Apple GPU, and the
/// reason half of them give points at a tile catalogue rather than at the
/// device.
///
/// Write-up, evidence and fix plan:
/// `crates/burn-cubecl/docs/accelerated-conv-candidates-decline-on-metal.md`.
///
/// ```bash
/// cargo test -p burn-cubecl --release --features metal \
///     candidate_survival -- --ignored --nocapture --test-threads=1
/// ```
///
/// No timing: each candidate is called once and its outcome recorded.
///
/// **This test is device-specific and passes while the defect is present.** On a
/// GPU whose accelerated candidates all compile it is *expected* to fail,
/// because there the problem does not exist. Read the printed table before the
/// assertion.
#[cfg(test)]
mod candidate_survival {
    use burn_backend::ops::ConvOptions;
    use burn_std::Shape;
    use cubek::convolution::{AcceleratedTileKind, components::ConvSetupError};

    use super::{conv_gemm_simple_async, conv_gemm_simple_sync, conv_gemm_simple_tma};
    use crate::{
        CubeDevice,
        kernel::conv::{conv_direct, conv_im2col_1x1},
        ops::numeric::full,
        tensor::CubeTensor,
    };

    /// The widest residual convolution of a dilated TCN, where this was found.
    /// A perfectly ordinary dense layer — every dimension a multiple of 8,
    /// channels a power of two — so nothing about the shape explains a decline.
    /// Small batch: this sets candidates up, it does not time them.
    const BATCH: usize = 8;
    const CHANNELS: usize = 128;
    const KERNEL: usize = 5;
    const DILATION: usize = 4;
    const PADDING: usize = 16;
    const LENGTH_IN: usize = 160;

    /// The two reasons the accelerated candidates give. They need different
    /// answers, which is why the test tells them apart: a missing tile size is
    /// a selector that never offers Metal the 8x8 `simdgroup_matrix` it has,
    /// while a missing async barrier is a capability the device really lacks.
    const NO_TILE: &str = "No tile size is available for the problem";
    const NO_BARRIER: &str = "Async barrier instructions are not available";

    fn options() -> ConvOptions<1> {
        ConvOptions::new([1], [PADDING], [DILATION], 1)
    }

    /// One candidate's fate, as autotune sees it: it set up and launched, or it
    /// declined with a reason.
    fn outcome(result: Result<CubeTensor, ConvSetupError>) -> Result<(), String> {
        result.map(|_| ()).map_err(|err| {
            format!("{err:?}")
                .lines()
                .next()
                .unwrap_or("(no reason given)")
                .trim()
                .to_string()
        })
    }

    /// Of the eight forward candidates a dense convolution is offered, how many
    /// set up.
    ///
    /// The five depthwise tunables in `forward/tune.rs` are left out: they
    /// decline anything that is not one filter per channel, by design, and a
    /// dense shape never had them.
    #[test]
    #[ignore = "queries a device's capabilities; run explicitly, see the module docs"]
    fn most_accelerated_candidates_decline_a_dense_convolution() {
        let device = CubeDevice::default();
        let filled = |shape: Shape| full::<f32>(shape, &device, 0.031_25);
        let input = filled(Shape::new([BATCH, LENGTH_IN, CHANNELS]));
        let weight = filled(Shape::new([CHANNELS, KERNEL, CHANNELS]));
        let bias = filled(Shape::new([CHANNELS]));

        /// One of the three accelerated launchers above. Named so that the
        /// table below reads as a list of candidates rather than of closures.
        type Launch = fn(
            CubeTensor,
            CubeTensor,
            Option<CubeTensor>,
            ConvOptions<1>,
            AcceleratedTileKind,
        ) -> Result<CubeTensor, ConvSetupError>;

        let call = |launch: Launch, kind| {
            outcome(launch(
                input.clone(),
                weight.clone(),
                Some(bias.clone()),
                options(),
                kind,
            ))
        };

        let outcomes = [
            (
                "conv_direct",
                outcome(conv_direct::<1>(
                    input.clone(),
                    weight.clone(),
                    Some(bias.clone()),
                    options(),
                )),
            ),
            (
                "conv_im2col_1x1",
                outcome(conv_im2col_1x1::<1>(
                    input.clone(),
                    weight.clone(),
                    Some(bias.clone()),
                    options(),
                )),
            ),
            (
                "simple_sync_cmma",
                call(conv_gemm_simple_sync::<1>, AcceleratedTileKind::Cmma),
            ),
            (
                "simple_sync_mma",
                call(conv_gemm_simple_sync::<1>, AcceleratedTileKind::Mma),
            ),
            (
                "simple_async_cmma",
                call(conv_gemm_simple_async::<1>, AcceleratedTileKind::Cmma),
            ),
            (
                "simple_async_mma",
                call(conv_gemm_simple_async::<1>, AcceleratedTileKind::Mma),
            ),
            (
                "simple_tma_cmma",
                call(conv_gemm_simple_tma::<1>, AcceleratedTileKind::Cmma),
            ),
            (
                "simple_tma_mma",
                call(conv_gemm_simple_tma::<1>, AcceleratedTileKind::Mma),
            ),
        ];

        println!("\n=== forward candidates for a dense convolution ===");
        println!("  {device:?}, kernel {KERNEL}, dilation {DILATION}, groups 1");
        for (name, outcome) in &outcomes {
            match outcome {
                Ok(()) => println!("  {name:<20} ok"),
                Err(reason) => println!("  {name:<20} declined: {reason}"),
            }
        }

        let alive = outcomes.iter().filter(|(_, o)| o.is_ok()).count();
        println!("  -> {alive} of {} survived", outcomes.len());

        // The actionable half. `cubecl-cpp`'s Metal dialect emits
        // `simdgroup_load` / `simdgroup_multiply_accumulate` /
        // `simdgroup_store` over 8x8 matrices, so the hardware and the code
        // generator both have a tile. A `mma` variant reporting that no tile
        // size is available is a claim about the selector's catalogue, not
        // about the device — and `cmma`, which uses the same matrices, works.
        let no_tile: Vec<_> = outcomes
            .iter()
            .filter_map(|(name, outcome)| {
                let reason = outcome.as_ref().err()?;
                reason.contains(NO_TILE).then_some(*name)
            })
            .collect();
        let no_barrier: Vec<_> = outcomes
            .iter()
            .filter_map(|(name, outcome)| {
                let reason = outcome.as_ref().err()?;
                reason.contains(NO_BARRIER).then_some(*name)
            })
            .collect();
        println!("  no tile size available:  {no_tile:?}");
        println!("  no async barrier:        {no_barrier:?}");

        assert!(
            outcomes.len() - alive >= 5,
            "only {} of {} candidates declined. On a device where the \
             accelerated ones compile this is the expected result and this test \
             is not for you; on an Apple GPU it means the selector improved, so \
             update \
             crates/burn-cubecl/docs/accelerated-conv-candidates-decline-on-metal.md",
            outcomes.len() - alive,
            outcomes.len(),
        );
        assert!(
            !no_tile.is_empty(),
            "no candidate declined for want of a tile size, so the half of this \
             that is a selector problem rather than a capability gap is fixed; \
             update \
             crates/burn-cubecl/docs/accelerated-conv-candidates-decline-on-metal.md",
        );
    }
}
