use crate::{
    kernel::utils::{address_type, decompose_linear, shape_divmod},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{Shape, ops::ConvTransposeOptions};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, tensor::layout::linear::LinearViewMut},
};
use cubek::convolution::components::ConvSetupError;

#[derive(CubeLaunch, CubeType)]
struct ConvArgs {
    conv_stride_0: usize,
    conv_stride_1: usize,
    dilation_0: usize,
    dilation_1: usize,
    padding_0: usize,
    padding_1: usize,
    groups: usize,
}

#[cube(launch, address_type = "dynamic")]
fn conv_transpose2d_direct_kernel<E: Numeric>(
    input: &Tensor<E>,
    weight: &Tensor<E>,
    bias: ComptimeOption<&[E]>,
    mut output: LinearViewMut<'_, E>,
    out_shape: Sequence<FastDivmod<usize>>,
    args: ConvArgs,
    #[define(E)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= output.shape() {
        terminate!();
    }

    let in_c_per_group = weight.shape(0) / args.groups;
    let out_c_per_group = weight.shape(1);
    let kernel_h = weight.shape(2);
    let kernel_w = weight.shape(3);

    let (_, pos) = decompose_linear(ABSOLUTE_POS, &out_shape);
    let [batch, oc_out, out_y, out_x] = *pos else {
        unreachable!()
    };

    let k = oc_out / out_c_per_group;
    let group = k % args.groups;
    let out_c = oc_out - out_c_per_group * group;

    let in_c_start = group * in_c_per_group;
    let in_c_end = in_c_start + in_c_per_group;

    let stride_0_i = args.conv_stride_0 as i32;
    let stride_1_i = args.conv_stride_1 as i32;

    let kms_h = (kernel_h * args.dilation_0) as i32 - stride_0_i;
    let kms_w = (kernel_w * args.dilation_1) as i32 - stride_1_i;

    let y_start = ((out_y + args.padding_0) as i32 - kms_h) / stride_0_i;
    let x_start = ((out_x + args.padding_1) as i32 - kms_w) / stride_1_i;

    let y_end = clamp(kms_h + y_start + 1, 0, input.shape(2) as i32) as usize;
    let x_end = clamp(kms_w + x_start + 1, 0, input.shape(3) as i32) as usize;
    let y_start = clamp_min(y_start, 0) as usize;
    let x_start = clamp_min(x_start, 0) as usize;

    let idx_input_batch = batch * input.stride(0);
    let idx_weight_oc = out_c * weight.stride(1);

    let bias: ComptimeOption<E> = bias.as_ref().map(|bias| bias[oc_out]);
    let mut sum = bias.unwrap_or_default();

    let numerator_h_base = out_y + args.padding_0;
    let numerator_w_base = out_x + args.padding_1;

    for in_c in in_c_start..in_c_end {
        let idx_input_ic = in_c * input.stride(1);
        let idx_weight_ic = in_c * weight.stride(0);

        for in_y in y_start..y_end {
            let numerator_tmp = in_y * args.conv_stride_0;
            let numerator_h = numerator_h_base - numerator_tmp;

            if numerator_h_base >= numerator_tmp && numerator_h.is_multiple_of(args.dilation_0) {
                let kernel_y = numerator_h / args.dilation_0;
                let idx_input_y = in_y * input.stride(2);
                let idx_weight_ky = kernel_y * weight.stride(2);

                for in_x in x_start..x_end {
                    let numerator_tmp = in_x * args.conv_stride_1;
                    let numerator_w = numerator_w_base - numerator_tmp;

                    if numerator_w_base >= numerator_tmp
                        && numerator_w.is_multiple_of(args.dilation_1)
                    {
                        let kernel_x = numerator_w / args.dilation_1;
                        let idx_input_x = in_x * input.stride(3);
                        let idx_weight_kx = kernel_x * weight.stride(3);

                        let index_input =
                            idx_input_batch + idx_input_ic + idx_input_y + idx_input_x;
                        let index_weight =
                            idx_weight_ic + idx_weight_oc + idx_weight_ky + idx_weight_kx;

                        let value = input[index_input];
                        let weight = weight[index_weight];

                        sum += value * weight;
                    }
                }
            }
        }
    }

    output.write(ABSOLUTE_POS, sum);
}

/// Perform a 2D convolution transposition using the direct algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv_transpose2d_direct(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvTransposeOptions<2>,
) -> Result<CubeTensor, ConvSetupError> {
    let [batch_size, _, in_height, in_width] = input.meta.shape().dims();
    let [_, out_channels, kernel_0, kernel_1] = weight.meta.shape().dims();

    let out_0 = (in_height - 1) * options.stride[0]
        + options.dilation[0] * (kernel_0 - 1)
        + options.padding_out[0]
        - 2 * options.padding[0]
        + 1;
    let out_1 = (in_width - 1) * options.stride[1]
        + options.dilation[1] * (kernel_1 - 1)
        + options.padding_out[1]
        - 2 * options.padding[1]
        + 1;

    let shape_out = Shape::new([batch_size, out_channels * options.groups, out_0, out_1]);

    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
        input.dtype,
    );

    let num_elems = output.meta.num_elements();
    let cube_dim = CubeDim::new(&input.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&input.client, num_elems, cube_dim);
    let dtype = input.dtype;

    conv_transpose2d_direct_kernel::launch(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(input, weight, bias, output),
        input.into_tensor_arg(),
        weight.into_tensor_arg(),
        bias.map(|bias| bias.into_buffer_arg()).into(),
        output.clone().into_linear_view(),
        shape_divmod(&output),
        ConvArgsLaunch::new(
            options.stride[0],
            options.stride[1],
            options.dilation[0],
            options.dilation[1],
            options.padding[0],
            options.padding[1],
            options.groups,
        ),
        dtype_to_storage_type(dtype),
    );

    Ok(output)
}

/// The loop above runs `kernel * dilation - stride + 1` times and does a
/// multiply-accumulate on `kernel` of them, so this kernel's cost is set by the
/// dilation it was handed rather than by the arithmetic it performs.
///
/// Write-up, evidence and fix plan:
/// `crates/burn-cubecl/docs/dgrad-transpose-loop-scales-with-dilation.md`.
///
/// ```bash
/// cargo test -p burn-cubecl --release --features metal \
///     dilation_loop_length -- --ignored --nocapture --test-threads=1
/// ```
///
/// **This test passes while the defect is present.** A failure because the sweep
/// went flat is the result the document asks for.
#[cfg(test)]
mod dilation_loop_length {
    use std::time::Instant;

    use burn_backend::ops::ConvTransposeOptions;
    use burn_std::Shape;
    use cubecl::client::Client;

    use super::conv_transpose2d_direct;
    use crate::{
        CubeDevice, kernel::conv::conv_transpose2d_col2im, ops::numeric::full, tensor::CubeTensor,
    };

    /// The widest residual convolution of a dilated TCN, where this was found:
    /// `[128, 128, 160] -> [128, 128, 176]`, `kernel = 5`, unit stride,
    /// `groups = 1`. Its data gradient is a transposed convolution from 176
    /// back to 160.
    const BATCH: usize = 128;
    const CHANNELS: usize = 128;
    const KERNEL: usize = 5;
    const STRIDE: usize = 1;
    const LENGTH_IN: usize = 160;
    const LENGTH_OUT: usize = 176;

    /// The dilations a TCN stacks.
    const DILATIONS: [usize; 3] = [1, 2, 4];

    /// The forward padding that lands a dilation on [`LENGTH_OUT`], inverting
    /// `out = in + 2 * padding - dilation * (kernel - 1)`.
    ///
    /// This is what makes the sweep a controlled experiment. Padding moves
    /// where the loop above *starts*, never how long it runs, so paying for the
    /// dilation here leaves the loop length as the only thing that varies:
    /// every row below runs on the same tensors, produces the same shape and
    /// performs the same multiply-accumulates.
    const fn padding(dilation: usize) -> usize {
        (LENGTH_OUT - LENGTH_IN + dilation * (KERNEL - 1)) / 2
    }

    /// How many times the loop above runs, read off its own bounds: `kms_h =
    /// kernel * dilation - stride`, and `y_end - y_start` spans `kms_h + 1`.
    /// Of these, `kernel` do a multiply-accumulate.
    const fn iterations(dilation: usize) -> usize {
        KERNEL * dilation - STRIDE + 1
    }

    /// Median of five timed runs after twenty untimed ones, with the queue
    /// drained inside each window.
    ///
    /// Twenty, not two: `col2im` reaches `matmul`'s autotune internally, and
    /// tuning resolves asynchronously — until it settles, a call runs whichever
    /// candidate the tuner provisionally picked, so a short warm-up measures
    /// the transient rather than the kernel. The kernel above needs no warm-up
    /// beyond compilation, but both columns are measured the same way.
    ///
    /// `op` hands its result back so that nothing it queued is eliminated as
    /// dead.
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

    /// The kernel above costs what its loop is long.
    ///
    /// Called directly, which is what isolates the kernel: reaching it through
    /// `conv_data_backward_fallback` goes by way of `conv_transpose2d`'s own
    /// tuner, which picks `col2im` for some of these dilations and this kernel
    /// for others, so an end-to-end sweep measures the choice as much as the
    /// kernel. The shapes are exactly the ones the gradient path hands over: a
    /// 1D problem reshaped to `width = 1`, `padding_out = 0`.
    ///
    /// A kernel priced by its arithmetic gives three equal rows. One priced by
    /// its loop gives a row per dilation, in proportion — and a flat
    /// `ms / iteration` column is what says which of the two this is.
    #[test]
    #[ignore = "measures a device; run explicitly, see the module docs"]
    fn the_transposed_convolution_costs_a_dilation_multiple_of_its_arithmetic() {
        let device = CubeDevice::default();
        let client = device.client();

        let filled = |shape: Shape| full::<f32>(shape, &device, 0.031_25);
        let out_grad = filled(Shape::new([BATCH, CHANNELS, LENGTH_OUT, 1]));
        let weight = filled(Shape::new([CHANNELS, CHANNELS, KERNEL, 1]));

        println!("\n=== conv_transpose2d_direct against dilation ===");
        println!(
            "  {device:?}, [{BATCH}, {CHANNELS}, {LENGTH_OUT}, 1] -> \
             [{BATCH}, {CHANNELS}, {LENGTH_IN}, 1], kernel {KERNEL}"
        );
        println!("  dilation  padding  loop  useful  direct ms  ms / iteration  col2im ms");

        let mut measured = Vec::new();
        for dilation in DILATIONS {
            let options = || {
                ConvTransposeOptions::new(
                    [STRIDE, 1],
                    [padding(dilation), 0],
                    [0, 0],
                    [dilation, 1],
                    1,
                )
            };
            let direct = median_ms(&client, || {
                conv_transpose2d_direct(out_grad.clone(), weight.clone(), None, options())
                    .expect("the direct transposed convolution declines nothing")
            });
            // The other candidate this file's tuner has, as the reference for
            // what the same gradient costs without the loop.
            let col2im = median_ms(&client, || {
                conv_transpose2d_col2im(out_grad.clone(), weight.clone(), None, options())
                    .expect("col2im declines nothing here")
            });

            println!(
                "  {dilation:>8}  {:>7}  {:>4}  {KERNEL:>6}  {direct:>9.3}  {:>14.4}  {col2im:>9.3}",
                padding(dilation),
                iterations(dilation),
                direct / iterations(dilation) as f64,
            );
            measured.push(direct);
        }

        // The extra iterations dilation 4 walks, priced at what the sweep says
        // an iteration costs. This, and not the ratio, is the thing to fix: the
        // loop-length model over-predicts because the kernel also carries a
        // fixed cost, but the waste is real either way.
        let waste = measured[2] - measured[0];
        let share = waste / measured[2] * 100.0;
        println!(
            "  dilation 4 / dilation 1  {:.2}x   (loop length ratio {:.0}x)",
            measured[2] / measured[0],
            iterations(DILATIONS[2]) as f64 / iterations(DILATIONS[0]) as f64,
        );
        println!(
            "  {} extra iterations, none of them arithmetic: {waste:.1} ms, {share:.0}% of the kernel",
            iterations(DILATIONS[2]) - iterations(DILATIONS[0]),
        );

        assert!(
            share >= 30.0,
            "the iterations dilation 4 adds over dilation 1 account for only \
             {share:.0}% of the kernel, on identical tensors; retire \
             crates/burn-cubecl/docs/dgrad-transpose-loop-scales-with-dilation.md",
        );
    }
}
