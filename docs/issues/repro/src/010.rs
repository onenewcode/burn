use burn_core::tensor::{Device, Tensor};
use burn_nn::GroupNormConfig;

fn main() {
    let device = Device::flex();
    let norm = GroupNormConfig::new(1, 2).init(&device);
    let small = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        norm.forward(Tensor::<3>::from_data([[[1.0], [3.0]]], &device))
            .into_data()
    }));
    assert!(small.is_err());
    println!("GroupNorm [1, 2, 1]: panicked (expected approximately [-1, 1])");
    let control = norm
        .forward(Tensor::<3>::from_data(
            [[[1.0], [3.0]], [[1.0], [3.0]]],
            &device,
        ))
        .into_data()
        .try_into_vec_as::<f32>()
        .unwrap();
    for (actual, expected) in control.iter().zip([-1.0, 1.0, -1.0, 1.0]) {
        assert!((actual - expected).abs() < 1e-4);
    }
    println!("duplicated batch control: {control:?}");
}
