use burn_core::{
    module::Param,
    tensor::{DType, Device, Tensor},
};
use burn_nn::RmsNormConfig;

fn main() {
    let device = Device::flex();
    let mut norm = RmsNormConfig::new(2).init(&device);
    norm.gamma = Param::from_tensor(norm.gamma.val().cast(DType::F64));
    let input = Tensor::<2>::from_data([[1e20f64, 1e20]], &device).cast(DType::F64);
    let reference = (input.clone() / (input.clone().square().mean_dim(1) + 1e-5).sqrt())
        .into_data()
        .try_into_vec_as::<f64>()
        .unwrap();
    let actual = norm
        .forward(input)
        .into_data()
        .try_into_vec_as::<f64>()
        .unwrap();
    println!("RmsNorm F64: {actual:?}; F64 formula: {reference:?}");
    assert_eq!(actual, vec![0.0, 0.0]);
    assert!(reference.iter().all(|v| (v - 1.0).abs() < 1e-12));
    let control = norm
        .forward(Tensor::<2>::from_data([[3.0f64, 4.0]], &device).cast(DType::F64))
        .into_data()
        .try_into_vec_as::<f64>()
        .unwrap();
    for (actual, input) in control.iter().zip([3.0, 4.0]) {
        assert!((actual - input / (12.5f64 + 1e-5).sqrt()).abs() < 1e-6);
    }
    println!("ordinary input control: {control:?}");
}
