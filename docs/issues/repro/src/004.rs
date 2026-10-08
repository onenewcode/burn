use burn_core::tensor::{Device, Tensor};
use burn_nn::loss::{GaussianNLLLossConfig, Reduction};

fn gradient(variance: f32) -> f32 {
    let device = Device::flex().autodiff();
    let var = Tensor::<1>::from_data([variance], &device).require_grad();
    let loss = GaussianNLLLossConfig::new().init().forward(
        Tensor::from_data([0.0], &device),
        Tensor::from_data([1.0], &device),
        var.clone(),
        Reduction::Mean,
    );
    var.grad(&loss.backward()).unwrap().into_scalar::<f32>()
}

fn main() {
    let actual = gradient(1e-8);
    let eps = 1e-6f64;
    let expected = 0.5 * (1.0 / eps - 1.0 / (eps * eps));
    println!("variance=1e-8: gradient={actual}, PyTorch-compatible expectation={expected}");
    assert_eq!(actual, 0.0);
    let control = gradient(0.5);
    println!("variance=0.5 control: gradient={control} (expected -1)");
    assert!((control + 1.0).abs() < 1e-6);
}
