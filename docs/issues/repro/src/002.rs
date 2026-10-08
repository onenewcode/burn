use burn_core::tensor::{Device, Tensor};
use burn_nn::loss::BinaryCrossEntropyLossConfig;

fn main() {
    let device = Device::flex().autodiff();
    let prediction = Tensor::<1>::from_data([0.0, 1.0], &device).require_grad();
    let loss = BinaryCrossEntropyLossConfig::new()
        .init(&device)
        .forward(prediction.clone(), Tensor::from_data([0, 1], &device));
    let value = loss.clone().into_scalar::<f32>();
    let gradient = prediction
        .grad(&loss.backward())
        .unwrap()
        .into_data()
        .try_into_vec_as::<f32>()
        .unwrap();
    println!("probability input: loss={value}, gradient={gradient:?}");
    assert_eq!(value, 0.0);
    assert!(gradient.iter().all(|v| v.is_nan()));

    let logits = Tensor::<1>::from_data([-100.0, 100.0], &device).require_grad();
    let loss = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init(&device)
        .forward(logits.clone(), Tensor::from_data([0, 1], &device));
    let gradient = logits
        .grad(&loss.backward())
        .unwrap()
        .into_data()
        .try_into_vec_as::<f32>()
        .unwrap();
    println!("logits control: gradient={gradient:?}");
    assert!(gradient.iter().all(|v| v.is_finite()));
}
