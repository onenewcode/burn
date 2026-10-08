use burn_core::tensor::{Device, Tensor};
use burn_nn::loss::{Reduction, TripletMarginLossConfig};

fn main() {
    let device = Device::flex().autodiff();
    let zero = Tensor::<2>::zeros([1, 2], &device).require_grad();
    let norm = burn_linalg::l2_norm(zero.clone(), 1).sum();
    let gradient = zero
        .grad(&norm.backward())
        .unwrap()
        .into_data()
        .try_into_vec_as::<f32>()
        .unwrap();
    println!("L2 at zero: gradient={gradient:?}");
    assert!(gradient.iter().all(|v| v.is_nan()));

    let anchor = Tensor::<2>::from_data([[0.0, 0.0]], &device).require_grad();
    let loss = TripletMarginLossConfig::new().init().forward(
        anchor.clone(),
        Tensor::from_data([[0.0, 0.0]], &device),
        Tensor::from_data([[2.0, 0.0]], &device),
        Reduction::Mean,
    );
    let value = loss.clone().into_scalar::<f32>();
    let gradient = anchor
        .grad(&loss.backward())
        .unwrap()
        .into_data()
        .try_into_vec_as::<f32>()
        .unwrap();
    println!("inactive triplet: loss={value}, gradient={gradient:?} (expected [0, 0])");
    assert_eq!(value, 0.0);
    assert!(gradient.iter().all(|v| v.is_nan()));
}
