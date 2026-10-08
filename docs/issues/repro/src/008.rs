use burn_core::tensor::{Device, Tensor};
use burn_vision::{Nms, NmsOptions};
fn main() {
    let device = Device::flex();
    let boxes = Tensor::<2>::from_data([[0.0f32, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]], &device);
    let scores = Tensor::<1>::from_data([0.9f32, 0.8], &device);
    println!("Calling public Nms::nms");
    let output = boxes.nms(scores, NmsOptions::default());
    println!("{:?}", output.into_data().try_into_vec_as::<i32>().unwrap());
}
