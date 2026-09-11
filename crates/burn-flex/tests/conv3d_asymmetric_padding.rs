//! Regression tests for https://github.com/tracel-ai/burn/issues/5601
//!
//! `Flex::conv3d` previously ignored `padding_end`. These tests call the backend
//! op directly because the public `burn_tensor::module::conv3d` still rejects
//! asymmetric 3D padding.

use burn_backend::ops::{ConvOptions, ModuleOps};
use burn_backend::TensorData;
use burn_flex::{Flex, FlexTensor};

#[test]
fn test_conv3d_asymmetric_end_padding() {
    // Exact repro from https://github.com/tracel-ai/burn/issues/5601
    let x = FlexTensor::from_data(TensorData::new(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![1, 1, 1, 1, 4],
    ));
    let w = FlexTensor::from_data(TensorData::new(vec![1.0f32, 1.0], vec![1, 1, 1, 1, 2]));
    let opts =
        ConvOptions::<3>::new_with_padding([1, 1, 1], [(0, 0), (0, 0), (0, 1)], [1, 1, 1], 1);
    let out = Flex::conv3d(x, w, None, opts);

    // Bug: shape [1,1,1,1,3] values [3,5,7]
    // Expected: shape [1,1,1,1,4] values [3,5,7,4]
    assert_eq!(out.layout().shape().to_vec(), vec![1, 1, 1, 1, 4]);
    let values: Vec<f32> = out.into_data().try_into_vec().unwrap();
    assert_eq!(values, vec![3.0, 5.0, 7.0, 4.0]);
}

#[test]
fn test_conv3d_asymmetric_padding_matches_conv2d() {
    // Issue 5601: the equivalent conv2d call with [(0, 0), (0, 1)] returns
    // shape [1,1,1,4] = [3,5,7,4]. conv3d must match that.
    let x3 = FlexTensor::from_data(TensorData::new(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![1, 1, 1, 1, 4],
    ));
    let w3 = FlexTensor::from_data(TensorData::new(vec![1.0f32, 1.0], vec![1, 1, 1, 1, 2]));
    let opts3 =
        ConvOptions::<3>::new_with_padding([1, 1, 1], [(0, 0), (0, 0), (0, 1)], [1, 1, 1], 1);
    let out3 = Flex::conv3d(x3, w3, None, opts3);

    let x2 = FlexTensor::from_data(TensorData::new(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![1, 1, 1, 4],
    ));
    let w2 = FlexTensor::from_data(TensorData::new(vec![1.0f32, 1.0], vec![1, 1, 1, 2]));
    let opts2 = ConvOptions::<2>::new_with_padding([1, 1], [(0, 0), (0, 1)], [1, 1], 1);
    let out2 = Flex::conv2d(x2, w2, None, opts2);

    assert_eq!(out2.layout().shape().to_vec(), vec![1, 1, 1, 4]);
    assert_eq!(out3.layout().shape().to_vec(), vec![1, 1, 1, 1, 4]);

    let values_2d: Vec<f32> = out2.into_data().try_into_vec().unwrap();
    let values_3d: Vec<f32> = out3.into_data().try_into_vec().unwrap();
    assert_eq!(values_2d, vec![3.0, 5.0, 7.0, 4.0]);
    assert_eq!(values_3d, values_2d);
}
