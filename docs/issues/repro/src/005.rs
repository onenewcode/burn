use burn_core::data::dataloader::{DataLoaderBuilder, batcher::Batcher};
use burn_core::data::dataset::{Dataset, DatasetError};
use burn_core::tensor::Device;

struct BrokenFirstItem;
impl Dataset<i32> for BrokenFirstItem {
    fn len(&self) -> usize {
        2
    }
    fn get(&self, index: usize) -> Result<i32, DatasetError> {
        println!("get({index})");
        if index == 0 {
            Err(DatasetError::new(std::io::Error::other("bad sample")))
        } else {
            Ok(42)
        }
    }
}
struct Identity;
impl Batcher<i32, Vec<i32>> for Identity {
    fn batch(&self, items: Vec<i32>, _: &Device) -> Vec<i32> {
        items
    }
}
fn main() {
    let loader = DataLoaderBuilder::new(Identity)
        .batch_size(1)
        .set_device(Device::flex())
        .build(BrokenFirstItem);
    let mut iter = loader.iter();
    for _ in 0..3 {
        let result = iter.next();
        println!("next: {result:?}");
        assert!(matches!(result, Some(Err(_))));
    }
    println!("items_processed={}", iter.progress().items_processed);
    assert_eq!(iter.progress().items_processed, 0);
}
