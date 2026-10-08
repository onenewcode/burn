use burn_core::data::dataloader::{DataLoaderBuilder, batcher::Batcher};
use burn_core::data::dataset::InMemDataset;
use burn_core::tensor::Device;
use std::sync::mpsc;
use std::time::Duration;

struct Identity;
impl Batcher<i32, Vec<i32>> for Identity {
    fn batch(&self, items: Vec<i32>, _: &Device) -> Vec<i32> {
        items
    }
}
fn main() {
    let loader = DataLoaderBuilder::new(Identity)
        .batch_size(1)
        .num_workers(1)
        .set_device(Device::flex())
        .build(InMemDataset::new((0..200).collect()));
    let mut first = loader.iter();
    assert_eq!(first.next().unwrap().unwrap(), vec![0]);

    let other = loader.clone();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut second = other.iter();
        ready_tx.send(()).unwrap();
        result_tx.send(second.next()).unwrap();
    });
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let blocked = matches!(
        result_rx.recv_timeout(Duration::from_secs(1)),
        Err(mpsc::RecvTimeoutError::Timeout)
    );
    println!("second iterator blocked while first is retained: {blocked}");
    // Release the first receiver so this reproduction itself cannot hang indefinitely.
    drop(first);
    let result = result_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    println!("after dropping first: {result:?}");
    reader.join().unwrap();
    assert!(blocked);
    assert_eq!(result.unwrap().unwrap(), vec![0]);

    // Ordinary sequential reuse works.
    assert_eq!(loader.iter().count(), 200);
    println!("sequential reuse control: 200 batches");
}
