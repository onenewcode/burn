use burn_core::data::dataloader::Progress;
use burn_core::tensor::{Device, Tensor};
use burn_train::metric::{
    BleuInput, BleuScore, Metric, MetricMetadata, Numeric, RougeLInput, RougeLScore, WerInput,
    WordErrorRate,
};

fn main() {
    let device = Device::flex();
    let metadata = MetricMetadata {
        progress: Progress::new(1, 1, None),
        iteration: None,
        lr: None,
    };
    let mut metric = WordErrorRate::new();
    metric
        .update(
            &WerInput::new(
                Tensor::from_data([[1, 2, 3]], &device),
                Tensor::from_data([[1, 2]], &device),
            ),
            &metadata,
        )
        .unwrap();
    let value = metric.value().unwrap().current();
    println!("longer prediction: WER={value}% (expected 50%)");
    assert_eq!(value, 0.0); // Confirm the current defect, not the desired behavior.

    let shorter = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        metric
            .update(
                &WerInput::new(
                    Tensor::from_data([[1]], &device),
                    Tensor::from_data([[1, 2]], &device),
                ),
                &metadata,
            )
            .unwrap();
    }));
    println!("shorter prediction panicked: {}", shorter.is_err());
    assert!(shorter.is_err());

    let mut bleu = BleuScore::with_max_n(1);
    let mut rouge = RougeLScore::new();
    let long = Tensor::from_data([[1, 2, 3]], &device);
    let reference = Tensor::from_data([[1, 2]], &device);
    bleu.update(&BleuInput::new(long.clone(), reference.clone()), &metadata)
        .unwrap();
    rouge
        .update(&RougeLInput::new(long, reference.clone()), &metadata)
        .unwrap();
    let b = bleu.value().unwrap().current();
    let r = rouge.value().unwrap().current();
    println!("longer prediction: BLEU-1={b}% (expected 66.6667%), ROUGE-L={r}% (expected 80%)");
    assert_eq!(b, 100.0);
    assert_eq!(r, 100.0);

    let short = Tensor::from_data([[1]], &device);
    let b = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bleu.update(&BleuInput::new(short.clone(), reference.clone()), &metadata)
            .unwrap();
    }));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rouge
            .update(&RougeLInput::new(short, reference.clone()), &metadata)
            .unwrap();
    }));
    assert!(b.is_err() && r.is_err());
    println!("shorter prediction: BLEU and ROUGE both panicked");

    // Equal-length control for all three metrics.
    let input = Tensor::from_data([[1, 2]], &device);
    metric
        .update(&WerInput::new(input.clone(), input.clone()), &metadata)
        .unwrap();
    bleu.update(&BleuInput::new(input.clone(), input.clone()), &metadata)
        .unwrap();
    rouge
        .update(&RougeLInput::new(input.clone(), input), &metadata)
        .unwrap();
    assert_eq!(metric.value().unwrap().current(), 0.0);
    assert_eq!(bleu.value().unwrap().current(), 100.0);
    assert_eq!(rouge.value().unwrap().current(), 100.0);
    println!("equal-length controls: WER=0%, BLEU-1=100%, ROUGE-L=100%");
}
