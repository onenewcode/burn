use burn_core::tensor::{Device, Tensor};
use burn_signal::{StftOptions, stft};

fn main() {
    let device = Device::flex();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stft(
            Tensor::<2>::zeros([1, 1024], &device),
            None,
            StftOptions::default(),
        )
    }));
    println!(
        "default n_fft={}, panicked={}",
        StftOptions::default().n_fft,
        failed.is_err()
    );
    assert!(failed.is_err());
    let result = stft(
        Tensor::<2>::zeros([1, 1024], &device),
        None,
        StftOptions::new(512),
    );
    println!("n_fft=512 control: shape={:?}", result.dims());
    assert_eq!(result.dims(), [1, 9, 257, 2]);
}
