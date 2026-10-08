use std::{
    io::{Read, Write},
    net::TcpListener,
    thread,
    time::Duration,
};

// A complete, deterministic localhost HTTP response; never accesses the public network.
pub fn download(response: &'static str) -> Result<Vec<u8>, Box<dyn std::any::Any + Send>> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/model.bin", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let mut chunk = [0; 1024];
        while !request.windows(4).any(|w| w == b"\r\n\r\n") {
            let n = stream.read(&mut chunk).unwrap();
            assert!(n > 0, "client closed before finishing request headers");
            request.extend_from_slice(&chunk[..n]);
        }
        stream.write_all(response.as_bytes()).unwrap();
    });
    let result = std::panic::catch_unwind(|| {
        burn_std::network::downloader::download_file_as_bytes(&url, "local reproduction")
    });
    server.join().unwrap();
    result
}
