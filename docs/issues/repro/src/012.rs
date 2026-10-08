mod http_fixture;

fn main() {
    let chunked = http_fixture::download(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\nok\r\n0\r\n\r\n",
    );
    println!("valid chunked HTTP 200: panicked={}", chunked.is_err());
    assert!(chunked.is_err());
    let control = http_fixture::download(
        "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
    )
    .unwrap();
    assert_eq!(control, b"ok");
    println!(
        "Content-Length control: {:?}",
        String::from_utf8(control).unwrap()
    );
}
