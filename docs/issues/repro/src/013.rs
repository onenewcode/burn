mod http_fixture;

fn main() {
    let body = http_fixture::download(
        "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found",
    )
    .unwrap();
    assert_eq!(body, b"not found");
    println!(
        "HTTP 404 returned as successful bytes: {:?}",
        String::from_utf8(body).unwrap()
    );
    let control = http_fixture::download(
        "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
    )
    .unwrap();
    assert_eq!(control, b"ok");
    println!(
        "HTTP 200 control: {:?}",
        String::from_utf8(control).unwrap()
    );
}
