mod support;

use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::time::{Duration, sleep};

#[tokio::test]
async fn captures_request_body_when_tcp_writes_are_segmented() {
    let (base_url, captured) = support::serve_capture_once(b"ok", "text/plain").await;
    let address = base_url.strip_prefix("http://").unwrap();
    let mut socket = TcpStream::connect(address).await.unwrap();

    socket
        .write_all(b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 11\r\n")
        .await
        .unwrap();
    socket.flush().await.unwrap();
    sleep(Duration::from_millis(20)).await;

    let _ = socket.write_all(b"\r\nhello world").await;
    let request = captured.await.unwrap().request;

    assert!(
        request.ends_with("\r\n\r\nhello world"),
        "captured request: {request:?}"
    );
}
