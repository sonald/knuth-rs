mod support;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::oneshot;

#[tokio::test]
async fn captures_request_body_when_tcp_writes_are_segmented() {
    const FIRST_SEGMENT: &[u8] = b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 11\r\n";
    const SECOND_SEGMENT: &[u8] = b"\r\nhello world";

    let (base_url, mut captured) = support::serve_capture_once(b"ok", "text/plain").await;
    let address = base_url.strip_prefix("http://").unwrap().to_string();
    let (first_written, first_written_rx) = oneshot::channel();
    let (write_second, write_second_rx) = oneshot::channel();
    let writer = tokio::spawn(async move {
        let mut socket = TcpStream::connect(address).await.unwrap();
        socket.write_all(FIRST_SEGMENT).await.unwrap();
        socket.flush().await.unwrap();
        first_written.send(()).unwrap();

        write_second_rx.await.unwrap();
        socket.write_all(SECOND_SEGMENT).await.unwrap();
        socket.flush().await.unwrap();

        let mut response = Vec::new();
        socket.read_to_end(&mut response).await.unwrap();
        response
    });

    first_written_rx.await.unwrap();
    assert!(matches!(
        captured.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    write_second.send(()).unwrap();
    let request = captured.await.unwrap().request;
    let response = writer.await.unwrap();

    assert!(
        request.ends_with("\r\n\r\nhello world"),
        "captured request: {request:?}"
    );
    assert!(response.starts_with(b"HTTP/1.1 200 OK\r\n"));
}
