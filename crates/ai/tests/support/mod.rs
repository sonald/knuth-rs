#![allow(dead_code)]

use std::ffi::OsString;
use std::sync::OnceLock;

use ai::{Api, KnownApi, Model, ModelCost, Provider};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

#[derive(Debug, Clone)]
pub struct CapturedRequest {
    pub request: String,
}

pub struct EnvVarGuard {
    name: &'static str,
    previous: Option<OsString>,
}

impl EnvVarGuard {
    pub fn set(name: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(name);
        unsafe {
            std::env::set_var(name, value);
        }
        Self { name, previous }
    }

    pub fn remove(name: &'static str) -> Self {
        let previous = std::env::var_os(name);
        unsafe {
            std::env::remove_var(name);
        }
        Self { name, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            match self.previous.take() {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }
}

pub const TEST_RSA_PRIVATE_KEY: &str = r#"-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDbWpf/I/lQhtSl
hMVzQGtUkBF9eq+MHxXzWpg8fu41Eka9spOsp833v4podDbCVZcgmsflr8UsR2NN
RnY5uem4yJTS0UD2F75QZXw3w/LIo7XJjNnSHqD5RKrbrJ436fni+SJQiQFmMeY7
nuy4qPy2ZZXNv9e8s3SsTZoYdhagDF4bSRL73TgvxrNiatJfu63NAqTt4Wp8F81F
c6EFsAqpSXYUi7Rn9kYzntwHwNn6oBb77Uw3TdopDCMCxvJYQNjyT26cJMkJ2wFj
BXyR1583QerD40ag1ixRcLe2UxdCrDYNnwzlMFzccGBE7u042E8X2fgFtTbXmNi8
2y1OQx+FAgMBAAECggEAMQ3Rb/1bg+ajJ2DJxzxgxEhzoNwO5gcNoZ5g7SZ1nui3
aTTGUZY1OXJcQX+7vznk0iXLDlKAhaZxTpazWbV5zxkMLxkcrewhY1lOrinj4Xq6
7JfTHmo7FYOFshqoR1jLyTZtthTtey0tj6e6yJEB8shE9/4vAMQhE2dHTrEZ3jB7
oRwneAH9a9K/MrvClmHDpiNfdrhU4DW4T92T6IRNM28fkevUl/BmMv7SV/EMWbGN
1mGwwBw1kDhYIFJDKKhVJIGXPW2Doyyhp8BAT2QBGEcC5Wc1GiC44uXyCdZMlx4Y
Q0jCwGfJHd4XtJMnRJK1YRs30UPLaq2p42EsT+TwAQKBgQD9ZldmjcFfj+r+2KLc
kFPKU2fBpdKDK91Cqump/zWx02tx2jtMktnXkTIK6gxBDBYUNzIeRBBf68RDAF/f
Oe0lS+o/9F/pECatIsnlNyDfokyBoN6FGHCts2TwmchzjtJ4rhb3JP639UROECWZ
Rbp7aM/aHl2OzwJJNcs4+nrOcQKBgQDdmtEawBwH5OLHcmSnPXF+ucMI1Z3K7+dj
YSS6jVJINhI53nJjS4WMMxsd2pJtJVpeTEkHWdxmdX3ylTm9IOvGbhpbY35tqF9I
t0GZUSLlp9JH8RhqUB68CfGzRcIklgoZk38JSzBcI/jUZTm/+M+t40qFF30qBDzE
46e3GnfUVQKBgQC7h6LFBcGHLGYYJlEY9ELeaC1QNZz+cFb2ALCem32sVa+deYkL
GV7YVt73DtD0zrIEUfjoRyzrH/uGLl/FPwRO5si8fekA/W/yD93koZDVkDIYeOpV
C4pQMoRQPy8GvjrrDsN2Mc3EbGIZd3+r19uzexTf8jsA9hhV/9afG1gJMQKBgHGk
Gudk7Pr/XWx6NTOuRq1+BY5aPXj8XeSQxI0GO9PcJqyWboKNEAc9jgJZPA3MwfLp
m+mxI11Hkzb7X4ilgUNY4xtKgmMpnPNlRrag7Qxoa2WJNcQPIjO7xb7xXwX0C2ni
QZs6e6pEqC4DWwIfTiEWFfj6eq05TxCIzlEPubOhAoGBAKXB8oBAmjcl+rzuWSHd
8qLCIPo/LWPwnnSTBikWvy38R1nbK8BV1RZFM6q0QB0IKyJnDnBlbZF29/SS/Urj
xNjs+VWQGqNS8/F6M0nqo8IjaLTvTlvFiMT2v+xBEKx6JiTcr0qJQthFIQSgqb9T
Y3+GdKlmV1FudX0aYfUua00a
-----END PRIVATE KEY-----
"#;

pub fn env_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

pub async fn serve_once(body: &'static [u8], content_type: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 8192];
        let _ = socket.read(&mut buf).await;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.write_all(body).await.unwrap();
        socket.flush().await.unwrap();
    });
    format!("http://{addr}")
}

pub async fn serve_capture_once(
    body: &'static [u8],
    content_type: &'static str,
) -> (String, oneshot::Receiver<CapturedRequest>) {
    serve_capture_once_with_status("200 OK", body, content_type).await
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut request = Vec::new();
    let header_end = loop {
        let mut chunk = [0u8; 4096];
        let n = socket.read(&mut chunk).await.unwrap_or(0);
        if n == 0 {
            break request.len();
        }
        request.extend_from_slice(&chunk[..n]);
        if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let content_length = String::from_utf8_lossy(&request[..header_end])
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    while request.len() < header_end + content_length {
        let mut chunk = [0u8; 4096];
        let n = socket.read(&mut chunk).await.unwrap_or(0);
        if n == 0 {
            break;
        }
        request.extend_from_slice(&chunk[..n]);
    }
    String::from_utf8_lossy(&request).to_string()
}

pub async fn serve_capture_once_with_status(
    status: &'static str,
    body: &'static [u8],
    content_type: &'static str,
) -> (String, oneshot::Receiver<CapturedRequest>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        let _ = tx.send(CapturedRequest { request });
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.write_all(body).await.unwrap();
        socket.flush().await.unwrap();
    });
    (format!("http://{addr}"), rx)
}

pub async fn serve_capture_hanging_once() -> (String, oneshot::Receiver<CapturedRequest>) {
    serve_capture_hanging_once_with_response(None).await
}

pub async fn serve_hanging_response_body_once() -> (String, oneshot::Receiver<CapturedRequest>) {
    serve_capture_hanging_once_with_response(Some(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 128\r\nConnection: close\r\n\r\n{",
    ))
    .await
}

async fn serve_capture_hanging_once_with_response(
    response: Option<&'static [u8]>,
) -> (String, oneshot::Receiver<CapturedRequest>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_request(&mut socket).await;
        if let Some(response) = response {
            socket.write_all(response).await.unwrap();
            socket.flush().await.unwrap();
        }
        let _ = tx.send(CapturedRequest { request });
        std::future::pending::<()>().await;
    });
    (format!("http://{addr}"), rx)
}

pub fn aws_eventstream_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
    let name = b":event-type";
    let mut headers = Vec::new();
    headers.push(name.len() as u8);
    headers.extend_from_slice(name);
    headers.push(7);
    headers.extend_from_slice(&(event_type.len() as u16).to_be_bytes());
    headers.extend_from_slice(event_type.as_bytes());

    let total_len = 12 + headers.len() + payload.len() + 4;
    let mut out = Vec::new();
    out.extend_from_slice(&(total_len as u32).to_be_bytes());
    out.extend_from_slice(&(headers.len() as u32).to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&headers);
    out.extend_from_slice(payload);
    out.extend_from_slice(&0u32.to_be_bytes());
    out
}

pub fn model(api: KnownApi, provider: &str, id: &str, base_url: String) -> Model {
    Model {
        id: id.into(),
        name: id.into(),
        api: Api::known(api),
        provider: Provider::from(provider),
        base_url,
        reasoning: false,
        thinking_level_map: None,
        input: vec![],
        cost: ModelCost {
            input: 1.0,
            output: 2.0,
            cache_read: 0.25,
            cache_write: 1.25,
        },
        context_window: 128_000,
        max_tokens: 4096,
        headers: None,
        compat: None,
    }
}
