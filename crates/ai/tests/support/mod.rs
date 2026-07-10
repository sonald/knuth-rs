#![allow(dead_code)]

use std::sync::OnceLock;

use ai::{Api, KnownApi, Model, ModelCost, Provider};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

#[derive(Debug, Clone)]
pub struct CapturedRequest {
    pub request: String,
}

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
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
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
        let _ = tx.send(CapturedRequest {
            request: String::from_utf8_lossy(&request).to_string(),
        });
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.write_all(body).await.unwrap();
        socket.flush().await.unwrap();
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
