//! OpenAI Codex / ChatGPT OAuth flow. TODO: 1:1 port of
//! `packages/ai/src/auth/oauth/openai-codex.ts`.

use serde::Deserialize;

use super::types::{OAuthCredentials, OAuthLoginCallbacks};

const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    expires_in: i64,
}

pub async fn login(_callbacks: OAuthLoginCallbacks) -> Result<OAuthCredentials, String> {
    Err("openai-codex OAuth not yet implemented".into())
}

pub async fn refresh(creds: &OAuthCredentials) -> Result<OAuthCredentials, String> {
    let client = crate::utils::node_http_proxy::build_client(Some(30_000))
        .map_err(|error| format!("OpenAI Codex OAuth client error: {error}"))?;
    refresh_with_url(creds, &client, TOKEN_URL).await
}

async fn refresh_with_url(
    creds: &OAuthCredentials,
    client: &reqwest::Client,
    token_url: &str,
) -> Result<OAuthCredentials, String> {
    let refresh_token = creds
        .refresh_token
        .as_deref()
        .filter(|token| !token.trim().is_empty())
        .ok_or_else(|| "OpenAI Codex OAuth refresh requires refresh_token".to_string())?;
    let response = client
        .post(token_url)
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", CLIENT_ID),
        ])
        .send()
        .await
        .map_err(|error| format!("OpenAI Codex OAuth refresh request failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "OpenAI Codex OAuth refresh failed with status {}",
            response.status()
        ));
    }

    let token: TokenResponse = response
        .json()
        .await
        .map_err(|error| format!("OpenAI Codex OAuth refresh returned invalid JSON: {error}"))?;
    if token.access_token.trim().is_empty() || token.refresh_token.trim().is_empty() {
        return Err("OpenAI Codex OAuth refresh response is missing token fields".to_string());
    }
    if token.expires_in <= 0 {
        return Err("OpenAI Codex OAuth refresh expires_in is invalid".to_string());
    }
    let expires_at = chrono::Utc::now()
        .timestamp_millis()
        .checked_add(
            token
                .expires_in
                .checked_mul(1_000)
                .ok_or_else(|| "OpenAI Codex OAuth refresh expires_in is invalid".to_string())?,
        )
        .ok_or_else(|| "OpenAI Codex OAuth refresh expires_in is invalid".to_string())?;

    Ok(OAuthCredentials {
        access_token: token.access_token,
        refresh_token: Some(token.refresh_token),
        expires_at: Some(expires_at),
        extra: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    async fn serve_once(
        status: &'static str,
        body: &'static str,
    ) -> (String, oneshot::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (request_tx, request_rx) = oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            loop {
                let read = socket.read(&mut buffer).await.unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                let request_text = String::from_utf8_lossy(&request);
                let Some((headers, body)) = request_text.split_once("\r\n\r\n") else {
                    continue;
                };
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        line.split_once(':').and_then(|(name, value)| {
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                    })
                    .unwrap_or_default();
                if body.len() >= content_length {
                    break;
                }
            }
            let _ = request_tx.send(String::from_utf8(request).unwrap());
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        (format!("http://{address}/oauth/token"), request_rx)
    }

    fn credentials(refresh_token: Option<&str>) -> OAuthCredentials {
        OAuthCredentials {
            access_token: "expired-access".to_string(),
            refresh_token: refresh_token.map(str::to_string),
            expires_at: Some(0),
            extra: None,
        }
    }

    #[tokio::test]
    async fn refresh_posts_form_and_returns_rotated_credentials() {
        let (url, request) = serve_once(
            "200 OK",
            "{\"access_token\":\"new-access\",\"refresh_token\":\"new-refresh\",\"expires_in\":3600}",
        )
        .await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let before = chrono::Utc::now().timestamp_millis();

        let refreshed = refresh_with_url(&credentials(Some("old-refresh")), &client, &url)
            .await
            .unwrap();

        let request = request.await.unwrap();
        assert!(request.starts_with("POST /oauth/token HTTP/1.1\r\n"));
        assert!(request.contains("content-type: application/x-www-form-urlencoded"));
        assert!(request.contains("grant_type=refresh_token"));
        assert!(request.contains("refresh_token=old-refresh"));
        assert!(request.contains(&format!("client_id={CLIENT_ID}")));
        assert_eq!(refreshed.access_token, "new-access");
        assert_eq!(refreshed.refresh_token.as_deref(), Some("new-refresh"));
        assert!(refreshed.expires_at.unwrap() >= before + 3_600_000);
    }

    #[tokio::test]
    async fn refresh_rejects_incomplete_token_response() {
        let (url, _request) = serve_once(
            "200 OK",
            "{\"access_token\":\"new-access\",\"expires_in\":3600}",
        )
        .await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();

        let error = refresh_with_url(&credentials(Some("old-refresh")), &client, &url)
            .await
            .unwrap_err();

        assert!(error.contains("invalid JSON"));
    }

    #[tokio::test]
    async fn refresh_rejects_nonpositive_expires_in() {
        let (url, _request) = serve_once(
            "200 OK",
            "{\"access_token\":\"new-access\",\"refresh_token\":\"new-refresh\",\"expires_in\":0}",
        )
        .await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();

        let error = refresh_with_url(&credentials(Some("old-refresh")), &client, &url)
            .await
            .unwrap_err();

        assert_eq!(error, "OpenAI Codex OAuth refresh expires_in is invalid");
    }

    #[tokio::test]
    async fn refresh_reports_http_failure_without_response_body() {
        let (url, _request) = serve_once("401 Unauthorized", "{\"error\":\"secret detail\"}").await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();

        let error = refresh_with_url(&credentials(Some("old-refresh")), &client, &url)
            .await
            .unwrap_err();

        assert!(error.contains("401 Unauthorized"));
        assert!(!error.contains("secret detail"));
    }
}
