//! Shared OAuth types. 1:1 stub of `packages/ai/src/utils/oauth/types.ts`.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OAuthProviderId {
    Anthropic,
    OpenAICodex,
    GitHubCopilot,
    Google,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct OAuthCredentials {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

impl fmt::Debug for OAuthCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OAuthCredentials")
            .field("access_token", &"<redacted>")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "<redacted>"),
            )
            .field("expires_at", &self.expires_at)
            .field("extra", &self.extra)
            .finish()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OAuthAuthInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<String>,
}

#[derive(Clone, Debug)]
pub struct OAuthSelectOption {
    pub id: String,
    pub label: String,
}

#[derive(Clone, Debug)]
pub enum OAuthPrompt {
    Url { url: String },
    Select { options: Vec<OAuthSelectOption> },
    Code { hint: String },
}

#[derive(Clone, Debug)]
pub struct OAuthLoginCallbacks {
    // TODO: callback wiring once we port the flows.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_debug_redacts_tokens() {
        let creds = OAuthCredentials {
            access_token: "access-secret".into(),
            refresh_token: Some("refresh-secret".into()),
            expires_at: Some(123),
            extra: None,
        };
        let rendered = format!("{creds:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("access-secret"));
        assert!(!rendered.contains("refresh-secret"));
    }
}
