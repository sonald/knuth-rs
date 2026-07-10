//! Cloudflare Workers AI / AI Gateway helpers. 1:1 port of
//! `packages/ai/src/providers/cloudflare.ts`.
//!
//! Cloudflare is not its own wire protocol — it rides on `openai-completions` (and Anthropic via
//! the gateway passthrough). This module only resolves the `{VAR}` placeholders in a model's
//! `base_url` from the environment; there is no standalone provider struct.

use crate::types::Model;

/// Workers AI direct endpoint.
pub const CLOUDFLARE_WORKERS_AI_BASE_URL: &str =
    "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1";

/// AI Gateway Unified (compat) API.
pub const CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL: &str =
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat";

/// AI Gateway → OpenAI passthrough.
pub const CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL: &str =
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai";

/// AI Gateway → Anthropic passthrough.
pub const CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL: &str = "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic";

pub fn is_cloudflare_provider(provider: &str) -> bool {
    provider == "cloudflare-workers-ai" || provider == "cloudflare-ai-gateway"
}

/// Substitute the supported Cloudflare ID placeholders in a base URL from the environment.
pub fn resolve_cloudflare_base_url(model: &Model) -> Result<String, String> {
    let url = &model.base_url;
    if !url.contains('{') && !url.contains('}') {
        return Ok(url.clone());
    }
    let mut out = String::with_capacity(url.len());
    let mut chars = url.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            let mut name = String::new();
            let mut closed = false;
            for nc in chars.by_ref() {
                if nc == '}' {
                    closed = true;
                    break;
                }
                if nc == '{' {
                    return Err("invalid nested Cloudflare placeholder".into());
                }
                name.push(nc);
            }
            if !closed {
                return Err("unterminated Cloudflare placeholder".into());
            }
            if !matches!(
                name.as_str(),
                "CLOUDFLARE_ACCOUNT_ID" | "CLOUDFLARE_GATEWAY_ID"
            ) {
                return Err(format!("unsupported Cloudflare placeholder {{{name}}}"));
            }
            match std::env::var(&name) {
                Ok(v) if !v.is_empty() => out.push_str(&v),
                _ => {
                    return Err(format!(
                        "{name} is required for provider {} but is not set.",
                        model.provider.0
                    ));
                }
            }
        } else if c == '}' {
            return Err("unmatched closing brace in Cloudflare base URL".into());
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn model_with_base(base: &str) -> Model {
        Model {
            id: "m".into(),
            name: "m".into(),
            api: Api::known(KnownApi::OpenAICompletions),
            provider: Provider::from("cloudflare-workers-ai"),
            base_url: base.into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![],
            cost: ModelCost::default(),
            context_window: 0,
            max_tokens: 0,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn passthrough_when_no_placeholder() {
        let m = model_with_base("https://example.com/v1");
        assert_eq!(
            resolve_cloudflare_base_url(&m).unwrap(),
            "https://example.com/v1"
        );
    }

    #[test]
    fn errors_on_missing_allowed_env() {
        const NAME: &str = "CLOUDFLARE_GATEWAY_ID";
        let previous = std::env::var_os(NAME);
        unsafe { std::env::remove_var(NAME) };
        let result =
            resolve_cloudflare_base_url(&model_with_base("https://x/{CLOUDFLARE_GATEWAY_ID}/v1"));
        unsafe {
            if let Some(value) = previous {
                std::env::set_var(NAME, value);
            }
        }

        assert!(result.is_err());
    }

    #[test]
    fn unknown_placeholder_is_rejected_even_when_environment_variable_exists() {
        const NAME: &str = "CLOUDFLARE_ARBITRARY_SECRET_FOR_TEST";
        const SECRET: &str = "arbitrary-secret-must-not-be-resolved";
        let previous = std::env::var_os(NAME);
        unsafe { std::env::set_var(NAME, SECRET) };
        let result =
            resolve_cloudflare_base_url(&model_with_base(&format!("https://x/{{{NAME}}}/v1")));
        unsafe {
            match previous {
                Some(value) => std::env::set_var(NAME, value),
                None => std::env::remove_var(NAME),
            }
        }

        let error = result.expect_err("unknown placeholders must be rejected");
        assert!(!error.contains(SECRET));
    }

    #[test]
    fn incomplete_placeholders_are_rejected() {
        const NAME: &str = "CLOUDFLARE_ACCOUNT_ID";
        let previous = std::env::var_os(NAME);
        unsafe { std::env::set_var(NAME, "account-for-incomplete-placeholder") };
        let missing_close =
            resolve_cloudflare_base_url(&model_with_base("https://x/{CLOUDFLARE_ACCOUNT_ID"));
        unsafe {
            match previous {
                Some(value) => std::env::set_var(NAME, value),
                None => std::env::remove_var(NAME),
            }
        }

        assert!(missing_close.is_err());
        assert!(resolve_cloudflare_base_url(&model_with_base("https://x/stray}")).is_err());
    }
}
