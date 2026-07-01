//! OpenAI-compatible base URL normalization shared by chat-completions and responses providers.

fn base_has_version_suffix(base: &str) -> bool {
    let Some(last) = base.rsplit('/').next() else {
        return false;
    };
    let Some(digits) = last.strip_prefix('v') else {
        return false;
    };
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

fn base_already_versioned(base: &str) -> bool {
    base.ends_with("/v1") || base.contains("/v1/") || base_has_version_suffix(base)
}

/// Normalize an OpenAI-compatible API root before appending route segments.
pub(crate) fn normalize_openai_compat_base(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    if base_already_versioned(trimmed) {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

pub(crate) fn build_chat_completions_url(base: &str) -> String {
    format!("{}/chat/completions", normalize_openai_compat_base(base))
}

pub(crate) fn build_responses_url(base: &str) -> String {
    format!("{}/responses", normalize_openai_compat_base(base))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_root_gets_v1() {
        assert_eq!(
            normalize_openai_compat_base("https://api.openai.com"),
            "https://api.openai.com/v1"
        );
    }

    #[test]
    fn v1_suffix_unchanged() {
        assert_eq!(
            normalize_openai_compat_base("https://api.openai.com/v1"),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            normalize_openai_compat_base("https://api.groq.com/openai/v1"),
            "https://api.groq.com/openai/v1"
        );
    }

    #[test]
    fn non_v1_version_suffix_unchanged() {
        assert_eq!(
            normalize_openai_compat_base("https://ark.cn-beijing.volces.com/api/coding/v3"),
            "https://ark.cn-beijing.volces.com/api/coding/v3"
        );
        assert_eq!(
            normalize_openai_compat_base("https://api.z.ai/api/coding/paas/v4"),
            "https://api.z.ai/api/coding/paas/v4"
        );
    }

    #[test]
    fn chat_completions_url_for_volces_coding() {
        assert_eq!(
            build_chat_completions_url("https://ark.cn-beijing.volces.com/api/coding/v3"),
            "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"
        );
    }

    #[test]
    fn responses_url_does_not_double_v1() {
        assert_eq!(
            build_responses_url("https://api.openai.com"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            build_responses_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            build_responses_url("https://gateway.example.com/acct/gw/openai"),
            "https://gateway.example.com/acct/gw/openai/v1/responses"
        );
    }
}
