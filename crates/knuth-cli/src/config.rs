use ai::{Api, InputModality, KnownApi, Model, ModelCost, Provider, StreamOptions, get_model};
use anyhow::{Result, anyhow};
use std::env;

pub struct UserSettings {
    pub model: Model,
    pub options: StreamOptions,
}

impl UserSettings {
    pub fn load(model_override: Option<&str>) -> Result<Self> {
        let model = Self::load_model_from_env(model_override)?;
        let options = StreamOptions {
            max_tokens: Some(1024),
            api_key: env_value("KNUTH_API_KEY"),
            ..Default::default()
        };
        Ok(Self { model, options })
    }

    fn load_model_from_env(model_override: Option<&str>) -> Result<Model> {
        let selector = model_override
            .map(str::to_string)
            .or_else(|| env_value("KNUTH_MODEL"))
            .ok_or_else(|| anyhow!("KNUTH_MODEL is not set; pass --model or set KNUTH_MODEL"))?;

        load_model(&selector, env_value)
    }
}

fn load_model<F>(selector: &str, env_var: F) -> Result<Model>
where
    F: Fn(&str) -> Option<String>,
{
    let selector = selector.trim();
    if selector.is_empty() {
        return Err(anyhow!("model selector is empty"));
    }

    let (selector_provider, model_id) = match selector.split_once('/') {
        Some((provider, model)) if !provider.trim().is_empty() && !model.trim().is_empty() => {
            (Some(provider.trim().to_string()), model.trim().to_string())
        }
        _ => (None, selector.to_string()),
    };
    let api_override = env_value_from(&env_var, "KNUTH_API").map(|api| api_from_name(&api));
    let base_url = env_value_from(&env_var, "KNUTH_BASE_URL");
    // ponytail: one selector plus env overrides; add named profiles when editing env is the bottleneck.
    let provider = selector_provider
        .or_else(|| env_value_from(&env_var, "KNUTH_PROVIDER"))
        .unwrap_or_else(|| default_provider_for_api(api_override.as_ref()).to_string());

    let provider = Provider::from(provider);
    let mut model = match get_model(&provider, &model_id) {
        Some(model) => model,
        None => {
            custom_model(
                provider.clone(),
                model_id.clone(),
                api_override
                    .clone()
                    .unwrap_or_else(|| default_api_for_provider(&provider.0)),
                base_url.clone(),
            )?
        }
    };

    if let Some(api) = api_override {
        model.api = api;
    }
    if let Some(base_url) = base_url {
        model.base_url = base_url;
    }

    Ok(model)
}

fn custom_model(
    provider: Provider,
    model_id: String,
    api: Api,
    base_url: Option<String>,
) -> Result<Model> {
    let base_url = base_url.ok_or_else(|| {
        anyhow!(
            "model '{}/{}' was not found; set KNUTH_BASE_URL for a custom model",
            provider.0,
            model_id
        )
    })?;

    Ok(Model {
        id: model_id.clone(),
        name: model_id,
        api,
        provider,
        base_url,
        reasoning: true,
        thinking_level_map: None,
        input: vec![InputModality::Text],
        cost: ModelCost::default(),
        context_window: 1_000_000,
        max_tokens: 10_000,
        headers: None,
        compat: None,
    })
}

fn api_from_name(name: &str) -> Api {
    match name.trim() {
        "openai" | "responses" | "openai-responses" => Api::known(KnownApi::OpenAIResponses),
        "completions" | "openai-completions" => Api::known(KnownApi::OpenAICompletions),
        "chatgpt" | "codex" | "openai-codex" | "openai-codex-responses" => {
            Api::known(KnownApi::OpenAICodexResponses)
        }
        "anthropic" | "anthropic-messages" => Api::known(KnownApi::AnthropicMessages),
        other => Api::from(other),
    }
}

fn default_api_for_provider(provider: &str) -> Api {
    match provider {
        "anthropic" => Api::known(KnownApi::AnthropicMessages),
        "chatgpt" | "codex" | "openai-codex" => Api::known(KnownApi::OpenAICodexResponses),
        _ => Api::known(KnownApi::OpenAIResponses),
    }
}

fn default_provider_for_api(api: Option<&Api>) -> &'static str {
    match api.map(|api| api.0.as_str()) {
        Some("anthropic-messages") => "anthropic",
        Some("openai-codex-responses") => "openai-codex",
        _ => "openai",
    }
}

fn env_value(name: &str) -> Option<String> {
    env::var(name).ok().and_then(nonempty)
}

fn env_value_from<F>(env_var: &F, name: &str) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    env_var(name).and_then(nonempty)
}

fn nonempty(value: String) -> Option<String> {
    let value = value.trim().to_string();
    if value.is_empty() { None } else { Some(value) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[test]
    fn bare_model_with_base_url_uses_openai_responses() {
        let model = load_model(
            "gpt-5.4-mini",
            env(&[("KNUTH_BASE_URL", "https://aicoding.2233.ai")]),
        )
        .unwrap();

        assert_eq!(model.provider.0, "openai");
        assert_eq!(model.api.0, "openai-responses");
        assert_eq!(model.base_url, "https://aicoding.2233.ai");
    }

    #[test]
    fn provider_prefix_selects_codex_catalog_model() {
        let model = load_model(
            "openai-codex/gpt-5.4-mini",
            env(&[("KNUTH_BASE_URL", "https://aicoding.2233.ai")]),
        )
        .unwrap();

        assert_eq!(model.provider.0, "openai-codex");
        assert_eq!(model.api.0, "openai-codex-responses");
        assert_eq!(model.base_url, "https://aicoding.2233.ai");
    }

    #[test]
    fn chatgpt_provider_alias_uses_codex_protocol() {
        let model = load_model(
            "chatgpt/gpt-5.4-mini",
            env(&[("KNUTH_BASE_URL", "https://aicoding.2233.ai")]),
        )
        .unwrap();

        assert_eq!(model.provider.0, "chatgpt");
        assert_eq!(model.api.0, "openai-codex-responses");
        assert_eq!(model.base_url, "https://aicoding.2233.ai");
    }

    #[test]
    fn provider_prefix_can_use_third_party_anthropic_base_url() {
        let model = load_model(
            "anthropic/claude-haiku-4-5-20251001",
            env(&[("KNUTH_BASE_URL", "https://aicoding.2233.ai")]),
        )
        .unwrap();

        assert_eq!(model.provider.0, "anthropic");
        assert_eq!(model.api.0, "anthropic-messages");
        assert_eq!(model.base_url, "https://aicoding.2233.ai");
    }

    #[test]
    fn api_env_can_create_custom_anthropic_model() {
        let model = load_model(
            "local-claude",
            env(&[
                ("KNUTH_API", "anthropic"),
                ("KNUTH_BASE_URL", "https://anthropic.example.test"),
            ]),
        )
        .unwrap();

        assert_eq!(model.provider.0, "anthropic");
        assert_eq!(model.api.0, "anthropic-messages");
        assert_eq!(model.id, "local-claude");
    }
}
