use ai::{
    Api, CacheRetention, InputModality, KnownApi, Model, ModelCost, Provider, StreamOptions,
    get_model,
};
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

const CONFIG_DIR_NAME: &str = "knuth";
const CONFIG_FILE_NAME: &str = "knuth.yaml";

pub struct UserSettings {
    pub model: Model,
    pub options: StreamOptions,
}

impl UserSettings {
    pub fn load(model_override: Option<&str>, config_override: Option<&Path>) -> Result<Self> {
        let config = load_file_config(
            config_override
                .map(Path::to_path_buf)
                .or_else(|| env_value("KNUTH_CONFIG").map(PathBuf::from)),
        )?;
        let model = Self::load_model_from_env(model_override, &config)?;
        let options = load_options(&config);
        Ok(Self { model, options })
    }

    fn load_model_from_env(model_override: Option<&str>, config: &FileConfig) -> Result<Model> {
        let selector = model_override
            .map(str::to_string)
            .or_else(|| env_value("KNUTH_MODEL"))
            .or_else(|| config.model.clone())
            .ok_or_else(|| anyhow!("KNUTH_MODEL is not set; pass --model or set KNUTH_MODEL"))?;

        load_model(
            &selector,
            env_value("KNUTH_PROVIDER").or_else(|| config.provider.clone()),
            env_value("KNUTH_API").or_else(|| config.api.clone()),
            env_value("KNUTH_BASE_URL").or_else(|| config.base_url.clone()),
        )
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct FileConfig {
    model: Option<String>,
    provider: Option<String>,
    api: Option<String>,
    base_url: Option<String>,
    options: Option<FileOptions>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct FileOptions {
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    cache_retention: Option<CacheRetention>,
    headers: Option<HashMap<String, String>>,
    reasoning_effort: Option<String>,
    thinking: Option<FileThinking>,
    provider_extras: Option<HashMap<String, Value>>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct FileThinking {
    enabled: Option<bool>,
    budget_tokens: Option<u32>,
}

fn load_file_config(path_override: Option<PathBuf>) -> Result<FileConfig> {
    let explicit = path_override.is_some();
    let path = match path_override {
        Some(path) => PathBuf::from(path),
        None => default_config_file()?,
    };

    if !path.exists() {
        if explicit {
            return Err(anyhow!("config file does not exist: {}", path.display()));
        }
        return Ok(FileConfig::default());
    }

    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file {}", path.display()))?;
    serde_yaml::from_str(&text)
        .with_context(|| format!("failed to parse config file {}", path.display()))
}

fn default_config_file() -> Result<PathBuf> {
    default_config_dir().map(|dir| dir.join(CONFIG_FILE_NAME))
}

fn default_config_dir() -> Result<PathBuf> {
    platform_config_base()
        .map(|base| base.join(CONFIG_DIR_NAME))
        .ok_or_else(|| anyhow!("could not determine user config directory"))
}

fn platform_config_base() -> Option<PathBuf> {
    platform_config_base_from_env(env_value)
}

fn platform_config_base_from_env<F>(env_var: F) -> Option<PathBuf>
where
    F: Fn(&str) -> Option<String>,
{
    #[cfg(target_os = "windows")]
    {
        return env_var("APPDATA")
            .or_else(|| env_var("USERPROFILE").map(|home| format!("{home}\\AppData\\Roaming")))
            .map(PathBuf::from);
    }

    #[cfg(target_os = "macos")]
    {
        return env_var("HOME")
            .map(PathBuf::from)
            .map(|home| home.join("Library").join("Application Support"));
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        return env_var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| env_var("HOME").map(|home| PathBuf::from(home).join(".config")));
    }

    #[cfg(not(any(unix, target_os = "windows")))]
    {
        env_var("HOME").map(|home| PathBuf::from(home).join(".config"))
    }
}

fn load_options(config: &FileConfig) -> StreamOptions {
    let file = config.options.as_ref();
    let mut options = StreamOptions {
        max_tokens: file.and_then(|o| o.max_tokens).or(Some(1024)),
        temperature: file.and_then(|o| o.temperature),
        cache_retention: file.and_then(|o| o.cache_retention),
        headers: file.and_then(|o| o.headers.clone()),
        api_key: env_value("KNUTH_API_KEY"),
        provider_extras: file
            .and_then(|o| o.provider_extras.clone())
            .unwrap_or_default(),
        ..Default::default()
    };

    if let Some(effort) = file
        .and_then(|o| o.reasoning_effort.as_ref())
        .and_then(nonempty_ref)
    {
        options
            .provider_extras
            .insert("reasoning_effort".to_string(), json!(effort));
    }

    if let Some(thinking) = file.and_then(|o| o.thinking.as_ref()) {
        let value = if thinking.enabled.unwrap_or(true) {
            json!({
                "type": "enabled",
                "budget_tokens": thinking.budget_tokens.unwrap_or(4096),
            })
        } else {
            json!({ "type": "disabled" })
        };
        options
            .provider_extras
            .insert("thinking".to_string(), value);
    }

    options
}

fn load_model(
    selector: &str,
    provider_override: Option<String>,
    api_override: Option<String>,
    base_url: Option<String>,
) -> Result<Model> {
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
    let api_override = api_override.map(|api| api_from_name(&api));
    // ponytail: one selector plus env overrides; add named profiles when editing env is the bottleneck.
    let provider = selector_provider
        .or(provider_override)
        .unwrap_or_else(|| default_provider_for_api(api_override.as_ref()).to_string());

    let provider = Provider::from(provider);
    let mut model = match get_model(&provider, &model_id) {
        Some(model) => model,
        None => custom_model(
            provider.clone(),
            model_id.clone(),
            api_override
                .clone()
                .unwrap_or_else(|| default_api_for_provider(&provider.0)),
            base_url.clone(),
        )?,
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

fn nonempty(value: String) -> Option<String> {
    let value = value.trim().to_string();
    if value.is_empty() { None } else { Some(value) }
}

fn nonempty_ref(value: &String) -> Option<String> {
    nonempty(value.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_model_with_base_url_uses_openai_responses() {
        let model = load_model(
            "gpt-5.4-mini",
            None,
            None,
            Some("https://aicoding.2233.ai".to_string()),
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
            None,
            None,
            Some("https://aicoding.2233.ai".to_string()),
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
            None,
            None,
            Some("https://aicoding.2233.ai".to_string()),
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
            None,
            None,
            Some("https://aicoding.2233.ai".to_string()),
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
            None,
            Some("anthropic".to_string()),
            Some("https://anthropic.example.test".to_string()),
        )
        .unwrap();

        assert_eq!(model.provider.0, "anthropic");
        assert_eq!(model.api.0, "anthropic-messages");
        assert_eq!(model.id, "local-claude");
    }

    #[test]
    fn yaml_options_fill_stream_options() {
        let config = FileConfig {
            options: Some(FileOptions {
                max_tokens: Some(8192),
                temperature: Some(0.2),
                cache_retention: Some(CacheRetention::Long),
                reasoning_effort: Some("high".to_string()),
                thinking: Some(FileThinking {
                    enabled: Some(true),
                    budget_tokens: Some(12_000),
                }),
                provider_extras: Some(HashMap::from([("service_tier".to_string(), json!("auto"))])),
                headers: Some(HashMap::from([(
                    "HTTP-Referer".to_string(),
                    "https://knuth.local".to_string(),
                )])),
            }),
            ..Default::default()
        };

        let options = load_options(&config);

        assert_eq!(options.max_tokens, Some(8192));
        assert_eq!(options.temperature, Some(0.2));
        assert_eq!(options.cache_retention, Some(CacheRetention::Long));
        assert_eq!(options.provider_extras["reasoning_effort"], json!("high"));
        assert_eq!(
            options.provider_extras["thinking"],
            json!({ "type": "enabled", "budget_tokens": 12_000 })
        );
        assert_eq!(options.provider_extras["service_tier"], json!("auto"));
        assert_eq!(
            options.headers.unwrap()["HTTP-Referer"],
            "https://knuth.local"
        );
    }

    #[test]
    fn yaml_file_shape_parses() {
        let config: FileConfig = serde_yaml::from_str(
            r#"
model: openrouter/anthropic/claude-sonnet-4.5
api: openai-completions
base_url: https://openrouter.ai/api/v1
options:
  max_tokens: 4096
  temperature: 0.2
  cache_retention: long
  reasoning_effort: high
  thinking:
    enabled: true
    budget_tokens: 8192
  headers:
    HTTP-Referer: https://knuth.local
  provider_extras:
    service_tier: auto
"#,
        )
        .unwrap();

        assert_eq!(
            config.model.unwrap(),
            "openrouter/anthropic/claude-sonnet-4.5"
        );
        assert_eq!(config.api.unwrap(), "openai-completions");
        let options = config.options.unwrap();
        assert_eq!(options.cache_retention, Some(CacheRetention::Long));
        assert_eq!(options.reasoning_effort.unwrap(), "high");
        assert_eq!(options.thinking.unwrap().budget_tokens, Some(8192));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn default_config_path_uses_macos_application_support() {
        let base = platform_config_base_from_env(|name| {
            (name == "HOME").then(|| "/Users/tester".to_string())
        })
        .unwrap();

        assert_eq!(
            base.join(CONFIG_DIR_NAME).join(CONFIG_FILE_NAME),
            PathBuf::from("/Users/tester/Library/Application Support/knuth/knuth.yaml")
        );
    }

    #[test]
    #[cfg(all(unix, not(target_os = "macos")))]
    fn default_config_path_uses_xdg_config_home() {
        let base = platform_config_base_from_env(|name| {
            (name == "XDG_CONFIG_HOME").then(|| "/tmp/config".to_string())
        })
        .unwrap();

        assert_eq!(
            base.join(CONFIG_DIR_NAME).join(CONFIG_FILE_NAME),
            PathBuf::from("/tmp/config/knuth/knuth.yaml")
        );
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn default_config_path_uses_appdata() {
        let base = platform_config_base_from_env(|name| {
            (name == "APPDATA").then(|| "C:\\Users\\tester\\AppData\\Roaming".to_string())
        })
        .unwrap();

        assert_eq!(
            base.join(CONFIG_DIR_NAME).join(CONFIG_FILE_NAME),
            PathBuf::from("C:\\Users\\tester\\AppData\\Roaming")
                .join("knuth")
                .join("knuth.yaml")
        );
    }
}
