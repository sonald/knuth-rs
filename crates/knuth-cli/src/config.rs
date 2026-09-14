use crate::output::OutputStyle;
use crate::policy::PolicyMode;
use ai::{
    Api, CacheRetention, InputModality, KnownApi, Model, ModelCost, Provider, StreamOptions,
    get_model,
    oauth::{OAuthCredentials, openai_codex},
};
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    env, fs,
    future::Future,
    io::Write,
    path::{Path, PathBuf},
};

const CONFIG_DIR_NAME: &str = "knuth";
const CONFIG_FILE_NAME: &str = "knuth.yaml";
const HISTORY_FILE_NAME: &str = "history.txt";
const AUTH_FILE_NAME: &str = "auth.json";
const CODEX_ACCOUNT_EXTRA: &str = "chatgpt_account_id";

pub struct UserSettings {
    pub model: Model,
    pub options: StreamOptions,
    pub policy_mode: PolicyMode,
    pub output_style: OutputStyle,
}

impl UserSettings {
    pub async fn load(
        model_override: Option<&str>,
        config_override: Option<&Path>,
    ) -> Result<Self> {
        let (config, config_path) = load_file_config(
            config_override
                .map(Path::to_path_buf)
                .or_else(|| env_value("KNUTH_CONFIG").map(PathBuf::from)),
        )?;
        let model = Self::load_model_from_env(model_override, &config)?;
        let mut options = load_options(&config);
        load_codex_auth_json(&model, &mut options, &config_path).await?;
        let policy_mode = env_value("KNUTH_POLICY_MODE")
            .map(|s| s.parse::<PolicyMode>())
            .transpose()
            .map_err(|e| anyhow!("KNUTH_POLICY_MODE: {e}"))?
            .or(config.policy_mode)
            .unwrap_or_default();
        let output_style = env_value("KNUTH_OUTPUT_STYLE")
            .map(|s| s.parse::<OutputStyle>())
            .transpose()
            .map_err(|e| anyhow!("KNUTH_OUTPUT_STYLE: {e}"))?
            .or(config.output_style)
            .unwrap_or_default();
        Ok(Self {
            model,
            options,
            policy_mode,
            output_style,
        })
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
    policy_mode: Option<PolicyMode>,
    output_style: Option<OutputStyle>,
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

fn load_file_config(path_override: Option<PathBuf>) -> Result<(FileConfig, PathBuf)> {
    let explicit = path_override.is_some();
    let path = match path_override {
        Some(path) => PathBuf::from(path),
        None => default_config_file()?,
    };

    if !path.exists() {
        if explicit {
            return Err(anyhow!("config file does not exist: {}", path.display()));
        }
        return Ok((FileConfig::default(), path));
    }

    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file {}", path.display()))?;
    let config = serde_yaml::from_str(&text)
        .with_context(|| format!("failed to parse config file {}", path.display()))?;
    Ok((config, path))
}

fn default_config_file() -> Result<PathBuf> {
    default_config_dir().map(|dir| dir.join(CONFIG_FILE_NAME))
}

fn default_config_dir() -> Result<PathBuf> {
    platform_config_base()
        .map(|base| base.join(CONFIG_DIR_NAME))
        .ok_or_else(|| anyhow!("could not determine user config directory"))
}

pub fn default_history_file() -> Result<PathBuf> {
    default_config_dir().map(|dir| dir.join(HISTORY_FILE_NAME))
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

fn default_auth_file(config_path: &Path) -> Option<PathBuf> {
    config_path.parent().map(|dir| dir.join(AUTH_FILE_NAME))
}

async fn load_codex_auth_json(
    model: &Model,
    options: &mut StreamOptions,
    config_path: &Path,
) -> Result<()> {
    if model.api.0 != KnownApi::OpenAICodexResponses.as_str() {
        return Ok(());
    }

    let needs_token = options.api_key.is_none() && env_value("CODEX_AUTH_TOKEN").is_none();
    if !needs_token {
        return Ok(());
    }
    let needs_account = !has_chatgpt_account_id(&options.provider_extras)
        && env_value("CODEX_ACCOUNT_ID").is_none();

    let Some(path) = default_auth_file(config_path) else {
        return Ok(());
    };
    apply_codex_auth_json(&path, options, needs_token, needs_account).await
}

async fn apply_codex_auth_json(
    path: &Path,
    options: &mut StreamOptions,
    needs_token: bool,
    needs_account: bool,
) -> Result<()> {
    apply_codex_auth_json_with_refresher(
        path,
        options,
        needs_token,
        needs_account,
        |credentials| async move { openai_codex::refresh(&credentials).await },
    )
    .await
}

async fn apply_codex_auth_json_with_refresher<F, Fut>(
    path: &Path,
    options: &mut StreamOptions,
    needs_token: bool,
    needs_account: bool,
    refresh: F,
) -> Result<()>
where
    F: FnOnce(OAuthCredentials) -> Fut,
    Fut: Future<Output = Result<OAuthCredentials, String>>,
{
    if !path.exists() {
        return Ok(());
    }

    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read Codex auth file {}", path.display()))?;
    let auth: Value = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse Codex auth file {}", path.display()))?;
    let mut auth = auth;

    if needs_token {
        let access_token = auth
            .get("access_token")
            .and_then(Value::as_str)
            .and_then(nonempty_str)
            .ok_or_else(|| anyhow!("Codex auth file {} is missing access_token", path.display()))?;
        let expires_at = parse_codex_auth_expires_at(&auth, path)?;
        let credentials = OAuthCredentials {
            access_token,
            refresh_token: auth
                .get("refresh_token")
                .and_then(Value::as_str)
                .and_then(nonempty_str),
            expires_at,
            extra: None,
        };
        let credentials = match expires_at {
            Some(expires_at) if expires_at <= chrono::Utc::now().timestamp_millis() => {
                if credentials.refresh_token.is_none() {
                    return Err(anyhow!(
                        "Codex auth file {} is expired and missing refresh_token; refresh it or set CODEX_AUTH_TOKEN",
                        path.display()
                    ));
                }
                let refreshed = refresh(credentials).await.map_err(|error| {
                    anyhow!(
                        "failed to refresh expired Codex auth file {}: {error}",
                        path.display()
                    )
                })?;
                write_refreshed_codex_auth(path, &mut auth, &refreshed)?;
                refreshed
            }
            _ => credentials,
        };
        options.api_key = Some(credentials.access_token);
    }

    if needs_account {
        if let Some(account_id) = auth
            .get("account_id")
            .and_then(Value::as_str)
            .and_then(nonempty_str)
        {
            options
                .provider_extras
                .insert(CODEX_ACCOUNT_EXTRA.to_string(), json!(account_id));
        }
    }

    Ok(())
}

fn has_chatgpt_account_id(provider_extras: &HashMap<String, Value>) -> bool {
    provider_extras
        .get(CODEX_ACCOUNT_EXTRA)
        .and_then(Value::as_str)
        .and_then(nonempty_str)
        .is_some()
}

fn parse_codex_auth_expires_at(auth: &Value, path: &Path) -> Result<Option<i64>> {
    let Some(raw_expires_at) = auth.get("expires_at") else {
        return Ok(None);
    };
    let expires_at = parse_expires_at_millis(raw_expires_at)
        .ok_or_else(|| anyhow!("Codex auth file {} has invalid expires_at", path.display()))?;
    Ok(Some(expires_at))
}

fn write_refreshed_codex_auth(
    path: &Path,
    auth: &mut Value,
    credentials: &OAuthCredentials,
) -> Result<()> {
    let refresh_token = credentials
        .refresh_token
        .as_deref()
        .filter(|token| !token.trim().is_empty())
        .ok_or_else(|| anyhow!("refreshed Codex credentials are missing refresh_token"))?;
    let expires_at = credentials
        .expires_at
        .ok_or_else(|| anyhow!("refreshed Codex credentials are missing expires_at"))?;
    if credentials.access_token.trim().is_empty() {
        return Err(anyhow!(
            "refreshed Codex credentials are missing access_token"
        ));
    }
    let object = auth
        .as_object_mut()
        .ok_or_else(|| anyhow!("Codex auth file {} must be a JSON object", path.display()))?;
    object.insert("access_token".to_string(), json!(credentials.access_token));
    object.insert("refresh_token".to_string(), json!(refresh_token));
    object.insert("expires_at".to_string(), json!(expires_at));
    let contents = serde_json::to_vec_pretty(auth)
        .context("failed to serialize refreshed Codex credentials")?;
    replace_file_atomically(path, &contents)
}

fn replace_file_atomically(path: &Path, contents: &[u8]) -> Result<()> {
    let permissions = fs::metadata(path)
        .with_context(|| format!("failed to inspect Codex auth file {}", path.display()))?
        .permissions();
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Codex auth file {} has no parent directory", path.display()))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("Codex auth file {} has an invalid name", path.display()))?;
    let temporary = parent.join(format!(
        ".{name}.knuth-refresh-{}.tmp",
        uuid::Uuid::new_v4()
    ));
    let result = (|| -> std::io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        fs::set_permissions(&temporary, permissions)?;
        file.write_all(contents)?;
        file.sync_all()?;
        fs::rename(&temporary, path)
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error).with_context(|| {
            format!(
                "failed to atomically replace Codex auth file {}",
                path.display()
            )
        });
    }
    Ok(())
}

fn parse_expires_at_millis(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => n.as_i64().map(normalize_epoch_millis),
        Value::String(s) => s
            .parse::<i64>()
            .ok()
            .map(normalize_epoch_millis)
            .or_else(|| {
                chrono::DateTime::parse_from_rfc3339(s)
                    .ok()
                    .map(|dt| dt.timestamp_millis())
            }),
        _ => None,
    }
}

fn normalize_epoch_millis(value: i64) -> i64 {
    if value < 10_000_000_000 {
        value * 1000
    } else {
        value
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

fn nonempty_str(value: &str) -> Option<String> {
    nonempty(value.to_string())
}

fn nonempty_ref(value: &String) -> Option<String> {
    nonempty(value.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
            let saved = vars
                .iter()
                .map(|(name, _)| (*name, env::var(name).ok()))
                .collect();
            for (name, value) in vars {
                unsafe {
                    match value {
                        Some(value) => env::set_var(name, value),
                        None => env::remove_var(name),
                    }
                }
            }
            Self { saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in &self.saved {
                unsafe {
                    match value {
                        Some(value) => env::set_var(name, value),
                        None => env::remove_var(name),
                    }
                }
            }
        }
    }

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
policy_mode: read-only
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
        assert_eq!(config.policy_mode.unwrap().as_str(), "read-only");
        assert!(config.output_style.is_none());
        let options = config.options.unwrap();
        assert_eq!(options.cache_retention, Some(CacheRetention::Long));
        assert_eq!(options.reasoning_effort.unwrap(), "high");
        assert_eq!(options.thinking.unwrap().budget_tokens, Some(8192));
    }

    #[tokio::test]
    async fn policy_mode_defaults_to_auto_and_env_overrides() {
        let _lock = ENV_LOCK.lock().unwrap();
        let dir = env::temp_dir().join(format!("knuth-policy-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("knuth.yaml");
        fs::write(
            &config_path,
            r#"
model: chatgpt/gpt-5.4-mini
base_url: https://aicoding.2233.ai
"#,
        )
        .unwrap();

        let cleared = &[
            ("KNUTH_MODEL", None),
            ("KNUTH_BASE_URL", None),
            ("KNUTH_API_KEY", None),
            ("CODEX_AUTH_TOKEN", None),
            ("CODEX_ACCOUNT_ID", None),
            ("KNUTH_CONFIG", None),
            ("KNUTH_PROVIDER", None),
            ("KNUTH_API", None),
            ("KNUTH_POLICY_MODE", None),
        ];

        let _env = EnvGuard::set(cleared);
        let settings = UserSettings::load(None, Some(&config_path)).await.unwrap();
        assert_eq!(settings.policy_mode.as_str(), "auto");
        drop(_env);

        let _env = EnvGuard::set(&[
            ("KNUTH_POLICY_MODE", Some("bypass-permissions")),
            ("KNUTH_MODEL", None),
            ("KNUTH_API_KEY", None),
            ("KNUTH_CONFIG", None),
            ("KNUTH_BASE_URL", None),
        ]);
        let settings = UserSettings::load(None, Some(&config_path)).await.unwrap();
        assert_eq!(settings.policy_mode.as_str(), "bypass-permissions");
        drop(_env);

        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn output_style_defaults_to_default_and_env_overrides_yaml() {
        let _lock = ENV_LOCK.lock().unwrap();
        let dir = env::temp_dir().join(format!("knuth-output-style-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("knuth.yaml");
        fs::write(
            &config_path,
            r#"
model: chatgpt/gpt-5.4-mini
base_url: https://aicoding.2233.ai
output_style: concise
"#,
        )
        .unwrap();

        let _env = EnvGuard::set(&[
            ("KNUTH_MODEL", None),
            ("KNUTH_BASE_URL", None),
            ("KNUTH_API_KEY", None),
            ("CODEX_AUTH_TOKEN", None),
            ("CODEX_ACCOUNT_ID", None),
            ("KNUTH_CONFIG", None),
            ("KNUTH_PROVIDER", None),
            ("KNUTH_API", None),
            ("KNUTH_POLICY_MODE", None),
            ("KNUTH_OUTPUT_STYLE", None),
        ]);
        let settings = UserSettings::load(None, Some(&config_path)).await.unwrap();
        assert_eq!(settings.output_style, OutputStyle::Concise);
        drop(_env);

        let _env = EnvGuard::set(&[
            ("KNUTH_OUTPUT_STYLE", Some("default")),
            ("KNUTH_MODEL", None),
            ("KNUTH_API_KEY", None),
            ("KNUTH_CONFIG", None),
            ("KNUTH_BASE_URL", None),
        ]);
        let settings = UserSettings::load(None, Some(&config_path)).await.unwrap();
        assert_eq!(settings.output_style, OutputStyle::Default);

        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn codex_model_reads_auth_json_next_to_config_file() {
        let _lock = ENV_LOCK.lock().unwrap();
        let dir = env::temp_dir().join(format!("knuth-auth-json-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("knuth.yaml");
        fs::write(
            &config_path,
            r#"
model: chatgpt/gpt-5.4-mini
base_url: https://aicoding.2233.ai
"#,
        )
        .unwrap();
        fs::write(
            dir.join("auth.json"),
            r#"{
  "access_token": "codex-access",
  "account_id": "account-123",
  "expires_at": 4102444800000,
  "id_token": "ignored",
  "refresh_token": "ignored"
}"#,
        )
        .unwrap();

        let _env = EnvGuard::set(&[
            ("KNUTH_MODEL", None),
            ("KNUTH_BASE_URL", None),
            ("KNUTH_API_KEY", None),
            ("CODEX_AUTH_TOKEN", None),
            ("CODEX_ACCOUNT_ID", None),
            ("KNUTH_CONFIG", None),
            ("KNUTH_PROVIDER", None),
            ("KNUTH_API", None),
        ]);

        let settings = UserSettings::load(None, Some(&config_path)).await.unwrap();

        assert_eq!(settings.options.api_key.as_deref(), Some("codex-access"));
        assert_eq!(
            settings.options.provider_extras["chatgpt_account_id"],
            json!("account-123")
        );

        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn codex_auth_json_without_expires_at_is_accepted() {
        let _lock = ENV_LOCK.lock().unwrap();
        let dir = env::temp_dir().join(format!("knuth-auth-json-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("knuth.yaml");
        fs::write(
            &config_path,
            r#"
model: chatgpt/gpt-5.4-mini
base_url: https://aicoding.2233.ai
"#,
        )
        .unwrap();
        fs::write(
            dir.join("auth.json"),
            r#"{
  "access_token": "codex-access",
  "account_id": "account-123"
}"#,
        )
        .unwrap();

        let _env = EnvGuard::set(&[
            ("KNUTH_MODEL", None),
            ("KNUTH_BASE_URL", None),
            ("KNUTH_API_KEY", None),
            ("CODEX_AUTH_TOKEN", None),
            ("CODEX_ACCOUNT_ID", None),
            ("KNUTH_CONFIG", None),
            ("KNUTH_PROVIDER", None),
            ("KNUTH_API", None),
        ]);

        let settings = UserSettings::load(None, Some(&config_path)).await.unwrap();

        assert_eq!(settings.options.api_key.as_deref(), Some("codex-access"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn explicit_codex_api_key_skips_auth_json() {
        let dir = env::temp_dir().join(format!("knuth-auth-json-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("knuth.yaml");
        fs::write(dir.join("auth.json"), "not valid JSON").unwrap();
        let model = load_model(
            "chatgpt/gpt-5.4-mini",
            None,
            None,
            Some("https://aicoding.2233.ai".to_string()),
        )
        .unwrap();
        let mut options = StreamOptions {
            api_key: Some("explicit-token".to_string()),
            ..Default::default()
        };

        load_codex_auth_json(&model, &mut options, &config_path)
            .await
            .unwrap();

        assert_eq!(options.api_key.as_deref(), Some("explicit-token"));
        assert_eq!(
            fs::read_to_string(dir.join("auth.json")).unwrap(),
            "not valid JSON"
        );

        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn expired_codex_auth_json_refreshes_and_preserves_other_fields() {
        let dir = env::temp_dir().join(format!("knuth-auth-json-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        fs::write(
            &auth_path,
            r#"{
  "access_token": "expired-access",
  "refresh_token": "old-refresh",
  "expires_at": 0,
  "account_id": "account-123",
  "id_token": "preserve-me"
}"#,
        )
        .unwrap();
        let mut options = StreamOptions::default();

        apply_codex_auth_json_with_refresher(&auth_path, &mut options, true, true, |credentials| {
            assert_eq!(credentials.access_token, "expired-access");
            assert_eq!(credentials.refresh_token.as_deref(), Some("old-refresh"));
            std::future::ready(Ok(OAuthCredentials {
                access_token: "new-access".to_string(),
                refresh_token: Some("new-refresh".to_string()),
                expires_at: Some(4_102_444_800_000),
                extra: None,
            }))
        })
        .await
        .unwrap();

        assert_eq!(options.api_key.as_deref(), Some("new-access"));
        assert_eq!(
            options.provider_extras[CODEX_ACCOUNT_EXTRA],
            json!("account-123")
        );
        let saved: Value = serde_json::from_str(&fs::read_to_string(&auth_path).unwrap()).unwrap();
        assert_eq!(saved["access_token"], json!("new-access"));
        assert_eq!(saved["refresh_token"], json!("new-refresh"));
        assert_eq!(saved["expires_at"], json!(4_102_444_800_000_i64));
        assert_eq!(saved["account_id"], json!("account-123"));
        assert_eq!(saved["id_token"], json!("preserve-me"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn failed_codex_auth_refresh_leaves_auth_file_unchanged() {
        let dir = env::temp_dir().join(format!("knuth-auth-json-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        let original = r#"{
  "access_token": "expired-access",
  "refresh_token": "old-refresh",
  "expires_at": 0
}"#;
        fs::write(&auth_path, original).unwrap();
        let mut options = StreamOptions::default();

        let error =
            apply_codex_auth_json_with_refresher(&auth_path, &mut options, true, false, |_| {
                std::future::ready(Err("OAuth server rejected refresh".to_string()))
            })
            .await
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("failed to refresh expired Codex auth file")
        );
        assert_eq!(fs::read_to_string(&auth_path).unwrap(), original);
        assert!(options.api_key.is_none());

        fs::remove_dir_all(dir).unwrap();
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
