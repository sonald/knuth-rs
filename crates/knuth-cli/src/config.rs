use ai::{Api, KnownApi, Model, ModelCost, Provider, StreamOptions, get_model};
use anyhow::Result;
use std::env;
pub struct UserSettings {
    pub model: Model,
    pub options: StreamOptions,
}

impl UserSettings {
    pub fn load() -> Result<Self> {
        let model = Self::load_model_from_env()?;
        let options = StreamOptions {
            max_tokens: Some(1024),
            api_key: Some(env::var("KNUTH_API_KEY")?),
            ..Default::default()
        };
        Ok(Self { model, options })
    }

    fn load_model_from_env() -> Result<Model> {
        let model = env::var("KNUTH_MODEL")?;
        let base_url = env::var("KNUTH_BASE_URL")?;

        let model = if model.contains('/') {
            let parts: Vec<&str> = model.split('/').collect();
            let provider = parts[0];
            let model = parts[1];
            get_model(&Provider::from(provider), model)
        } else {
            //this is custom model
            let model = Model {
                id: model.to_string(),
                name: model,
                api: Api::known(KnownApi::OpenAICompletions),
                // api: Api::known(KnownApi::OpenAIResponses),
                provider: Provider::from("custom"),
                base_url: base_url,
                reasoning: true,
                thinking_level_map: None,
                input: vec![ai::InputModality::Text],
                cost: ModelCost::default(),
                context_window: 1_000_000,
                max_tokens: 10_000,
                headers: None,
                compat: None,
            };
            Some(model)
        };

        model.ok_or_else(|| anyhow::anyhow!("Model not found"))
    }
}
