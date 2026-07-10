//! Model registry. 1:1 stub of `packages/ai/src/models.ts`.
//!
//! TS exposes `getModel(provider, id)`, `listModels()`, custom-model registration, and
//! OpenAI-compat overrides. Here we provide just the surface; the data comes from
//! `models_generated.rs` (which is currently empty — populate via `build.rs` once we port
//! `scripts/generate-models.ts`).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::models_generated::BUILTIN_MODELS;
use crate::types::{Api, Model, Provider, Usage};

fn custom_registry() -> &'static Mutex<HashMap<String, Model>> {
    static CELL: OnceLock<Mutex<HashMap<String, Model>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(HashMap::new()))
}

fn key(provider: &Provider, id: &str) -> String {
    format!("{}/{}", provider.0, id)
}

pub fn get_model(provider: &Provider, id: &str) -> Option<Model> {
    let custom = custom_registry().lock().expect("registry poisoned");
    if let Some(m) = custom.get(&key(provider, id)) {
        return Some(m.clone());
    }
    BUILTIN_MODELS
        .iter()
        .find(|m| m.provider == *provider && m.id == id)
        .cloned()
}

pub fn list_models() -> Vec<Model> {
    let custom = custom_registry().lock().expect("registry poisoned");
    let mut out: Vec<Model> = BUILTIN_MODELS.iter().cloned().collect();
    out.extend(custom.values().cloned());
    out
}

pub fn register_custom_model(model: Model) {
    let k = key(&model.provider, &model.id);
    let mut reg = custom_registry().lock().expect("registry poisoned");
    reg.insert(k, model);
}

pub fn unregister_custom_model(provider: &Provider, id: &str) {
    let mut reg = custom_registry().lock().expect("registry poisoned");
    reg.remove(&key(provider, id));
}

pub fn list_apis() -> Vec<Api> {
    let mut out = std::collections::HashSet::new();
    for m in BUILTIN_MODELS.iter() {
        out.insert(m.api.clone());
    }
    out.into_iter().collect()
}

pub fn calculate_usage_cost(model: &Model, usage: &mut Usage) {
    usage.cost.input = usage.input as f64 * model.cost.input / 1_000_000.0;
    usage.cost.output = usage.output as f64 * model.cost.output / 1_000_000.0;
    usage.cost.cache_read = usage.cache_read as f64 * model.cost.cache_read / 1_000_000.0;
    usage.cost.cache_write = usage.cache_write as f64 * model.cost.cache_write / 1_000_000.0;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{KnownApi, ModelCost, Usage};

    #[test]
    fn calculate_usage_cost_uses_per_million_prices() {
        let model = Model {
            id: "priced-model".into(),
            name: "Priced Model".into(),
            api: Api::known(KnownApi::OpenAICompletions),
            provider: Provider::from("openai"),
            base_url: String::new(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![],
            cost: ModelCost {
                input: 2.0,
                output: 4.0,
                cache_read: 0.5,
                cache_write: 1.5,
            },
            context_window: 128_000,
            max_tokens: 4096,
            headers: None,
            compat: None,
        };
        let mut usage = Usage {
            input: 500_000,
            output: 250_000,
            cache_read: 2_000_000,
            cache_write: 1_000_000,
            ..Default::default()
        };

        calculate_usage_cost(&model, &mut usage);

        assert_eq!(usage.cost.input, 1.0);
        assert_eq!(usage.cost.output, 1.0);
        assert_eq!(usage.cost.cache_read, 1.0);
        assert_eq!(usage.cost.cache_write, 1.5);
        assert_eq!(usage.cost.total, 4.5);
    }
}
