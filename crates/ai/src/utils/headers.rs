//! Header merging + User-Agent construction. 1:1 port of `packages/ai/src/utils/headers.ts`.

use std::collections::HashMap;

pub fn user_agent() -> String {
    format!("pie-ai-rs/{}", env!("CARGO_PKG_VERSION"))
}

/// Merge two header maps. Right side wins on collision (mirrors TS `{ ...a, ...b }`).
pub fn merge_headers(
    base: &HashMap<String, String>,
    overrides: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut out = base.clone();
    if let Some(over) = overrides {
        for (k, v) in over {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

pub fn merged_model_and_option_headers(
    model_headers: Option<&HashMap<String, String>>,
    option_headers: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    match model_headers {
        Some(base) => merge_headers(base, option_headers),
        None => option_headers.cloned().unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_headers_are_used_without_option_headers() {
        let model = HashMap::from([("x-model".to_string(), "model".to_string())]);

        assert_eq!(merged_model_and_option_headers(Some(&model), None), model);
    }

    #[test]
    fn option_headers_override_model_headers() {
        let model = HashMap::from([
            ("x-model".to_string(), "model".to_string()),
            ("x-shared".to_string(), "model".to_string()),
        ]);
        let options = HashMap::from([
            ("x-options".to_string(), "options".to_string()),
            ("x-shared".to_string(), "options".to_string()),
        ]);

        assert_eq!(
            merged_model_and_option_headers(Some(&model), Some(&options)),
            HashMap::from([
                ("x-model".to_string(), "model".to_string()),
                ("x-options".to_string(), "options".to_string()),
                ("x-shared".to_string(), "options".to_string()),
            ])
        );
    }

    #[test]
    fn option_headers_are_used_without_model_headers() {
        let options = HashMap::from([("x-options".to_string(), "options".to_string())]);

        assert_eq!(
            merged_model_and_option_headers(None, Some(&options)),
            options
        );
    }
}
