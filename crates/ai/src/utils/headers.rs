//! Header merging + User-Agent construction. 1:1 port of `packages/ai/src/utils/headers.ts`.

use std::collections::HashMap;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

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

pub(crate) fn merged_model_and_option_headers(
    model_headers: Option<&HashMap<String, String>>,
    option_headers: Option<&HashMap<String, String>>,
) -> Result<HeaderMap, String> {
    let mut merged = HeaderMap::new();
    for headers in [model_headers, option_headers].into_iter().flatten() {
        for (name, value) in headers {
            let parsed_name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|error| format!("invalid header name {name:?}: {error}"))?;
            let parsed_value = HeaderValue::from_str(value)
                .map_err(|error| format!("invalid value for header {name:?}: {error}"))?;
            merged.insert(parsed_name, parsed_value);
        }
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_headers_are_used_without_option_headers() {
        let model = HashMap::from([("x-model".to_string(), "model".to_string())]);

        let merged = merged_model_and_option_headers(Some(&model), None).unwrap();
        assert_eq!(merged.get("x-model").unwrap(), "model");
    }

    #[test]
    fn option_headers_override_model_headers_case_insensitively() {
        let model = HashMap::from([
            ("x-model".to_string(), "model".to_string()),
            ("X-Shared".to_string(), "model".to_string()),
        ]);
        let options = HashMap::from([
            ("x-options".to_string(), "options".to_string()),
            ("x-shared".to_string(), "options".to_string()),
        ]);

        let merged = merged_model_and_option_headers(Some(&model), Some(&options)).unwrap();
        assert_eq!(merged.get("x-model").unwrap(), "model");
        assert_eq!(merged.get("x-options").unwrap(), "options");
        assert_eq!(merged.get("x-shared").unwrap(), "options");
        assert_eq!(merged.get_all("x-shared").iter().count(), 1);
    }

    #[test]
    fn option_headers_are_used_without_model_headers() {
        let options = HashMap::from([("x-options".to_string(), "options".to_string())]);

        let merged = merged_model_and_option_headers(None, Some(&options)).unwrap();
        assert_eq!(merged.get("x-options").unwrap(), "options");
    }

    #[test]
    fn invalid_header_name_is_rejected() {
        let model = HashMap::from([("invalid header name".to_string(), "value".to_string())]);

        assert!(merged_model_and_option_headers(Some(&model), None).is_err());
    }

    #[test]
    fn invalid_header_value_is_rejected() {
        let options = HashMap::from([("x-options".to_string(), "bad\nvalue".to_string())]);

        assert!(merged_model_and_option_headers(None, Some(&options)).is_err());
    }
}
