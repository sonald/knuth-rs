//! Tool input validation.
//!
//! TS uses typebox schemas — Rust uses raw `serde_json::Value` schemas per Q3:A. Validation is
//! intentionally unsupported until a real caller needs a JSON Schema dependency.

use serde_json::Value;

#[derive(Clone, Debug)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
}

/// Validate `input` against `schema`.
///
/// This fails closed instead of pretending validation succeeded.
pub fn validate(_input: &Value, _schema: &Value) -> ValidationResult {
    ValidationResult {
        valid: false,
        errors: vec![ValidationError {
            path: String::new(),
            message: "JSON Schema validation is not implemented".into(),
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validation_fails_closed_while_unsupported() {
        let result = validate(&json!({ "x": 1 }), &json!({ "type": "object" }));
        assert!(!result.valid);
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].message.contains("not implemented"));
    }
}
