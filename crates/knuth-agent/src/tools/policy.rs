use ai::{Tool, ToolCall};
use async_trait::async_trait;
use std::str::FromStr;
use tokio_util::sync::CancellationToken;

use crate::ToolResult;

#[derive(Clone)]
pub struct PolicyContext {
    pub mode: PolicyMode,
}

#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PolicyMode {
    ReadOnly,
    AcceptEdits,
    #[serde(alias = "plan")]
    PlanMode,
    #[default]
    Auto,
    BypassPermissions,
}

impl PolicyMode {
    /// Canonical string form, used when rendering the effective config.
    pub fn as_str(&self) -> &'static str {
        match self {
            PolicyMode::ReadOnly => "read-only",
            PolicyMode::AcceptEdits => "accept-edits",
            PolicyMode::PlanMode => "plan",
            PolicyMode::Auto => "auto",
            PolicyMode::BypassPermissions => "bypass-permissions",
        }
    }
}

impl FromStr for PolicyMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "read-only" | "read_only" | "readonly" => Ok(PolicyMode::ReadOnly),
            "accept-edits" | "accept_edits" | "acceptedits" => Ok(PolicyMode::AcceptEdits),
            "plan" | "plan-mode" | "plan_mode" => Ok(PolicyMode::PlanMode),
            "auto" | "default" => Ok(PolicyMode::Auto),
            "bypass-permissions" | "bypass_permissions" | "bypass" => {
                Ok(PolicyMode::BypassPermissions)
            }
            other => Err(format!("unknown policy mode '{other}'")),
        }
    }
}

#[async_trait]
pub trait PolicyEngineTrait: Send + Sync {
    fn schemas(&self) -> Vec<Tool>;
    async fn execute(
        &self,
        tool_call: &ToolCall,
        cancel: CancellationToken,
        ctx: &PolicyContext,
    ) -> ToolResult;
}

#[cfg(test)]
mod tests {
    use super::PolicyMode;
    use std::str::FromStr;

    #[test]
    fn policy_mode_parses_canonical_names() {
        assert_eq!(
            "read-only".parse::<PolicyMode>().unwrap().as_str(),
            "read-only"
        );
        assert_eq!(
            "accept-edits".parse::<PolicyMode>().unwrap().as_str(),
            "accept-edits"
        );
        assert_eq!("plan".parse::<PolicyMode>().unwrap().as_str(), "plan");
        assert_eq!("auto".parse::<PolicyMode>().unwrap().as_str(), "auto");
        assert_eq!(
            "bypass-permissions".parse::<PolicyMode>().unwrap().as_str(),
            "bypass-permissions"
        );
    }

    #[test]
    fn policy_mode_parses_aliases() {
        assert_eq!("plan-mode".parse::<PolicyMode>().unwrap().as_str(), "plan");
        assert_eq!(
            "ReadOnly".parse::<PolicyMode>().unwrap().as_str(),
            "read-only"
        );
        assert_eq!("default".parse::<PolicyMode>().unwrap().as_str(), "auto");
    }

    #[test]
    fn policy_mode_rejects_unknown() {
        assert!("sideways".parse::<PolicyMode>().is_err());
    }

    #[test]
    fn policy_mode_deserializes_from_scalar() {
        let mode: PolicyMode = serde_json::from_str("\"plan\"").unwrap();
        assert_eq!(mode.as_str(), "plan");

        let mode: PolicyMode = serde_json::from_str("\"plan-mode\"").unwrap();
        assert_eq!(mode.as_str(), "plan");

        let mode: PolicyMode = serde_json::from_str("\"read-only\"").unwrap();
        assert_eq!(mode.as_str(), "read-only");
    }

    #[test]
    fn policy_mode_defaults_to_auto() {
        assert_eq!(PolicyMode::default().as_str(), "auto");
        let mode: PolicyMode = PolicyMode::from_str("auto").unwrap();
        assert!(matches!(mode, PolicyMode::Auto));
    }
}
