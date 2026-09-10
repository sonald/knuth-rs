use ai::{Tool, ToolCall};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::ToolResult;

#[derive(Clone, Debug, Default)]
pub struct PolicyContext {}

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
