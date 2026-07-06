//! OpenRouter image generation placeholder. The Rust images API is explicitly unsupported.

use crate::types::ImagesModel;

#[derive(Copy, Clone)]
pub struct ImagesEntry;

impl ImagesEntry {
    pub async fn generate(&self, _model: &ImagesModel) -> Result<(), String> {
        Err("image generation is not supported in the Rust ai crate".into())
    }
}
