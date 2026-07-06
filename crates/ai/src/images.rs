//! Top-level images API. Image generation is not implemented in this Rust port yet.

use crate::types::{AssistantImages, ImagesContext, ImagesModel};

pub async fn images(
    _model: &ImagesModel,
    _context: &ImagesContext,
) -> Result<AssistantImages, String> {
    Err("image generation is not supported in the Rust ai crate".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ImagesApi, ImagesProvider, ModelCost};

    #[tokio::test]
    async fn images_returns_clear_unsupported_error() {
        let model = ImagesModel {
            id: "image-model".into(),
            name: "Image Model".into(),
            api: ImagesApi("openrouter-images".into()),
            provider: ImagesProvider("openrouter".into()),
            base_url: String::new(),
            input: vec![],
            output: vec![],
            cost: ModelCost::default(),
            headers: None,
        };
        let err = images(&model, &ImagesContext::default()).await.unwrap_err();
        assert!(err.contains("not supported"));
    }
}
