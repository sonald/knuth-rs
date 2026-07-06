//! Image provider registration placeholder. No image providers are supported yet.

use std::sync::OnceLock;

static ENSURED: OnceLock<()> = OnceLock::new();

pub fn ensure() {
    ENSURED.get_or_init(|| {});
}
