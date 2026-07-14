//! Runtime components for the Knuth project.

pub mod harness;
pub use harness::*;

pub mod agent_step;
pub use agent_step::*;

pub mod actor;
pub use actor::*;

pub mod event_log;
pub use event_log::*;

pub mod tools;
pub use tools::*;

#[cfg(test)]
pub(crate) mod test_support {
    /// The faux provider's response queue is process-global; tests that use it
    /// must hold this lock so parallel tests don't consume each other's responses.
    pub(crate) fn faux_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }
}
