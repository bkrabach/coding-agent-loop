//! Error types for the coding-agent-loop crate.
//!
//! Three-level hierarchy:
//! - [`AgentError`]: session-level failures
//! - [`EnvError`]: execution environment failures
//! - [`ToolError`]: tool executor failures

use std::time::Duration;

/// Session-level error.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("session closed")]
    SessionClosed,

    #[error("session aborted")]
    Aborted,

    #[error("turn limit reached: {0}")]
    TurnLimit(String),

    #[error("LLM error: {0}")]
    Llm(#[from] unified_llm::UnifiedLlmError),

    #[error("environment error: {0}")]
    Environment(#[from] EnvError),

    #[error("subagent error: {0}")]
    SubAgent(String),

    #[error("configuration error: {0}")]
    Configuration(String),
}

/// Execution environment error.
#[derive(Debug, thiserror::Error)]
pub enum EnvError {
    #[error("file not found: {0}")]
    FileNotFound(String),

    #[error("permission denied: {0}")]
    PermissionDenied(String),

    #[error("command timeout after {0:?}")]
    CommandTimeout(Duration),

    #[error("command failed (exit {exit_code}): {message}")]
    CommandFailed { exit_code: i32, message: String },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Tool executor error.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("file not found: {0}")]
    FileNotFound(String),

    #[error("edit conflict: {0}")]
    EditConflict(String),

    #[error("patch parse error: {0}")]
    PatchParse(String),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("environment error: {0}")]
    Environment(#[from] EnvError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn error_types_are_send_sync() {
        assert_send_sync::<AgentError>();
        assert_send_sync::<EnvError>();
        assert_send_sync::<ToolError>();
    }

    #[test]
    fn env_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let env_err: EnvError = io_err.into();
        assert!(matches!(env_err, EnvError::Io(_)));
    }

    #[test]
    fn agent_error_from_env_error() {
        let env_err = EnvError::FileNotFound("foo.txt".into());
        let agent_err: AgentError = env_err.into();
        assert!(matches!(agent_err, AgentError::Environment(_)));
    }

    #[test]
    fn tool_error_from_env_error() {
        let env_err = EnvError::PermissionDenied("bar".into());
        let tool_err: ToolError = env_err.into();
        assert!(matches!(tool_err, ToolError::Environment(_)));
    }

    #[test]
    fn error_display_messages() {
        let e = EnvError::FileNotFound("foo.txt".into());
        assert_eq!(e.to_string(), "file not found: foo.txt");

        let e = EnvError::CommandTimeout(Duration::from_secs(10));
        assert!(e.to_string().contains("command timeout after"));

        let e = EnvError::CommandFailed {
            exit_code: 1,
            message: "oops".into(),
        };
        assert_eq!(e.to_string(), "command failed (exit 1): oops");

        let e = AgentError::TurnLimit("max 10 rounds".into());
        assert_eq!(e.to_string(), "turn limit reached: max 10 rounds");
    }
}
