//! Session configuration and defaults.
//!
//! [`SessionConfig`] holds all tunable parameters for an agent session.
//! [`SessionConfigPatch`] allows mid-session updates to a subset of fields.

use std::collections::HashMap;
use std::time::Duration;

/// All tunable parameters for a coding agent session.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Maximum total dialogue turns (User + Assistant). 0 = unlimited.
    pub max_turns: u32,
    /// Maximum tool execution rounds per user input. 0 = unlimited.
    pub max_tool_rounds_per_input: u32,
    /// Default timeout for command execution (applied when no per-call override).
    pub default_command_timeout: Duration,
    /// Hard cap on per-call command timeout overrides.
    pub max_command_timeout: Duration,
    /// Reasoning effort level: `"low"`, `"medium"`, `"high"`, or `None`.
    pub reasoning_effort: Option<String>,
    /// Per-tool character output limits. Keys are tool names; empty = use defaults.
    pub tool_output_limits: HashMap<String, usize>,
    /// Per-tool line output limits (secondary pass after character truncation).
    pub tool_line_limits: HashMap<String, usize>,
    /// Whether to run the loop detection algorithm after each tool round.
    pub enable_loop_detection: bool,
    /// Number of consecutive tool calls to inspect for repeating patterns.
    pub loop_detection_window: usize,
    /// Maximum subagent nesting depth. 1 = no sub-subagents.
    pub max_subagent_depth: u32,
    /// User instruction overrides appended last in the system prompt (highest priority).
    ///
    /// When set, this text is appended after all profile instructions and project docs.
    /// Per NLSpec §9.8: "User instruction overrides are appended last (highest priority)."
    pub user_instructions: Option<String>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_turns: 0,
            max_tool_rounds_per_input: 0,
            default_command_timeout: Duration::from_secs(10),
            max_command_timeout: Duration::from_secs(600),
            reasoning_effort: None,
            tool_output_limits: HashMap::new(),
            tool_line_limits: HashMap::new(),
            enable_loop_detection: true,
            loop_detection_window: 10,
            max_subagent_depth: 1,
            user_instructions: None,
        }
    }
}

impl SessionConfig {
    /// Apply a patch, updating only the fields that are `Some`.
    pub fn apply_patch(&mut self, patch: SessionConfigPatch) {
        if let Some(v) = patch.reasoning_effort {
            self.reasoning_effort = v;
        }
        if let Some(v) = patch.max_tool_rounds_per_input {
            self.max_tool_rounds_per_input = v;
        }
        if let Some(v) = patch.max_turns {
            self.max_turns = v;
        }
        if let Some(v) = patch.default_command_timeout {
            self.default_command_timeout = v;
        }
        if let Some(v) = patch.enable_loop_detection {
            self.enable_loop_detection = v;
        }
        if let Some(v) = patch.loop_detection_window {
            self.loop_detection_window = v;
        }
        if let Some(v) = patch.user_instructions {
            self.user_instructions = v;
        }
    }
}

/// A partial update for [`SessionConfig`].
///
/// Each field is `Option<T>`. `None` means "don't change". `Some(v)` means "set to v".
/// For `Option<String>` fields, `Some(None)` means "clear the value".
#[derive(Debug, Clone, Default)]
pub struct SessionConfigPatch {
    /// Set to `Some(Some("high"))` to change, `Some(None)` to clear.
    pub reasoning_effort: Option<Option<String>>,
    pub max_tool_rounds_per_input: Option<u32>,
    pub max_turns: Option<u32>,
    pub default_command_timeout: Option<Duration>,
    pub enable_loop_detection: Option<bool>,
    pub loop_detection_window: Option<usize>,
    /// Set to `Some(Some("instructions"))` to set, `Some(None)` to clear.
    pub user_instructions: Option<Option<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let cfg = SessionConfig::default();
        assert_eq!(cfg.max_turns, 0);
        assert_eq!(cfg.max_tool_rounds_per_input, 0);
        assert_eq!(cfg.default_command_timeout, Duration::from_secs(10));
        assert_eq!(cfg.max_command_timeout, Duration::from_secs(600));
        assert!(cfg.reasoning_effort.is_none());
        assert!(cfg.enable_loop_detection);
        assert_eq!(cfg.loop_detection_window, 10);
        assert_eq!(cfg.max_subagent_depth, 1);
        assert!(cfg.tool_output_limits.is_empty());
        assert!(cfg.tool_line_limits.is_empty());
    }

    // ── GAP-CAL-006: Default command timeout is explicitly 10 seconds ────────
    //
    // NLSpec §9.4: "Command timeout default is 10 seconds."
    // The shell tool reads this from SessionConfig::default_command_timeout.

    #[test]
    fn default_command_timeout_is_10_seconds() {
        let cfg = SessionConfig::default();
        assert_eq!(
            cfg.default_command_timeout,
            Duration::from_secs(10),
            "NLSpec §9.4 requires the default command timeout to be 10 seconds"
        );
    }

    #[test]
    fn apply_patch_noop() {
        let mut cfg = SessionConfig::default();
        let before_timeout = cfg.default_command_timeout;
        cfg.apply_patch(SessionConfigPatch::default());
        assert_eq!(cfg.default_command_timeout, before_timeout);
        assert!(cfg.reasoning_effort.is_none());
    }

    #[test]
    fn apply_patch_sets_reasoning_effort() {
        let mut cfg = SessionConfig::default();
        cfg.apply_patch(SessionConfigPatch {
            reasoning_effort: Some(Some("high".into())),
            ..Default::default()
        });
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn apply_patch_clears_reasoning_effort() {
        let mut cfg = SessionConfig {
            reasoning_effort: Some("medium".into()),
            ..Default::default()
        };
        cfg.apply_patch(SessionConfigPatch {
            reasoning_effort: Some(None),
            ..Default::default()
        });
        assert!(cfg.reasoning_effort.is_none());
    }

    #[test]
    fn apply_patch_multiple_fields() {
        let mut cfg = SessionConfig::default();
        cfg.apply_patch(SessionConfigPatch {
            max_turns: Some(50),
            enable_loop_detection: Some(false),
            ..Default::default()
        });
        assert_eq!(cfg.max_turns, 50);
        assert!(!cfg.enable_loop_detection);
    }
}
