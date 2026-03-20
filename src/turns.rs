//! Conversation turn types and history.
//!
//! A [`Turn`] is a single entry in the conversation history. The five variants
//! map directly to NLSpec §2.4. [`History`] is an ordered `Vec<Turn>` with
//! helper methods used by the agentic loop.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use unified_llm::Usage;

// ── Per-turn data types ──────────────────────────────────────────────────────

/// A tool call requested by the assistant (name + id + parsed arguments).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantToolCall {
    /// Provider-assigned call ID (used to match results).
    pub id: String,
    /// The tool name.
    pub name: String,
    /// Parsed JSON arguments.
    pub arguments: serde_json::Value,
}

/// The result of executing a single tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Matches the `id` of the [`AssistantToolCall`] this answers.
    pub tool_call_id: String,
    /// Text content returned by the tool (possibly truncated).
    pub content: String,
    /// Whether the tool reported an error.
    pub is_error: bool,
}

// ── Turn structs ─────────────────────────────────────────────────────────────

/// A turn submitted by the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserTurn {
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

impl UserTurn {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            timestamp: Utc::now(),
        }
    }
}

/// A turn produced by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantTurn {
    /// Textual output (may be empty when the model only requests tool calls).
    pub content: String,
    /// Tool invocations requested by the model.
    pub tool_calls: Vec<AssistantToolCall>,
    /// Reasoning / thinking text if the provider makes it available.
    pub reasoning: Option<String>,
    /// Token usage for this turn (None if the provider omitted usage).
    pub usage: Option<Usage>,
    /// Provider-assigned response ID.
    pub response_id: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl AssistantTurn {
    pub fn new(content: impl Into<String>, tool_calls: Vec<AssistantToolCall>) -> Self {
        Self {
            content: content.into(),
            tool_calls,
            reasoning: None,
            usage: None,
            response_id: None,
            timestamp: Utc::now(),
        }
    }
}

/// A turn that carries one result per preceding tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultsTurn {
    pub results: Vec<ToolResult>,
    pub timestamp: DateTime<Utc>,
}

impl ToolResultsTurn {
    pub fn new(results: Vec<ToolResult>) -> Self {
        Self {
            results,
            timestamp: Utc::now(),
        }
    }
}

/// A system-level turn (injected at the start of a session or for metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemTurn {
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

impl SystemTurn {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            timestamp: Utc::now(),
        }
    }
}

/// A steering message injected by the host between tool rounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteeringTurn {
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

impl SteeringTurn {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            timestamp: Utc::now(),
        }
    }
}

// ── Turn enum ────────────────────────────────────────────────────────────────

/// A single entry in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Turn {
    User(UserTurn),
    Assistant(AssistantTurn),
    ToolResults(ToolResultsTurn),
    System(SystemTurn),
    Steering(SteeringTurn),
}

// ── History ──────────────────────────────────────────────────────────────────

/// Ordered conversation history for a session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct History {
    turns: Vec<Turn>,
}

impl History {
    pub fn new() -> Self {
        Self { turns: Vec::new() }
    }

    /// Append a turn to the end of the history.
    pub fn push(&mut self, turn: Turn) {
        self.turns.push(turn);
    }

    /// All turns in insertion order.
    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// Total number of turns (all types).
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Count only `User` and `Assistant` turns for `max_turns` enforcement.
    ///
    /// Steering / System / ToolResults turns do not count against the limit.
    pub fn dialogue_turn_count(&self) -> usize {
        self.turns
            .iter()
            .filter(|t| matches!(t, Turn::User(_) | Turn::Assistant(_)))
            .count()
    }

    /// Collect the most recent `window` [`AssistantToolCall`]s (newest first).
    ///
    /// Iterates turns newest-first; within each turn iterates calls newest-first
    /// (last element of `tool_calls` is treated as the most-recently-requested).
    /// Used by loop detection to inspect the recent call signature window.
    pub fn recent_tool_calls(&self, window: usize) -> Vec<&AssistantToolCall> {
        if window == 0 {
            return vec![];
        }
        self.turns
            .iter()
            .rev()
            .flat_map(|t| match t {
                Turn::Assistant(a) => a.tool_calls.iter().rev().collect::<Vec<_>>(),
                _ => vec![],
            })
            .take(window)
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn history_default_is_empty() {
        let h = History::default();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert_eq!(h.dialogue_turn_count(), 0);
    }

    #[test]
    fn dialogue_turn_count_only_counts_user_and_assistant() {
        let mut h = History::new();
        h.push(Turn::System(SystemTurn::new("sys")));
        h.push(Turn::User(UserTurn::new("hello")));
        h.push(Turn::Assistant(AssistantTurn::new("hi", vec![])));
        h.push(Turn::Steering(SteeringTurn::new("steer")));
        h.push(Turn::ToolResults(ToolResultsTurn::new(vec![])));
        h.push(Turn::User(UserTurn::new("again")));

        assert_eq!(h.len(), 6);
        assert_eq!(h.dialogue_turn_count(), 3); // 2 User + 1 Assistant
    }

    #[test]
    fn recent_tool_calls_returns_correct_window() {
        let mut h = History::new();

        let call1 = AssistantToolCall {
            id: "1".into(),
            name: "read_file".into(),
            arguments: json!({}),
        };
        let call2 = AssistantToolCall {
            id: "2".into(),
            name: "write_file".into(),
            arguments: json!({}),
        };
        let call3 = AssistantToolCall {
            id: "3".into(),
            name: "shell".into(),
            arguments: json!({}),
        };

        h.push(Turn::Assistant(AssistantTurn::new("", vec![call1])));
        h.push(Turn::Assistant(AssistantTurn::new("", vec![call2, call3])));

        let recent = h.recent_tool_calls(2);
        assert_eq!(recent.len(), 2);
        // Newest first: call3 (index 1 in last assistant), then call2
        assert_eq!(recent[0].name, "shell");
        assert_eq!(recent[1].name, "write_file");
    }

    #[test]
    fn recent_tool_calls_zero_window() {
        let mut h = History::new();
        h.push(Turn::Assistant(AssistantTurn::new(
            "",
            vec![AssistantToolCall {
                id: "1".into(),
                name: "shell".into(),
                arguments: json!({}),
            }],
        )));
        assert_eq!(h.recent_tool_calls(0).len(), 0);
    }

    #[test]
    fn recent_tool_calls_larger_window_than_available() {
        let mut h = History::new();
        h.push(Turn::Assistant(AssistantTurn::new(
            "",
            vec![AssistantToolCall {
                id: "1".into(),
                name: "read_file".into(),
                arguments: json!({}),
            }],
        )));
        // Ask for 10 but only 1 available
        let recent = h.recent_tool_calls(10);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn turns_serde_round_trip() {
        let mut h = History::new();
        h.push(Turn::User(UserTurn::new("hello")));
        h.push(Turn::Assistant(AssistantTurn {
            content: "done".into(),
            tool_calls: vec![AssistantToolCall {
                id: "c1".into(),
                name: "read_file".into(),
                arguments: json!({"path": "/foo.txt"}),
            }],
            reasoning: Some("thinking...".into()),
            usage: None,
            response_id: Some("resp-abc".into()),
            timestamp: Utc::now(),
        }));

        let json = serde_json::to_string(&h).unwrap();
        let restored: History = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.len(), 2);
        assert_eq!(restored.dialogue_turn_count(), 2);
    }

    #[test]
    fn tool_result_is_error_preserved() {
        let r = ToolResult {
            tool_call_id: "x".into(),
            content: "file not found".into(),
            is_error: true,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: ToolResult = serde_json::from_str(&json).unwrap();
        assert!(r2.is_error);
    }
}
