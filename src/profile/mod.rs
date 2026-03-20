//! Provider profile trait and factory functions.
//!
//! A [`ProviderProfile`] encapsulates the tool set, system prompt, and
//! capability flags for one LLM provider family. Three built-in profiles are
//! provided via factory functions: [`openai_profile`], [`anthropic_profile`],
//! and [`gemini_profile`].
//!
//! See NLSpec §3 for the full specification.

pub mod anthropic;
pub mod gemini;
pub mod openai;

use crate::environment::ExecutionEnvironment;
use crate::tools::{ToolDefinition, ToolRegistry};

// ── ProjectDoc ────────────────────────────────────────────────────────────────

/// A project instruction document discovered by the doc-discovery layer.
#[derive(Debug, Clone)]
pub struct ProjectDoc {
    /// Path of the file, relative to the project root.
    pub path: String,
    /// Content of the file.
    pub content: String,
}

// ── ProviderProfile trait ─────────────────────────────────────────────────────

/// Provider-specific agent profile.
///
/// Encapsulates the tool set, system prompt, and capability flags for one
/// LLM provider family. Implement this trait to create custom profiles.
pub trait ProviderProfile: Send + Sync {
    /// Short identifier: `"openai"`, `"anthropic"`, or `"gemini"`.
    fn id(&self) -> &str;

    /// Model identifier sent in LLM requests (e.g., `"codex-4o"`).
    fn model(&self) -> &str;

    /// Tool registry containing all tools available to this profile.
    fn tool_registry(&self) -> &ToolRegistry;

    /// Mutable access to the tool registry for host-application extensions.
    fn tool_registry_mut(&mut self) -> &mut ToolRegistry;

    /// Build the complete system prompt for the current environment and project.
    ///
    /// Called every time the session loop builds a new LLM request.
    /// `git_context` is the result of `build_git_context()` (None when not in a repo).
    fn build_system_prompt(
        &self,
        env: &dyn ExecutionEnvironment,
        project_docs: &[ProjectDoc],
        git_context: Option<&crate::prompt::GitContext>,
    ) -> String;

    /// Tool definitions to include in the LLM request.
    ///
    /// Derived from `tool_registry().definitions()`.
    fn tools(&self) -> Vec<ToolDefinition> {
        self.tool_registry()
            .definitions()
            .into_iter()
            .cloned()
            .collect()
    }

    /// Optional provider-specific options passed through to unified-llm.
    fn provider_options(&self) -> Option<serde_json::Value> {
        None
    }

    /// True if this model family supports extended reasoning (thinking tokens).
    fn supports_reasoning(&self) -> bool;

    /// True if this profile supports streaming responses.
    fn supports_streaming(&self) -> bool;

    /// True if this profile supports parallel tool call execution.
    fn supports_parallel_tool_calls(&self) -> bool;

    /// Approximate context window size in tokens (for usage warnings).
    fn context_window_size(&self) -> u32;
}

// ── Factory functions ─────────────────────────────────────────────────────────

/// Create an OpenAI provider profile for the given model.
///
/// Registers: `read_file`, `apply_patch`, `write_file`, `shell` (10s), `grep`, `glob`.
pub fn openai_profile(model: &str) -> Box<dyn ProviderProfile> {
    Box::new(openai::OpenAiProfile::new(model))
}

/// Create an Anthropic provider profile for the given model.
///
/// Registers: `read_file`, `write_file`, `edit_file`, `shell` (120s), `grep`, `glob`.
pub fn anthropic_profile(model: &str) -> Box<dyn ProviderProfile> {
    Box::new(anthropic::AnthropicProfile::new(model))
}

/// Create a Gemini provider profile for the given model.
///
/// Registers: `read_file`, `write_file`, `edit_file`, `shell` (10s), `grep`, `glob`.
pub fn gemini_profile(model: &str) -> Box<dyn ProviderProfile> {
    Box::new(gemini::GeminiProfile::new(model))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_profile_id_and_tools() {
        let p = openai_profile("gpt-4o");
        assert_eq!(p.id(), "openai");
        assert_eq!(p.model(), "gpt-4o");
        assert_eq!(
            p.tools().len(),
            6,
            "openai tools: {:?}",
            p.tools().iter().map(|t| &t.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn anthropic_profile_id_and_tools() {
        let p = anthropic_profile("claude-opus-4-5");
        assert_eq!(p.id(), "anthropic");
        assert_eq!(p.model(), "claude-opus-4-5");
        assert_eq!(p.tools().len(), 6);
    }

    #[test]
    fn gemini_profile_id_and_tools() {
        let p = gemini_profile("gemini-2.0-flash");
        assert_eq!(p.id(), "gemini");
        assert_eq!(p.model(), "gemini-2.0-flash");
        assert_eq!(p.tools().len(), 6);
    }

    #[test]
    fn openai_supports_parallel_tool_calls() {
        assert!(openai_profile("gpt-4o").supports_parallel_tool_calls());
    }

    #[test]
    fn anthropic_does_not_support_parallel_tool_calls() {
        assert!(!anthropic_profile("claude-opus-4-5").supports_parallel_tool_calls());
    }

    #[test]
    fn gemini_supports_parallel_tool_calls() {
        assert!(gemini_profile("gemini-2.0-flash").supports_parallel_tool_calls());
    }

    #[test]
    fn custom_tool_registration_via_mut() {
        use crate::error::ToolError;
        use crate::tools::{RegisteredTool, ToolDefinition, ToolExecutor};
        use async_trait::async_trait;
        use serde_json::{Value, json};

        struct Noop;
        #[async_trait]
        impl ToolExecutor for Noop {
            async fn execute(
                &self,
                _: Value,
                _: &dyn crate::environment::ExecutionEnvironment,
            ) -> Result<String, ToolError> {
                Ok("ok".into())
            }
        }

        let mut p = openai_profile("gpt-4o");
        let before = p.tools().len();
        p.tool_registry_mut().register(RegisteredTool {
            definition: ToolDefinition {
                name: "custom_tool".into(),
                description: "test".into(),
                parameters: json!({"type": "object", "properties": {}}),
            },
            executor: Box::new(Noop),
        });
        assert_eq!(p.tools().len(), before + 1);
    }

    #[test]
    fn build_system_prompt_contains_environment_block() {
        use crate::testing::MockExecutionEnvironment;
        let env = MockExecutionEnvironment::new("/work");
        let p = anthropic_profile("claude-opus-4-5");
        let prompt = p.build_system_prompt(&env, &[], None);
        assert!(
            prompt.contains("<environment>"),
            "prompt: {}",
            &prompt[..200]
        );
    }

    // ── GAP-CAL-001: OpenAI profile's tools() list includes apply_patch ──────

    #[test]
    fn openai_profile_tools_include_apply_patch() {
        let p = openai_profile("gpt-4o");
        let tools = p.tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"apply_patch"),
            "apply_patch missing from OpenAI tools: {:?}",
            names
        );
    }

    // ── GAP-CAL-002: Custom tool registration appears by name ─────────────────

    #[test]
    fn custom_tool_appears_by_name_in_tools_list() {
        use crate::error::ToolError;
        use crate::tools::{RegisteredTool, ToolDefinition, ToolExecutor};
        use async_trait::async_trait;
        use serde_json::{Value, json};

        struct MyTool;
        #[async_trait]
        impl ToolExecutor for MyTool {
            async fn execute(
                &self,
                _: Value,
                _: &dyn crate::environment::ExecutionEnvironment,
            ) -> Result<String, ToolError> {
                Ok("my_output".into())
            }
        }

        let mut p = anthropic_profile("claude-opus-4-5");
        p.tool_registry_mut().register(RegisteredTool {
            definition: ToolDefinition {
                name: "my_custom_tool".into(),
                description: "A custom tool added at runtime".into(),
                parameters: json!({"type": "object", "properties": {}}),
            },
            executor: Box::new(MyTool),
        });

        let tools = p.tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"my_custom_tool"),
            "custom tool missing from tool list: {:?}",
            names
        );
    }

    // ── GAP-CAL-003: Custom tool with same name overrides profile default ─────

    #[test]
    fn custom_tool_overrides_profile_default() {
        use crate::error::ToolError;
        use crate::tools::{RegisteredTool, ToolDefinition, ToolExecutor};
        use async_trait::async_trait;
        use serde_json::{Value, json};

        struct ReplacementShell;
        #[async_trait]
        impl ToolExecutor for ReplacementShell {
            async fn execute(
                &self,
                _: Value,
                _: &dyn crate::environment::ExecutionEnvironment,
            ) -> Result<String, ToolError> {
                Ok("replacement".into())
            }
        }

        let mut p = openai_profile("gpt-4o");
        let before_count = p.tools().len();

        // Re-register "shell" with a different implementation.
        p.tool_registry_mut().register(RegisteredTool {
            definition: ToolDefinition {
                name: "shell".into(),
                description: "Custom shell override".into(),
                parameters: json!({"type": "object", "properties": {}}),
            },
            executor: Box::new(ReplacementShell),
        });

        // Count must NOT grow (same name → replace, not append).
        assert_eq!(
            p.tools().len(),
            before_count,
            "re-registering a tool should not change the tool count"
        );

        // Description must reflect the override.
        let shell_def = p
            .tools()
            .into_iter()
            .find(|t| t.name == "shell")
            .expect("shell tool must still be present");
        assert_eq!(shell_def.description, "Custom shell override");
    }
}
