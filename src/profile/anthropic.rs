//! Anthropic provider profile (Claude Code-aligned).
//!
//! Tools: read_file, write_file, edit_file, shell (120s), grep, glob.
//! Does NOT support parallel tool calls.

use std::time::Duration;

use crate::environment::ExecutionEnvironment;
use crate::profile::{ProjectDoc, ProviderProfile};
use crate::prompt::build_environment_context;
use crate::tools::core::{
    EditFileExecutor, GlobExecutor, GrepExecutor, ReadFileExecutor, ShellExecutor,
    WriteFileExecutor,
};
use crate::tools::{ToolDefinition, ToolRegistry};

pub struct AnthropicProfile {
    model: String,
    registry: ToolRegistry,
}

impl AnthropicProfile {
    pub fn new(model: &str) -> Self {
        let mut registry = ToolRegistry::new();
        registry.register(ReadFileExecutor::registered_tool());
        registry.register(WriteFileExecutor::registered_tool());
        registry.register(EditFileExecutor::registered_tool());
        // Claude Code uses a 120s default shell timeout.
        registry.register(ShellExecutor::registered_tool(
            Duration::from_secs(120),
            Duration::from_secs(600),
        ));
        registry.register(GrepExecutor::registered_tool());
        registry.register(GlobExecutor::registered_tool());

        Self {
            model: model.to_owned(),
            registry,
        }
    }
}

impl ProviderProfile for AnthropicProfile {
    fn id(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn tool_registry(&self) -> &ToolRegistry {
        &self.registry
    }

    fn tool_registry_mut(&mut self) -> &mut ToolRegistry {
        &mut self.registry
    }

    fn build_system_prompt(
        &self,
        env: &dyn ExecutionEnvironment,
        project_docs: &[ProjectDoc],
        git_context: Option<&crate::prompt::GitContext>,
    ) -> String {
        let identity = "You are Claude, an AI assistant made by Anthropic. \
            You help users write, modify, and debug code.\n\
            You work autonomously using tools to read files, apply changes, and run commands.";

        let tool_guide = "## Tool Guidelines\n\n\
            - ALWAYS read a file before editing it. Use `read_file` to understand the current content.\n\
            - Prefer `edit_file` over `write_file` for modifying existing files.\n\
            - Use `write_file` only when creating new files or completely replacing file content.\n\
            - Use `shell` to run commands, compile code, run tests, and inspect the environment.\n\
            - Use `grep` to search for patterns across the codebase.\n\
            - Use `glob` to find files by pattern.\n\n\
            ## edit_file Format\n\n\
            `edit_file` performs an exact string search-and-replace:\n\
            - `old_string` must appear EXACTLY ONCE in the file (unless replace_all=true)\n\
            - Include enough surrounding context in `old_string` to make it unique\n\
            - If `old_string` matches multiple locations, the tool returns an error — \
              read the file again and add more context\n\
            - Whitespace and indentation must be exact\n\n\
            ## File Operation Preferences\n\n\
            1. Always read files before editing\n\
            2. Make targeted, minimal changes — edit specific sections rather than rewriting whole files\n\
            3. Prefer editing existing files over creating new ones\n\
            4. Verify changes after making them by reading the modified file";

        let env_ctx = build_environment_context(env, git_context);

        let best_practices = "## Best Practices\n\n\
            - Run tests after making changes to verify correctness.\n\
            - Keep changes focused and minimal.\n\
            - Explain your reasoning before making changes.\n\
            - Handle edge cases and error conditions.";

        let mut all_parts: Vec<String> = vec![
            identity.to_owned(),
            tool_guide.to_owned(),
            env_ctx,
            best_practices.to_owned(),
        ];

        for doc in project_docs {
            all_parts.push(format!(
                "## Project Instructions ({})\n\n{}",
                doc.path, doc.content
            ));
        }

        all_parts
            .iter()
            .filter(|p| !p.trim().is_empty())
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        self.registry.definitions().into_iter().cloned().collect()
    }

    fn provider_options(&self) -> Option<serde_json::Value> {
        None
    }

    fn supports_reasoning(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        false
    }

    fn context_window_size(&self) -> u32 {
        200_000
    }
}
