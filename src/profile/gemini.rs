//! Gemini provider profile (gemini-cli-aligned).
//!
//! Tools: read_file, write_file, edit_file, shell (10s), grep, glob.
//! Supports parallel tool calls.

use std::time::Duration;

use crate::environment::ExecutionEnvironment;
use crate::profile::{ProjectDoc, ProviderProfile};
use crate::prompt::build_environment_context;
use crate::tools::core::{
    EditFileExecutor, GlobExecutor, GrepExecutor, ReadFileExecutor, ShellExecutor,
    WriteFileExecutor,
};
use crate::tools::{ToolDefinition, ToolRegistry};

pub struct GeminiProfile {
    model: String,
    registry: ToolRegistry,
}

impl GeminiProfile {
    pub fn new(model: &str) -> Self {
        let mut registry = ToolRegistry::new();
        registry.register(ReadFileExecutor::registered_tool());
        registry.register(WriteFileExecutor::registered_tool());
        registry.register(EditFileExecutor::registered_tool());
        registry.register(ShellExecutor::registered_tool(
            Duration::from_secs(10),
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

impl ProviderProfile for GeminiProfile {
    fn id(&self) -> &str {
        "gemini"
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
        let identity = "You are Gemini, a helpful AI assistant built by Google. \
            You help users write, modify, and debug code.\n\
            You work autonomously using tools to read files, apply changes, and run commands.";

        let tool_guide = "## Tool Guidelines\n\n\
            - Use `read_file` to read file contents before making changes.\n\
            - Use `edit_file` to make targeted changes to existing files using exact string replacement.\n\
            - Use `write_file` to create new files or completely replace file content.\n\
            - Use `shell` to run commands, build projects, execute tests, and inspect the environment.\n\
            - Use `grep` to search for text patterns across files.\n\
            - Use `glob` to list files matching a pattern.\n\n\
            ## edit_file Usage\n\n\
            When editing files:\n\
            - `old_string` must be an exact, unique substring of the file\n\
            - Include enough surrounding lines to make the match unique\n\
            - Verify changes by reading the file after editing\n\n\
            ## Coding Approach\n\n\
            - Understand the codebase structure before making changes\n\
            - Make targeted, minimal changes\n\
            - Respect existing code style and conventions\n\
            - Run tests after making changes\n\
            - Check the GEMINI.md file in the project root for project-specific instructions";

        let env_ctx = build_environment_context(env, git_context);

        let mut all_parts: Vec<String> = vec![identity.to_owned(), tool_guide.to_owned(), env_ctx];

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
        true
    }

    fn context_window_size(&self) -> u32 {
        1_000_000
    }
}
