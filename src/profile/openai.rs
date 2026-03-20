//! OpenAI provider profile (codex-rs-aligned).
//!
//! Tools: read_file, apply_patch, write_file, shell (10s), grep, glob.
//! Supports parallel tool calls.

use std::time::Duration;

use chrono::Utc;

use crate::environment::ExecutionEnvironment;
use crate::profile::{ProjectDoc, ProviderProfile};
use crate::prompt::build_environment_context;
use crate::tools::apply_patch::ApplyPatchExecutor;
use crate::tools::core::{
    GlobExecutor, GrepExecutor, ReadFileExecutor, ShellExecutor, WriteFileExecutor,
};
use crate::tools::{ToolDefinition, ToolRegistry};

pub struct OpenAiProfile {
    model: String,
    registry: ToolRegistry,
}

impl OpenAiProfile {
    pub fn new(model: &str) -> Self {
        let mut registry = ToolRegistry::new();
        registry.register(ReadFileExecutor::registered_tool());
        registry.register(ApplyPatchExecutor::registered_tool());
        registry.register(WriteFileExecutor::registered_tool());
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

impl ProviderProfile for OpenAiProfile {
    fn id(&self) -> &str {
        "openai"
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
        let identity = "You are a coding agent. You help users write, modify, and debug code.\n\
            You work autonomously using tools to read files, apply changes, and run commands.";

        let tool_guide = "## Tool Guidelines\n\n\
            - Use `read_file` to inspect files before modifying them.\n\
            - Use `apply_patch` to create new files and modify existing files. This is your primary editing tool.\n\
            - Use `write_file` only for creating new files when apply_patch is not appropriate.\n\
            - Use `shell` to run commands, build code, run tests, and inspect the environment.\n\
            - Use `grep` to search for patterns across files.\n\
            - Use `glob` to find files matching a pattern.\n\n\
            ## apply_patch Format\n\n\
            Use the v4a patch format. The patch string must start with `*** Begin Patch` and end with `*** End Patch`.\n\n\
            Add a new file:\n\
              *** Begin Patch\n\
              *** Add File: path/to/new_file.py\n\
              +line one\n\
              +line two\n\
              *** End Patch\n\n\
            Modify an existing file (always provide 3 lines of context above and below each change):\n\
              *** Begin Patch\n\
              *** Update File: path/to/existing_file.py\n\
              @@ context_hint\n\
               unchanged context line\n\
              -line to remove\n\
              +line to add\n\
              *** End Patch\n\n\
            Delete a file:\n\
              *** Begin Patch\n\
              *** Delete File: path/to/file.py\n\
              *** End Patch";

        // V2-CAL-008: include model info so the LLM knows which model it is.
        let model_info = format!("Model: {}", self.model);

        let env_ctx = build_environment_context(env, git_context);

        let best_practices = "## Best Practices\n\n\
            - Read files before editing them.\n\
            - Make targeted, minimal changes.\n\
            - Verify changes by reading the modified file after applying patches.\n\
            - Run tests after making changes when a test suite is available.\n\
            - Prefer editing existing files over creating new ones.";

        let doc_sections: Vec<String> = project_docs
            .iter()
            .map(|doc| format!("## Project Instructions ({})\n\n{}", doc.path, doc.content))
            .collect();

        let mut all_parts: Vec<&str> = vec![
            identity,
            &model_info,
            tool_guide,
            env_ctx.as_str(),
            best_practices,
        ];
        all_parts.extend(doc_sections.iter().map(String::as_str));

        all_parts
            .iter()
            .filter(|p| !p.trim().is_empty())
            .cloned()
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
        128_000
    }
}

// Suppress unused import warning for Utc used only if we ever add date to prompt.
const _: () = {
    let _ = Utc::now;
};
