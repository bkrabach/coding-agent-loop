//! Tool registry and executor trait.
//!
//! [`ToolRegistry`] maps tool names to [`RegisteredTool`] entries.
//! [`ToolExecutor`] is the async trait that individual tool implementations satisfy.
//! Latest-wins semantics: re-registering a tool by name replaces the existing one.

pub mod apply_patch;
pub mod core;

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use crate::environment::ExecutionEnvironment;
use crate::error::ToolError;

// ── Types ─────────────────────────────────────────────────────────────────────

/// JSON Schema-based tool definition sent to the LLM.
///
/// `parameters` must be a JSON Schema object (root `"type": "object"`);
/// this is not enforced at this layer.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Unique tool name (as the LLM will call it).
    pub name: String,
    /// Human/LLM-readable description of what the tool does.
    pub description: String,
    /// JSON Schema describing accepted arguments.
    pub parameters: Value,
}

/// A tool with its definition and an async executor.
pub struct RegisteredTool {
    pub definition: ToolDefinition,
    pub executor: Box<dyn ToolExecutor>,
}

// ── ToolExecutor trait ────────────────────────────────────────────────────────

/// Implement this trait to create a callable tool.
///
/// `execute` is called with the parsed arguments (as `serde_json::Value`) and
/// a reference to the active execution environment.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(
        &self,
        args: Value,
        env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError>;
}

// ── ToolRegistry ──────────────────────────────────────────────────────────────

/// Registry of all tools available to a provider profile.
///
/// Tools are keyed by name. Re-registering a tool replaces the existing entry
/// (latest-wins semantics, enabling host-application overrides).
pub struct ToolRegistry {
    tools: HashMap<String, RegisteredTool>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool, replacing any existing tool with the same name.
    pub fn register(&mut self, tool: RegisteredTool) {
        self.tools.insert(tool.definition.name.clone(), tool);
    }

    /// Remove a tool by name. No-op if the name is not registered.
    pub fn unregister(&mut self, name: &str) {
        self.tools.remove(name);
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name)
    }

    /// Return all tool definitions (for building LLM [`Request`] objects).
    ///
    /// Order is unspecified (HashMap order). Callers should sort if needed.
    pub fn definitions(&self) -> Vec<&ToolDefinition> {
        self.tools.values().map(|t| &t.definition).collect()
    }

    /// Return all registered tool names.
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns `true` if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Minimal executor for tests.
    struct EchoExecutor(String);

    #[async_trait]
    impl ToolExecutor for EchoExecutor {
        async fn execute(
            &self,
            _args: Value,
            _env: &dyn ExecutionEnvironment,
        ) -> Result<String, ToolError> {
            Ok(self.0.clone())
        }
    }

    fn make_tool(name: &str) -> RegisteredTool {
        RegisteredTool {
            definition: ToolDefinition {
                name: name.to_owned(),
                description: format!("{name} tool"),
                parameters: json!({"type": "object", "properties": {}}),
            },
            executor: Box::new(EchoExecutor(name.to_owned())),
        }
    }

    #[test]
    fn new_registry_is_empty() {
        let r = ToolRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn register_and_get() {
        let mut r = ToolRegistry::new();
        r.register(make_tool("read_file"));
        assert_eq!(r.len(), 1);
        assert!(r.get("read_file").is_some());
        assert!(r.get("write_file").is_none());
    }

    #[test]
    fn latest_wins_on_duplicate_name() {
        let mut r = ToolRegistry::new();
        r.register(make_tool("shell"));
        r.register(make_tool("shell")); // second registration
        assert_eq!(r.len(), 1, "duplicate name must not grow the registry");
    }

    #[test]
    fn unregister_removes_tool() {
        let mut r = ToolRegistry::new();
        r.register(make_tool("grep"));
        r.unregister("grep");
        assert!(r.is_empty());
    }

    #[test]
    fn unregister_missing_is_noop() {
        let mut r = ToolRegistry::new();
        r.unregister("nonexistent"); // must not panic
        assert!(r.is_empty());
    }

    #[test]
    fn definitions_returns_one_per_tool() {
        let mut r = ToolRegistry::new();
        r.register(make_tool("a"));
        r.register(make_tool("b"));
        r.register(make_tool("c"));
        assert_eq!(r.definitions().len(), 3);
    }

    #[test]
    fn names_returns_all_names() {
        let mut r = ToolRegistry::new();
        r.register(make_tool("read_file"));
        r.register(make_tool("write_file"));
        let mut names = r.names();
        names.sort();
        assert_eq!(names, vec!["read_file", "write_file"]);
    }

    #[test]
    fn default_is_empty() {
        let r = ToolRegistry::default();
        assert!(r.is_empty());
    }

    #[test]
    fn box_dyn_tool_executor_compiles() {
        let _: Box<dyn ToolExecutor> = Box::new(EchoExecutor("hi".into()));
    }
}
