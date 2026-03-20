//! Subagent tools: spawn_agent, send_input, wait, close_agent.
//!
//! A subagent is a child [`Session`] spawned by the parent to handle a scoped
//! task. It gets its own conversation history but shares the parent's
//! execution environment (same filesystem / working directory).
//!
//! The `spawn_agent` tool runs the child session **synchronously** — it
//! submits the task and awaits completion, returning the final output as the
//! tool result. The other three tools (`send_input`, `wait`, `close_agent`)
//! operate on already-completed (or restarted) sessions stored in a shared
//! registry.
//!
//! See NLSpec §7 for the full specification.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::config::SessionConfig;
use crate::environment::ExecutionEnvironment;
use crate::error::ToolError;
use crate::profile::ProviderProfile;
use crate::session::Session;
use crate::tools::{RegisteredTool, ToolDefinition, ToolExecutor};
use async_trait::async_trait;
use serde_json::{Value, json};

// ── SubAgentStatus / Handle / Result ─────────────────────────────────────────

/// Lifecycle status of a spawned subagent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubAgentStatus {
    Running,
    Completed,
    Failed,
}

/// A live or completed subagent entry in the registry.
pub struct SubAgentHandle {
    /// Unique identifier (UUID v4).
    pub id: String,
    /// Last known status.
    pub status: SubAgentStatus,
    /// The underlying child session (may have been closed).
    pub session: Session,
    /// The final output text from the last submit call.
    pub last_output: String,
}

/// Result returned to the LLM by the `wait` tool.
#[derive(Debug, Clone)]
pub struct SubAgentResult {
    pub output: String,
    pub success: bool,
    pub turns_used: u32,
}

// ── Registry ──────────────────────────────────────────────────────────────────

/// Shared registry of active subagents.
///
/// Wrapped in `Arc<Mutex<…>>` so all four tool executors can share it.
pub type SubAgentRegistry = Arc<Mutex<HashMap<String, SubAgentHandle>>>;

// ── Factory types ─────────────────────────────────────────────────────────────

/// A factory closure that creates a new provider profile.
pub type ProfileFactory = Arc<dyn Fn() -> Box<dyn ProviderProfile> + Send + Sync>;

/// A factory closure that creates a new execution environment.
pub type EnvFactory = Arc<dyn Fn() -> Box<dyn ExecutionEnvironment> + Send + Sync>;

// ── Public constructor ────────────────────────────────────────────────────────

/// Create the four subagent tool executors and return them as [`RegisteredTool`]s
/// plus the shared [`SubAgentRegistry`] for external inspection.
///
/// # Arguments
/// * `profile_factory` — closure that creates a fresh provider profile for the child
/// * `env_factory`     — closure that creates a fresh (but same-dir) env for the child
/// * `client`          — cloneable LLM client shared with child sessions
/// * `config`          — parent session config (used as template for child)
/// * `parent_depth`    — nesting depth of the parent session (child gets `+1`)
pub fn make_subagent_tools(
    profile_factory: ProfileFactory,
    env_factory: EnvFactory,
    client: unified_llm::Client,
    config: SessionConfig,
    parent_depth: u32,
) -> (SubAgentRegistry, Vec<RegisteredTool>) {
    let registry: SubAgentRegistry = Arc::new(Mutex::new(HashMap::new()));

    let spawn = RegisteredTool {
        definition: ToolDefinition {
            name: "spawn_agent".into(),
            description: "Spawn a subagent to handle a scoped task autonomously. \
                          The subagent runs to completion and its output is returned."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Natural language task description for the subagent."
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "Maximum turns for the subagent (0 = unlimited, default: 0)."
                    }
                },
                "required": ["task"]
            }),
        },
        executor: Box::new(SpawnAgentExecutor {
            profile_factory: Arc::clone(&profile_factory),
            env_factory: Arc::clone(&env_factory),
            client: client.clone(),
            config: config.clone(),
            parent_depth,
            registry: Arc::clone(&registry),
        }),
    };

    let send = RegisteredTool {
        definition: ToolDefinition {
            name: "send_input".into(),
            description: "Send a message to a previously spawned subagent.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The subagent ID returned by spawn_agent."
                    },
                    "message": {
                        "type": "string",
                        "description": "Message to send to the subagent."
                    }
                },
                "required": ["agent_id", "message"]
            }),
        },
        executor: Box::new(SendInputExecutor {
            registry: Arc::clone(&registry),
        }),
    };

    let wait = RegisteredTool {
        definition: ToolDefinition {
            name: "wait".into(),
            description: "Wait for a subagent to complete and return its final output.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The subagent ID to wait for."
                    }
                },
                "required": ["agent_id"]
            }),
        },
        executor: Box::new(WaitExecutor {
            registry: Arc::clone(&registry),
        }),
    };

    let close = RegisteredTool {
        definition: ToolDefinition {
            name: "close_agent".into(),
            description: "Terminate a subagent and remove it from the registry.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The subagent ID to close."
                    }
                },
                "required": ["agent_id"]
            }),
        },
        executor: Box::new(CloseAgentExecutor {
            registry: Arc::clone(&registry),
        }),
    };

    (registry, vec![spawn, send, wait, close])
}

// ── Executors ─────────────────────────────────────────────────────────────────

// ── spawn_agent ───────────────────────────────────────────────────────────────

struct SpawnAgentExecutor {
    profile_factory: ProfileFactory,
    env_factory: EnvFactory,
    client: unified_llm::Client,
    config: SessionConfig,
    parent_depth: u32,
    registry: SubAgentRegistry,
}

#[async_trait]
impl ToolExecutor for SpawnAgentExecutor {
    async fn execute(
        &self,
        args: Value,
        _env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError> {
        let task = args
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::Validation("spawn_agent: missing required arg 'task'".into())
            })?
            .to_owned();

        let max_turns = args
            .get("max_turns")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(0);

        // Depth check — enforce max_subagent_depth.
        let child_depth = self.parent_depth + 1;
        if self.config.max_subagent_depth > 0 && self.parent_depth >= self.config.max_subagent_depth
        {
            return Err(ToolError::Validation(format!(
                "Subagent depth limit reached (max: {}). \
                 Cannot spawn nested subagents.",
                self.config.max_subagent_depth
            )));
        }

        // Build child config.
        let mut child_config = self.config.clone();
        child_config.max_turns = max_turns;
        // Child cannot spawn further subagents beyond remaining depth budget.
        child_config.max_subagent_depth = self.config.max_subagent_depth;

        // Create child session.
        let profile = (self.profile_factory)();
        let env = (self.env_factory)();
        let mut child_session =
            Session::new_at_depth(child_config, profile, env, self.client.clone(), child_depth);

        let agent_id = child_session.id().to_owned();

        // Run to completion.
        let submit_result = child_session.submit(&task).await;

        // Extract the final output from the last AssistantTurn.
        let last_output = child_session
            .history()
            .turns()
            .iter()
            .rev()
            .find_map(|t| {
                if let crate::turns::Turn::Assistant(a) = t {
                    Some(a.content.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let turns_used = child_session.history().dialogue_turn_count() as u32;

        let (status, output_for_caller) = match submit_result {
            Ok(()) => (SubAgentStatus::Completed, last_output.clone()),
            Err(e) => {
                let msg = format!("Subagent failed: {}", e);
                (SubAgentStatus::Failed, msg)
            }
        };

        // Store in registry.
        self.registry.lock().unwrap().insert(
            agent_id.clone(),
            SubAgentHandle {
                id: agent_id.clone(),
                status,
                session: child_session,
                last_output: last_output.clone(),
            },
        );

        Ok(format!(
            "Agent {agent_id} completed ({turns_used} turns).\n\n{output_for_caller}"
        ))
    }
}

// ── send_input ────────────────────────────────────────────────────────────────

struct SendInputExecutor {
    registry: SubAgentRegistry,
}

#[async_trait]
impl ToolExecutor for SendInputExecutor {
    async fn execute(
        &self,
        args: Value,
        _env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError> {
        let agent_id = args
            .get("agent_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::Validation("send_input: missing required arg 'agent_id'".into())
            })?
            .to_owned();

        let message = args
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::Validation("send_input: missing required arg 'message'".into())
            })?
            .to_owned();

        // Pull the handle out of the registry so we can call submit() on it.
        let mut handle = {
            let mut reg = self.registry.lock().unwrap();
            reg.remove(&agent_id)
                .ok_or_else(|| ToolError::Validation(format!("Agent {agent_id} not found.")))?
        };

        // Run the new input.
        let submit_result = handle.session.submit(&message).await;

        let last_output = handle
            .session
            .history()
            .turns()
            .iter()
            .rev()
            .find_map(|t| {
                if let crate::turns::Turn::Assistant(a) = t {
                    Some(a.content.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let (status, reply) = match submit_result {
            Ok(()) => (SubAgentStatus::Completed, last_output.clone()),
            Err(e) => (SubAgentStatus::Failed, format!("Error: {e}")),
        };

        handle.status = status;
        handle.last_output = last_output;

        // Put the handle back.
        self.registry
            .lock()
            .unwrap()
            .insert(agent_id.clone(), handle);

        Ok(format!("Agent {agent_id}: {reply}"))
    }
}

// ── wait ─────────────────────────────────────────────────────────────────────

struct WaitExecutor {
    registry: SubAgentRegistry,
}

#[async_trait]
impl ToolExecutor for WaitExecutor {
    async fn execute(
        &self,
        args: Value,
        _env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError> {
        let agent_id = args
            .get("agent_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::Validation("wait: missing required arg 'agent_id'".into()))?
            .to_owned();

        let reg = self.registry.lock().unwrap();
        match reg.get(&agent_id) {
            None => Ok(format!("Agent {agent_id} not found or already closed.")),
            Some(handle) => {
                let turns = handle.session.history().dialogue_turn_count() as u32;
                let success = handle.status == SubAgentStatus::Completed;
                Ok(format!(
                    "Agent {agent_id} {} ({} turns).\n\n{}",
                    if success { "completed" } else { "failed" },
                    turns,
                    handle.last_output
                ))
            }
        }
    }
}

// ── close_agent ───────────────────────────────────────────────────────────────

struct CloseAgentExecutor {
    registry: SubAgentRegistry,
}

#[async_trait]
impl ToolExecutor for CloseAgentExecutor {
    async fn execute(
        &self,
        args: Value,
        _env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError> {
        let agent_id = args
            .get("agent_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::Validation("close_agent: missing required arg 'agent_id'".into())
            })?
            .to_owned();

        let removed = self.registry.lock().unwrap().remove(&agent_id);
        match removed {
            None => Ok(format!("Agent {agent_id} not found.")),
            Some(_) => Ok(format!("Agent {agent_id} closed.")),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use unified_llm::{ClientBuilder, testing::MockProviderAdapter};

    use crate::config::SessionConfig;
    use crate::profile::openai_profile;
    use crate::testing::MockExecutionEnvironment;

    /// Register mock under "openai" to match openai_profile's provider id.
    async fn make_client(mock: MockProviderAdapter) -> unified_llm::Client {
        ClientBuilder::new()
            .provider("openai", mock)
            .build()
            .await
            .unwrap()
    }

    fn make_profile_factory() -> ProfileFactory {
        Arc::new(|| openai_profile("mock-model"))
    }

    fn make_env_factory() -> EnvFactory {
        let env = MockExecutionEnvironment::new("/work");
        Arc::new(move || Box::new(env.clone()))
    }

    // ── AC-2: spawn_agent runs child, returns output ──────────────────────────

    #[tokio::test]
    async fn spawn_agent_runs_and_returns_output() {
        let mock = MockProviderAdapter::default().push_text_response("Subagent done.");
        let client = make_client(mock).await;

        let config = SessionConfig::default();
        let (_registry, tools) = make_subagent_tools(
            make_profile_factory(),
            make_env_factory(),
            client,
            config,
            0, // parent depth
        );

        // Find spawn_agent tool
        let spawn = tools
            .into_iter()
            .find(|t| t.definition.name == "spawn_agent")
            .unwrap();

        let env = MockExecutionEnvironment::new("/work");
        let result = spawn
            .executor
            .execute(json!({ "task": "do something" }), &env)
            .await
            .unwrap();

        assert!(result.contains("Subagent done."), "got: {}", result);
    }

    // ── AC-3: Child session has independent history ───────────────────────────

    #[tokio::test]
    async fn child_has_independent_history() {
        let mock = MockProviderAdapter::default().push_text_response("Child output.");
        let client = make_client(mock).await;
        let config = SessionConfig::default();

        let (registry, tools) = make_subagent_tools(
            make_profile_factory(),
            make_env_factory(),
            client,
            config,
            0,
        );

        let spawn = tools
            .into_iter()
            .find(|t| t.definition.name == "spawn_agent")
            .unwrap();
        let env = MockExecutionEnvironment::new("/work");
        spawn
            .executor
            .execute(json!({ "task": "child task" }), &env)
            .await
            .unwrap();

        let reg = registry.lock().unwrap();
        let handle = reg.values().next().unwrap();

        // Child history: User + Assistant (2 dialogue turns)
        assert_eq!(handle.session.history().dialogue_turn_count(), 2);
    }

    // ── AC-5: Depth limiting ──────────────────────────────────────────────────

    #[tokio::test]
    async fn depth_limit_prevents_nested_spawn() {
        let mock = MockProviderAdapter::default();
        let client = make_client(mock).await;

        let config = SessionConfig {
            max_subagent_depth: 1,
            ..Default::default()
        };

        // Parent is at depth 1 (already at the limit).
        let (_registry, tools) = make_subagent_tools(
            make_profile_factory(),
            make_env_factory(),
            client,
            config,
            1, // parent_depth = 1 = max
        );

        let spawn = tools
            .into_iter()
            .find(|t| t.definition.name == "spawn_agent")
            .unwrap();
        let env = MockExecutionEnvironment::new("/work");
        let result = spawn
            .executor
            .execute(json!({ "task": "nested task" }), &env)
            .await;

        assert!(
            matches!(result, Err(ToolError::Validation(_))),
            "expected depth-limit error, got: {:?}",
            result
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("depth limit"), "got: {}", err_msg);
    }

    // ── AC-6: wait returns last output ────────────────────────────────────────

    #[tokio::test]
    async fn wait_returns_last_output() {
        let mock = MockProviderAdapter::default().push_text_response("Final answer.");
        let client = make_client(mock).await;
        let config = SessionConfig::default();

        let (registry, tools) = make_subagent_tools(
            make_profile_factory(),
            make_env_factory(),
            client,
            config,
            0,
        );

        let spawn = tools
            .iter()
            .find(|t| t.definition.name == "spawn_agent")
            .unwrap();
        let env = MockExecutionEnvironment::new("/work");
        let spawn_result = spawn
            .executor
            .execute(json!({ "task": "the task" }), &env)
            .await
            .unwrap();

        // Extract agent_id from spawn result
        let agent_id = {
            let reg = registry.lock().unwrap();
            reg.keys().next().unwrap().clone()
        };

        let wait = tools.iter().find(|t| t.definition.name == "wait").unwrap();
        let wait_result = wait
            .executor
            .execute(json!({ "agent_id": agent_id }), &env)
            .await
            .unwrap();

        assert!(
            wait_result.contains("Final answer."),
            "wait result: {}",
            wait_result
        );
        let _ = spawn_result; // used
    }

    // ── AC-7: close_agent removes from registry ───────────────────────────────

    #[tokio::test]
    async fn close_agent_removes_from_registry() {
        let mock = MockProviderAdapter::default().push_text_response("Done.");
        let client = make_client(mock).await;
        let config = SessionConfig::default();

        let (registry, tools) = make_subagent_tools(
            make_profile_factory(),
            make_env_factory(),
            client,
            config,
            0,
        );

        let spawn = tools
            .iter()
            .find(|t| t.definition.name == "spawn_agent")
            .unwrap();
        let env = MockExecutionEnvironment::new("/work");
        spawn
            .executor
            .execute(json!({ "task": "something" }), &env)
            .await
            .unwrap();

        let agent_id = {
            let reg = registry.lock().unwrap();
            reg.keys().next().unwrap().clone()
        };

        assert_eq!(registry.lock().unwrap().len(), 1);

        let close = tools
            .iter()
            .find(|t| t.definition.name == "close_agent")
            .unwrap();
        let close_result = close
            .executor
            .execute(json!({ "agent_id": agent_id }), &env)
            .await
            .unwrap();

        assert!(close_result.contains("closed"), "got: {}", close_result);
        assert_eq!(registry.lock().unwrap().len(), 0);
    }

    // ── AC-8: Unknown agent_id returns error string (not ToolError) ───────────

    #[tokio::test]
    async fn wait_with_unknown_agent_returns_not_found_string() {
        let mock = MockProviderAdapter::default();
        let client = make_client(mock).await;
        let config = SessionConfig::default();

        let (_registry, tools) = make_subagent_tools(
            make_profile_factory(),
            make_env_factory(),
            client,
            config,
            0,
        );

        let env = MockExecutionEnvironment::new("/work");
        let wait = tools.iter().find(|t| t.definition.name == "wait").unwrap();
        let result = wait
            .executor
            .execute(json!({ "agent_id": "no-such-id" }), &env)
            .await
            .unwrap(); // should return Ok with a "not found" message

        assert!(result.contains("not found"), "got: {}", result);
    }

    // ── send_input resumes a completed subagent ───────────────────────────────

    #[tokio::test]
    async fn send_input_resumes_subagent() {
        // First mock: child's initial spawn submit
        // Second mock: child's send_input submit
        let mock = MockProviderAdapter::default()
            .push_text_response("Initial done.")
            .push_text_response("Follow-up done.");

        let client = make_client(mock).await;
        let config = SessionConfig::default();

        let (registry, tools) = make_subagent_tools(
            make_profile_factory(),
            make_env_factory(),
            client,
            config,
            0,
        );

        let env = MockExecutionEnvironment::new("/work");

        // Spawn
        let spawn = tools
            .iter()
            .find(|t| t.definition.name == "spawn_agent")
            .unwrap();
        spawn
            .executor
            .execute(json!({ "task": "initial task" }), &env)
            .await
            .unwrap();

        let agent_id = {
            let reg = registry.lock().unwrap();
            reg.keys().next().unwrap().clone()
        };

        // Send follow-up input
        let send = tools
            .iter()
            .find(|t| t.definition.name == "send_input")
            .unwrap();
        let result = send
            .executor
            .execute(json!({ "agent_id": agent_id, "message": "do more" }), &env)
            .await
            .unwrap();

        assert!(
            result.contains("Follow-up done."),
            "send_input result: {}",
            result
        );
    }

    // ── spawn_agent with max_turns applies to child ───────────────────────────

    #[tokio::test]
    async fn spawn_agent_max_turns_applied() {
        let mock = MockProviderAdapter::default().push_text_response("Done quickly.");
        let client = make_client(mock).await;
        let config = SessionConfig::default();

        let (registry, tools) = make_subagent_tools(
            make_profile_factory(),
            make_env_factory(),
            client,
            config,
            0,
        );

        let env = MockExecutionEnvironment::new("/work");
        let spawn = tools
            .iter()
            .find(|t| t.definition.name == "spawn_agent")
            .unwrap();
        spawn
            .executor
            .execute(json!({ "task": "fast task", "max_turns": 10 }), &env)
            .await
            .unwrap();

        // Verify the child completed (it did because mock returned text response)
        let reg = registry.lock().unwrap();
        let handle = reg.values().next().unwrap();
        assert_eq!(handle.status, SubAgentStatus::Completed);
    }
}
