//! Autonomous coding agent loop library.
//!
//! Implements the core agentic loop: LLM call → tool execution → repeat.

pub mod config;
pub mod environment;
pub mod error;
pub mod events;
pub mod loop_detection;
pub mod profile;
pub mod prompt;
pub mod session;
pub mod subagent;
pub mod testing;
pub mod tools;
pub mod truncation;
pub mod turns;

pub use config::{SessionConfig, SessionConfigPatch};
pub use environment::{
    DirEntry, ExecResult, ExecutionEnvironment, GrepOptions, LocalExecutionEnvironment,
};
pub use error::{AgentError, EnvError, ToolError};
pub use events::{EVENT_CHANNEL_CAPACITY, EventKind, EventSender, SessionEvent};
pub use session::{Session, SessionState};
pub use subagent::{
    EnvFactory, ProfileFactory, SubAgentHandle, SubAgentRegistry, SubAgentResult, SubAgentStatus,
    make_subagent_tools,
};
pub use testing::{EnvCall, MockCommandResponse, MockExecutionEnvironment};
pub use tools::{RegisteredTool, ToolDefinition, ToolExecutor, ToolRegistry};
pub use turns::{
    AssistantToolCall, AssistantTurn, History, SteeringTurn, SystemTurn, ToolResult,
    ToolResultsTurn, Turn, UserTurn,
};
