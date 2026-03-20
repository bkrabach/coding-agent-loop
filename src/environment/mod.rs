//! Execution environment abstraction.
//!
//! All tool operations pass through the [`ExecutionEnvironment`] trait.
//! This decouples tool logic from where it runs — local filesystem,
//! Docker, SSH, WASM, etc.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;

use crate::error::EnvError;

pub mod local;
pub use local::LocalExecutionEnvironment;

// ── Supporting data types ────────────────────────────────────────────────────

/// A single filesystem directory entry.
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    /// File size in bytes. `None` for directories or when unavailable.
    pub size: Option<u64>,
}

/// Result from executing a shell command.
#[derive(Debug, Clone)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    /// Exit code from the process, or -1 on timeout/signal.
    pub exit_code: i32,
    /// Whether the command was terminated due to a timeout.
    pub timed_out: bool,
    /// Wall-clock duration of the command execution.
    pub duration: Duration,
}

/// Options for the grep search operation.
#[derive(Debug, Clone, Default)]
pub struct GrepOptions {
    /// Whether to match case-insensitively.
    pub case_insensitive: bool,
    /// Maximum number of results to return. 0 = use implementation default.
    pub max_results: usize,
    /// Optional glob filter to restrict which files are searched (e.g. `"*.rs"`).
    pub glob_filter: Option<String>,
}

// ── Trait definition ─────────────────────────────────────────────────────────

/// Abstraction over where tool operations execute.
///
/// Implement this trait to support different execution targets:
/// local filesystem, Docker containers, SSH remotes, WASM, etc.
/// Changing the environment should not require changing any tool logic.
///
/// # Provided implementations
/// - [`LocalExecutionEnvironment`]: runs on the local machine (default, fully tested).
/// - [`crate::testing::MockExecutionEnvironment`]: in-memory mock for unit tests.
///
/// # GAP-CAL-008: Out-of-scope environment variants
/// Docker, Kubernetes, and SSH execution environments are architecturally
/// supported by this trait but are **out of scope** for the current milestone.
/// They can be added by downstream crates without modifying tool or session
/// logic. See NLSpec §9.4.
///
/// # Object Safety
/// The trait uses `#[async_trait]` for object safety with `dyn ExecutionEnvironment`.
/// Implementations must be `Send + Sync`.
#[async_trait]
pub trait ExecutionEnvironment: Send + Sync {
    // ── File operations ──────────────────────────────────────────────────

    /// Read a file and return its text content, optionally with line numbers.
    ///
    /// - `offset`: 1-based starting line number (`None` = read from line 1).
    /// - `limit`: maximum number of lines to return (`None` = no limit).
    ///
    /// Returns `EnvError::FileNotFound` if the path does not exist.
    async fn read_file(
        &self,
        path: &str,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<String, EnvError>;

    /// Write `content` to `path`, creating parent directories as needed.
    /// Overwrites existing files.
    async fn write_file(&self, path: &str, content: &str) -> Result<(), EnvError>;

    /// Returns `true` if the path exists (file or directory), `false` otherwise.
    async fn file_exists(&self, path: &str) -> Result<bool, EnvError>;

    /// List directory contents up to `depth` levels deep (1 = immediate children).
    async fn list_directory(&self, path: &str, depth: u32) -> Result<Vec<DirEntry>, EnvError>;

    // ── Command execution ────────────────────────────────────────────────

    /// Execute a shell command with a timeout.
    ///
    /// - `timeout`: how long to wait before killing the process.
    /// - `working_dir`: override the working directory (`None` = use environment default).
    /// - `env_vars`: extra environment variables merged on top of the filtered system env.
    async fn exec_command(
        &self,
        command: &str,
        timeout: Duration,
        working_dir: Option<&str>,
        env_vars: Option<&HashMap<String, String>>,
    ) -> Result<ExecResult, EnvError>;

    // ── Search operations ────────────────────────────────────────────────

    /// Search file contents using a regex pattern.
    /// Returns matching lines in `filename:line_num:content` format.
    async fn grep(
        &self,
        pattern: &str,
        path: &str,
        options: &GrepOptions,
    ) -> Result<String, EnvError>;

    /// Find files matching a glob pattern under `path`.
    /// Returns paths sorted by modification time (newest first).
    async fn glob(&self, pattern: &str, path: &str) -> Result<Vec<String>, EnvError>;

    // ── Lifecycle ────────────────────────────────────────────────────────

    /// Initialize the environment (allocate resources, verify connectivity).
    /// No-op for local environments.
    async fn initialize(&mut self) -> Result<(), EnvError>;

    /// Clean up resources (kill background processes, release connections).
    async fn cleanup(&mut self) -> Result<(), EnvError>;

    // ── Metadata ─────────────────────────────────────────────────────────

    /// The working directory used to resolve relative paths.
    fn working_directory(&self) -> &str;

    /// Platform string: `"linux"`, `"macos"`, or `"windows"`.
    fn platform(&self) -> &str;

    /// Human-readable OS version string (informational).
    fn os_version(&self) -> &str;
}

// ── Compile-time check ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_object_safe(_: &dyn ExecutionEnvironment) {}

    #[allow(dead_code)]
    fn _check_object_safety(env: &dyn ExecutionEnvironment) {
        assert_object_safe(env);
    }

    #[test]
    fn grep_options_default() {
        let opts = GrepOptions::default();
        assert!(!opts.case_insensitive);
        assert_eq!(opts.max_results, 0);
        assert!(opts.glob_filter.is_none());
    }

    #[test]
    fn dir_entry_is_debug_clone() {
        let entry = DirEntry {
            name: "foo.rs".into(),
            is_dir: false,
            size: Some(1234),
        };
        let cloned = entry.clone();
        assert_eq!(cloned.name, "foo.rs");
        assert!(!cloned.is_dir);
        assert_eq!(cloned.size, Some(1234));
    }

    #[test]
    fn exec_result_is_debug_clone() {
        let r = ExecResult {
            stdout: "hello".into(),
            stderr: "".into(),
            exit_code: 0,
            timed_out: false,
            duration: Duration::from_millis(100),
        };
        let cloned = r.clone();
        assert_eq!(cloned.stdout, "hello");
        assert!(!cloned.timed_out);
    }
}
