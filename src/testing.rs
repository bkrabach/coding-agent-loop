//! Test utilities for the coding-agent-loop crate.
//!
//! [`MockExecutionEnvironment`] provides an in-memory execution environment
//! for unit and integration tests, with a virtual filesystem, configurable
//! command responses, and call recording.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use regex::Regex;

use crate::environment::ExecutionEnvironment;
use crate::environment::{DirEntry, ExecResult, GrepOptions};
use crate::error::EnvError;

// ── Call recording ────────────────────────────────────────────────────────────

/// A single recorded call to the mock environment, for test assertions.
#[derive(Debug, Clone)]
pub enum EnvCall {
    ReadFile {
        path: String,
        offset: Option<usize>,
        limit: Option<usize>,
    },
    WriteFile {
        path: String,
        content: String,
    },
    FileExists {
        path: String,
    },
    ListDirectory {
        path: String,
        depth: u32,
    },
    ExecCommand {
        command: String,
        timeout: Duration,
    },
    Grep {
        pattern: String,
        path: String,
    },
    Glob {
        pattern: String,
        path: String,
    },
}

// ── MockCommandResponse ───────────────────────────────────────────────────────

/// Canned response for a command prefix match.
#[derive(Debug, Clone, Default)]
pub struct MockCommandResponse {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
    pub duration: Duration,
}

impl MockCommandResponse {
    /// Convenience: success response with stdout.
    pub fn success(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            exit_code: 0,
            ..Default::default()
        }
    }

    /// Convenience: failure response with exit code and stderr.
    pub fn failure(exit_code: i32, stderr: impl Into<String>) -> Self {
        Self {
            stderr: stderr.into(),
            exit_code,
            ..Default::default()
        }
    }
}

// ── MockExecutionEnvironment ──────────────────────────────────────────────────

/// In-memory [`ExecutionEnvironment`] for testing.
///
/// - Virtual filesystem: `HashMap<path, content>`
/// - Configurable command responses keyed by command prefix (longest-match wins)
/// - Call recording via `recorded_calls()`
///
/// Thread-safe via `Arc<Mutex<...>>` — cheap to clone.
#[derive(Debug, Clone)]
pub struct MockExecutionEnvironment {
    /// Virtual filesystem: canonical path → content.
    pub files: Arc<Mutex<HashMap<String, String>>>,
    /// Command responses: command prefix → response.
    pub command_responses: Arc<Mutex<HashMap<String, MockCommandResponse>>>,
    /// Ordered log of all calls made.
    pub calls: Arc<Mutex<Vec<EnvCall>>>,
    /// The working directory string returned by `working_directory()`.
    pub working_dir: String,
    /// The platform string returned by `platform()`.
    pub platform_str: String,
}

impl MockExecutionEnvironment {
    /// Create a new mock environment with the given working directory.
    pub fn new(working_dir: impl Into<String>) -> Self {
        Self {
            files: Arc::new(Mutex::new(HashMap::new())),
            command_responses: Arc::new(Mutex::new(HashMap::new())),
            calls: Arc::new(Mutex::new(Vec::new())),
            working_dir: working_dir.into(),
            platform_str: "linux".to_owned(),
        }
    }

    /// Builder: pre-populate a file in the virtual filesystem.
    pub fn with_file(self, path: impl Into<String>, content: impl Into<String>) -> Self {
        self.files
            .lock()
            .unwrap()
            .insert(path.into(), content.into());
        self
    }

    /// Builder: add a canned command response for commands starting with `prefix`.
    pub fn with_command_response(
        self,
        prefix: impl Into<String>,
        response: MockCommandResponse,
    ) -> Self {
        self.command_responses
            .lock()
            .unwrap()
            .insert(prefix.into(), response);
        self
    }

    /// Mutating: add a file to the virtual filesystem in place.
    pub fn add_file(&self, path: impl Into<String>, content: impl Into<String>) {
        self.files
            .lock()
            .unwrap()
            .insert(path.into(), content.into());
    }

    /// Mutating: add a canned command response in place.
    pub fn add_command_response(&self, prefix: impl Into<String>, response: MockCommandResponse) {
        self.command_responses
            .lock()
            .unwrap()
            .insert(prefix.into(), response);
    }

    /// Return all recorded calls in order.
    pub fn recorded_calls(&self) -> Vec<EnvCall> {
        self.calls.lock().unwrap().clone()
    }

    /// Clear the recorded call log.
    pub fn clear_calls(&self) {
        self.calls.lock().unwrap().clear();
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn record(&self, call: EnvCall) {
        self.calls.lock().unwrap().push(call);
    }

    /// Find the longest-matching command prefix, returning the associated response.
    fn find_command_response(&self, command: &str) -> Option<MockCommandResponse> {
        let responses = self.command_responses.lock().unwrap();
        responses
            .iter()
            .filter(|(prefix, _)| command.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map(|(_, response)| response.clone())
    }
}

// ── ExecutionEnvironment impl ─────────────────────────────────────────────────

#[async_trait]
impl ExecutionEnvironment for MockExecutionEnvironment {
    async fn read_file(
        &self,
        path: &str,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<String, EnvError> {
        self.record(EnvCall::ReadFile {
            path: path.to_owned(),
            offset,
            limit,
        });

        let files = self.files.lock().unwrap();
        let content = files
            .get(path)
            .ok_or_else(|| EnvError::FileNotFound(path.to_owned()))?
            .clone();
        drop(files);

        // Apply offset (1-based) and limit.
        let start = offset.map(|o| o.saturating_sub(1)).unwrap_or(0);
        let lines: Vec<&str> = content.lines().collect();
        let slice = &lines[start.min(lines.len())..];
        let slice = match limit {
            Some(n) => &slice[..n.min(slice.len())],
            None => slice,
        };

        let numbered: Vec<String> = slice
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:4} | {}", start + i + 1, line))
            .collect();

        Ok(numbered.join("\n"))
    }

    async fn write_file(&self, path: &str, content: &str) -> Result<(), EnvError> {
        self.record(EnvCall::WriteFile {
            path: path.to_owned(),
            content: content.to_owned(),
        });
        self.files
            .lock()
            .unwrap()
            .insert(path.to_owned(), content.to_owned());
        Ok(())
    }

    async fn file_exists(&self, path: &str) -> Result<bool, EnvError> {
        self.record(EnvCall::FileExists {
            path: path.to_owned(),
        });
        Ok(self.files.lock().unwrap().contains_key(path))
    }

    async fn list_directory(&self, path: &str, depth: u32) -> Result<Vec<DirEntry>, EnvError> {
        self.record(EnvCall::ListDirectory {
            path: path.to_owned(),
            depth,
        });

        let files = self.files.lock().unwrap();
        let prefix = if path == "." || path.is_empty() {
            String::new()
        } else {
            format!("{path}/")
        };

        let mut entries: Vec<DirEntry> = files
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .map(|k| {
                let relative = k.strip_prefix(&prefix).unwrap_or(k);
                let top_level = relative.split('/').next().unwrap_or(relative);
                let is_dir = relative.contains('/');
                DirEntry {
                    name: top_level.to_owned(),
                    is_dir,
                    size: if is_dir {
                        None
                    } else {
                        Some(files[k].len() as u64)
                    },
                }
            })
            .collect();

        // Deduplicate (directories may appear for multiple files).
        entries.dedup_by(|a, b| a.name == b.name);
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    async fn exec_command(
        &self,
        command: &str,
        timeout: Duration,
        _working_dir: Option<&str>,
        _env_vars: Option<&HashMap<String, String>>,
    ) -> Result<ExecResult, EnvError> {
        self.record(EnvCall::ExecCommand {
            command: command.to_owned(),
            timeout,
        });

        let response = self
            .find_command_response(command)
            .unwrap_or_else(|| MockCommandResponse {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: 0,
                timed_out: false,
                duration: Duration::ZERO,
            });

        Ok(ExecResult {
            stdout: response.stdout,
            stderr: response.stderr,
            exit_code: response.exit_code,
            timed_out: response.timed_out,
            duration: response.duration,
        })
    }

    async fn grep(
        &self,
        pattern: &str,
        path: &str,
        _options: &GrepOptions,
    ) -> Result<String, EnvError> {
        self.record(EnvCall::Grep {
            pattern: pattern.to_owned(),
            path: path.to_owned(),
        });

        let re = Regex::new(pattern).map_err(|e| {
            EnvError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                e.to_string(),
            ))
        })?;

        let files = self.files.lock().unwrap();
        let prefix = if path == "." || path.is_empty() {
            String::new()
        } else {
            format!("{path}/")
        };

        let mut results = Vec::new();
        let mut keys: Vec<&String> = files.keys().filter(|k| k.starts_with(&prefix)).collect();
        keys.sort();

        for file_path in keys {
            for (line_no, line) in files[file_path].lines().enumerate() {
                if re.is_match(line) {
                    results.push(format!("{}:{}:{}", file_path, line_no + 1, line));
                }
            }
        }

        Ok(results.join("\n"))
    }

    async fn glob(&self, pattern: &str, path: &str) -> Result<Vec<String>, EnvError> {
        self.record(EnvCall::Glob {
            pattern: pattern.to_owned(),
            path: path.to_owned(),
        });

        let glob = globset::Glob::new(pattern).map_err(|e| {
            EnvError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                e.to_string(),
            ))
        })?;
        let glob_set = globset::GlobSetBuilder::new()
            .add(glob)
            .build()
            .map_err(|e| {
                EnvError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    e.to_string(),
                ))
            })?;

        let files = self.files.lock().unwrap();
        let prefix = if path == "." || path.is_empty() {
            String::new()
        } else {
            format!("{path}/")
        };

        let mut matches: Vec<String> = files
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .map(|k| k.strip_prefix(&prefix).unwrap_or(k).to_owned())
            .filter(|rel| glob_set.is_match(rel))
            .collect();
        matches.sort();

        Ok(matches)
    }

    async fn initialize(&mut self) -> Result<(), EnvError> {
        Ok(())
    }

    async fn cleanup(&mut self) -> Result<(), EnvError> {
        Ok(())
    }

    fn working_directory(&self) -> &str {
        &self.working_dir
    }

    fn platform(&self) -> &str {
        &self.platform_str
    }

    fn os_version(&self) -> &str {
        "mock-os-1.0"
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn mock_env_is_send_sync() {
        assert_send_sync::<MockExecutionEnvironment>();
    }

    #[test]
    fn with_file_builder() {
        let env = MockExecutionEnvironment::new("/work").with_file("foo.txt", "hello world");
        let files = env.files.lock().unwrap();
        assert!(files.contains_key("foo.txt"));
        assert_eq!(files["foo.txt"], "hello world");
    }

    #[tokio::test]
    async fn write_read_roundtrip() {
        let env = MockExecutionEnvironment::new("/work");
        env.write_file("out.txt", "content here").await.unwrap();
        let content = env.read_file("out.txt", None, None).await.unwrap();
        assert!(content.contains("content here"));
    }

    #[tokio::test]
    async fn read_file_not_found() {
        let env = MockExecutionEnvironment::new("/work");
        let err = env.read_file("missing.txt", None, None).await.unwrap_err();
        assert!(matches!(err, EnvError::FileNotFound(_)));
    }

    #[tokio::test]
    async fn file_exists_true_after_write() {
        let env = MockExecutionEnvironment::new("/work");
        assert!(!env.file_exists("x.txt").await.unwrap());
        env.write_file("x.txt", "").await.unwrap();
        assert!(env.file_exists("x.txt").await.unwrap());
    }

    #[tokio::test]
    async fn command_with_matching_response() {
        let env = MockExecutionEnvironment::new("/work")
            .with_command_response("echo", MockCommandResponse::success("hello\n"));
        let result = env
            .exec_command("echo hello", Duration::from_secs(5), None, None)
            .await
            .unwrap();
        assert_eq!(result.stdout, "hello\n");
        assert_eq!(result.exit_code, 0);
    }

    #[tokio::test]
    async fn command_no_match_returns_default_success() {
        let env = MockExecutionEnvironment::new("/work");
        let result = env
            .exec_command("ls", Duration::from_secs(5), None, None)
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(!result.timed_out);
    }

    #[tokio::test]
    async fn longest_prefix_wins() {
        let env = MockExecutionEnvironment::new("/work")
            .with_command_response("echo", MockCommandResponse::success("short"))
            .with_command_response("echo hello", MockCommandResponse::success("long"));
        let result = env
            .exec_command("echo hello world", Duration::from_secs(5), None, None)
            .await
            .unwrap();
        assert_eq!(result.stdout, "long");
    }

    #[tokio::test]
    async fn recorded_calls_in_order() {
        let env = MockExecutionEnvironment::new("/work");
        env.write_file("a.txt", "x").await.unwrap();
        env.read_file("a.txt", None, None).await.unwrap();
        env.file_exists("a.txt").await.unwrap();

        let calls = env.recorded_calls();
        assert_eq!(calls.len(), 3);
        assert!(matches!(calls[0], EnvCall::WriteFile { .. }));
        assert!(matches!(calls[1], EnvCall::ReadFile { .. }));
        assert!(matches!(calls[2], EnvCall::FileExists { .. }));
    }

    #[tokio::test]
    async fn clear_calls() {
        let env = MockExecutionEnvironment::new("/work");
        env.write_file("f.txt", "").await.unwrap();
        assert_eq!(env.recorded_calls().len(), 1);
        env.clear_calls();
        assert_eq!(env.recorded_calls().len(), 0);
    }

    #[tokio::test]
    async fn glob_finds_matching_files() {
        let env = MockExecutionEnvironment::new("/work")
            .with_file("src/main.rs", "")
            .with_file("src/lib.rs", "")
            .with_file("README.md", "");

        let matches = env.glob("*.rs", "src").await.unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().any(|p| p.ends_with("main.rs")));
        assert!(matches.iter().any(|p| p.ends_with("lib.rs")));
    }

    #[tokio::test]
    async fn grep_finds_pattern() {
        let env = MockExecutionEnvironment::new("/work")
            .with_file("code.rs", "fn main() {\n    println!(\"hello\");\n}\n");

        let result = env
            .grep("fn main", ".", &GrepOptions::default())
            .await
            .unwrap();
        assert!(result.contains("fn main"));
    }

    #[tokio::test]
    async fn read_file_with_offset_limit() {
        let env = MockExecutionEnvironment::new("/work").with_file("lines.txt", "a\nb\nc\nd\ne");

        // Lines 2–3
        let content = env.read_file("lines.txt", Some(2), Some(2)).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("b"));
        assert!(lines[1].contains("c"));
    }
}
