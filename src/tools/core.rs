//! Core tool executors: read_file, write_file, edit_file, shell, grep, glob.
//!
//! Each executor wraps `ExecutionEnvironment` methods and provides a
//! `registered_tool()` factory returning a fully wired `RegisteredTool`.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::environment::{ExecutionEnvironment, GrepOptions};
use crate::error::ToolError;
use crate::tools::{RegisteredTool, ToolDefinition, ToolExecutor};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Strip line-number prefixes inserted by `ExecutionEnvironment::read_file`.
///
/// Handles two formats produced by different environment implementations:
/// - `"     N\tcontent"` (tab separator — `LocalExecutionEnvironment`)
/// - `"   N | content"` (` | ` separator — `MockExecutionEnvironment`)
///
/// Returns the raw file content with original line endings preserved.
pub(crate) fn strip_line_numbers(numbered: &str) -> String {
    let has_trailing_nl = numbered.ends_with('\n');
    let raw_lines: Vec<&str> = numbered.split('\n').collect();

    // Skip the last empty element that split('\n') creates when input ends with '\n'.
    let n = if has_trailing_nl && raw_lines.last() == Some(&"") {
        raw_lines.len() - 1
    } else {
        raw_lines.len()
    };

    let stripped: Vec<&str> = raw_lines[..n]
        .iter()
        .map(|line| strip_one_line(line))
        .collect();

    let mut result = stripped.join("\n");
    if has_trailing_nl {
        result.push('\n');
    }
    result
}

/// Strip the line-number prefix from a single line (without its newline).
fn strip_one_line(line: &str) -> &str {
    // Tab format: "     N\tcontent"
    if let Some(idx) = line.find('\t') {
        let prefix = &line[..idx];
        if !prefix.is_empty() && prefix.trim().chars().all(|c| c.is_ascii_digit()) {
            return &line[idx + 1..];
        }
    }
    // " | " format: "   N | content"
    if let Some(idx) = line.find(" | ") {
        let prefix = &line[..idx];
        if !prefix.is_empty() && prefix.trim().chars().all(|c| c.is_ascii_digit()) {
            return &line[idx + 3..];
        }
    }
    line
}

// ── ReadFileExecutor ──────────────────────────────────────────────────────────

/// Executor for the `read_file` tool.
pub struct ReadFileExecutor;

impl ReadFileExecutor {
    /// Return a ready-to-register `RegisteredTool`.
    pub fn registered_tool() -> RegisteredTool {
        RegisteredTool {
            definition: ToolDefinition {
                name: "read_file".to_owned(),
                description: "Read a file from the filesystem. Returns line-numbered content."
                    .to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the file."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "1-based line number to start reading from."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read (default: 2000)."
                        }
                    },
                    "required": ["file_path"]
                }),
            },
            executor: Box::new(ReadFileExecutor),
        }
    }
}

#[async_trait]
impl ToolExecutor for ReadFileExecutor {
    async fn execute(
        &self,
        args: Value,
        env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError> {
        let file_path = args["file_path"]
            .as_str()
            .ok_or_else(|| ToolError::Validation("missing required arg: file_path".to_owned()))?
            .to_owned();

        let offset = args["offset"].as_u64().map(|n| n as usize);
        let limit = args["limit"].as_u64().map(|n| n as usize);

        env.read_file(&file_path, offset, limit)
            .await
            .map_err(|e| match e {
                crate::error::EnvError::FileNotFound(_) => ToolError::FileNotFound(file_path),
                other => ToolError::Environment(other),
            })
    }
}

// ── WriteFileExecutor ─────────────────────────────────────────────────────────

/// Executor for the `write_file` tool.
pub struct WriteFileExecutor;

impl WriteFileExecutor {
    pub fn registered_tool() -> RegisteredTool {
        RegisteredTool {
            definition: ToolDefinition {
                name: "write_file".to_owned(),
                description:
                    "Write content to a file, creating it (and parent directories) if it does not \
                     exist. Overwrites existing files."
                        .to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the file."
                        },
                        "content": {
                            "type": "string",
                            "description": "Full file content to write."
                        }
                    },
                    "required": ["file_path", "content"]
                }),
            },
            executor: Box::new(WriteFileExecutor),
        }
    }
}

#[async_trait]
impl ToolExecutor for WriteFileExecutor {
    async fn execute(
        &self,
        args: Value,
        env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError> {
        let file_path = args["file_path"]
            .as_str()
            .ok_or_else(|| ToolError::Validation("missing required arg: file_path".to_owned()))?
            .to_owned();
        let content = args["content"]
            .as_str()
            .ok_or_else(|| ToolError::Validation("missing required arg: content".to_owned()))?
            .to_owned();

        let byte_count = content.len();
        env.write_file(&file_path, &content)
            .await
            .map_err(ToolError::Environment)?;

        Ok(format!("Written {} bytes to {}", byte_count, file_path))
    }
}

// ── EditFileExecutor ──────────────────────────────────────────────────────────

/// Executor for the `edit_file` tool.
///
/// Performs exact-string search-and-replace. When `replace_all` is false,
/// `old_string` must appear exactly once in the file.
pub struct EditFileExecutor;

impl EditFileExecutor {
    pub fn registered_tool() -> RegisteredTool {
        RegisteredTool {
            definition: ToolDefinition {
                name: "edit_file".to_owned(),
                description:
                    "Replace an exact string occurrence in a file. When replace_all is false \
                     (default), old_string must appear exactly once."
                        .to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the file."
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Exact text to find and replace."
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement text."
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "If true, replace all occurrences. Default: false."
                        }
                    },
                    "required": ["file_path", "old_string", "new_string"]
                }),
            },
            executor: Box::new(EditFileExecutor),
        }
    }
}

#[async_trait]
impl ToolExecutor for EditFileExecutor {
    async fn execute(
        &self,
        args: Value,
        env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError> {
        let file_path = args["file_path"]
            .as_str()
            .ok_or_else(|| ToolError::Validation("missing required arg: file_path".to_owned()))?
            .to_owned();
        let old_string = args["old_string"]
            .as_str()
            .ok_or_else(|| ToolError::Validation("missing required arg: old_string".to_owned()))?
            .to_owned();
        let new_string = args["new_string"]
            .as_str()
            .ok_or_else(|| ToolError::Validation("missing required arg: new_string".to_owned()))?
            .to_owned();
        let replace_all = args["replace_all"].as_bool().unwrap_or(false);

        // Read the file (returns line-numbered content).
        let numbered = env
            .read_file(&file_path, None, None)
            .await
            .map_err(|e| match e {
                crate::error::EnvError::FileNotFound(_) => {
                    ToolError::FileNotFound(file_path.clone())
                }
                other => ToolError::Environment(other),
            })?;

        // Strip line numbers to get raw content.
        let raw = strip_line_numbers(&numbered);

        // Count occurrences.
        let count = raw.matches(old_string.as_str()).count();

        let new_content;
        let replacements;

        if replace_all {
            new_content = raw.replace(old_string.as_str(), new_string.as_str());
            replacements = count;
        } else {
            match count {
                0 => {
                    // Fuzzy fallback: whitespace normalization.
                    if let Some(result) = fuzzy_replace(&raw, &old_string, &new_string) {
                        new_content = result;
                        replacements = 1;
                    } else {
                        return Err(ToolError::EditConflict(format!(
                            "old_string not found in {}",
                            file_path
                        )));
                    }
                }
                1 => {
                    new_content = raw.replacen(old_string.as_str(), new_string.as_str(), 1);
                    replacements = 1;
                }
                n => {
                    return Err(ToolError::EditConflict(format!(
                        "old_string found {} times in {}; add more context to make it unique or \
                         use replace_all=true",
                        n, file_path
                    )));
                }
            }
        }

        env.write_file(&file_path, &new_content)
            .await
            .map_err(ToolError::Environment)?;

        Ok(format!(
            "Replaced {} occurrence(s) in {}",
            replacements, file_path
        ))
    }
}

/// Whitespace-normalizing fuzzy replace. Returns `Some(new_content)` if exactly
/// one fuzzy match is found, `None` otherwise.
fn fuzzy_replace(content: &str, old: &str, new: &str) -> Option<String> {
    let norm_content = normalize_whitespace(content);
    let norm_old = normalize_whitespace(old);

    let count = norm_content.matches(norm_old.as_str()).count();
    if count != 1 {
        return None;
    }

    // Find the position in the original content that corresponds to the
    // normalized match. Simple approach: find the normalized old in normalized
    // content, then map back.
    // For simplicity, we do line-by-line mapping.
    let orig_lines: Vec<&str> = content.lines().collect();
    let old_lines: Vec<&str> = old.lines().collect();

    if old_lines.is_empty() {
        return None;
    }

    // Search for a block of orig_lines whose normalized versions match norm_old_lines.
    let norm_old_lines: Vec<String> = old_lines.iter().map(|l| normalize_whitespace(l)).collect();

    for start in 0..orig_lines.len() {
        let end = start + old_lines.len();
        if end > orig_lines.len() {
            break;
        }
        let candidate: Vec<String> = orig_lines[start..end]
            .iter()
            .map(|l| normalize_whitespace(l))
            .collect();
        if candidate == norm_old_lines {
            // Reconstruct: replace lines [start..end] with new lines.
            let mut result_lines: Vec<&str> = orig_lines[..start].to_vec();
            result_lines.extend(new.lines());
            result_lines.extend(orig_lines[end..].iter().copied());
            let mut result = result_lines.join("\n");
            // Preserve trailing newline.
            if content.ends_with('\n') {
                result.push('\n');
            }
            return Some(result);
        }
    }

    None
}

fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── ShellExecutor ─────────────────────────────────────────────────────────────

/// Executor for the `shell` tool.
pub struct ShellExecutor {
    pub default_timeout: Duration,
    pub max_timeout: Duration,
}

impl ShellExecutor {
    pub fn new(default_timeout: Duration, max_timeout: Duration) -> Self {
        Self {
            default_timeout,
            max_timeout,
        }
    }

    pub fn registered_tool(default_timeout: Duration, max_timeout: Duration) -> RegisteredTool {
        RegisteredTool {
            definition: ToolDefinition {
                name: "shell".to_owned(),
                description:
                    "Execute a shell command. Returns stdout, stderr, exit code, and duration."
                        .to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to run."
                        },
                        "timeout_ms": {
                            "type": "integer",
                            "description": "Override command timeout in milliseconds."
                        },
                        "description": {
                            "type": "string",
                            "description": "Human-readable description of what this command does."
                        }
                    },
                    "required": ["command"]
                }),
            },
            executor: Box::new(ShellExecutor {
                default_timeout,
                max_timeout,
            }),
        }
    }
}

#[async_trait]
impl ToolExecutor for ShellExecutor {
    async fn execute(
        &self,
        args: Value,
        env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| ToolError::Validation("missing required arg: command".to_owned()))?
            .to_owned();

        // Resolve effective timeout.
        let timeout = if let Some(ms) = args["timeout_ms"].as_u64() {
            if ms == 0 {
                self.default_timeout
            } else {
                let requested = Duration::from_millis(ms);
                requested.min(self.max_timeout)
            }
        } else {
            self.default_timeout
        };

        let result = env
            .exec_command(&command, timeout, None, None)
            .await
            .map_err(ToolError::Environment)?;

        let mut output = format!(
            "exit_code: {}\nduration: {}ms\nstdout:\n{}\nstderr:\n{}",
            result.exit_code,
            result.duration.as_millis(),
            result.stdout,
            result.stderr,
        );

        if result.timed_out {
            output.push_str(&format!(
                "\n[ERROR: Command timed out after {}ms. Partial output is shown above.\n\
                 You can retry with a longer timeout by setting the timeout_ms parameter.]",
                timeout.as_millis()
            ));
        }

        Ok(output)
    }
}

// ── GrepExecutor ──────────────────────────────────────────────────────────────

/// Executor for the `grep` tool.
pub struct GrepExecutor;

impl GrepExecutor {
    pub fn registered_tool() -> RegisteredTool {
        RegisteredTool {
            definition: ToolDefinition {
                name: "grep".to_owned(),
                description:
                    "Search file contents using a regex pattern. Returns matching lines with file \
                     paths and line numbers."
                        .to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for."
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory or file path to search in. Default: working directory."
                        },
                        "glob_filter": {
                            "type": "string",
                            "description": "File name pattern filter (e.g. '*.rs')."
                        },
                        "case_insensitive": {
                            "type": "boolean",
                            "description": "Match case-insensitively. Default: false."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return. Default: 100."
                        }
                    },
                    "required": ["pattern"]
                }),
            },
            executor: Box::new(GrepExecutor),
        }
    }
}

#[async_trait]
impl ToolExecutor for GrepExecutor {
    async fn execute(
        &self,
        args: Value,
        env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::Validation("missing required arg: pattern".to_owned()))?
            .to_owned();

        let path = args["path"].as_str().unwrap_or(".");
        let glob_filter = args["glob_filter"].as_str().map(str::to_owned);
        let case_insensitive = args["case_insensitive"].as_bool().unwrap_or(false);
        let max_results = args["max_results"].as_u64().unwrap_or(100) as usize;

        let options = GrepOptions {
            case_insensitive,
            max_results,
            glob_filter,
        };

        let result = env
            .grep(&pattern, path, &options)
            .await
            .map_err(ToolError::Environment)?;

        if result.is_empty() {
            Ok("No matches found.".to_owned())
        } else {
            Ok(result)
        }
    }
}

// ── GlobExecutor ──────────────────────────────────────────────────────────────

/// Executor for the `glob` tool.
pub struct GlobExecutor;

impl GlobExecutor {
    pub fn registered_tool() -> RegisteredTool {
        RegisteredTool {
            definition: ToolDefinition {
                name: "glob".to_owned(),
                description: "Find files matching a glob pattern. Returns file paths sorted by \
                     modification time (newest first)."
                    .to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern (e.g. '**/*.rs')."
                        },
                        "path": {
                            "type": "string",
                            "description": "Base directory to search from. Default: working directory."
                        }
                    },
                    "required": ["pattern"]
                }),
            },
            executor: Box::new(GlobExecutor),
        }
    }
}

#[async_trait]
impl ToolExecutor for GlobExecutor {
    async fn execute(
        &self,
        args: Value,
        env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::Validation("missing required arg: pattern".to_owned()))?
            .to_owned();

        let path = args["path"].as_str().unwrap_or(".");

        let paths = env
            .glob(&pattern, path)
            .await
            .map_err(ToolError::Environment)?;

        if paths.is_empty() {
            Ok("No files found.".to_owned())
        } else {
            Ok(paths.join("\n"))
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{MockCommandResponse, MockExecutionEnvironment};

    // ── read_file ──

    #[tokio::test]
    async fn read_file_missing_arg_returns_validation_error() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = ReadFileExecutor;
        let result = executor.execute(json!({}), &env).await;
        assert!(matches!(result, Err(ToolError::Validation(_))));
    }

    #[tokio::test]
    async fn read_file_not_found_returns_file_not_found_error() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = ReadFileExecutor;
        let result = executor
            .execute(json!({"file_path": "/nonexistent.txt"}), &env)
            .await;
        assert!(matches!(result, Err(ToolError::FileNotFound(_))));
    }

    #[tokio::test]
    async fn read_file_success_returns_content() {
        let env = MockExecutionEnvironment::new("/tmp");
        env.add_file("/tmp/hello.txt", "hello world");
        let executor = ReadFileExecutor;
        let result = executor
            .execute(json!({"file_path": "/tmp/hello.txt"}), &env)
            .await
            .unwrap();
        assert!(result.contains("hello world"));
    }

    #[test]
    fn read_file_registered_tool_has_correct_name() {
        let t = ReadFileExecutor::registered_tool();
        assert_eq!(t.definition.name, "read_file");
    }

    // ── write_file ──

    #[tokio::test]
    async fn write_file_missing_file_path_validation_error() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = WriteFileExecutor;
        let result = executor.execute(json!({"content": "hi"}), &env).await;
        assert!(matches!(result, Err(ToolError::Validation(_))));
    }

    #[tokio::test]
    async fn write_file_missing_content_validation_error() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = WriteFileExecutor;
        let result = executor
            .execute(json!({"file_path": "/tmp/out.txt"}), &env)
            .await;
        assert!(matches!(result, Err(ToolError::Validation(_))));
    }

    #[tokio::test]
    async fn write_file_success_returns_byte_count() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = WriteFileExecutor;
        let result = executor
            .execute(
                json!({"file_path": "/tmp/out.txt", "content": "hello"}),
                &env,
            )
            .await
            .unwrap();
        assert!(result.contains("5 bytes"));
        assert!(result.contains("/tmp/out.txt"));
    }

    #[test]
    fn write_file_registered_tool_has_correct_name() {
        let t = WriteFileExecutor::registered_tool();
        assert_eq!(t.definition.name, "write_file");
    }

    // ── edit_file ──

    #[tokio::test]
    async fn edit_file_missing_arg_returns_validation_error() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = EditFileExecutor;
        let result = executor.execute(json!({}), &env).await;
        assert!(matches!(result, Err(ToolError::Validation(_))));
    }

    #[tokio::test]
    async fn edit_file_not_found_returns_file_not_found() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = EditFileExecutor;
        let result = executor
            .execute(
                json!({"file_path": "/nonexistent.txt", "old_string": "x", "new_string": "y"}),
                &env,
            )
            .await;
        assert!(matches!(result, Err(ToolError::FileNotFound(_))));
    }

    #[tokio::test]
    async fn edit_file_old_string_not_found_returns_conflict() {
        let env = MockExecutionEnvironment::new("/tmp");
        env.add_file("/tmp/f.txt", "hello world");
        let executor = EditFileExecutor;
        let result = executor
            .execute(
                json!({"file_path": "/tmp/f.txt", "old_string": "NOTHERE", "new_string": "x"}),
                &env,
            )
            .await;
        assert!(matches!(result, Err(ToolError::EditConflict(_))));
    }

    #[tokio::test]
    async fn edit_file_multiple_matches_returns_conflict() {
        let env = MockExecutionEnvironment::new("/tmp");
        env.add_file("/tmp/f.txt", "foo foo foo");
        let executor = EditFileExecutor;
        let result = executor
            .execute(
                json!({"file_path": "/tmp/f.txt", "old_string": "foo", "new_string": "bar"}),
                &env,
            )
            .await;
        assert!(matches!(result, Err(ToolError::EditConflict(_))));
    }

    #[tokio::test]
    async fn edit_file_single_match_succeeds() {
        let env = MockExecutionEnvironment::new("/tmp");
        env.add_file("/tmp/f.txt", "hello world");
        let executor = EditFileExecutor;
        let result = executor
            .execute(
                json!({"file_path": "/tmp/f.txt", "old_string": "hello", "new_string": "goodbye"}),
                &env,
            )
            .await
            .unwrap();
        assert!(result.contains("1 occurrence(s)"));
    }

    #[tokio::test]
    async fn edit_file_replace_all_replaces_multiple() {
        let env = MockExecutionEnvironment::new("/tmp");
        env.add_file("/tmp/f.txt", "foo foo foo");
        let executor = EditFileExecutor;
        let result = executor
            .execute(
                json!({"file_path": "/tmp/f.txt", "old_string": "foo", "new_string": "bar", "replace_all": true}),
                &env,
            )
            .await
            .unwrap();
        assert!(result.contains("3 occurrence(s)"));
    }

    #[test]
    fn edit_file_registered_tool_has_correct_name() {
        let t = EditFileExecutor::registered_tool();
        assert_eq!(t.definition.name, "edit_file");
    }

    // ── shell ──

    #[tokio::test]
    async fn shell_missing_command_returns_validation_error() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = ShellExecutor::new(Duration::from_secs(10), Duration::from_secs(600));
        let result = executor.execute(json!({}), &env).await;
        assert!(matches!(result, Err(ToolError::Validation(_))));
    }

    #[tokio::test]
    async fn shell_success_returns_formatted_output() {
        let env = MockExecutionEnvironment::new("/tmp");
        env.add_command_response(
            "echo hello",
            MockCommandResponse {
                stdout: "hello\n".to_owned(),
                stderr: "".to_owned(),
                exit_code: 0,
                timed_out: false,
                duration: Duration::from_millis(5),
            },
        );
        let executor = ShellExecutor::new(Duration::from_secs(10), Duration::from_secs(600));
        let result = executor
            .execute(json!({"command": "echo hello"}), &env)
            .await
            .unwrap();
        assert!(result.contains("exit_code: 0"));
        assert!(result.contains("hello"));
    }

    #[tokio::test]
    async fn shell_timeout_appends_error_marker() {
        let env = MockExecutionEnvironment::new("/tmp");
        env.add_command_response(
            "sleep 60",
            MockCommandResponse {
                stdout: "partial".to_owned(),
                stderr: "".to_owned(),
                exit_code: -1,
                timed_out: true,
                duration: Duration::from_secs(10),
            },
        );
        let executor = ShellExecutor::new(Duration::from_secs(10), Duration::from_secs(600));
        let result = executor
            .execute(json!({"command": "sleep 60"}), &env)
            .await
            .unwrap();
        assert!(result.contains("[ERROR: Command timed out"));
    }

    #[tokio::test]
    async fn shell_nonzero_exit_still_returns_ok() {
        let env = MockExecutionEnvironment::new("/tmp");
        env.add_command_response(
            "false",
            MockCommandResponse {
                stdout: "".to_owned(),
                stderr: "error".to_owned(),
                exit_code: 1,
                timed_out: false,
                duration: Duration::from_millis(1),
            },
        );
        let executor = ShellExecutor::new(Duration::from_secs(10), Duration::from_secs(600));
        let result = executor.execute(json!({"command": "false"}), &env).await;
        assert!(result.is_ok()); // non-zero exit is not a ToolError
        assert!(result.unwrap().contains("exit_code: 1"));
    }

    #[test]
    fn shell_registered_tool_has_correct_name() {
        let t = ShellExecutor::registered_tool(Duration::from_secs(10), Duration::from_secs(600));
        assert_eq!(t.definition.name, "shell");
    }

    // ── grep ──

    #[tokio::test]
    async fn grep_missing_pattern_returns_validation_error() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = GrepExecutor;
        let result = executor.execute(json!({}), &env).await;
        assert!(matches!(result, Err(ToolError::Validation(_))));
    }

    #[tokio::test]
    async fn grep_no_matches_returns_no_matches_message() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = GrepExecutor;
        let result = executor
            .execute(json!({"pattern": "NOMATCH"}), &env)
            .await
            .unwrap();
        assert_eq!(result, "No matches found.");
    }

    #[test]
    fn grep_registered_tool_has_correct_name() {
        let t = GrepExecutor::registered_tool();
        assert_eq!(t.definition.name, "grep");
    }

    // ── glob ──

    #[tokio::test]
    async fn glob_missing_pattern_returns_validation_error() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = GlobExecutor;
        let result = executor.execute(json!({}), &env).await;
        assert!(matches!(result, Err(ToolError::Validation(_))));
    }

    #[tokio::test]
    async fn glob_no_files_returns_no_files_message() {
        let env = MockExecutionEnvironment::new("/tmp");
        let executor = GlobExecutor;
        let result = executor
            .execute(json!({"pattern": "*.nonexistent"}), &env)
            .await
            .unwrap();
        assert_eq!(result, "No files found.");
    }

    #[test]
    fn glob_registered_tool_has_correct_name() {
        let t = GlobExecutor::registered_tool();
        assert_eq!(t.definition.name, "glob");
    }

    // ── strip_line_numbers ──

    #[test]
    fn strip_line_numbers_removes_prefix() {
        // Simulate what env.read_file returns
        let numbered = "     1\thello\n     2\tworld\n";
        let raw = strip_line_numbers(numbered);
        assert_eq!(raw, "hello\nworld\n");
    }

    #[test]
    fn strip_line_numbers_handles_empty() {
        assert_eq!(strip_line_numbers(""), "");
    }
}
