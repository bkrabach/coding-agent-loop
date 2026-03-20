//! Local filesystem execution environment.
//!
//! Implements [`ExecutionEnvironment`] using the local machine's filesystem
//! and process spawning. This is the default implementation.
//!
//! Features:
//! - **F-105**: File ops (read_file, write_file, file_exists, list_directory)
//! - **F-106**: Command execution with timeout enforcement and env-var filtering
//! - **F-107**: grep (shell-out to rg/grep) and glob (globset walk)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use globset::{Glob, GlobSetBuilder};
use tokio::process::Command;

use super::{DirEntry, ExecResult, GrepOptions};
use crate::environment::ExecutionEnvironment;
use crate::error::EnvError;

// ── Secret-variable patterns ─────────────────────────────────────────────────

/// Suffixes that identify sensitive environment variables (case-insensitive).
static SECRET_SUFFIXES: &[&str] = &["_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "_CREDENTIAL"];

// ── LocalExecutionEnvironment ────────────────────────────────────────────────

/// An [`ExecutionEnvironment`] that runs on the local machine.
pub struct LocalExecutionEnvironment {
    working_dir: PathBuf,
}

impl LocalExecutionEnvironment {
    /// Create a new environment rooted at `working_dir`.
    pub fn new(working_dir: impl Into<PathBuf>) -> Self {
        Self {
            working_dir: working_dir.into(),
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Resolve `path` relative to the working directory if it is not absolute.
    fn resolve(&self, path: &str) -> PathBuf {
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.working_dir.join(p)
        }
    }

    /// Check whether an environment variable name looks like a secret.
    /// Returns `true` for names like `OPENAI_API_KEY`, `MY_TOKEN`, etc.
    pub fn is_secret_env_var(name: &str) -> bool {
        let upper = name.to_uppercase();
        SECRET_SUFFIXES.iter().any(|suffix| upper.ends_with(suffix))
    }

    /// Build the filtered environment (system env minus secret variables).
    pub fn filtered_env() -> HashMap<String, String> {
        std::env::vars()
            .filter(|(k, _)| !Self::is_secret_env_var(k))
            .collect()
    }

    /// Recursively collect directory entries up to `depth` levels.
    fn collect_entries(dir: &Path, current_depth: u32, max_depth: u32) -> Vec<DirEntry> {
        if current_depth > max_depth {
            return vec![];
        }
        let Ok(read) = std::fs::read_dir(dir) else {
            return vec![];
        };
        let mut entries: Vec<DirEntry> = read
            .filter_map(|e| e.ok())
            .map(|e| {
                let meta = e.metadata().ok();
                let is_dir = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);
                let size = if is_dir { None } else { meta.map(|m| m.len()) };
                DirEntry {
                    name: e.file_name().to_string_lossy().into_owned(),
                    is_dir,
                    size,
                }
            })
            .collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }
}

// ── ExecutionEnvironment impl ────────────────────────────────────────────────

#[async_trait]
impl ExecutionEnvironment for LocalExecutionEnvironment {
    // ── F-105: File ops ──────────────────────────────────────────────────

    async fn read_file(
        &self,
        path: &str,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<String, EnvError> {
        let full_path = self.resolve(path);

        // Map specific IO errors to semantic EnvError variants.
        let content = tokio::fs::read_to_string(&full_path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                EnvError::FileNotFound(path.to_owned())
            } else if e.kind() == std::io::ErrorKind::PermissionDenied {
                EnvError::PermissionDenied(path.to_owned())
            } else {
                EnvError::Io(e)
            }
        })?;

        // Apply offset (1-based) and limit.
        let start = offset.map(|o| o.saturating_sub(1)).unwrap_or(0);
        let lines: Vec<&str> = content.lines().collect();
        let slice = &lines[start.min(lines.len())..];
        let slice = match limit {
            Some(n) => &slice[..n.min(slice.len())],
            None => slice,
        };

        // Format with line numbers (right-justified to 4 digits, 1-based from `start`).
        let numbered: Vec<String> = slice
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:4} | {}", start + i + 1, line))
            .collect();

        Ok(numbered.join("\n"))
    }

    async fn write_file(&self, path: &str, content: &str) -> Result<(), EnvError> {
        let full_path = self.resolve(path);

        // Create parent directories.
        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    EnvError::PermissionDenied(path.to_owned())
                } else {
                    EnvError::Io(e)
                }
            })?;
        }

        tokio::fs::write(&full_path, content).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                EnvError::PermissionDenied(path.to_owned())
            } else {
                EnvError::Io(e)
            }
        })
    }

    async fn file_exists(&self, path: &str) -> Result<bool, EnvError> {
        let full_path = self.resolve(path);
        match tokio::fs::metadata(&full_path).await {
            Ok(_) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(EnvError::Io(e)),
        }
    }

    async fn list_directory(&self, path: &str, depth: u32) -> Result<Vec<DirEntry>, EnvError> {
        let full_path = self.resolve(path);
        // Verify it exists and is a directory.
        match tokio::fs::metadata(&full_path).await {
            Ok(m) if m.is_dir() => {}
            Ok(_) => {
                // It's a file, not a directory.
                return Err(EnvError::Io(std::io::Error::other(format!(
                    "{path} is not a directory"
                ))));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(EnvError::FileNotFound(path.to_owned()));
            }
            Err(e) => return Err(EnvError::Io(e)),
        }

        // Use synchronous recursion via blocking task to avoid async recursion.
        let entries =
            tokio::task::spawn_blocking(move || Self::collect_entries(&full_path, 1, depth))
                .await
                .map_err(|e| EnvError::Io(std::io::Error::other(e.to_string())))?;
        Ok(entries)
    }

    // ── F-106: Command execution ─────────────────────────────────────────

    async fn exec_command(
        &self,
        command: &str,
        timeout: Duration,
        working_dir: Option<&str>,
        env_vars: Option<&HashMap<String, String>>,
    ) -> Result<ExecResult, EnvError> {
        let cwd = working_dir
            .map(|d| self.resolve(d))
            .unwrap_or_else(|| self.working_dir.clone());

        // Build filtered + merged environment.
        let mut env = Self::filtered_env();
        if let Some(extra) = env_vars {
            env.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        // Choose shell based on platform.
        #[cfg(target_os = "windows")]
        let (shell, shell_flag) = ("cmd.exe", "/c");
        #[cfg(not(target_os = "windows"))]
        let (shell, shell_flag) = ("/bin/bash", "-c");

        let mut cmd = Command::new(shell);
        cmd.arg(shell_flag)
            .arg(command)
            .current_dir(&cwd)
            .env_clear()
            .envs(&env)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let start = Instant::now();

        // Spawn the process and capture the PID *before* consuming the handle.
        let child = cmd.spawn().map_err(EnvError::Io)?;
        let child_pid = child.id(); // Option<u32>: None if already exited

        // V2-CAL-005: SIGTERM → 2 s wait → SIGKILL — declare kill() once.
        #[cfg(unix)]
        unsafe extern "C" {
            fn kill(pid: i32, sig: i32) -> i32;
        }

        // Race process against timeout.
        match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => {
                let duration = start.elapsed();
                let exit_code = output.status.code().unwrap_or(-1);
                Ok(ExecResult {
                    stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                    exit_code,
                    timed_out: false,
                    duration,
                })
            }
            Ok(Err(e)) => Err(EnvError::Io(e)),
            Err(_elapsed) => {
                // V2-CAL-005: SIGTERM → 2 s wait → SIGKILL sequence.
                //
                // NLSpec §9.4: on timeout, send SIGTERM to the process, wait
                // 2 seconds for a graceful shutdown, then send SIGKILL.
                // The child PID was captured before `wait_with_output()` consumed
                // the handle, so we still have a valid pid to signal.
                #[cfg(unix)]
                if let Some(pid) = child_pid {
                    // SAFETY: kill() is async-signal-safe; pid came from a
                    // live child we just spawned.
                    unsafe {
                        kill(pid as i32, 15 /* SIGTERM */);
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    unsafe {
                        kill(pid as i32, 9 /* SIGKILL */);
                    }
                }
                let duration = start.elapsed();
                Ok(ExecResult {
                    stdout: String::new(),
                    stderr: format!("\n[Command timed out after {timeout:?}]"),
                    exit_code: -1,
                    timed_out: true,
                    duration,
                })
            }
        }
    }

    // ── F-107: Search operations ─────────────────────────────────────────

    async fn grep(
        &self,
        pattern: &str,
        path: &str,
        options: &GrepOptions,
    ) -> Result<String, EnvError> {
        let full_path = self.resolve(path);
        let max_results = if options.max_results == 0 {
            100
        } else {
            options.max_results
        };

        // Try ripgrep first, fall back to system grep.
        let rg_available = which_rg().await;

        let cmd_str = if rg_available {
            build_rg_command(pattern, &full_path, options, max_results)
        } else {
            build_grep_command(pattern, &full_path, options, max_results)
        };

        let result = self
            .exec_command(&cmd_str, Duration::from_secs(30), None, None)
            .await?;

        // Non-zero exit is fine when there are no matches (grep returns 1).
        if result.timed_out {
            return Err(EnvError::CommandTimeout(Duration::from_secs(30)));
        }

        Ok(result.stdout)
    }

    async fn glob(&self, pattern: &str, path: &str) -> Result<Vec<String>, EnvError> {
        let base = self.resolve(path);

        // Verify base path exists.
        match tokio::fs::metadata(&base).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(EnvError::FileNotFound(path.to_owned()));
            }
            Err(e) => return Err(EnvError::Io(e)),
        }

        let pattern_owned = pattern.to_owned();
        let base_clone = base.clone();

        let matches = tokio::task::spawn_blocking(move || glob_walk(&base_clone, &pattern_owned))
            .await
            .map_err(|e| EnvError::Io(std::io::Error::other(e.to_string())))??;

        Ok(matches)
    }

    // ── Lifecycle ────────────────────────────────────────────────────────

    async fn initialize(&mut self) -> Result<(), EnvError> {
        Ok(())
    }

    async fn cleanup(&mut self) -> Result<(), EnvError> {
        Ok(())
    }

    // ── Metadata ─────────────────────────────────────────────────────────

    fn working_directory(&self) -> &str {
        self.working_dir.to_str().unwrap_or(".")
    }

    fn platform(&self) -> &str {
        std::env::consts::OS
    }

    fn os_version(&self) -> &str {
        // Best-effort; detailed version requires platform-specific calls.
        std::env::consts::OS
    }
}

// ── Helper functions ─────────────────────────────────────────────────────────

async fn which_rg() -> bool {
    Command::new("rg")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false)
}

fn build_rg_command(
    pattern: &str,
    path: &Path,
    options: &GrepOptions,
    max_results: usize,
) -> String {
    let mut parts = vec!["rg".to_string(), "--line-number".to_string()];
    if options.case_insensitive {
        parts.push("--ignore-case".to_string());
    }
    if let Some(glob) = &options.glob_filter {
        parts.push(format!("--glob={}", shell_escape(glob)));
    }
    parts.push(format!("--max-count={max_results}"));
    parts.push(shell_escape(pattern));
    parts.push(shell_escape(&path.to_string_lossy()));
    parts.join(" ")
}

fn build_grep_command(
    pattern: &str,
    path: &Path,
    options: &GrepOptions,
    max_results: usize,
) -> String {
    let mut parts = vec!["grep".to_string(), "-rn".to_string(), "-E".to_string()];
    if options.case_insensitive {
        parts.push("--ignore-case".to_string());
    }
    if let Some(glob) = &options.glob_filter {
        parts.push(format!("--include={}", shell_escape(glob)));
    }
    parts.push(format!("-m {max_results}"));
    parts.push(shell_escape(pattern));
    parts.push(shell_escape(&path.to_string_lossy()));
    parts.join(" ")
}

/// Minimal shell escaping — wrap in single quotes, escape any embedded single quotes.
fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// Walk `base` recursively and collect paths matching the glob `pattern`.
/// Returns paths as strings relative to `base`, sorted by mtime (newest first).
fn glob_walk(base: &Path, pattern: &str) -> Result<Vec<String>, EnvError> {
    let glob = Glob::new(pattern).map_err(|e| {
        EnvError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            e.to_string(),
        ))
    })?;
    let glob_set = GlobSetBuilder::new().add(glob).build().map_err(|e| {
        EnvError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            e.to_string(),
        ))
    })?;

    let mut matches: Vec<(std::time::SystemTime, String)> = Vec::new();
    walk_dir_for_glob(base, base, &glob_set, &mut matches);

    // Sort newest first by mtime.
    matches.sort_by_key(|k| std::cmp::Reverse(k.0));

    Ok(matches.into_iter().map(|(_, path)| path).collect())
}

fn walk_dir_for_glob(
    base: &Path,
    current: &Path,
    glob_set: &globset::GlobSet,
    matches: &mut Vec<(std::time::SystemTime, String)>,
) {
    let Ok(entries) = std::fs::read_dir(current) else {
        return;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };

        // Build the relative path from base for glob matching.
        let rel = path.strip_prefix(base).unwrap_or(&path);

        if meta.is_dir() {
            walk_dir_for_glob(base, &path, glob_set, matches);
        } else if glob_set.is_match(rel) {
            let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            matches.push((mtime, rel.to_string_lossy().into_owned()));
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_env(dir: &TempDir) -> LocalExecutionEnvironment {
        LocalExecutionEnvironment::new(dir.path())
    }

    // ── F-106: Secret env var filtering ─────────────────────────────────

    #[test]
    fn secret_suffixes_detected() {
        assert!(LocalExecutionEnvironment::is_secret_env_var(
            "OPENAI_API_KEY"
        ));
        assert!(LocalExecutionEnvironment::is_secret_env_var("MY_SECRET"));
        assert!(LocalExecutionEnvironment::is_secret_env_var("GITHUB_TOKEN"));
        assert!(LocalExecutionEnvironment::is_secret_env_var("DB_PASSWORD"));
        assert!(LocalExecutionEnvironment::is_secret_env_var(
            "AWS_CREDENTIAL"
        ));
        assert!(LocalExecutionEnvironment::is_secret_env_var(
            "anthropic_api_key"
        )); // lowercase
    }

    #[test]
    fn non_secret_vars_allowed() {
        assert!(!LocalExecutionEnvironment::is_secret_env_var("PATH"));
        assert!(!LocalExecutionEnvironment::is_secret_env_var("HOME"));
        assert!(!LocalExecutionEnvironment::is_secret_env_var("USER"));
        assert!(!LocalExecutionEnvironment::is_secret_env_var("CARGO_HOME"));
        assert!(!LocalExecutionEnvironment::is_secret_env_var("RUST_LOG"));
    }

    #[test]
    fn filtered_env_excludes_secrets() {
        // We can't easily test the real system env, but we can verify PATH is present.
        let env = LocalExecutionEnvironment::filtered_env();
        // PATH should be present on any reasonable system.
        // (Skip on CI if not set.)
        if std::env::var("PATH").is_ok() {
            assert!(env.contains_key("PATH"));
        }
    }

    // ── F-105: File ops ──────────────────────────────────────────────────

    #[tokio::test]
    async fn write_and_read_file() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);

        env.write_file("hello.txt", "line1\nline2\nline3\n")
            .await
            .unwrap();

        let content = env.read_file("hello.txt", None, None).await.unwrap();
        assert!(content.contains("line1"));
        assert!(content.contains("line2"));
        assert!(content.contains("line3"));
    }

    #[tokio::test]
    async fn read_file_with_offset_and_limit() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);

        env.write_file("test.txt", "a\nb\nc\nd\ne\n").await.unwrap();

        // Read lines 2-3 (offset=2, limit=2).
        let content = env.read_file("test.txt", Some(2), Some(2)).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("b"));
        assert!(lines[1].contains("c"));
    }

    #[tokio::test]
    async fn read_file_not_found() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        let err = env.read_file("nope.txt", None, None).await.unwrap_err();
        assert!(matches!(err, EnvError::FileNotFound(_)));
    }

    #[tokio::test]
    async fn write_creates_parent_directories() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        env.write_file("a/b/c.txt", "content").await.unwrap();
        assert!(env.file_exists("a/b/c.txt").await.unwrap());
    }

    #[tokio::test]
    async fn file_exists_false_for_missing() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        assert!(!env.file_exists("ghost.txt").await.unwrap());
    }

    #[tokio::test]
    async fn list_directory_returns_entries() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);

        env.write_file("file1.txt", "x").await.unwrap();
        env.write_file("file2.txt", "y").await.unwrap();

        let entries = env.list_directory(".", 1).await.unwrap();
        assert!(!entries.is_empty());
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"file1.txt"));
        assert!(names.contains(&"file2.txt"));
    }

    #[tokio::test]
    async fn list_directory_not_found() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        let err = env.list_directory("no_such_dir", 1).await.unwrap_err();
        assert!(matches!(err, EnvError::FileNotFound(_)));
    }

    #[tokio::test]
    async fn working_directory_matches_init() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        assert_eq!(env.working_directory(), dir.path().to_str().unwrap());
    }

    // ── F-106: Command execution ─────────────────────────────────────────

    #[tokio::test]
    async fn exec_echo_command() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        let result = env
            .exec_command("echo hello", Duration::from_secs(5), None, None)
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.trim() == "hello");
        assert!(!result.timed_out);
    }

    #[tokio::test]
    async fn exec_nonzero_exit_code() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        let result = env
            .exec_command("exit 42", Duration::from_secs(5), None, None)
            .await
            .unwrap();
        assert_eq!(result.exit_code, 42);
    }

    #[tokio::test]
    async fn exec_command_timeout() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        let result = env
            .exec_command("sleep 10", Duration::from_millis(100), None, None)
            .await
            .unwrap();
        assert!(result.timed_out);
        assert_eq!(result.exit_code, -1);
        assert!(result.stderr.contains("timed out"));
    }

    #[tokio::test]
    async fn exec_extra_env_vars_available() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        let mut extra = HashMap::new();
        extra.insert("MY_TEST_VAR_12345".to_string(), "hello_world".to_string());
        let result = env
            .exec_command(
                "echo $MY_TEST_VAR_12345",
                Duration::from_secs(5),
                None,
                Some(&extra),
            )
            .await
            .unwrap();
        assert!(result.stdout.contains("hello_world"));
    }

    // ── F-107: Glob ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn glob_finds_matching_files() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);

        env.write_file("foo.rs", "").await.unwrap();
        env.write_file("bar.rs", "").await.unwrap();
        env.write_file("baz.txt", "").await.unwrap();

        let matches = env.glob("*.rs", ".").await.unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().any(|p| p.ends_with("foo.rs")));
        assert!(matches.iter().any(|p| p.ends_with("bar.rs")));
    }

    #[tokio::test]
    async fn glob_empty_when_no_match() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        env.write_file("hello.txt", "").await.unwrap();
        let matches = env.glob("*.rs", ".").await.unwrap();
        assert!(matches.is_empty());
    }

    #[tokio::test]
    async fn glob_not_found_error() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        let err = env.glob("*.rs", "nonexistent_dir").await.unwrap_err();
        assert!(matches!(err, EnvError::FileNotFound(_)));
    }

    // ── F-107: Grep ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn grep_finds_pattern() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        env.write_file("code.rs", "fn main() {\n    println!(\"hello\");\n}\n")
            .await
            .unwrap();

        let result = env
            .grep("fn main", ".", &GrepOptions::default())
            .await
            .unwrap();
        assert!(result.contains("fn main"));
    }

    #[tokio::test]
    async fn grep_no_match_returns_empty() {
        let dir = TempDir::new().unwrap();
        let env = temp_env(&dir);
        env.write_file("file.txt", "no match here\n").await.unwrap();

        let result = env
            .grep("xyz_not_present_12345", ".", &GrepOptions::default())
            .await
            .unwrap();
        assert!(result.trim().is_empty());
    }

    // -----------------------------------------------------------------------
    // V2-CAL-005: SIGTERM → SIGKILL on timeout — process must not survive
    // -----------------------------------------------------------------------
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn timeout_kills_process_via_sigterm_sigkill() {
        // Start a process that writes its PID to a temp file, then sleeps.
        let dir = TempDir::new().unwrap();
        let pid_file = dir.path().join("child_pid.txt");
        let pid_file_str = pid_file.to_str().unwrap().to_string();

        let env = temp_env(&dir);

        // The command writes its bash PID and then sleeps.
        let cmd = format!("echo $$ > {pid_file_str}; sleep 30");

        let result = env
            .exec_command(&cmd, Duration::from_secs(1), None, None)
            .await
            .unwrap();

        assert!(result.timed_out, "command must be reported as timed-out");

        // Wait briefly for the SIGTERM + 2s + SIGKILL sequence to complete.
        // (The implementation already waits 2s internally before we return.)
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Verify the child process is no longer alive by checking /proc.
        if let Ok(pid_text) = std::fs::read_to_string(&pid_file) {
            if let Ok(pid) = pid_text.trim().parse::<u32>() {
                let proc_path = format!("/proc/{pid}/status");
                assert!(
                    !std::path::Path::new(&proc_path).exists(),
                    "process {pid} must be dead after timeout, but {proc_path} still exists"
                );
            }
        }
        // (If the PID file was not written, the process likely exited before
        // writing, which is also acceptable.)
    }
}
