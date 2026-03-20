//! System prompt building utilities.
//!
//! Provides `build_environment_context()` (sync, metadata only),
//! `build_git_context()` (async, runs git commands), and
//! `discover_project_docs()` (async, filesystem walk).
//!
//! See NLSpec §6 for the full specification.

use std::path::Path;
use std::time::Duration;

use chrono::Utc;

use crate::environment::ExecutionEnvironment;
use crate::profile::ProjectDoc;

// ── Environment context block ─────────────────────────────────────────────────

/// Generate the `<environment>` XML context block for inclusion in system prompts.
///
/// Synchronous — uses env metadata methods plus optional pre-collected git context.
/// The git context (branch, status, recent commits) is appended after the platform/date
/// fields inside the XML block.
pub fn build_environment_context(
    env: &dyn ExecutionEnvironment,
    git: Option<&GitContext>,
) -> String {
    let date = Utc::now().format("%Y-%m-%d").to_string();
    let base = format!(
        "Working directory: {}\nPlatform: {}\nOS version: {}\nToday's date: {}",
        env.working_directory(),
        env.platform(),
        env.os_version(),
        date,
    );
    match git {
        None => format!("<environment>\n{}\n</environment>", base),
        Some(g) => {
            let branch = g.branch.as_deref().unwrap_or("(detached HEAD)");
            let commits = if g.recent_commits.is_empty() {
                "(none)".to_owned()
            } else {
                g.recent_commits.join("\n")
            };
            format!(
                "<environment>\n{}\nGit branch: {}\nGit status: {}\nRecent commits:\n{}\n</environment>",
                base, branch, g.status_summary, commits
            )
        }
    }
}

// ── Git context ───────────────────────────────────────────────────────────────

/// Information collected from git at session start.
#[derive(Debug, Clone)]
pub struct GitContext {
    pub is_git_repo: bool,
    pub branch: Option<String>,
    pub status_summary: String,
    pub recent_commits: Vec<String>,
}

/// Run git commands to build a brief snapshot.
///
/// Returns `None` when not in a git repository or if git is unavailable.
pub async fn build_git_context(env: &dyn ExecutionEnvironment) -> Option<GitContext> {
    let timeout = Duration::from_secs(5);

    // Is this a git repo?
    let check = env
        .exec_command("git rev-parse --is-inside-work-tree", timeout, None, None)
        .await
        .ok()?;
    if check.exit_code != 0 {
        return None;
    }

    // Current branch.
    let branch = env
        .exec_command("git branch --show-current", timeout, None, None)
        .await
        .ok()
        .filter(|r| r.exit_code == 0)
        .map(|r| r.stdout.trim().to_owned())
        .filter(|s| !s.is_empty());

    // Status summary.
    let status_raw = env
        .exec_command("git status --short", timeout, None, None)
        .await
        .ok()
        .filter(|r| r.exit_code == 0)
        .map(|r| r.stdout.clone())
        .unwrap_or_default();

    let status_summary = if status_raw.trim().is_empty() {
        "clean".to_owned()
    } else {
        let modified = status_raw
            .lines()
            .filter(|l| !l.trim_start().starts_with('?'))
            .count();
        let untracked = status_raw
            .lines()
            .filter(|l| l.trim_start().starts_with("??"))
            .count();
        format!("{} modified, {} untracked", modified, untracked)
    };

    // Recent commits.
    let log_raw = env
        .exec_command("git log --oneline -5", timeout, None, None)
        .await
        .ok()
        .filter(|r| r.exit_code == 0)
        .map(|r| r.stdout.clone())
        .unwrap_or_default();

    let recent_commits: Vec<String> = log_raw
        .lines()
        .map(|l| l.trim().to_owned())
        .filter(|l| !l.is_empty())
        .collect();

    Some(GitContext {
        is_git_repo: true,
        branch,
        status_summary,
        recent_commits,
    })
}

// ── System prompt assembly ────────────────────────────────────────────────────

/// Concatenate non-empty prompt parts with double newlines.
pub fn assemble_system_prompt(parts: &[&str]) -> String {
    parts
        .iter()
        .filter(|p| !p.trim().is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n\n")
}

// ── Project doc discovery ─────────────────────────────────────────────────────

/// Discover project instruction documents for the given provider profile.
///
/// Walks from the git root (or working_dir if not in a git repo) down to
/// `working_dir`, collecting instruction files in root-first order.
///
/// `provider_id` controls which files are loaded:
/// - `"openai"`:    `AGENTS.md`, `.codex/instructions.md`
/// - `"anthropic"`: `AGENTS.md`, `CLAUDE.md`
/// - `"gemini"`:    `AGENTS.md`, `GEMINI.md`
/// - other:         `AGENTS.md` only
///
/// Total content is capped at 32 KiB.
pub async fn discover_project_docs(
    working_dir: &str,
    provider_id: &str,
    env: &dyn ExecutionEnvironment,
) -> Vec<ProjectDoc> {
    const MAX_BYTES: usize = 32 * 1024;

    // Determine target filenames.
    let mut targets: Vec<&str> = vec!["AGENTS.md"];
    match provider_id {
        "openai" => targets.push(".codex/instructions.md"),
        "anthropic" => targets.push("CLAUDE.md"),
        "gemini" => targets.push("GEMINI.md"),
        _ => {}
    }

    // Detect git root.
    let git_root = env
        .exec_command(
            "git rev-parse --show-toplevel",
            Duration::from_secs(5),
            None,
            None,
        )
        .await
        .ok()
        .filter(|r| r.exit_code == 0)
        .map(|r| r.stdout.trim().to_owned())
        .filter(|s| !s.is_empty());

    let root = git_root.as_deref().unwrap_or(working_dir);

    // Build list of directories from root → working_dir.
    let dirs = path_ancestors_from_root(root, working_dir);

    let mut docs: Vec<ProjectDoc> = Vec::new();
    let mut total_bytes: usize = 0;

    'outer: for dir in &dirs {
        for &target in &targets {
            let file_path = if dir.is_empty() || dir == "." {
                target.to_owned()
            } else {
                format!("{}/{}", dir, target)
            };

            let exists = env.file_exists(&file_path).await.unwrap_or(false);
            if !exists {
                continue;
            }

            let numbered = match env.read_file(&file_path, None, None).await {
                Ok(s) => s,
                Err(_) => continue,
            };
            let content = strip_line_numbers_prompt(&numbered);

            let remaining = MAX_BYTES.saturating_sub(total_bytes);
            if remaining == 0 {
                break 'outer;
            }

            let (final_content, truncated) = if content.len() > remaining {
                let truncated_content = format!(
                    "{}\n\n[Project instructions truncated at 32KB]",
                    &content[..remaining.saturating_sub(60)]
                );
                (truncated_content, true)
            } else {
                (content.clone(), false)
            };

            let byte_count = final_content.len();

            // Compute relative path for display.
            let rel_path = relative_path(root, &file_path);
            docs.push(ProjectDoc {
                path: rel_path,
                content: final_content,
            });

            total_bytes += byte_count;

            if truncated {
                break 'outer;
            }
        }
    }

    docs
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build the list of directories from `root` to `leaf`, inclusive, ordered
/// root-first. Both paths should be absolute or both relative.
fn path_ancestors_from_root(root: &str, leaf: &str) -> Vec<String> {
    let root_path = Path::new(root);
    let leaf_path = Path::new(leaf);

    // If leaf is not under root (or they're equal), just return [root].
    if leaf_path == root_path {
        return vec![root.to_owned()];
    }

    // Try to get the relative path from root to leaf.
    let rel = match leaf_path.strip_prefix(root_path) {
        Ok(r) => r,
        Err(_) => {
            // Not under root — just search in working_dir.
            return vec![leaf.to_owned()];
        }
    };

    let mut dirs = vec![root.to_owned()];
    let mut current = root_path.to_path_buf();
    for component in rel.components() {
        current = current.join(component);
        dirs.push(current.display().to_string());
    }
    dirs
}

/// Compute a relative path of `file` with respect to `base`.
fn relative_path(base: &str, file: &str) -> String {
    let base_path = Path::new(base);
    let file_path = Path::new(file);
    match file_path.strip_prefix(base_path) {
        Ok(rel) => rel.display().to_string(),
        Err(_) => file.to_owned(),
    }
}

/// Strip line-number prefixes from `env.read_file` output.
///
/// Handles both formats: `"     N\tcontent"` (tab) and `"   N | content"` (` | `).
/// Preserves the original trailing-newline state.
fn strip_line_numbers_prompt(numbered: &str) -> String {
    let has_trailing_nl = numbered.ends_with('\n');
    let raw_lines: Vec<&str> = numbered.split('\n').collect();

    let n = if has_trailing_nl && raw_lines.last() == Some(&"") {
        raw_lines.len() - 1
    } else {
        raw_lines.len()
    };

    let stripped: Vec<&str> = raw_lines[..n]
        .iter()
        .map(|line| strip_one_line_prompt(line))
        .collect();

    let mut result = stripped.join("\n");
    if has_trailing_nl {
        result.push('\n');
    }
    result
}

fn strip_one_line_prompt(line: &str) -> &str {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockExecutionEnvironment;

    #[test]
    fn build_environment_context_has_required_fields() {
        let env = MockExecutionEnvironment::new("/work/proj");
        let ctx = build_environment_context(&env, None);
        assert!(ctx.contains("<environment>"), "open tag: {ctx}");
        assert!(ctx.contains("</environment>"), "close tag: {ctx}");
        assert!(ctx.contains("Working directory: /work/proj"), "wd: {ctx}");
        assert!(ctx.contains("Platform: linux"), "platform: {ctx}");
        // Date is runtime-dependent; just check the format exists.
        assert!(ctx.contains("Today's date:"), "date: {ctx}");
    }

    #[test]
    fn assemble_system_prompt_filters_empty() {
        let result = assemble_system_prompt(&["a", "", "b", "  ", "c"]);
        assert_eq!(result, "a\n\nb\n\nc");
    }

    #[test]
    fn assemble_system_prompt_single_part() {
        assert_eq!(assemble_system_prompt(&["hello"]), "hello");
    }

    #[test]
    fn assemble_system_prompt_all_empty() {
        assert_eq!(assemble_system_prompt(&["", "  ", ""]), "");
    }

    #[tokio::test]
    async fn build_git_context_returns_none_when_not_git_repo() {
        use crate::testing::MockCommandResponse;
        let env = MockExecutionEnvironment::new("/tmp");
        env.add_command_response(
            "git rev-parse --is-inside-work-tree",
            MockCommandResponse::failure(128, "not a git repo"),
        );
        let ctx = build_git_context(&env).await;
        assert!(ctx.is_none());
    }

    #[tokio::test]
    async fn build_git_context_returns_branch_name() {
        use crate::testing::MockCommandResponse;
        let env = MockExecutionEnvironment::new("/work");
        env.add_command_response(
            "git rev-parse --is-inside-work-tree",
            MockCommandResponse::success("true\n"),
        );
        env.add_command_response(
            "git branch --show-current",
            MockCommandResponse::success("main\n"),
        );
        env.add_command_response("git status --short", MockCommandResponse::success(""));
        env.add_command_response(
            "git log --oneline -5",
            MockCommandResponse::success("abc123 first commit\n"),
        );
        let ctx = build_git_context(&env).await.unwrap();
        assert_eq!(ctx.branch.as_deref(), Some("main"));
        assert!(ctx.is_git_repo);
    }

    #[tokio::test]
    async fn discover_project_docs_anthropic_loads_agents_and_claude() {
        let env = MockExecutionEnvironment::new("/repo");
        env.add_command_response(
            "git rev-parse --show-toplevel",
            crate::testing::MockCommandResponse::success("/repo\n"),
        );
        env.add_file("/repo/AGENTS.md", "# Agents\nsome instructions");
        env.add_file("/repo/CLAUDE.md", "# Claude\nanthropic instructions");

        let docs = discover_project_docs("/repo", "anthropic", &env).await;
        assert_eq!(docs.len(), 2, "expected 2 docs: {:?}", docs);
        let paths: Vec<&str> = docs.iter().map(|d| d.path.as_str()).collect();
        assert!(
            paths.iter().any(|p| p.ends_with("AGENTS.md")),
            "AGENTS.md missing: {:?}",
            paths
        );
        assert!(
            paths.iter().any(|p| p.ends_with("CLAUDE.md")),
            "CLAUDE.md missing: {:?}",
            paths
        );
    }

    // ── GAP-CAL-013: Anthropic profile does NOT load GEMINI.md ───────────────

    #[tokio::test]
    async fn anthropic_profile_does_not_load_gemini_md() {
        let env = MockExecutionEnvironment::new("/repo");
        env.add_command_response(
            "git rev-parse --show-toplevel",
            crate::testing::MockCommandResponse::success("/repo\n"),
        );
        // Both CLAUDE.md and GEMINI.md exist in the repo.
        env.add_file("/repo/AGENTS.md", "# Agents");
        env.add_file("/repo/CLAUDE.md", "# Claude instructions");
        env.add_file(
            "/repo/GEMINI.md",
            "# Gemini instructions — should NOT be loaded",
        );

        let docs = discover_project_docs("/repo", "anthropic", &env).await;

        let paths: Vec<&str> = docs.iter().map(|d| d.path.as_str()).collect();

        // CLAUDE.md must be present.
        assert!(
            paths.iter().any(|p| p.ends_with("CLAUDE.md")),
            "Anthropic profile must load CLAUDE.md: {:?}",
            paths
        );
        // GEMINI.md must NOT be present.
        assert!(
            !paths.iter().any(|p| p.ends_with("GEMINI.md")),
            "Anthropic profile must NOT load GEMINI.md: {:?}",
            paths
        );
    }

    #[tokio::test]
    async fn discover_project_docs_missing_files_skipped() {
        let env = MockExecutionEnvironment::new("/repo");
        env.add_command_response(
            "git rev-parse --show-toplevel",
            crate::testing::MockCommandResponse::success("/repo\n"),
        );
        // Only AGENTS.md exists; no CLAUDE.md
        env.add_file("/repo/AGENTS.md", "# Agents");
        let docs = discover_project_docs("/repo", "anthropic", &env).await;
        assert_eq!(docs.len(), 1);
        assert!(docs[0].path.ends_with("AGENTS.md"));
    }

    #[tokio::test]
    async fn discover_project_docs_empty_when_no_files() {
        let env = MockExecutionEnvironment::new("/repo");
        env.add_command_response(
            "git rev-parse --show-toplevel",
            crate::testing::MockCommandResponse::success("/repo\n"),
        );
        let docs = discover_project_docs("/repo", "openai", &env).await;
        assert!(docs.is_empty());
    }

    #[tokio::test]
    async fn discover_project_docs_32kb_budget_enforced() {
        let env = MockExecutionEnvironment::new("/repo");
        env.add_command_response(
            "git rev-parse --show-toplevel",
            crate::testing::MockCommandResponse::success("/repo\n"),
        );
        // Large content: 40KB
        let large_content = "x".repeat(40 * 1024);
        env.add_file("/repo/AGENTS.md", &large_content);
        let docs = discover_project_docs("/repo", "openai", &env).await;
        assert!(!docs.is_empty());
        let total: usize = docs.iter().map(|d| d.content.len()).sum();
        assert!(
            total <= 32 * 1024 + 100, // small buffer for the truncation marker
            "total={total}"
        );
        assert!(
            docs[0]
                .content
                .contains("[Project instructions truncated at 32KB]"),
            "no truncation marker"
        );
    }
}
