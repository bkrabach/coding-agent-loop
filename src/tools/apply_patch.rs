//! apply_patch tool executor — v4a patch format parser and applier.
//!
//! Supports Add File, Delete File, Update File (with context-based hunks),
//! and Move to (rename). Context matching uses exact search with fuzzy
//! (whitespace-normalization) fallback.
//!
//! See NLSpec Appendix A for the full v4a grammar.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::environment::ExecutionEnvironment;
use crate::error::ToolError;
use crate::tools::core::strip_line_numbers;
use crate::tools::{RegisteredTool, ToolDefinition, ToolExecutor};

// ── Patch data types ──────────────────────────────────────────────────────────

/// A parsed v4a patch ready to apply.
#[derive(Debug)]
pub struct Patch {
    pub operations: Vec<PatchOperation>,
}

/// A single file operation within a patch.
#[derive(Debug)]
pub enum PatchOperation {
    AddFile {
        path: String,
        content: String,
    },
    DeleteFile {
        path: String,
    },
    UpdateFile {
        old_path: String,
        new_path: Option<String>,
        hunks: Vec<Hunk>,
    },
}

/// A single `@@ context_hint` block with its change lines.
#[derive(Debug)]
pub struct Hunk {
    pub context_hint: String,
    pub lines: Vec<HunkLine>,
}

/// One line within a hunk.
#[derive(Debug, Clone)]
pub enum HunkLine {
    /// Space-prefixed — unchanged context line.
    Context(String),
    /// Minus-prefixed — line to remove.
    Delete(String),
    /// Plus-prefixed — line to add.
    Add(String),
}

// ── v4a markers ───────────────────────────────────────────────────────────────

const BEGIN_PATCH: &str = "*** Begin Patch";
const END_PATCH: &str = "*** End Patch";
const ADD_FILE: &str = "*** Add File: ";
const DELETE_FILE: &str = "*** Delete File: ";
const UPDATE_FILE: &str = "*** Update File: ";
const MOVE_TO: &str = "*** Move to: ";
const END_OF_FILE: &str = "*** End of File";

// ── Parser ────────────────────────────────────────────────────────────────────

/// Parse a v4a patch string.
///
/// Returns `Err` with a description of the first parse failure.
pub fn parse_patch(patch_text: &str) -> Result<Patch, String> {
    let lines: Vec<&str> = patch_text.lines().collect();
    if lines.is_empty() {
        return Err("empty patch".to_owned());
    }

    // Check begin/end markers.
    if !lines[0].trim_end().eq(BEGIN_PATCH) {
        return Err(format!(
            "line 1: expected '{}', got '{}'",
            BEGIN_PATCH,
            lines[0].trim_end()
        ));
    }

    let last = lines.iter().rposition(|l| l.trim_end().eq(END_PATCH));
    let end_line = last.ok_or_else(|| format!("missing '{}' marker", END_PATCH))?;

    let body = &lines[1..end_line];
    let mut ops: Vec<PatchOperation> = Vec::new();
    let mut i = 0usize;

    while i < body.len() {
        let line = body[i];

        if let Some(path) = line.strip_prefix(ADD_FILE) {
            // Collect all following `+` lines.
            i += 1;
            let mut content_lines: Vec<String> = Vec::new();
            while i < body.len() && !is_operation_header(body[i]) {
                let l = body[i];
                if l == END_OF_FILE {
                    i += 1;
                    break;
                }
                if let Some(rest) = l.strip_prefix('+') {
                    content_lines.push(rest.to_owned());
                }
                // Lines without '+' prefix in an add block are ignored.
                i += 1;
            }
            ops.push(PatchOperation::AddFile {
                path: path.trim().to_owned(),
                content: content_lines.join("\n"),
            });
        } else if let Some(path) = line.strip_prefix(DELETE_FILE) {
            ops.push(PatchOperation::DeleteFile {
                path: path.trim().to_owned(),
            });
            i += 1;
        } else if let Some(path) = line.strip_prefix(UPDATE_FILE) {
            let old_path = path.trim().to_owned();
            i += 1;

            // Optional "*** Move to:" line.
            let new_path = if i < body.len() {
                if let Some(np) = body[i].strip_prefix(MOVE_TO) {
                    let p = np.trim().to_owned();
                    i += 1;
                    Some(p)
                } else {
                    None
                }
            } else {
                None
            };

            // Parse hunks.
            let mut hunks: Vec<Hunk> = Vec::new();
            while i < body.len() && !is_operation_header(body[i]) {
                let l = body[i];
                if l == END_OF_FILE {
                    i += 1;
                    break;
                }
                if let Some(hint) = l
                    .strip_prefix("@@ ")
                    .or_else(|| if l == "@@" { Some("") } else { None })
                {
                    // New hunk.
                    let hunk_hint = hint.trim().to_owned();
                    i += 1;
                    let mut hunk_lines: Vec<HunkLine> = Vec::new();
                    while i < body.len() {
                        let hl = body[i];
                        if hl == END_OF_FILE
                            || hl.starts_with("@@ ")
                            || hl == "@@"
                            || is_operation_header(hl)
                        {
                            break;
                        }
                        let parsed = hl
                            .strip_prefix(' ')
                            .map(|rest| HunkLine::Context(rest.to_owned()))
                            .or_else(|| {
                                hl.strip_prefix('-')
                                    .map(|rest| HunkLine::Delete(rest.to_owned()))
                            })
                            .or_else(|| {
                                hl.strip_prefix('+')
                                    .map(|rest| HunkLine::Add(rest.to_owned()))
                            });
                        if let Some(hl_parsed) = parsed {
                            hunk_lines.push(hl_parsed);
                        }
                        i += 1;
                    }
                    if !hunk_lines.is_empty() {
                        hunks.push(Hunk {
                            context_hint: hunk_hint,
                            lines: hunk_lines,
                        });
                    }
                } else {
                    i += 1; // skip unexpected lines between hunks
                }
            }
            ops.push(PatchOperation::UpdateFile {
                old_path,
                new_path,
                hunks,
            });
        } else {
            // Skip unrecognized lines (blank lines between ops, etc.).
            i += 1;
        }
    }

    Ok(Patch { operations: ops })
}

fn is_operation_header(line: &str) -> bool {
    line.starts_with(ADD_FILE) || line.starts_with(DELETE_FILE) || line.starts_with(UPDATE_FILE)
}

// ── Applier ───────────────────────────────────────────────────────────────────

/// Apply a parsed patch via the execution environment.
/// Returns a summary of operations performed.
pub async fn apply_patch_to_env(
    patch: &Patch,
    env: &dyn ExecutionEnvironment,
) -> Result<String, ToolError> {
    let mut summary_lines: Vec<String> = Vec::new();

    for op in &patch.operations {
        match op {
            PatchOperation::AddFile { path, content } => {
                env.write_file(path, content)
                    .await
                    .map_err(ToolError::Environment)?;
                summary_lines.push(format!("  Created: {}", path));
            }

            PatchOperation::DeleteFile { path } => {
                let exists = env
                    .file_exists(path)
                    .await
                    .map_err(ToolError::Environment)?;
                if !exists {
                    return Err(ToolError::FileNotFound(path.clone()));
                }
                let cmd = format!("rm -f '{}'", path.replace('\'', "'\\''"));
                env.exec_command(&cmd, Duration::from_secs(10), None, None)
                    .await
                    .map_err(ToolError::Environment)?;
                summary_lines.push(format!("  Deleted: {}", path));
            }

            PatchOperation::UpdateFile {
                old_path,
                new_path,
                hunks,
            } => {
                let numbered = env
                    .read_file(old_path, None, None)
                    .await
                    .map_err(|e| match e {
                        crate::error::EnvError::FileNotFound(_) => {
                            ToolError::FileNotFound(old_path.clone())
                        }
                        other => ToolError::Environment(other),
                    })?;
                let raw = strip_line_numbers(&numbered);

                let mut content = raw;
                for hunk in hunks {
                    content = apply_hunk(&content, hunk, old_path)?;
                }

                // Write to destination (new_path if renaming, else old_path).
                let dest = new_path.as_deref().unwrap_or(old_path.as_str());
                env.write_file(dest, &content)
                    .await
                    .map_err(ToolError::Environment)?;

                if let Some(np) = new_path {
                    // Delete the original file if renaming.
                    if np != old_path {
                        let cmd = format!("rm -f '{}'", old_path.replace('\'', "'\\''"));
                        env.exec_command(&cmd, Duration::from_secs(10), None, None)
                            .await
                            .map_err(ToolError::Environment)?;
                        summary_lines.push(format!("  Renamed: {} -> {}", old_path, np));
                    } else {
                        summary_lines.push(format!("  Updated: {}", old_path));
                    }
                } else {
                    summary_lines.push(format!("  Updated: {}", old_path));
                }
            }
        }
    }

    Ok(format!("Applied patch:\n{}", summary_lines.join("\n")))
}

/// Apply a single hunk to file content. Returns the modified content.
fn apply_hunk(content: &str, hunk: &Hunk, path: &str) -> Result<String, ToolError> {
    // Build the "search block": context lines + delete lines (in order).
    let search_lines: Vec<&str> = hunk
        .lines
        .iter()
        .filter_map(|l| match l {
            HunkLine::Context(s) | HunkLine::Delete(s) => Some(s.as_str()),
            HunkLine::Add(_) => None,
        })
        .collect();

    if search_lines.is_empty() {
        // No context/delete lines → only additions; append after context_hint.
        return apply_hunk_additions_only(content, hunk);
    }

    let search_block = search_lines.join("\n");

    // Build the replacement block: context lines + add lines (in order, deletes removed).
    let replacement_lines: Vec<&str> = hunk
        .lines
        .iter()
        .filter_map(|l| match l {
            HunkLine::Context(s) | HunkLine::Add(s) => Some(s.as_str()),
            HunkLine::Delete(_) => None,
        })
        .collect();
    let replacement_block = replacement_lines.join("\n");

    // Try exact match first.
    if let Some(pos) = content.find(&search_block) {
        let before = &content[..pos];
        let after = &content[pos + search_block.len()..];
        return Ok(format!("{}{}{}", before, replacement_block, after));
    }

    // Fuzzy fallback: normalize whitespace per line, then search.
    if let Some(result) = fuzzy_apply(content, &search_block, &replacement_block) {
        return Ok(result);
    }

    Err(ToolError::PatchParse(format!(
        "hunk context not found near '{}' in {}",
        hunk.context_hint, path
    )))
}

/// For hunks that have only Add lines (no Context/Delete), append the additions
/// at the position hinted by context_hint.
fn apply_hunk_additions_only(content: &str, hunk: &Hunk) -> Result<String, ToolError> {
    let add_lines: Vec<&str> = hunk
        .lines
        .iter()
        .filter_map(|l| {
            if let HunkLine::Add(s) = l {
                Some(s.as_str())
            } else {
                None
            }
        })
        .collect();
    let addition = add_lines.join("\n");

    // Insert before the context_hint if found, otherwise append.
    if !hunk.context_hint.is_empty() {
        if let Some(pos) = content.find(&hunk.context_hint) {
            let before = &content[..pos];
            let after = &content[pos..];
            return Ok(format!("{}\n{}{}", before, addition, after));
        }
    }
    // Append at end.
    if content.ends_with('\n') {
        Ok(format!("{}{}\n", content, addition))
    } else {
        Ok(format!("{}\n{}", content, addition))
    }
}

/// Fuzzy line-by-line match: normalize whitespace on each line.
fn fuzzy_apply(content: &str, search: &str, replacement: &str) -> Option<String> {
    let content_lines: Vec<&str> = content.lines().collect();
    let search_lines: Vec<&str> = search.lines().collect();

    if search_lines.is_empty() {
        return None;
    }

    let norm_search: Vec<String> = search_lines.iter().map(|l| normalize_ws(l)).collect();

    for start in 0..content_lines.len() {
        let end = start + search_lines.len();
        if end > content_lines.len() {
            break;
        }
        let candidate: Vec<String> = content_lines[start..end]
            .iter()
            .map(|l| normalize_ws(l))
            .collect();
        if candidate == norm_search {
            let mut result_lines: Vec<&str> = content_lines[..start].to_vec();
            result_lines.extend(replacement.lines());
            result_lines.extend(content_lines[end..].iter().copied());
            let mut result = result_lines.join("\n");
            if content.ends_with('\n') {
                result.push('\n');
            }
            return Some(result);
        }
    }
    None
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── ApplyPatchExecutor ────────────────────────────────────────────────────────

/// Executor for the `apply_patch` tool.
pub struct ApplyPatchExecutor;

impl ApplyPatchExecutor {
    pub fn registered_tool() -> RegisteredTool {
        RegisteredTool {
            definition: ToolDefinition {
                name: "apply_patch".to_owned(),
                description:
                    "Apply code changes using the v4a patch format. Supports creating, deleting, \
                     and modifying files in a single operation."
                        .to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "patch": {
                            "type": "string",
                            "description": "The patch content in v4a format, starting with '*** Begin Patch' and ending with '*** End Patch'."
                        }
                    },
                    "required": ["patch"]
                }),
            },
            executor: Box::new(ApplyPatchExecutor),
        }
    }
}

#[async_trait]
impl ToolExecutor for ApplyPatchExecutor {
    async fn execute(
        &self,
        args: Value,
        env: &dyn ExecutionEnvironment,
    ) -> Result<String, ToolError> {
        let patch_text = args["patch"]
            .as_str()
            .ok_or_else(|| ToolError::Validation("missing required arg: patch".to_owned()))?;

        let patch = parse_patch(patch_text).map_err(ToolError::PatchParse)?;
        apply_patch_to_env(&patch, env).await
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockExecutionEnvironment;

    fn make_patch(text: &str) -> Patch {
        parse_patch(text).expect("parse failed")
    }

    fn begin_end(body: &str) -> String {
        format!("{}\n{}\n{}", BEGIN_PATCH, body, END_PATCH)
    }

    // ── Parser tests ──

    #[test]
    fn parse_add_file() {
        let p = make_patch(&begin_end(
            "*** Add File: src/foo.rs\n+fn foo() {}\n+// end",
        ));
        assert_eq!(p.operations.len(), 1);
        if let PatchOperation::AddFile { path, content } = &p.operations[0] {
            assert_eq!(path, "src/foo.rs");
            assert!(content.contains("fn foo()"), "content: {content}");
        } else {
            panic!("expected AddFile");
        }
    }

    #[test]
    fn parse_delete_file() {
        let p = make_patch(&begin_end("*** Delete File: old.rs"));
        assert_eq!(p.operations.len(), 1);
        if let PatchOperation::DeleteFile { path } = &p.operations[0] {
            assert_eq!(path, "old.rs");
        } else {
            panic!("expected DeleteFile");
        }
    }

    #[test]
    fn parse_update_file_single_hunk() {
        let body = "*** Update File: main.rs\n@@ fn main()\n hello\n-return 0;\n+return 1;";
        let p = make_patch(&begin_end(body));
        assert_eq!(p.operations.len(), 1);
        if let PatchOperation::UpdateFile { hunks, .. } = &p.operations[0] {
            assert_eq!(hunks.len(), 1);
            let hunk = &hunks[0];
            assert_eq!(hunk.context_hint, "fn main()");
        } else {
            panic!("expected UpdateFile");
        }
    }

    #[test]
    fn parse_update_file_multiple_hunks() {
        let body = "*** Update File: lib.rs\n@@ fn a()\n a\n-b\n+c\n@@ fn d()\n d\n-e\n+f";
        let p = make_patch(&begin_end(body));
        if let PatchOperation::UpdateFile { hunks, .. } = &p.operations[0] {
            assert_eq!(hunks.len(), 2);
        } else {
            panic!("expected UpdateFile");
        }
    }

    #[test]
    fn parse_move_to() {
        let body = "*** Update File: old.rs\n*** Move to: new.rs\n@@ fn x()\n x\n-y\n+z";
        let p = make_patch(&begin_end(body));
        if let PatchOperation::UpdateFile {
            old_path, new_path, ..
        } = &p.operations[0]
        {
            assert_eq!(old_path, "old.rs");
            assert_eq!(new_path.as_deref(), Some("new.rs"));
        } else {
            panic!("expected UpdateFile");
        }
    }

    #[test]
    fn parse_missing_begin_returns_error() {
        let result = parse_patch("just some text");
        assert!(result.is_err());
    }

    #[test]
    fn parse_missing_end_returns_error() {
        let result = parse_patch(&format!("{}\n*** Add File: foo.rs\n+x", BEGIN_PATCH));
        assert!(result.is_err());
    }

    #[test]
    fn parse_multi_operation_patch() {
        let body = "*** Add File: a.rs\n+fn a() {}\n*** Delete File: b.rs";
        let p = make_patch(&begin_end(body));
        assert_eq!(p.operations.len(), 2);
    }

    // ── Applier tests ──

    #[tokio::test]
    async fn apply_add_file_creates_file() {
        let env = MockExecutionEnvironment::new("/work");
        let body = "*** Add File: /work/hello.rs\n+fn hello() {}";
        let patch = make_patch(&begin_end(body));
        let result = apply_patch_to_env(&patch, &env).await.unwrap();
        assert!(result.contains("Created"), "result: {result}");
        let files = env.files.lock().unwrap();
        assert!(files.contains_key("/work/hello.rs"), "file not created");
    }

    #[tokio::test]
    async fn apply_delete_file_removes_file() {
        let env = MockExecutionEnvironment::new("/work");
        env.add_file("/work/old.rs", "content");
        env.add_command_response("rm -f", crate::testing::MockCommandResponse::success(""));
        let body = "*** Delete File: /work/old.rs";
        let patch = make_patch(&begin_end(body));
        let result = apply_patch_to_env(&patch, &env).await.unwrap();
        assert!(result.contains("Deleted"), "result: {result}");
    }

    #[tokio::test]
    async fn apply_delete_file_not_found_returns_error() {
        let env = MockExecutionEnvironment::new("/work");
        let body = "*** Delete File: /work/missing.rs";
        let patch = make_patch(&begin_end(body));
        let result = apply_patch_to_env(&patch, &env).await;
        assert!(matches!(result, Err(ToolError::FileNotFound(_))));
    }

    #[tokio::test]
    async fn apply_update_file_single_hunk() {
        let env = MockExecutionEnvironment::new("/work");
        // The mock env returns line-numbered content.
        // We need to set up the file so read_file works.
        env.add_file("/work/main.rs", "fn main() {\n    return 0;\n}\n");
        let body = "*** Update File: /work/main.rs\n@@ fn main\n fn main() {\n-    return 0;\n+    return 1;\n }";
        let patch = make_patch(&begin_end(body));
        let result = apply_patch_to_env(&patch, &env).await.unwrap();
        assert!(result.contains("Updated"), "result: {result}");
    }

    #[tokio::test]
    async fn apply_update_file_hunk_not_found_returns_patch_parse_error() {
        let env = MockExecutionEnvironment::new("/work");
        env.add_file("/work/main.rs", "fn main() {}\n");
        let body = "*** Update File: /work/main.rs\n@@ nothing\n NOTHERE\n-ALSO_NOTHERE\n+NEW";
        let patch = make_patch(&begin_end(body));
        let result = apply_patch_to_env(&patch, &env).await;
        assert!(matches!(result, Err(ToolError::PatchParse(_))));
    }

    #[tokio::test]
    async fn apply_patch_executor_missing_patch_arg_returns_validation_error() {
        let env = MockExecutionEnvironment::new("/work");
        let executor = ApplyPatchExecutor;
        let result = executor.execute(json!({}), &env).await;
        assert!(matches!(result, Err(ToolError::Validation(_))));
    }

    #[test]
    fn apply_patch_registered_tool_name() {
        let t = ApplyPatchExecutor::registered_tool();
        assert_eq!(t.definition.name, "apply_patch");
    }

    // ── Fuzzy matching ──

    #[test]
    fn fuzzy_apply_matches_with_extra_spaces() {
        // Search has trailing spaces, content doesn't.
        let content = "fn foo() {\n    let x = 1;\n}\n";
        let search = "fn foo() {\n    let x = 1;\n}";
        let replacement = "fn foo() {\n    let x = 2;\n}";
        let result = fuzzy_apply(content, search, replacement);
        assert!(result.is_some(), "fuzzy should match");
        let r = result.unwrap();
        assert!(r.contains("let x = 2"), "replacement: {r}");
    }
}
