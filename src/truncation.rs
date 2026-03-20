//! Tool output truncation engine.
//!
//! Character-based truncation (head/tail split) ALWAYS runs first.
//! Line-based truncation runs second as a readability pass.
//!
//! See NLSpec §5 for the full specification.

use crate::config::SessionConfig;

// ── Truncation mode ───────────────────────────────────────────────────────────

/// How to truncate when character limit is exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationMode {
    /// Keep first half + last half; insert warning marker in the middle.
    HeadTail,
    /// Keep only the tail; prepend warning marker.
    Tail,
}

// ── Default limits ────────────────────────────────────────────────────────────

fn default_char_limit(tool_name: &str) -> usize {
    match tool_name {
        "read_file" => 50_000,
        "shell" => 30_000,
        "grep" => 20_000,
        "glob" => 20_000,
        "edit_file" => 10_000,
        "apply_patch" => 10_000,
        "write_file" => 1_000,
        "spawn_agent" => 20_000,
        _ => 50_000,
    }
}

fn default_truncation_mode(tool_name: &str) -> TruncationMode {
    match tool_name {
        "read_file" => TruncationMode::HeadTail,
        "shell" => TruncationMode::HeadTail,
        "grep" => TruncationMode::Tail,
        "glob" => TruncationMode::Tail,
        "edit_file" => TruncationMode::Tail,
        "apply_patch" => TruncationMode::Tail,
        "write_file" => TruncationMode::Tail,
        "spawn_agent" => TruncationMode::HeadTail,
        _ => TruncationMode::HeadTail,
    }
}

/// Returns `Some(max_lines)` for tools that have a line limit, `None` otherwise.
fn default_line_limit(tool_name: &str) -> Option<usize> {
    match tool_name {
        "shell" => Some(256),
        "grep" => Some(200),
        "glob" => Some(500),
        _ => None,
    }
}

// ── Core truncation functions ─────────────────────────────────────────────────

/// Apply character-based truncation to `output`.
///
/// If `output` has at most `max_chars` Unicode scalar values, returns it unchanged.
/// `max_chars = 0` means no limit → returns `output` unchanged.
pub fn truncate_output(output: &str, max_chars: usize, mode: TruncationMode) -> String {
    if max_chars == 0 {
        return output.to_owned();
    }

    let char_count = output.chars().count();
    if char_count <= max_chars {
        return output.to_owned();
    }

    let removed = char_count - max_chars;

    match mode {
        TruncationMode::HeadTail => {
            let half = max_chars / 2;
            let tail_chars = max_chars - half; // avoids off-by-one when max_chars is odd

            // Find byte offset for head end
            let head_end = char_index_to_byte(output, half);
            // Find byte offset for tail start (counting from end)
            let tail_start = char_index_from_end_to_byte(output, tail_chars);

            let head = &output[..head_end];
            let tail = &output[tail_start..];

            let marker = format!(
                "\n\n[WARNING: Tool output was truncated. {} characters were removed from the \
                 middle. The full output is available in the event stream. If you need to see \
                 specific parts, re-run the tool with more targeted parameters.]\n\n",
                removed
            );

            format!("{}{}{}", head, marker, tail)
        }
        TruncationMode::Tail => {
            let tail_start = char_index_from_end_to_byte(output, max_chars);
            let tail = &output[tail_start..];

            let marker = format!(
                "[WARNING: Tool output was truncated. First {} characters were removed. \
                 The full output is available in the event stream.]\n\n",
                removed
            );

            format!("{}{}", marker, tail)
        }
    }
}

/// Apply line-based truncation to `output`.
///
/// If `output` has at most `max_lines` lines, returns it unchanged.
/// `max_lines = 0` means no limit → returns `output` unchanged.
pub fn truncate_lines(output: &str, max_lines: usize) -> String {
    if max_lines == 0 {
        return output.to_owned();
    }

    let lines: Vec<&str> = output.split('\n').collect();
    if lines.len() <= max_lines {
        return output.to_owned();
    }

    let head_count = max_lines / 2;
    let tail_count = max_lines - head_count;
    let omitted = lines.len() - head_count - tail_count;

    let head = lines[..head_count].join("\n");
    let tail = lines[lines.len() - tail_count..].join("\n");
    let marker = format!("\n[... {} lines omitted ...]\n", omitted);

    format!("{}{}{}", head, marker, tail)
}

/// Full truncation pipeline for a named tool's output.
///
/// Step 1: character-based truncation (always runs first; handles all size
///         concerns including pathological cases like 2-line 10MB CSV files).
/// Step 2: line-based truncation (readability pass; only runs when char
///         truncation did NOT fire, so it never removes the char-truncation
///         warning marker).
///
/// Limits come from `config` fields, falling back to built-in defaults.
pub fn truncate_tool_output(output: &str, tool_name: &str, config: &SessionConfig) -> String {
    // Step 1: character truncation
    let max_chars = config
        .tool_output_limits
        .get(tool_name)
        .copied()
        .unwrap_or_else(|| default_char_limit(tool_name));
    let mode = default_truncation_mode(tool_name);

    let char_count = if max_chars > 0 {
        output.chars().count()
    } else {
        0
    };
    let char_truncation_fired = max_chars > 0 && char_count > max_chars;

    let result = truncate_output(output, max_chars, mode);

    // Step 2: line truncation — only when char truncation did NOT fire.
    // If char truncation already inserted a warning marker, line truncation
    // would remove it (the marker is in the middle of the head/tail output).
    if char_truncation_fired {
        return result;
    }

    let max_lines = config
        .tool_line_limits
        .get(tool_name)
        .copied()
        .or_else(|| default_line_limit(tool_name))
        .unwrap_or(0);
    truncate_lines(&result, max_lines)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return the byte offset in `s` corresponding to the end of the `n`-th Unicode
/// scalar value (0-indexed). Clamps to `s.len()` if `n >= char_count`.
fn char_index_to_byte(s: &str, n: usize) -> usize {
    s.char_indices()
        .nth(n)
        .map(|(byte_idx, _)| byte_idx)
        .unwrap_or(s.len())
}

/// Return the byte offset where the LAST `n` Unicode scalar values begin.
/// I.e., the tail is `s[result..]`.
fn char_index_from_end_to_byte(s: &str, n: usize) -> usize {
    let total = s.chars().count();
    if n >= total {
        return 0;
    }
    char_index_to_byte(s, total - n)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_truncation_when_under_limit() {
        let s = "hello world";
        assert_eq!(truncate_output(s, 100, TruncationMode::HeadTail), s);
        assert_eq!(truncate_output(s, 100, TruncationMode::Tail), s);
    }

    #[test]
    fn no_truncation_when_exactly_at_limit() {
        let s = "abc";
        assert_eq!(truncate_output(s, 3, TruncationMode::HeadTail), s);
    }

    #[test]
    fn head_tail_inserts_warning_in_middle() {
        // 10 chars, max 4 → head=2, tail=2
        let s = "0123456789";
        let result = truncate_output(s, 4, TruncationMode::HeadTail);
        assert!(result.starts_with("01"), "head: {result}");
        assert!(result.ends_with("89"), "tail: {result}");
        assert!(result.contains("[WARNING:"), "marker: {result}");
        assert!(
            result.contains("6 characters were removed"),
            "count: {result}"
        );
    }

    #[test]
    fn tail_mode_prepends_warning() {
        let s = "0123456789";
        let result = truncate_output(s, 4, TruncationMode::Tail);
        assert!(result.starts_with("[WARNING:"), "marker: {result}");
        assert!(result.ends_with("6789"), "tail: {result}");
        assert!(
            result.contains("6 characters were removed"),
            "count: {result}"
        );
    }

    #[test]
    fn zero_max_chars_no_truncation() {
        let s = "a".repeat(100_000);
        assert_eq!(truncate_output(&s, 0, TruncationMode::HeadTail), s);
    }

    #[test]
    fn truncate_lines_under_limit() {
        let s = "a\nb\nc";
        assert_eq!(truncate_lines(s, 5), s);
    }

    #[test]
    fn truncate_lines_exactly_at_limit() {
        let s = "a\nb\nc";
        assert_eq!(truncate_lines(s, 3), s);
    }

    #[test]
    fn truncate_lines_over_limit() {
        let lines: Vec<String> = (0..20).map(|i| format!("line{}", i)).collect();
        let s = lines.join("\n");
        let result = truncate_lines(&s, 10);
        assert!(
            result.contains("[... 10 lines omitted ...]"),
            "marker: {result}"
        );
        assert!(result.contains("line0"), "head: {result}");
        assert!(result.contains("line19"), "tail: {result}");
    }

    #[test]
    fn zero_max_lines_no_truncation() {
        let s = "a\n".repeat(1000);
        assert_eq!(truncate_lines(&s, 0), s);
    }

    #[test]
    fn truncate_tool_output_applies_char_then_line() {
        use std::collections::HashMap;
        // 600 lines, each 100 chars = 60_000 chars total
        let line = "x".repeat(99);
        let output = (0..600)
            .map(|_| line.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let config = SessionConfig {
            tool_output_limits: HashMap::new(),
            tool_line_limits: HashMap::new(),
            ..Default::default()
        };

        let result = truncate_tool_output(&output, "shell", &config);
        // shell: 30k char limit → char truncation fires
        // After char truncation the result is shorter, then line truncation (256) may fire
        assert!(
            result.contains("[WARNING:"),
            "char truncation: {}",
            &result[..200]
        );
    }

    #[test]
    fn truncate_tool_output_unknown_tool_uses_fallback() {
        let output = "x".repeat(60_000);
        let config = SessionConfig::default();
        let result = truncate_tool_output(&output, "unknown_tool", &config);
        // fallback: 50k chars, HeadTail
        assert!(result.contains("[WARNING:"));
        assert!(result.contains("10000 characters were removed"));
    }

    #[test]
    fn per_config_char_limit_overrides_default() {
        let mut config = SessionConfig::default();
        config.tool_output_limits.insert("shell".to_owned(), 100);
        let output = "x".repeat(200);
        let result = truncate_tool_output(&output, "shell", &config);
        assert!(result.contains("[WARNING:"));
        assert!(result.contains("100 characters were removed"));
    }

    #[test]
    fn unicode_chars_counted_not_bytes() {
        // "é" is 2 bytes but 1 char
        let s = "é".repeat(10); // 10 chars = 20 bytes
        let result = truncate_output(&s, 6, TruncationMode::HeadTail);
        // Should truncate 4 chars, not 4 bytes
        assert!(
            result.contains("4 characters were removed"),
            "result: {result}"
        );
    }

    // ── GAP-CAL-009: Default char/line limits match NLSpec §5.2 table ─────────

    #[test]
    fn default_char_limit_read_file_is_50k() {
        // read_file: 50 000 chars (HeadTail mode)
        let limit = default_char_limit("read_file");
        assert_eq!(limit, 50_000);
        let mode = default_truncation_mode("read_file");
        assert_eq!(mode, TruncationMode::HeadTail);
    }

    #[test]
    fn default_char_limit_shell_is_30k() {
        // shell: 30 000 chars (HeadTail mode)
        let limit = default_char_limit("shell");
        assert_eq!(limit, 30_000);
        let mode = default_truncation_mode("shell");
        assert_eq!(mode, TruncationMode::HeadTail);
    }

    #[test]
    fn default_char_limit_grep_is_20k() {
        // grep: 20 000 chars (Tail mode)
        let limit = default_char_limit("grep");
        assert_eq!(limit, 20_000);
        let mode = default_truncation_mode("grep");
        assert_eq!(mode, TruncationMode::Tail);
    }

    #[test]
    fn default_line_limit_glob_is_500() {
        // glob: 500 lines (NLSpec §5.2)
        let line_limit = default_line_limit("glob");
        assert_eq!(line_limit, Some(500));
    }

    #[test]
    fn default_line_limit_shell_is_256() {
        let line_limit = default_line_limit("shell");
        assert_eq!(line_limit, Some(256));
    }

    #[test]
    fn default_line_limit_grep_is_200() {
        let line_limit = default_line_limit("grep");
        assert_eq!(line_limit, Some(200));
    }

    #[test]
    fn default_line_limit_read_file_is_none() {
        // read_file has no line limit (only char limit)
        let line_limit = default_line_limit("read_file");
        assert_eq!(line_limit, None);
    }

    #[test]
    fn pathological_two_long_lines_char_truncated_not_line_truncated() {
        // 2 lines, each 30_001 chars = 60_002 total
        // shell limit: 30_000 chars, 256 lines
        // char truncation must fire (not bypassed by 2-line count)
        let line = "x".repeat(30_001);
        let output = format!("{}\n{}", line, line);
        let config = SessionConfig::default();
        let result = truncate_tool_output(&output, "shell", &config);
        assert!(
            result.contains("[WARNING:"),
            "char truncation did not fire: {}",
            &result[..100]
        );
    }
}
