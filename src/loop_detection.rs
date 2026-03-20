//! Loop detection for the agentic loop.
//!
//! Tracks tool call signatures (name + SHA-256 hash of arguments) and detects
//! repeating patterns of length 1, 2, or 3 in the most recent N calls.
//!
//! See NLSpec §2.10 for the full specification.

use sha2::{Digest, Sha256};

use crate::turns::{AssistantToolCall, History};

// ── Public API ────────────────────────────────────────────────────────────────

/// Detect whether the most recent tool calls form a repeating pattern.
///
/// Returns `true` if the last `window_size` tool-call signatures contain a
/// repeating subsequence of length 1, 2, or 3.
/// Returns `false` if fewer than `window_size` calls are available, or if
/// `window_size` is 0.
pub fn detect_loop(history: &History, window_size: usize) -> bool {
    if window_size == 0 {
        return false;
    }

    let sigs = recent_signatures(history, window_size);
    if sigs.len() < window_size {
        return false;
    }

    // Check for repeating patterns of length 1, 2, or 3.
    for pattern_len in 1usize..=3 {
        if window_size % pattern_len != 0 {
            continue;
        }
        let pattern = &sigs[..pattern_len];
        let mut all_match = true;
        let mut i = pattern_len;
        while i < window_size {
            if sigs[i..i + pattern_len] != *pattern {
                all_match = false;
                break;
            }
            i += pattern_len;
        }
        if all_match {
            return true;
        }
    }

    false
}

/// Compute a stable, deterministic signature string for a tool call.
///
/// Format: `"{tool_name}:{first_8_hex_chars_of_sha256(args_json)}"`
pub fn tool_call_signature(call: &AssistantToolCall) -> String {
    let args_json = serde_json::to_string(&call.arguments).unwrap_or_default();
    let hash = Sha256::digest(args_json.as_bytes());
    // Take first 8 hex chars (32 bits) — sufficient for loop detection.
    let hex = format!("{:x}", hash);
    let short = &hex[..8.min(hex.len())];
    format!("{}:{}", call.name, short)
}

/// Return signatures for the most recent `count` tool calls (newest first).
pub fn recent_signatures(history: &History, count: usize) -> Vec<String> {
    history
        .recent_tool_calls(count)
        .into_iter()
        .map(tool_call_signature)
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turns::{AssistantTurn, Turn};
    use serde_json::json;

    fn make_call(name: &str, arg: i32) -> AssistantToolCall {
        AssistantToolCall {
            id: format!("{name}-{arg}"),
            name: name.to_owned(),
            arguments: json!({ "x": arg }),
        }
    }

    fn history_from_calls(calls: Vec<AssistantToolCall>) -> History {
        let mut h = History::new();
        for call in calls {
            h.push(Turn::Assistant(AssistantTurn::new("", vec![call])));
        }
        h
    }

    #[test]
    fn no_loop_when_too_few_calls() {
        let h = history_from_calls(vec![make_call("shell", 1), make_call("grep", 2)]);
        assert!(!detect_loop(&h, 10));
    }

    #[test]
    fn pattern_length_1_detected() {
        // Same call 10 times
        let calls: Vec<_> = (0..10).map(|_| make_call("shell", 42)).collect();
        let h = history_from_calls(calls);
        assert!(detect_loop(&h, 10));
    }

    #[test]
    fn pattern_length_2_detected() {
        // A-B alternating 5 times = 10 calls
        let mut calls = Vec::new();
        for _ in 0..5 {
            calls.push(make_call("read_file", 1));
            calls.push(make_call("edit_file", 2));
        }
        let h = history_from_calls(calls);
        assert!(detect_loop(&h, 10));
    }

    #[test]
    fn pattern_length_3_detected() {
        // A-B-C cycling 3 times = 9 calls
        let mut calls = Vec::new();
        for _ in 0..3 {
            calls.push(make_call("read_file", 1));
            calls.push(make_call("shell", 2));
            calls.push(make_call("edit_file", 3));
        }
        let h = history_from_calls(calls);
        assert!(detect_loop(&h, 9));
    }

    #[test]
    fn no_loop_for_distinct_calls() {
        let calls: Vec<_> = (0..10).map(|i| make_call("shell", i)).collect();
        let h = history_from_calls(calls);
        assert!(!detect_loop(&h, 10));
    }

    #[test]
    fn window_size_not_divisible_skips_pattern() {
        // 7 calls, A-B alternating — window=7 is not divisible by 2
        // so length-2 patterns skipped; length-1 and length-3 patterns checked
        let mut calls = Vec::new();
        for _ in 0..4 {
            calls.push(make_call("read_file", 1));
            calls.push(make_call("edit_file", 2));
        }
        // push one extra to make it odd
        calls.push(make_call("read_file", 1));
        let h = history_from_calls(calls);
        // window=7: 7%2 != 0 so len-2 skipped; 7%3 != 0 so len-3 skipped; 7%1==0 but not all same
        assert!(!detect_loop(&h, 7));
    }

    #[test]
    fn window_size_zero_returns_false() {
        let h = history_from_calls(vec![make_call("shell", 1)]);
        assert!(!detect_loop(&h, 0));
    }

    #[test]
    fn same_args_same_signature() {
        let c1 = make_call("shell", 99);
        let c2 = make_call("shell", 99);
        assert_eq!(tool_call_signature(&c1), tool_call_signature(&c2));
    }

    #[test]
    fn different_args_different_signature() {
        let c1 = make_call("shell", 1);
        let c2 = make_call("shell", 2);
        assert_ne!(tool_call_signature(&c1), tool_call_signature(&c2));
    }

    #[test]
    fn signature_format_is_name_colon_hash() {
        let c = make_call("grep", 5);
        let sig = tool_call_signature(&c);
        assert!(sig.starts_with("grep:"), "format: {sig}");
        assert_eq!(sig.len(), "grep:".len() + 8, "length: {sig}");
    }
}
