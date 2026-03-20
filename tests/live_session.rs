//! Live integration tests for the coding-agent-loop Session.
//!
//! These tests make real network calls to OpenAI and incur API costs.
//! They are gated behind the `LIVE_TEST=1` environment variable so they are
//! **never** run in CI unless explicitly enabled.
//!
//! # Usage
//! ```bash
//! LIVE_TEST=1 cargo test -p coding-agent-loop --test live_session -- --nocapture
//! ```

use coding_agent_loop::profile::openai::OpenAiProfile;
use coding_agent_loop::{LocalExecutionEnvironment, Session, SessionConfig};
use tempfile::TempDir;
use unified_llm::Client;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Returns `true` when live tests should run.
fn live_test_enabled() -> bool {
    std::env::var("LIVE_TEST")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Build a Session with OpenAI gpt-4o-mini + LocalExecutionEnvironment rooted
/// at `dir`. Sets `max_tool_rounds_per_input = 10`.
fn make_session(dir: &TempDir) -> Session {
    let client =
        Client::from_env().expect("Client::from_env should succeed with OPENAI_API_KEY set");
    let config = SessionConfig {
        max_tool_rounds_per_input: 10,
        ..Default::default()
    };
    let profile = Box::new(OpenAiProfile::new("gpt-4o-mini"));
    let env = Box::new(LocalExecutionEnvironment::new(dir.path()));
    Session::new(config, profile, env, client)
}

// ─── Test 1: Simple file creation ────────────────────────────────────────────
//
// Submits a natural-language request to create hello.py and verifies the file
// was written with the expected content.

#[tokio::test]
async fn test_live_simple_file_creation() {
    if !live_test_enabled() {
        println!("Skipping — set LIVE_TEST=1 to run");
        return;
    }

    let dir = TempDir::new().expect("tempdir creation should succeed");
    let mut session = make_session(&dir);

    println!("[live-session-1] submitting: Create a file called hello.py that prints Hello World");
    let result = session
        .submit("Create a file called hello.py that prints Hello World")
        .await;

    assert!(
        result.is_ok(),
        "Session::submit should complete without error; got: {:?}",
        result
    );

    let hello_path = dir.path().join("hello.py");
    assert!(
        hello_path.exists(),
        "hello.py must have been created in the working directory; dir contents: {:?}",
        std::fs::read_dir(dir.path()).ok().map(|rd| rd
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .collect::<Vec<_>>())
    );

    let content = std::fs::read_to_string(&hello_path).expect("hello.py should be readable");
    println!("[live-session-1] hello.py contents:\n{content}");

    assert!(
        content.contains("Hello"),
        "hello.py must contain 'Hello'; got:\n{content}"
    );
}

// ─── Test 2: Read and edit ────────────────────────────────────────────────────
//
// Pre-creates hello.py with just the Hello print, then asks the LLM to read it
// and add a second print for Goodbye.  Verifies both strings appear.

#[tokio::test]
async fn test_live_read_and_edit() {
    if !live_test_enabled() {
        println!("Skipping — set LIVE_TEST=1 to run");
        return;
    }

    let dir = TempDir::new().expect("tempdir creation should succeed");

    // Seed hello.py so the session has something to read and edit.
    let hello_path = dir.path().join("hello.py");
    std::fs::write(&hello_path, "print(\"Hello World\")\n")
        .expect("pre-creating hello.py should succeed");

    let mut session = make_session(&dir);

    println!(
        "[live-session-2] submitting: Read hello.py and add a second print statement for Goodbye"
    );
    let result = session
        .submit("Read hello.py and add a second print statement that prints Goodbye")
        .await;

    assert!(
        result.is_ok(),
        "Session::submit should complete without error; got: {:?}",
        result
    );

    let content = std::fs::read_to_string(&hello_path).expect("hello.py should be readable");
    println!("[live-session-2] hello.py contents after edit:\n{content}");

    assert!(
        content.contains("Hello"),
        "hello.py must still contain 'Hello' after edit; got:\n{content}"
    );
    assert!(
        content.contains("Goodbye"),
        "hello.py must contain 'Goodbye' after edit; got:\n{content}"
    );
}

// ─── Test 3: Shell execution ──────────────────────────────────────────────────
//
// Pre-creates hello.py with both prints, then asks the LLM to run it with
// python3.  Verifies the session completes without error.

#[tokio::test]
async fn test_live_shell_execution() {
    if !live_test_enabled() {
        println!("Skipping — set LIVE_TEST=1 to run");
        return;
    }

    let dir = TempDir::new().expect("tempdir creation should succeed");

    // Seed hello.py with both prints so the session can run it.
    let hello_path = dir.path().join("hello.py");
    std::fs::write(&hello_path, "print(\"Hello World\")\nprint(\"Goodbye\")\n")
        .expect("pre-creating hello.py should succeed");

    let mut session = make_session(&dir);

    println!("[live-session-3] submitting: Run hello.py with python3 and show the output");
    let result = session
        .submit("Run hello.py with python3 and show me the output")
        .await;

    assert!(
        result.is_ok(),
        "Session::submit should complete without error for shell execution; got: {:?}",
        result
    );

    println!("[live-session-3] session completed OK");
}
