# coding-agent-loop

Agentic tool loop for LLM-powered coding agents in Rust.

[![CI](https://github.com/bkrabach/coding-agent-loop/actions/workflows/ci.yaml/badge.svg)](https://github.com/bkrabach/coding-agent-loop/actions)
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

Provides a `Session`-based agentic loop that:

1. Sends a prompt to an LLM (via [unified-llm](https://github.com/bkrabach/unified-llm))
2. Parses tool calls from the response (shell commands, file operations, etc.)
3. Executes the tool calls in a sandboxed environment
4. Feeds tool results back to the LLM
5. Repeats until the LLM produces a final response without tool calls

## Features

- **Tool execution** — Shell commands, file read/write, directory listing
- **Provider profiles** — Pre-configured profiles for OpenAI, Anthropic, and Gemini
- **Local execution environment** — Sandboxed file and command execution
- **Configurable** — Max turns, token limits, timeout control
- **Async** — Built on Tokio

## Quick Start

```bash
git clone https://github.com/bkrabach/coding-agent-loop.git
cd coding-agent-loop
cargo build
```

## Dependencies

- [unified-llm](https://github.com/bkrabach/unified-llm) — Multi-provider LLM client

## Related

- [attractor](https://github.com/bkrabach/attractor) — Pipeline engine (uses coding-agent-loop for agentic nodes)
- [attractor-server](https://github.com/bkrabach/attractor-server) — HTTP API server
