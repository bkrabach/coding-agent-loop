# coding-agent-loop

Agentic tool loop for LLM-powered coding agents in Rust.

[![CI](https://github.com/bkrabach/coding-agent-loop/actions/workflows/ci.yaml/badge.svg)](https://github.com/bkrabach/coding-agent-loop/actions)
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

`coding-agent-loop` provides a `Session`-based agentic loop that pairs an LLM with developer tools. It sends a prompt to an LLM via [unified-llm](https://github.com/bkrabach/unified-llm), parses tool calls from the response (shell commands, file operations, etc.), executes them in a sandboxed environment, feeds results back, and repeats until the LLM produces a final response without tool calls.

## Features

- **Agentic loop** — Automatic prompt → tool call → result → re-prompt cycle until completion
- **Tool execution** — Shell commands, file read/write, directory listing
- **Provider profiles** — Pre-configured profiles for OpenAI, Anthropic, and Gemini
- **Local execution environment** — Sandboxed file and command execution
- **Configurable limits** — Max turns, token limits, timeout control
- **Session-based** — Each task runs in an isolated `Session` with its own context
- **Async** — Built on Tokio for async I/O

## Quick Start

```bash
git clone https://github.com/bkrabach/coding-agent-loop.git
cd coding-agent-loop
cargo build
```

## Dependencies

- [unified-llm](https://github.com/bkrabach/unified-llm) — Multi-provider LLM client

## Origin

This project was built from the [Coding Agent Loop Specification](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md) (NLSpec) by [strongDM](https://github.com/strongdm). The NLSpec defines a language-agnostic specification for building a coding agent that pairs an LLM with developer tools through an agentic loop.

## Ecosystem

| Project | Description |
|---------|-------------|
| [attractor](https://github.com/bkrabach/attractor) | DOT-based pipeline engine |
| [attractor-server](https://github.com/bkrabach/attractor-server) | HTTP API server |
| [attractor-ui](https://github.com/bkrabach/attractor-ui) | Web frontend |
| [unified-llm](https://github.com/bkrabach/unified-llm) | Multi-provider LLM client |
| [coding-agent-loop](https://github.com/bkrabach/coding-agent-loop) | Agentic tool loop |

## License

MIT
