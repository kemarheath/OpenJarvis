# OpenJarvis

**Your AI stack, your rules.**

A modular, pluggable AI assistant backend. Compose your own stack across five pillars — Intelligence, Learning, Memory, Agents, and Inference — then swap any piece without touching the rest.

> **Status: Early Development** — Interfaces are being defined. Not yet usable.

## What is this?

OpenJarvis lets you build a personal AI assistant from composable parts:

- **Intelligence** — multi-model management with automatic routing (Qwen3, GPT OSS, Kimi-K2.5, Claude, GPT-5, Gemini)
- **Memory** — persistent, searchable storage with multiple backends (SQLite, FAISS, ColBERTv2, BM25, hybrid)
- **Agents** — pluggable reasoning and tool use (OpenClaw Pi agent, simple, orchestrator, custom)
- **Inference** — hardware-aware engine selection (vLLM, SGLang, Ollama, llama.cpp, MLX)
- **Learning** — router that improves over time (heuristic now, learned later)

## Documentation

- **[VISION.md](VISION.md)** — Project vision, architecture, design principles, and detailed pillar descriptions
- **[ROADMAP.md](ROADMAP.md)** — Phased development plan with deliverables and version milestones

## Quick orientation

```
src/openjarvis/
├── core/          # Registry, types, config, event bus
├── intelligence/  # Model management, routing
├── memory/        # Storage backends (SQLite, FAISS, ColBERT, BM25, hybrid)
├── agents/        # Agent implementations + tool system
├── engine/        # Inference engine wrappers
├── learning/      # Router policy (placeholder)
└── cli/           # CLI entry points (jarvis ask, serve, model, memory)
```

## Requirements

- Python 3.10+
- An inference backend: [Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm), or [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Node.js 22+ (only if using OpenClaw agent)

## License

TBD
