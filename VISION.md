# OpenJarvis

**Your AI stack, your rules.**

OpenJarvis is a modular, pluggable AI assistant backend. Instead of locking you into one model, one memory system, or one inference engine, OpenJarvis lets you compose your own stack across five pillars — then swap any piece without touching the rest.

Built for developers who want full control and researchers who need reproducible, measurable AI systems.

---

## The Five Pillars

OpenJarvis is organized around five composable pillars. Each pillar defines a clear interface; implementations are discovered at runtime via a decorator-based registry system.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          OpenJarvis                                 │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ Intelligence │  │  Learning   │  │   Memory /  │                │
│  │   (Models)   │  │  Approach   │  │   Storage   │                │
│  │             │  │  (Router)   │  │             │                │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │
│         │                │                │                        │
│         ▼                ▼                ▼                        │
│  ┌──────────────────────────────────────────────┐                  │
│  │              Agentic Logic                    │                  │
│  │     (Orchestration, Tools, Reasoning)         │                  │
│  └──────────────────────┬───────────────────────┘                  │
│                         │                                          │
│                         ▼                                          │
│  ┌──────────────────────────────────────────────┐                  │
│  │            Inference Engine                   │                  │
│  │    (vLLM, Ollama, llama.cpp, SGLang, MLX)     │                  │
│  └──────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 1. Intelligence (Model Layer)

**What it does:** Manages available language models — local and cloud — and routes queries to the best model for the task.

**What's pluggable:** Models, model providers, routing heuristics.

**Supported at launch:**

| Category | Models |
|----------|--------|
| Open-source (local) | Qwen3 8B, Qwen3 32B, GPT OSS 120B, Kimi-K2.5, MiniMax-M2.5 |
| Cloud APIs | Claude (Anthropic), GPT-4o / GPT-5 (OpenAI), Gemini (Google) |

**Key components:**

- **`ModelRegistry`** — decorator-based registry mapping model keys to `ModelSpec` objects (parameter count, quantization, hardware compatibility, context length)
- **Heuristic Router (V0)** — rule-based routing: short queries → small model, complex reasoning → large model, code → code specialist, fallback chains for unavailable models
- **Auto-discovery** — detects models from running inference engines (Ollama, vLLM) and available API keys

---

### 2. Learning Approach (Router Policy)

**What it does:** Determines *which* model handles a given query. V1 is heuristic; future versions will learn from usage data.

**What's pluggable:** Routing policy, reward functions, training pipeline.

**V1 (Placeholder):**
- Heuristic routing based on query characteristics (length, complexity keywords, domain detection)
- Rule-based fallback chains
- All telemetry logged to SQLite for future training

**Future (Post-V1):**
- Learned router via GRPO (Group Relative Policy Optimization)
- Preference learning from user feedback
- Continual fine-tuning on accumulated trajectories
- Multi-objective optimization: quality vs. latency vs. energy vs. cost

---

### 3. Memory / Storage

**What it does:** Provides persistent, searchable memory across conversations, documents, and personal notes. Memory is automatically injected into prompts with source attribution.

**What's pluggable:** Storage backends, retrieval strategies, embedding models, chunking strategies.

**Memory types:**
- **Conversation Memory** — sliding window with automatic summarization of older turns
- **Knowledge Base** — indexed documents (PDF, Markdown, code, text) with multi-backend search
- **Personal Notes** — user-created persistent notes and preferences
- **Episodic Memory** — records of past interactions, tool uses, and outcomes

**Backend implementations:**

| Backend | Type | Description |
|---------|------|-------------|
| **SQLite** (default) | Keyword + FTS | FTS5 full-text search. Zero dependencies, zero config. Always available. |
| **FAISS** | Dense retrieval | Neural semantic search via `sentence-transformers` + FAISS indexes. |
| **ColBERTv2** | Late interaction | Token-level MaxSim matching with 2-bit residual compression. Best retrieval quality. Uses `colbert-ai` package with `Indexer` for offline indexing and `Searcher` for millisecond-latency queries. |
| **BM25** | Sparse retrieval | Classic keyword search baseline. Fast, no GPU needed. |
| **Hybrid** | Fusion | BM25 + dense (or ColBERT) with Reciprocal Rank Fusion (RRF). Best of both worlds. |
| **Vector DB adapters** | Dense retrieval | Qdrant, ChromaDB connectors for users with existing vector infrastructure. |

**ColBERTv2 details:**
- Late interaction model: queries and documents are encoded independently, then matched at the token level via MaxSim
- 2-bit residual compression keeps indexes compact while preserving quality
- Offline indexing via `Indexer(checkpoint="colbertv2.0", config=ColBERTConfig(nbits=2))`
- Millisecond query latency via `Searcher(index=name).search(query, k=10)`
- Substantially better retrieval quality than single-vector dense methods on complex queries

---

### 4. Agentic Logic

**What it does:** Orchestrates multi-turn reasoning, tool calling, and task execution. The agent layer sits between the user and the model, managing context, tools, and conversation flow.

**What's pluggable:** Agent implementations, tools, tool registries, execution strategies.

**Agent implementations:**

| Agent | Description |
|-------|-------------|
| **`OpenClawAgent`** (default) | Wraps OpenClaw's Pi agent runtime. Multi-turn reasoning, tool calling, streaming responses, skill composition, context compaction. Two modes: **HTTP** (WebSocket to OpenClaw gateway on `:18789`) or **subprocess** (invoke `node` with `runEmbeddedPiAgent()`, JSON over stdin/stdout). Requires Node.js 22+. |
| **`SimpleAgent`** | Single-turn: query → model → response. No tool calling. Works without Node.js. Good for quick answers and testing. |
| **`OrchestratorAgent`** | Multi-turn with per-step model selection. Adapted from IPW's executor pattern. Routes each reasoning step to the optimal model. |
| **`CustomAgent`** | Template for user-defined agent logic. Subclass `BaseAgent`, implement `run()`, register with `AgentRegistry`. |

**Tool system:**
- `BaseTool` ABC with `ToolSpec` metadata (category, cost estimate, latency estimate, capabilities)
- `ToolRegistry` — runtime-discoverable tool catalog
- Built-in tools: Calculator, WebSearch, CodeInterpreter, FileRead/Write, Think, Retrieval (wired to memory backends), LLM-as-tool
- MCP (Model Context Protocol) compatible

**API server:**
- OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints
- Streaming via Server-Sent Events (SSE)
- Drop-in replacement for any OpenAI-compatible client

---

### 5. Inference Engine

**What it does:** Manages the actual LLM inference runtime — loading models, generating tokens, managing GPU memory.

**What's pluggable:** Engine backends, hardware profiles, quantization strategies.

**Supported engines:**

| Engine | Best for | GPU | CPU |
|--------|----------|-----|-----|
| **vLLM** | High-throughput server, multi-GPU, production | NVIDIA, AMD | — |
| **SGLang** | Structured generation, constrained decoding | NVIDIA, AMD | — |
| **Ollama** | Easy setup, Apple Silicon, single-model | NVIDIA, Apple | Yes |
| **llama.cpp** | Maximum hardware compatibility, GGUF models | NVIDIA, AMD, Apple | Yes |
| **MLX** | Apple Silicon native, Metal acceleration | Apple | Apple |

**Hardware auto-detection:**
- Detects GPU vendor (NVIDIA/AMD/Apple), model, VRAM, compute capability
- Recommends the best engine for detected hardware
- Apple Silicon → Ollama or MLX; NVIDIA datacenter → vLLM; AMD → vLLM with ROCm; CPU-only → llama.cpp

---

## Query Flow

```
User query
    │
    ▼
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Agentic │────▶│ Memory   │────▶│ Context  │
│  Logic   │     │ Retrieve │     │ Inject   │
└────┬─────┘     └──────────┘     └────┬─────┘
     │                                  │
     ▼                                  ▼
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Learning │────▶│  Model   │────▶│ Inference│
│ (Router) │     │ Select   │     │ Engine   │
└──────────┘     └──────────┘     └────┬─────┘
                                       │
                                       ▼
                                  ┌──────────┐
                                  │ Response │
                                  │ + Telem. │
                                  └──────────┘
```

1. **Agentic Logic** receives the user query, determines if tools or memory are needed
2. **Memory** retrieves relevant context (conversation history, documents, notes)
3. **Context Injection** assembles the full prompt with retrieved content and source attribution
4. **Learning/Router** selects the best model for this query based on routing policy
5. **Inference Engine** runs the selected model and streams the response
6. **Telemetry** records timing, token counts, energy (if available), and cost

---

## User Scenarios

### Developer on M4 Max MacBook Pro (128 GB unified memory)

```toml
# ~/.openjarvis/config.toml
[engine]
backend = "ollama"          # Native Apple Silicon support

[intelligence]
default_model = "qwen3-32b" # Fits in 128 GB unified memory
fallback = "qwen3-8b"

[memory]
backend = "sqlite"          # Zero-config, always works
retrieval = "hybrid"        # BM25 + FAISS for local docs

[agent]
type = "openclaw"           # Full agent capabilities
mode = "subprocess"         # No separate gateway needed
```

Day-to-day: codes with `jarvis ask`, indexes project docs with `jarvis memory index`, runs a local OpenAI-compatible server with `jarvis serve` for editor integration.

### Researcher on DGX Spark (2x B200, 384 GB GPU memory)

```toml
[engine]
backend = "vllm"
tensor_parallel = 2

[intelligence]
default_model = "qwen3-235b-a22b"
router = "heuristic"        # Route small queries to 8B, large to 235B

[memory]
backend = "colbert"         # Best retrieval quality for papers
knowledge_base = "~/papers/"

[agent]
type = "orchestrator"       # Multi-model orchestration
```

Running benchmarks with `jarvis bench`, profiling energy per query, comparing model efficiency across hardware configurations.

### Privacy-Focused Offline Setup

```toml
[engine]
backend = "llamacpp"        # No server needed
network = "offline"

[intelligence]
default_model = "qwen3-8b-q4"  # Quantized to fit available RAM

[memory]
backend = "sqlite"          # Everything local
retrieval = "bm25"          # No neural models needed

[agent]
type = "simple"             # No external dependencies
```

Fully air-gapped. No cloud APIs, no network calls, no telemetry export. All data stays on the machine.

---

## Comparison

| Feature | OpenJarvis | Ollama | LangChain | OpenClaw | vLLM |
|---------|-----------|--------|-----------|----------|------|
| **Focus** | Composable AI backend | Model runner | LLM app framework | AI coding assistant | Inference server |
| **Model management** | Multi-engine, auto-detect | Single engine | Bring your own | Cloud-first | Single engine |
| **Memory** | Multi-backend retrieval | None | Vector store wrappers | Conversation only | None |
| **Agents** | Pluggable (Pi, custom) | None | Chain-based | Pi agent (built-in) | None |
| **Inference** | vLLM/SGLang/Ollama/llama.cpp/MLX | Ollama only | External | External | vLLM only |
| **Hardware-aware** | Auto-detect + recommend | Manual | No | No | Manual |
| **Telemetry** | Energy, latency, cost | None | Callbacks | Basic | Metrics |
| **Offline** | Full support | Full support | Partial | No | Full support |
| **API** | OpenAI-compatible | OpenAI-compatible | Custom | Custom | OpenAI-compatible |
| **Language** | Python | Go | Python | TypeScript | Python |

OpenJarvis is **not** a replacement for these tools — it *composes* them. Ollama and vLLM are inference engine options. OpenClaw's Pi agent is the default agentic logic. LangChain-style chains can be implemented as custom agents.

---

## Design Principles

1. **Pluggable everything** — every component is registered and discoverable at runtime. Swap models, engines, memory backends, and agents without code changes.

2. **Registry-driven** — `RegistryBase[T]` pattern (adapted from IPW) provides type-safe, decorator-based registration for all extensible components: `ModelRegistry`, `EngineRegistry`, `MemoryRegistry`, `AgentRegistry`, `ToolRegistry`.

3. **Offline-first** — works without network access. Cloud APIs are optional enhancements, never requirements.

4. **Telemetry-native** — every inference call records timing, token counts, and (when hardware supports it) energy consumption. Data lands in SQLite for analysis.

5. **Hardware-aware** — auto-detects GPU vendor, model, VRAM, and platform. Recommends the best engine and model configuration for your hardware.

6. **Python-first** — core is pure Python (3.10+). Node.js required only for OpenClaw agent integration. No Java, no JVM, no heavy runtimes.

7. **OpenAI-compatible API** — `jarvis serve` exposes `/v1/chat/completions` and `/v1/models`. Any client that speaks OpenAI protocol works out of the box.

8. **Standalone** — OpenJarvis is a self-contained backend. OpenClaw is one possible frontend; so is `curl`, a Python SDK call, or any OpenAI-compatible client.
