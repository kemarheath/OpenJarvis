# Speech-to-Text Design

**Date:** 2026-03-03
**Status:** Approved
**Scope:** Add voice input (speech-to-text) to OpenJarvis — desktop app and browser

## Overview

Add a new Speech subsystem to OpenJarvis that lets users speak commands instead of typing them. The system transcribes audio to text using local open-source models (Faster-Whisper) by default, with cloud backends (OpenAI, Deepgram) as alternatives. Transcribed text is inserted into the input box for user review before sending.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | STT only (no TTS) | Ship voice input first; TTS follows later |
| Runtime | Separate process | Avoid VRAM conflicts with the LLM engine |
| Surfaces | Desktop app + browser | Both use the same React frontend |
| UX mode | Record-then-transcribe | Simpler to implement, works with all backends |
| Architecture | New Speech subsystem | Fits OpenJarvis patterns (ABC + registry + decorator) |

## Architecture

### New Module: `src/openjarvis/speech/`

```
src/openjarvis/speech/
├── __init__.py          # Imports, ensure_registered()
├── _stubs.py            # SpeechBackend ABC, TranscriptionResult dataclass
├── faster_whisper.py    # FasterWhisperBackend (local, default)
├── whisper_cpp.py       # WhisperCppBackend (local, llama.cpp ecosystem)
├── openai_whisper.py    # OpenAIWhisperBackend (cloud)
├── deepgram.py          # DeepgramBackend (cloud)
└── _discovery.py        # Auto-discover available backend (local preferred)
```

### Core Types (`_stubs.py`)

```python
@dataclass
class Segment:
    text: str
    start: float           # Start time in seconds
    end: float             # End time in seconds
    confidence: float | None

@dataclass
class TranscriptionResult:
    text: str               # The transcribed text
    language: str | None     # Detected language code (e.g., "en")
    confidence: float | None # Overall confidence [0, 1]
    duration_seconds: float  # Audio duration
    segments: list[Segment]  # Word/phrase-level timing (optional)

class SpeechBackend(ABC):
    backend_id: str

    @abstractmethod
    def transcribe(self, audio: bytes, *, format: str = "wav",
                   language: str | None = None) -> TranscriptionResult: ...

    @abstractmethod
    def health(self) -> bool: ...

    @abstractmethod
    def supported_formats(self) -> list[str]: ...
```

### Registry

New `SpeechRegistry` added to `core/registry.py` using `RegistryBase[T]`. Backends register via `@SpeechRegistry.register("faster-whisper")`.

### Discovery (`_discovery.py`)

Priority order (local-first):
1. Faster-Whisper (if `faster-whisper` package installed)
2. WhisperCpp (if `whisper-cpp-python` package installed)
3. OpenAI Whisper API (if `OPENAI_API_KEY` set)
4. Deepgram (if `DEEPGRAM_API_KEY` set)

Function: `get_speech_backend(config) -> SpeechBackend | None`

### Config

New `[speech]` section in `JarvisConfig`:

```toml
[speech]
backend = "auto"           # "auto", "faster-whisper", "whisper-cpp", "openai", "deepgram"
model = "base"             # Whisper model size: tiny, base, small, medium, large-v3
language = ""              # Empty = auto-detect
device = "auto"            # "auto", "cpu", "cuda"
compute_type = "float16"   # "float16", "int8", "float32"
```

New `SpeechConfig` dataclass in `core/config.py`.

## API Layer

### New Endpoints

```
POST /v1/speech/transcribe
  Content-Type: multipart/form-data
  Body: audio file (field name: "file")
  Optional form fields: language, model

  Response 200: {
    "text": "Hello, what's the weather like?",
    "language": "en",
    "confidence": 0.94,
    "duration_seconds": 2.3
  }

GET /v1/speech/backends
  Response 200: {
    "backends": ["faster-whisper"],
    "active": "faster-whisper",
    "model": "base"
  }

GET /v1/speech/health
  Response 200: {"available": true, "backend": "faster-whisper", "model": "base"}
  Response 200: {"available": false, "reason": "No speech backend installed"}
```

### Server Wiring

- `SystemBuilder.with_speech(backend=..., model=...)` — configure speech
- `JarvisSystem.speech` — active `SpeechBackend` instance or `None`
- Speech backend initializes lazily on first `transcribe()` call
- Routes added to `server/api_routes.py`

## Frontend Integration

### Web Frontend (`frontend/src/`)

**New component: `MicButton.tsx`**
- Sits next to the Send button in `InputArea.tsx`
- States: idle (mic icon), recording (pulsing red), transcribing (spinner)
- Uses `MediaRecorder` API to capture audio
- Sends audio blob as multipart to `/v1/speech/transcribe`
- Appends transcribed text to input textarea
- Hidden if `/v1/speech/health` returns `available: false`

**New hook: `useSpeech.ts`**
- Manages microphone permissions, `MediaRecorder` lifecycle
- `startRecording()`, `stopRecording()`, `isRecording`, `isTranscribing`, `error`
- Checks `navigator.mediaDevices` support

**Changes to existing components:**
- `InputArea.tsx` — add `MicButton` next to send button
- `App.tsx` — check speech availability on load

### Desktop App (`desktop/`)

Same React UI works in the Tauri WebView.

**New Tauri command: `transcribe_audio`** in `lib.rs` — proxies multipart POST to backend for cases where WebView fetch has CORS issues.

### Audio Format

Record as WebM/Opus (native browser format). Faster-Whisper handles WebM directly. Server-side conversion to WAV as fallback if a backend requires it.

### User Flow

```
User clicks mic button
  → Browser requests microphone permission (first time)
  → Recording starts (button pulses red)
User clicks mic button again (or releases)
  → Recording stops
  → Button shows spinner
  → Audio blob sent to POST /v1/speech/transcribe
  → Response text inserted into input textarea
  → User reviews/edits text
  → User clicks Send (normal flow)
```

## Backend Implementations

### Faster-Whisper (Default Local)

```python
@SpeechRegistry.register("faster-whisper")
class FasterWhisperBackend(SpeechBackend):
    backend_id = "faster-whisper"
    # Uses CTranslate2-based Faster-Whisper (4x faster than original Whisper)
    # Lazy model loading on first transcribe() call
    # Model sizes: tiny (39M), base (74M), small (244M), medium (769M), large-v3 (1.5B)
    # GPU: device="cuda", compute_type="float16" or "int8"
    # CPU: device="cpu", compute_type="int8"
```

Optional dep: `openjarvis[speech]` → `faster-whisper>=1.0`

### OpenAI Whisper API (Cloud)

```python
@SpeechRegistry.register("openai")
class OpenAIWhisperBackend(SpeechBackend):
    backend_id = "openai"
    # Uses openai.audio.transcriptions.create()
    # Requires OPENAI_API_KEY
    # Uses existing openai dependency
```

### Deepgram (Cloud)

```python
@SpeechRegistry.register("deepgram")
class DeepgramBackend(SpeechBackend):
    backend_id = "deepgram"
    # Uses deepgram-sdk
    # Requires DEEPGRAM_API_KEY
```

Optional dep: `openjarvis[speech-deepgram]` → `deepgram-sdk>=3.0`

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Mic permission denied | Toast error in frontend, no crash |
| No speech backend available | Mic button hidden, health endpoint returns `available: false` |
| Audio too short / silence | Return empty text with low confidence |
| Model loading failure | Health returns false, error logged |
| Network failure (cloud) | Error response, frontend shows retry option |
| Unsupported audio format | Server converts to WAV, or returns 400 with message |

## Testing

### Backend Tests
- `tests/speech/test_faster_whisper.py` — mock `faster_whisper` import, test transcribe with synthetic WAV
- `tests/speech/test_openai_whisper.py` — mock `openai` client, test transcribe
- `tests/speech/test_deepgram.py` — mock `deepgram` client, test transcribe
- `tests/speech/test_discovery.py` — test auto-discovery priority order

### API Tests
- `tests/server/test_speech_routes.py` — test endpoints with mocked backend

### Config Tests
- Test `SpeechConfig` defaults, TOML parsing, `[speech]` section

All optional-dep backends behind `pytest.importorskip()`.

## Optional Dependencies

```toml
[project.optional-dependencies]
speech = ["faster-whisper>=1.0"]
speech-deepgram = ["deepgram-sdk>=3.0"]
```

## Files Changed (Estimated)

### New Files (~12)
- `src/openjarvis/speech/__init__.py`
- `src/openjarvis/speech/_stubs.py`
- `src/openjarvis/speech/faster_whisper.py`
- `src/openjarvis/speech/openai_whisper.py`
- `src/openjarvis/speech/deepgram.py`
- `src/openjarvis/speech/_discovery.py`
- `frontend/src/components/MicButton.tsx`
- `frontend/src/hooks/useSpeech.ts`
- `tests/speech/test_faster_whisper.py`
- `tests/speech/test_openai_whisper.py`
- `tests/speech/test_deepgram.py`
- `tests/speech/test_discovery.py`
- `tests/server/test_speech_routes.py`

### Modified Files (~8)
- `src/openjarvis/core/registry.py` — add `SpeechRegistry`
- `src/openjarvis/core/config.py` — add `SpeechConfig`, `[speech]` section
- `src/openjarvis/system.py` — add `with_speech()`, wire speech backend
- `src/openjarvis/server/api_routes.py` — add speech endpoints
- `frontend/src/components/InputArea.tsx` — add MicButton
- `frontend/src/App.tsx` — check speech availability
- `desktop/src-tauri/src/lib.rs` — add `transcribe_audio` command
- `pyproject.toml` — add `[speech]` and `[speech-deepgram]` extras
- `tests/conftest.py` — add SpeechRegistry to `_clean_registries`
