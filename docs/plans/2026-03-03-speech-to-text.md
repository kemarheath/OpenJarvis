# Speech-to-Text Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Speech subsystem to OpenJarvis that lets users speak commands instead of typing them in the desktop app and browser.

**Architecture:** New `src/openjarvis/speech/` module following the ABC + registry pattern. `SpeechBackend` ABC with `transcribe()` method. Three backends: Faster-Whisper (local, default), OpenAI Whisper API (cloud), Deepgram (cloud). New `POST /v1/speech/transcribe` endpoint. Frontend `MicButton` component records audio and sends to backend.

**Tech Stack:** faster-whisper (CTranslate2), openai SDK (existing dep), deepgram-sdk, Web MediaRecorder API, FastAPI multipart uploads, Tauri commands (Rust/reqwest)

**Design doc:** `docs/plans/2026-03-03-speech-to-text-design.md`

---

### Task 1: Add SpeechRegistry to core

**Files:**
- Modify: `src/openjarvis/core/registry.py:136-152`
- Modify: `tests/conftest.py:12-35`
- Test: `tests/core/test_registry.py` (existing — verify it still passes)

**Step 1: Add SpeechRegistry class**

In `src/openjarvis/core/registry.py`, after line 137 (`class SkillRegistry`), add:

```python
class SpeechRegistry(RegistryBase[Any]):
    """Registry for speech backend implementations."""
```

Add `"SpeechRegistry"` to `__all__` list (alphabetically, after `"SkillRegistry"`).

**Step 2: Update conftest.py to clear SpeechRegistry**

In `tests/conftest.py`, add `SpeechRegistry` to the import block (line 12-21):

```python
from openjarvis.core.registry import (
    AgentRegistry,
    BenchmarkRegistry,
    ChannelRegistry,
    EngineRegistry,
    MemoryRegistry,
    ModelRegistry,
    RouterPolicyRegistry,
    SpeechRegistry,
    ToolRegistry,
)
```

Add `SpeechRegistry.clear()` after line 34 (`ChannelRegistry.clear()`).

**Step 3: Run tests to verify nothing broke**

Run: `uv run pytest tests/core/test_registry.py -v`
Expected: All existing registry tests PASS

**Step 4: Commit**

```bash
git add src/openjarvis/core/registry.py tests/conftest.py
git commit -m "feat(speech): add SpeechRegistry to core registry module"
```

---

### Task 2: Add SpeechConfig to config

**Files:**
- Modify: `src/openjarvis/core/config.py:824-853, 943-948`
- Test: `tests/core/test_config.py` (existing)

**Step 1: Write the failing test**

Create `tests/speech/__init__.py` (empty) and `tests/speech/test_config.py`:

```python
"""Tests for speech configuration."""

from openjarvis.core.config import JarvisConfig, SpeechConfig


def test_speech_config_defaults():
    cfg = SpeechConfig()
    assert cfg.backend == "auto"
    assert cfg.model == "base"
    assert cfg.language == ""
    assert cfg.device == "auto"
    assert cfg.compute_type == "float16"


def test_jarvis_config_has_speech():
    cfg = JarvisConfig()
    assert hasattr(cfg, "speech")
    assert isinstance(cfg.speech, SpeechConfig)
    assert cfg.speech.backend == "auto"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/speech/test_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'SpeechConfig'`

**Step 3: Add SpeechConfig dataclass**

In `src/openjarvis/core/config.py`, after line 831 (after `OperatorsConfig`), add:

```python
@dataclass(slots=True)
class SpeechConfig:
    """Speech-to-text settings."""

    backend: str = "auto"  # "auto", "faster-whisper", "whisper-cpp", "openai", "deepgram"
    model: str = "base"  # Whisper model size: tiny, base, small, medium, large-v3
    language: str = ""  # Empty = auto-detect
    device: str = "auto"  # "auto", "cpu", "cuda"
    compute_type: str = "float16"  # "float16", "int8", "float32"
```

Add to `JarvisConfig` (after line 853, `operators: OperatorsConfig`):

```python
    speech: SpeechConfig = field(default_factory=SpeechConfig)
```

Add `"speech"` to the `top_sections` tuple in `load_config()` (line 943-948):

```python
        top_sections = (
            "engine", "intelligence", "learning", "agent",
            "server", "telemetry", "traces", "security",
            "channel", "tools", "sandbox", "scheduler",
            "workflow", "sessions", "a2a", "operators",
            "speech",
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/speech/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/core/config.py tests/speech/
git commit -m "feat(speech): add SpeechConfig to configuration system"
```

---

### Task 3: Create SpeechBackend ABC and TranscriptionResult

**Files:**
- Create: `src/openjarvis/speech/__init__.py`
- Create: `src/openjarvis/speech/_stubs.py`
- Test: `tests/speech/test_stubs.py`

**Step 1: Write the failing test**

Create `tests/speech/test_stubs.py`:

```python
"""Tests for speech ABC and data types."""

from openjarvis.speech._stubs import Segment, SpeechBackend, TranscriptionResult


def test_transcription_result():
    result = TranscriptionResult(
        text="Hello world",
        language="en",
        confidence=0.95,
        duration_seconds=1.5,
        segments=[],
    )
    assert result.text == "Hello world"
    assert result.language == "en"
    assert result.confidence == 0.95
    assert result.duration_seconds == 1.5
    assert result.segments == []


def test_segment():
    seg = Segment(text="Hello", start=0.0, end=0.5, confidence=0.98)
    assert seg.text == "Hello"
    assert seg.start == 0.0
    assert seg.end == 0.5


def test_speech_backend_is_abstract():
    import pytest

    with pytest.raises(TypeError):
        SpeechBackend()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/speech/test_stubs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'openjarvis.speech'`

**Step 3: Create the module**

Create `src/openjarvis/speech/__init__.py`:

```python
"""Speech subsystem — speech-to-text backends."""
```

Create `src/openjarvis/speech/_stubs.py`:

```python
"""Abstract base classes and data types for the speech subsystem."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    """A timed segment of transcribed text."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: Optional[float] = None


@dataclass
class TranscriptionResult:
    """Result of a speech-to-text transcription."""

    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration_seconds: float = 0.0
    segments: List[Segment] = field(default_factory=list)


class SpeechBackend(ABC):
    """Abstract base class for speech-to-text backends."""

    backend_id: str = ""

    @abstractmethod
    def transcribe(
        self,
        audio: bytes,
        *,
        format: str = "wav",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio bytes to text."""

    @abstractmethod
    def health(self) -> bool:
        """Check if the backend is ready."""

    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Return list of supported audio formats."""
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/speech/test_stubs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/speech/
git commit -m "feat(speech): add SpeechBackend ABC, TranscriptionResult, Segment"
```

---

### Task 4: Implement FasterWhisperBackend

**Files:**
- Create: `src/openjarvis/speech/faster_whisper.py`
- Test: `tests/speech/test_faster_whisper.py`

**Step 1: Write the failing test**

Create `tests/speech/test_faster_whisper.py`:

```python
"""Tests for Faster-Whisper speech backend."""

from unittest.mock import MagicMock, patch

import pytest


def test_faster_whisper_backend_registers():
    """Backend registers itself in SpeechRegistry."""
    from openjarvis.core.registry import SpeechRegistry

    # Import triggers registration
    import openjarvis.speech.faster_whisper  # noqa: F401

    assert SpeechRegistry.contains("faster-whisper")


def test_faster_whisper_transcribe():
    """Transcribe returns a TranscriptionResult."""
    from openjarvis.speech._stubs import TranscriptionResult

    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = " Hello world"
    mock_segment.start = 0.0
    mock_segment.end = 1.2
    mock_segment.avg_logprob = -0.3

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95
    mock_info.duration = 1.5

    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    with patch(
        "openjarvis.speech.faster_whisper.WhisperModel",
        return_value=mock_model,
    ):
        from openjarvis.speech.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend(model_size="base", device="cpu")
        result = backend.transcribe(b"fake audio bytes")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration_seconds == 1.5


def test_faster_whisper_health_no_model():
    """Health returns False before model is loaded."""
    with patch(
        "openjarvis.speech.faster_whisper.WhisperModel",
        side_effect=ImportError("no module"),
    ):
        from openjarvis.speech.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend.__new__(FasterWhisperBackend)
        backend._model = None
        assert backend.health() is False


def test_faster_whisper_supported_formats():
    """Backend supports standard audio formats."""
    with patch("openjarvis.speech.faster_whisper.WhisperModel"):
        from openjarvis.speech.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend.__new__(FasterWhisperBackend)
        formats = backend.supported_formats()
        assert "wav" in formats
        assert "mp3" in formats
        assert "webm" in formats
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/speech/test_faster_whisper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'openjarvis.speech.faster_whisper'`

**Step 3: Implement FasterWhisperBackend**

Create `src/openjarvis/speech/faster_whisper.py`:

```python
"""Faster-Whisper speech-to-text backend (local, CTranslate2-based)."""

from __future__ import annotations

import io
import tempfile
from typing import List, Optional

from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech._stubs import Segment, SpeechBackend, TranscriptionResult

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment, misc]


@SpeechRegistry.register("faster-whisper")
class FasterWhisperBackend(SpeechBackend):
    """Local speech-to-text using Faster-Whisper (CTranslate2)."""

    backend_id = "faster-whisper"

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "float16",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    def _ensure_model(self) -> WhisperModel:
        """Lazy-load the Whisper model on first use."""
        if self._model is None:
            if WhisperModel is None:
                raise ImportError(
                    "faster-whisper is not installed. "
                    "Install with: pip install 'openjarvis[speech]'"
                )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio: bytes,
        *,
        format: str = "wav",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio bytes using Faster-Whisper."""
        model = self._ensure_model()

        # Write audio to a temp file (faster-whisper needs a file path)
        suffix = f".{format}" if not format.startswith(".") else format
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio)
            tmp.flush()

            kwargs = {}
            if language:
                kwargs["language"] = language

            segments_iter, info = model.transcribe(tmp.name, **kwargs)
            segments_list = list(segments_iter)

        # Build result
        text = "".join(seg.text for seg in segments_list).strip()
        segments = [
            Segment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                confidence=None,
            )
            for seg in segments_list
        ]

        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            confidence=getattr(info, "language_probability", None),
            duration_seconds=getattr(info, "duration", 0.0),
            segments=segments,
        )

    def health(self) -> bool:
        """Check if model is loaded or loadable."""
        if self._model is not None:
            return True
        return WhisperModel is not None

    def supported_formats(self) -> List[str]:
        """Supported audio formats (same as ffmpeg/Whisper)."""
        return ["wav", "mp3", "m4a", "ogg", "flac", "webm"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/speech/test_faster_whisper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/speech/faster_whisper.py tests/speech/test_faster_whisper.py
git commit -m "feat(speech): implement FasterWhisperBackend (local STT)"
```

---

### Task 5: Implement OpenAIWhisperBackend

**Files:**
- Create: `src/openjarvis/speech/openai_whisper.py`
- Test: `tests/speech/test_openai_whisper.py`

**Step 1: Write the failing test**

Create `tests/speech/test_openai_whisper.py`:

```python
"""Tests for OpenAI Whisper API speech backend."""

from unittest.mock import MagicMock, patch

from openjarvis.speech._stubs import TranscriptionResult


def test_openai_whisper_registers():
    from openjarvis.core.registry import SpeechRegistry
    import openjarvis.speech.openai_whisper  # noqa: F401

    assert SpeechRegistry.contains("openai")


def test_openai_whisper_transcribe():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello from OpenAI"
    mock_response.language = "en"
    mock_response.duration = 2.0
    mock_client.audio.transcriptions.create.return_value = mock_response

    with patch("openjarvis.speech.openai_whisper.OpenAI", return_value=mock_client):
        from openjarvis.speech.openai_whisper import OpenAIWhisperBackend

        backend = OpenAIWhisperBackend(api_key="test-key")
        result = backend.transcribe(b"fake audio", format="wav")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello from OpenAI"
        assert result.language == "en"


def test_openai_whisper_health():
    with patch("openjarvis.speech.openai_whisper.OpenAI"):
        from openjarvis.speech.openai_whisper import OpenAIWhisperBackend

        backend = OpenAIWhisperBackend(api_key="test-key")
        assert backend.health() is True


def test_openai_whisper_health_no_key():
    with patch("openjarvis.speech.openai_whisper.OpenAI"):
        from openjarvis.speech.openai_whisper import OpenAIWhisperBackend

        backend = OpenAIWhisperBackend.__new__(OpenAIWhisperBackend)
        backend._client = None
        backend._api_key = ""
        assert backend.health() is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/speech/test_openai_whisper.py -v`
Expected: FAIL

**Step 3: Implement OpenAIWhisperBackend**

Create `src/openjarvis/speech/openai_whisper.py`:

```python
"""OpenAI Whisper API speech-to-text backend (cloud)."""

from __future__ import annotations

import io
import os
from typing import List, Optional

from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech._stubs import SpeechBackend, TranscriptionResult

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment, misc]


@SpeechRegistry.register("openai")
class OpenAIWhisperBackend(SpeechBackend):
    """Cloud speech-to-text using OpenAI Whisper API."""

    backend_id = "openai"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client: Optional[OpenAI] = None
        if self._api_key and OpenAI is not None:
            self._client = OpenAI(api_key=self._api_key)

    def transcribe(
        self,
        audio: bytes,
        *,
        format: str = "wav",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using OpenAI's Whisper API."""
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized (missing API key?)")

        ext = format if not format.startswith(".") else format[1:]
        audio_file = io.BytesIO(audio)
        audio_file.name = f"audio.{ext}"

        kwargs: dict = {"model": "whisper-1", "file": audio_file}
        if language:
            kwargs["language"] = language
        kwargs["response_format"] = "verbose_json"

        response = self._client.audio.transcriptions.create(**kwargs)

        return TranscriptionResult(
            text=getattr(response, "text", str(response)),
            language=getattr(response, "language", None),
            confidence=None,
            duration_seconds=getattr(response, "duration", 0.0),
            segments=[],
        )

    def health(self) -> bool:
        return self._client is not None and bool(self._api_key)

    def supported_formats(self) -> List[str]:
        return ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/speech/test_openai_whisper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/speech/openai_whisper.py tests/speech/test_openai_whisper.py
git commit -m "feat(speech): implement OpenAIWhisperBackend (cloud STT)"
```

---

### Task 6: Implement DeepgramBackend

**Files:**
- Create: `src/openjarvis/speech/deepgram.py`
- Test: `tests/speech/test_deepgram.py`

**Step 1: Write the failing test**

Create `tests/speech/test_deepgram.py`:

```python
"""Tests for Deepgram speech backend."""

from unittest.mock import MagicMock, patch

from openjarvis.speech._stubs import TranscriptionResult


def test_deepgram_registers():
    from openjarvis.core.registry import SpeechRegistry
    import openjarvis.speech.deepgram  # noqa: F401

    assert SpeechRegistry.contains("deepgram")


def test_deepgram_transcribe():
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_channel = MagicMock()
    mock_alternative = MagicMock()
    mock_alternative.transcript = "Hello from Deepgram"
    mock_alternative.confidence = 0.92
    mock_channel.alternatives = [mock_alternative]
    mock_result.results.channels = [mock_channel]
    mock_result.metadata.duration = 1.8
    mock_result.results.channels[0].detected_language = "en"
    mock_client.listen.rest.v.return_value.transcribe_file.return_value = mock_result

    with patch("openjarvis.speech.deepgram.DeepgramClient", return_value=mock_client):
        from openjarvis.speech.deepgram import DeepgramSpeechBackend

        backend = DeepgramSpeechBackend(api_key="test-key")
        result = backend.transcribe(b"fake audio", format="wav")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello from Deepgram"


def test_deepgram_health():
    with patch("openjarvis.speech.deepgram.DeepgramClient"):
        from openjarvis.speech.deepgram import DeepgramSpeechBackend

        backend = DeepgramSpeechBackend(api_key="test-key")
        assert backend.health() is True


def test_deepgram_health_no_key():
    with patch("openjarvis.speech.deepgram.DeepgramClient"):
        from openjarvis.speech.deepgram import DeepgramSpeechBackend

        backend = DeepgramSpeechBackend.__new__(DeepgramSpeechBackend)
        backend._client = None
        backend._api_key = ""
        assert backend.health() is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/speech/test_deepgram.py -v`
Expected: FAIL

**Step 3: Implement DeepgramBackend**

Create `src/openjarvis/speech/deepgram.py`:

```python
"""Deepgram speech-to-text backend (cloud)."""

from __future__ import annotations

import os
from typing import List, Optional

from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech._stubs import SpeechBackend, TranscriptionResult

try:
    from deepgram import DeepgramClient, PrerecordedOptions
except ImportError:
    DeepgramClient = None  # type: ignore[assignment, misc]
    PrerecordedOptions = None  # type: ignore[assignment, misc]


@SpeechRegistry.register("deepgram")
class DeepgramSpeechBackend(SpeechBackend):
    """Cloud speech-to-text using Deepgram API."""

    backend_id = "deepgram"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self._client = None
        if self._api_key and DeepgramClient is not None:
            self._client = DeepgramClient(self._api_key)

    def transcribe(
        self,
        audio: bytes,
        *,
        format: str = "wav",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using Deepgram's API."""
        if self._client is None:
            raise RuntimeError("Deepgram client not initialized (missing API key?)")

        mime_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
            "webm": "audio/webm",
            "m4a": "audio/mp4",
        }
        mime_type = mime_map.get(format, "audio/wav")

        options_kwargs: dict = {"model": "nova-2", "smart_format": True}
        if language:
            options_kwargs["language"] = language
        else:
            options_kwargs["detect_language"] = True

        payload = {"buffer": audio, "mimetype": mime_type}

        if PrerecordedOptions is not None:
            options = PrerecordedOptions(**options_kwargs)
        else:
            options = options_kwargs

        response = self._client.listen.rest.v("1").transcribe_file(
            payload, options,
        )

        # Extract transcript from response
        channels = response.results.channels
        if channels and channels[0].alternatives:
            alt = channels[0].alternatives[0]
            text = alt.transcript
            confidence = getattr(alt, "confidence", None)
        else:
            text = ""
            confidence = None

        detected_lang = None
        if channels:
            detected_lang = getattr(channels[0], "detected_language", None)

        duration = getattr(response.metadata, "duration", 0.0)

        return TranscriptionResult(
            text=text,
            language=detected_lang,
            confidence=confidence,
            duration_seconds=duration,
            segments=[],
        )

    def health(self) -> bool:
        return self._client is not None and bool(self._api_key)

    def supported_formats(self) -> List[str]:
        return ["wav", "mp3", "ogg", "flac", "webm", "m4a"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/speech/test_deepgram.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/openjarvis/speech/deepgram.py tests/speech/test_deepgram.py
git commit -m "feat(speech): implement DeepgramSpeechBackend (cloud STT)"
```

---

### Task 7: Speech discovery and package init

**Files:**
- Create: `src/openjarvis/speech/_discovery.py`
- Modify: `src/openjarvis/speech/__init__.py`
- Test: `tests/speech/test_discovery.py`

**Step 1: Write the failing test**

Create `tests/speech/test_discovery.py`:

```python
"""Tests for speech backend auto-discovery."""

from unittest.mock import patch

from openjarvis.core.config import JarvisConfig, SpeechConfig


def test_get_speech_backend_explicit():
    """Explicit backend selection works."""
    from openjarvis.speech._discovery import get_speech_backend

    config = JarvisConfig()
    config.speech.backend = "faster-whisper"

    with patch("openjarvis.speech._discovery._create_backend") as mock_create:
        from openjarvis.speech._stubs import TranscriptionResult

        mock_backend = type("MockBackend", (), {
            "backend_id": "faster-whisper",
            "health": lambda self: True,
        })()
        mock_create.return_value = mock_backend

        result = get_speech_backend(config)
        assert result is not None
        assert result.backend_id == "faster-whisper"


def test_get_speech_backend_returns_none_if_nothing_available():
    """Returns None when no backend can be created."""
    from openjarvis.speech._discovery import get_speech_backend

    config = JarvisConfig()
    config.speech.backend = "nonexistent"

    result = get_speech_backend(config)
    assert result is None


def test_auto_discovery_priority():
    """Auto mode tries backends in priority order."""
    from openjarvis.speech._discovery import DISCOVERY_ORDER

    assert DISCOVERY_ORDER[0] == "faster-whisper"
    assert "openai" in DISCOVERY_ORDER
    assert "deepgram" in DISCOVERY_ORDER
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/speech/test_discovery.py -v`
Expected: FAIL

**Step 3: Implement discovery**

Create `src/openjarvis/speech/_discovery.py`:

```python
"""Auto-discover available speech-to-text backends."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from openjarvis.core.config import JarvisConfig
    from openjarvis.speech._stubs import SpeechBackend

# Priority order: local first, then cloud
DISCOVERY_ORDER = [
    "faster-whisper",
    "openai",
    "deepgram",
]


def _create_backend(
    key: str,
    config: "JarvisConfig",
) -> Optional["SpeechBackend"]:
    """Try to instantiate a speech backend by registry key."""
    from openjarvis.core.registry import SpeechRegistry

    if not SpeechRegistry.contains(key):
        return None

    try:
        backend_cls = SpeechRegistry.get(key)

        if key == "faster-whisper":
            return backend_cls(
                model_size=config.speech.model,
                device=config.speech.device,
                compute_type=config.speech.compute_type,
            )
        elif key == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return None
            return backend_cls(api_key=api_key)
        elif key == "deepgram":
            api_key = os.environ.get("DEEPGRAM_API_KEY", "")
            if not api_key:
                return None
            return backend_cls(api_key=api_key)
        else:
            return backend_cls()
    except Exception:
        return None


def get_speech_backend(config: "JarvisConfig") -> Optional["SpeechBackend"]:
    """Resolve the speech backend from config.

    If ``config.speech.backend`` is ``"auto"``, tries backends in
    priority order and returns the first healthy one.
    """
    # Trigger registration of built-in backends
    import openjarvis.speech  # noqa: F401

    backend_key = config.speech.backend

    if backend_key != "auto":
        return _create_backend(backend_key, config)

    # Auto-discovery: try each in priority order
    for key in DISCOVERY_ORDER:
        backend = _create_backend(key, config)
        if backend is not None:
            return backend

    return None
```

**Step 4: Update `src/openjarvis/speech/__init__.py`**

```python
"""Speech subsystem — speech-to-text backends."""

import importlib

# Optional backends — each registers itself via @SpeechRegistry.register()
for _mod in ("faster_whisper", "openai_whisper", "deepgram"):
    try:
        importlib.import_module(f".{_mod}", __name__)
    except ImportError:
        pass

__all__ = ["SpeechBackend", "TranscriptionResult", "Segment", "get_speech_backend"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/speech/test_discovery.py -v`
Expected: PASS

**Step 6: Run all speech tests**

Run: `uv run pytest tests/speech/ -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/openjarvis/speech/ tests/speech/test_discovery.py
git commit -m "feat(speech): add auto-discovery and wire speech package init"
```

---

### Task 8: Wire speech into SystemBuilder

**Files:**
- Modify: `src/openjarvis/system.py:16-43, 296-346, 447-471`
- Test: `tests/speech/test_system_integration.py`

**Step 1: Write the failing test**

Create `tests/speech/test_system_integration.py`:

```python
"""Tests for speech integration in SystemBuilder/JarvisSystem."""

from unittest.mock import MagicMock, patch

from openjarvis.system import JarvisSystem


def test_jarvis_system_has_speech_backend():
    """JarvisSystem has a speech_backend attribute."""
    system = JarvisSystem.__new__(JarvisSystem)
    assert hasattr(JarvisSystem, "__dataclass_fields__")
    assert "speech_backend" in JarvisSystem.__dataclass_fields__
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/speech/test_system_integration.py -v`
Expected: FAIL with `AssertionError`

**Step 3: Add speech_backend to JarvisSystem**

In `src/openjarvis/system.py`, add after line 42 (`operator_manager`):

```python
    speech_backend: Optional[Any] = None  # SpeechBackend
```

In `SystemBuilder.__init__()`, add after line 306 (`self._sessions`):

```python
        self._speech: Optional[bool] = None
```

Add builder method after line 346 (`sessions`):

```python
    def speech(self, enabled: bool) -> SystemBuilder:
        self._speech = enabled
        return self
```

In the `build()` method, before the `system = JarvisSystem(` call (around line 449), add:

```python
        # Set up speech backend
        speech_backend = None
        speech_enabled = self._speech if self._speech is not None else True
        if speech_enabled:
            try:
                from openjarvis.speech._discovery import get_speech_backend

                speech_backend = get_speech_backend(config)
            except Exception:
                pass
```

Add `speech_backend=speech_backend,` to the `JarvisSystem(...)` constructor call (after `operator_manager`-related line).

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/speech/test_system_integration.py -v`
Expected: PASS

**Step 5: Run full test suite to verify nothing broke**

Run: `uv run pytest tests/ -x -q`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/openjarvis/system.py tests/speech/test_system_integration.py
git commit -m "feat(speech): wire speech backend into SystemBuilder and JarvisSystem"
```

---

### Task 9: Add speech API endpoints

**Files:**
- Modify: `src/openjarvis/server/api_routes.py:571-599`
- Modify: `src/openjarvis/server/app.py:45-120`
- Test: `tests/server/test_speech_routes.py`

**Step 1: Write the failing test**

Create `tests/server/test_speech_routes.py`:

```python
"""Tests for speech API endpoints."""

import pytest

fastapi = pytest.importorskip("fastapi")

from io import BytesIO
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from openjarvis.speech._stubs import TranscriptionResult


@pytest.fixture
def mock_speech_backend():
    backend = MagicMock()
    backend.backend_id = "mock"
    backend.health.return_value = True
    backend.transcribe.return_value = TranscriptionResult(
        text="Hello world",
        language="en",
        confidence=0.95,
        duration_seconds=1.5,
        segments=[],
    )
    return backend


@pytest.fixture
def app_with_speech(mock_speech_backend):
    from fastapi import FastAPI
    from openjarvis.server.api_routes import speech_router

    app = FastAPI()
    app.state.speech_backend = mock_speech_backend
    app.include_router(speech_router)
    return app


@pytest.fixture
def client(app_with_speech):
    return TestClient(app_with_speech)


def test_transcribe_endpoint(client, mock_speech_backend):
    response = client.post(
        "/v1/speech/transcribe",
        files={"file": ("test.wav", b"fake audio data", "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Hello world"
    assert data["language"] == "en"
    assert data["confidence"] == 0.95
    assert data["duration_seconds"] == 1.5


def test_transcribe_no_file(client):
    response = client.post("/v1/speech/transcribe")
    assert response.status_code == 422  # FastAPI validation error


def test_health_endpoint(client):
    response = client.get("/v1/speech/health")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["backend"] == "mock"


def test_health_no_backend():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from openjarvis.server.api_routes import speech_router

    app = FastAPI()
    app.state.speech_backend = None
    app.include_router(speech_router)
    client = TestClient(app)

    response = client.get("/v1/speech/health")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/server/test_speech_routes.py -v`
Expected: FAIL with `ImportError: cannot import name 'speech_router'`

**Step 3: Add speech routes to api_routes.py**

In `src/openjarvis/server/api_routes.py`, add before the `include_all_routes` function (before line 573):

```python
# ---- Speech routes ----

speech_router = APIRouter(prefix="/v1/speech", tags=["speech"])


@speech_router.post("/transcribe")
async def transcribe_speech(request: Request):
    """Transcribe uploaded audio to text."""
    backend = getattr(request.app.state, "speech_backend", None)
    if backend is None:
        raise HTTPException(status_code=501, detail="Speech backend not configured")

    form = await request.form()
    audio_file = form.get("file")
    if audio_file is None:
        raise HTTPException(status_code=400, detail="Missing 'file' field")

    audio_bytes = await audio_file.read()
    language = form.get("language")

    # Detect format from filename
    filename = getattr(audio_file, "filename", "audio.wav")
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "wav"

    result = backend.transcribe(audio_bytes, format=ext, language=language or None)
    return {
        "text": result.text,
        "language": result.language,
        "confidence": result.confidence,
        "duration_seconds": result.duration_seconds,
    }


@speech_router.get("/health")
async def speech_health(request: Request):
    """Check if a speech backend is available."""
    backend = getattr(request.app.state, "speech_backend", None)
    if backend is None:
        return {"available": False, "reason": "No speech backend configured"}
    return {
        "available": backend.health(),
        "backend": backend.backend_id,
    }
```

Add `speech_router` to `include_all_routes()`:

```python
    app.include_router(speech_router)
```

Add `"speech_router"` to `__all__`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/server/test_speech_routes.py -v`
Expected: PASS

**Step 5: Wire speech backend into app state**

In `src/openjarvis/server/app.py`, add `speech_backend=None` parameter to `create_app()`:

```python
def create_app(
    engine,
    model: str,
    *,
    agent=None,
    bus=None,
    engine_name: str = "",
    agent_name: str = "",
    channel_bridge=None,
    config=None,
    speech_backend=None,
) -> FastAPI:
```

Add after line 119 (`app.state.session_start = time.time()`):

```python
    app.state.speech_backend = speech_backend
```

**Step 6: Wire speech into serve.py**

In `src/openjarvis/cli/serve.py`, add after the channel setup block (around line 161), before `create_app`:

```python
    # Set up speech backend
    speech_backend = None
    try:
        from openjarvis.speech._discovery import get_speech_backend

        speech_backend = get_speech_backend(config)
        if speech_backend:
            console.print(f"  Speech: [cyan]{speech_backend.backend_id}[/cyan]")
    except Exception:
        pass
```

Add `speech_backend=speech_backend` to the `create_app()` call:

```python
    app = create_app(
        engine, model_name, agent=agent, bus=bus,
        engine_name=engine_name, agent_name=agent_key or "",
        channel_bridge=channel_bridge, config=config,
        speech_backend=speech_backend,
    )
```

**Step 7: Run all tests**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS

**Step 8: Commit**

```bash
git add src/openjarvis/server/api_routes.py src/openjarvis/server/app.py src/openjarvis/cli/serve.py tests/server/test_speech_routes.py
git commit -m "feat(speech): add /v1/speech/transcribe and /health API endpoints"
```

---

### Task 10: Add speech client function to frontend API layer

**Files:**
- Modify: `frontend/src/api/client.ts`

**Step 1: Add transcribe and speech health functions**

In `frontend/src/api/client.ts`, add after the existing functions:

```typescript
export interface TranscriptionResult {
  text: string;
  language: string | null;
  confidence: number | null;
  duration_seconds: number;
}

export interface SpeechHealth {
  available: boolean;
  backend?: string;
  reason?: string;
}

export async function transcribeAudio(audioBlob: Blob, filename = 'recording.webm'): Promise<TranscriptionResult> {
  const formData = new FormData();
  formData.append('file', audioBlob, filename);
  const res = await fetch(`${BASE}/v1/speech/transcribe`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Transcription failed: ${res.status}`);
  return res.json();
}

export async function fetchSpeechHealth(): Promise<SpeechHealth> {
  const res = await fetch(`${BASE}/v1/speech/health`);
  if (!res.ok) return { available: false };
  return res.json();
}
```

**Step 2: Commit**

```bash
git add frontend/src/api/client.ts
git commit -m "feat(speech): add transcribeAudio and fetchSpeechHealth to frontend API"
```

---

### Task 11: Create useSpeech React hook

**Files:**
- Create: `frontend/src/hooks/useSpeech.ts`

**Step 1: Create the hook**

Create `frontend/src/hooks/useSpeech.ts`:

```typescript
import { useState, useCallback, useRef, useEffect } from 'react';
import { transcribeAudio, fetchSpeechHealth } from '../api/client';

export type SpeechState = 'idle' | 'recording' | 'transcribing';

export function useSpeech() {
  const [state, setState] = useState<SpeechState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [available, setAvailable] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  // Check if speech backend is available on mount
  useEffect(() => {
    fetchSpeechHealth()
      .then((health) => setAvailable(health.available))
      .catch(() => setAvailable(false));
  }, []);

  const startRecording = useCallback(async (): Promise<void> => {
    setError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Microphone not supported in this browser');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setState('recording');
    } catch (err) {
      setError('Microphone access denied');
      setState('idle');
    }
  }, []);

  const stopRecording = useCallback(async (): Promise<string> => {
    return new Promise((resolve, reject) => {
      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state !== 'recording') {
        reject(new Error('Not recording'));
        return;
      }

      recorder.onstop = async () => {
        setState('transcribing');

        // Stop all audio tracks
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;

        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        chunksRef.current = [];

        try {
          const result = await transcribeAudio(blob);
          setState('idle');
          resolve(result.text);
        } catch (err) {
          setState('idle');
          const msg = err instanceof Error ? err.message : 'Transcription failed';
          setError(msg);
          reject(err);
        }
      };

      recorder.stop();
    });
  }, []);

  return {
    state,
    error,
    available,
    startRecording,
    stopRecording,
    isRecording: state === 'recording',
    isTranscribing: state === 'transcribing',
  };
}
```

**Step 2: Commit**

```bash
git add frontend/src/hooks/useSpeech.ts
git commit -m "feat(speech): add useSpeech React hook for mic recording and transcription"
```

---

### Task 12: Create MicButton component and integrate into InputArea

**Files:**
- Create: `frontend/src/components/Chat/MicButton.tsx`
- Modify: `frontend/src/components/Chat/InputArea.tsx`

**Step 1: Create MicButton component**

Create `frontend/src/components/Chat/MicButton.tsx`:

```tsx
import type { SpeechState } from '../../hooks/useSpeech';

interface MicButtonProps {
  state: SpeechState;
  onClick: () => void;
  disabled?: boolean;
}

export function MicButton({ state, onClick, disabled }: MicButtonProps) {
  const title =
    state === 'recording'
      ? 'Stop recording'
      : state === 'transcribing'
        ? 'Transcribing...'
        : 'Voice input';

  return (
    <button
      className={`mic-btn ${state !== 'idle' ? `mic-${state}` : ''}`}
      onClick={onClick}
      disabled={disabled || state === 'transcribing'}
      title={title}
      style={{
        background: state === 'recording' ? '#e74c3c' : 'transparent',
        border: '1px solid var(--border, #555)',
        borderRadius: '8px',
        padding: '8px',
        cursor: disabled || state === 'transcribing' ? 'default' : 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minWidth: '36px',
        height: '36px',
        color: state === 'recording' ? '#fff' : 'var(--text, #cdd6f4)',
        opacity: disabled || state === 'transcribing' ? 0.5 : 1,
        animation: state === 'recording' ? 'pulse 1.5s ease-in-out infinite' : 'none',
      }}
    >
      {state === 'transcribing' ? (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="28" strokeDashoffset="10">
            <animateTransform attributeName="transform" type="rotate" from="0 8 8" to="360 8 8" dur="1s" repeatCount="indefinite" />
          </circle>
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M5 3a3 3 0 0 1 6 0v5a3 3 0 0 1-6 0V3z" />
          <path d="M3.5 6.5A.5.5 0 0 1 4 7v1a4 4 0 0 0 8 0V7a.5.5 0 0 1 1 0v1a5 5 0 0 1-4.5 4.975V15h3a.5.5 0 0 1 0 1h-7a.5.5 0 0 1 0-1h3v-2.025A5 5 0 0 1 3 8V7a.5.5 0 0 1 .5-.5z" />
        </svg>
      )}
    </button>
  );
}
```

**Step 2: Integrate MicButton into InputArea**

Replace `frontend/src/components/Chat/InputArea.tsx` with the version that includes the mic button.

Add import at the top:

```tsx
import { MicButton } from './MicButton';
import { useSpeech } from '../../hooks/useSpeech';
```

Inside the `InputArea` component, add the speech hook and handler:

```tsx
  const { state: speechState, available: speechAvailable, startRecording, stopRecording, error: speechError } = useSpeech();

  const handleMicClick = useCallback(async () => {
    if (speechState === 'recording') {
      try {
        const text = await stopRecording();
        if (text) {
          setTyped((prev) => (prev ? prev + ' ' + text : text));
        }
      } catch {
        // Error is captured in speechError
      }
    } else {
      await startRecording();
    }
  }, [speechState, startRecording, stopRecording]);
```

In the button area (lines 143-155), replace with:

```tsx
        {isStreaming ? (
          <button className="stop-btn" onClick={onStop}>
            Stop
          </button>
        ) : (
          <div style={{ display: 'flex', gap: '4px' }}>
            {speechAvailable && (
              <MicButton
                state={speechState}
                onClick={handleMicClick}
              />
            )}
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={!fullMessage.trim()}
            >
              Send
            </button>
          </div>
        )}
```

**Step 3: Commit**

```bash
git add frontend/src/components/Chat/MicButton.tsx frontend/src/components/Chat/InputArea.tsx
git commit -m "feat(speech): add MicButton component and integrate into InputArea"
```

---

### Task 13: Add Tauri command for speech transcription

**Files:**
- Modify: `desktop/src-tauri/src/lib.rs:168-243`

**Step 1: Add transcribe_audio command**

In `desktop/src-tauri/src/lib.rs`, after line 168 (after `run_jarvis_command`), add:

```rust
/// Transcribe audio via the speech API endpoint.
#[tauri::command]
async fn transcribe_audio(
    api_url: String,
    audio_data: Vec<u8>,
    filename: String,
) -> Result<serde_json::Value, String> {
    let url = format!("{}/v1/speech/transcribe", api_url);
    let client = reqwest::Client::new();

    let part = reqwest::multipart::Part::bytes(audio_data)
        .file_name(filename)
        .mime_str("audio/webm")
        .map_err(|e| format!("Failed to create multipart: {}", e))?;

    let form = reqwest::multipart::Form::new().part("file", part);

    let resp = client
        .post(&url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))?;
    Ok(body)
}

/// Check speech backend health.
#[tauri::command]
async fn speech_health(api_url: String) -> Result<serde_json::Value, String> {
    let url = format!("{}/v1/speech/health", api_url);
    let resp = reqwest::get(&url)
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))?;
    Ok(body)
}
```

Add to the `invoke_handler` list (line 231-243):

```rust
        .invoke_handler(tauri::generate_handler![
            check_health,
            fetch_energy,
            fetch_telemetry,
            fetch_traces,
            fetch_trace,
            fetch_learning_stats,
            fetch_learning_policy,
            fetch_memory_stats,
            search_memory,
            fetch_agents,
            run_jarvis_command,
            transcribe_audio,
            speech_health,
        ])
```

**Step 2: Commit**

```bash
git add desktop/src-tauri/src/lib.rs
git commit -m "feat(speech): add transcribe_audio and speech_health Tauri commands"
```

---

### Task 14: Add optional dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml:19-104`

**Step 1: Add speech extras**

In `pyproject.toml`, after line 99 (`dashboard = ["textual>=0.80"]`), add:

```toml
speech = ["faster-whisper>=1.0"]
speech-deepgram = ["deepgram-sdk>=3.0"]
```

The OpenAI backend uses the existing `openai` dep from `inference-cloud`.

**Step 2: Run tests to verify nothing broke**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(speech): add speech and speech-deepgram optional dependencies"
```

---

### Task 15: Run full test suite and update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All pass, no regressions

**Step 2: Run lint**

Run: `uv run ruff check src/openjarvis/speech/ tests/speech/`
Expected: Clean or fix any issues

**Step 3: Update CLAUDE.md**

Add speech subsystem to the Architecture section, add `[speech]` config documentation, add `jarvis serve` speech output mention, update file counts.

Key additions:
- Speech subsystem description in the Five Pillars or Cross-cutting Systems section
- `[speech]` config section documentation
- Speech API endpoints (`POST /v1/speech/transcribe`, `GET /v1/speech/health`)
- Speech optional dependencies (`openjarvis[speech]`, `openjarvis[speech-deepgram]`)

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with speech subsystem documentation"
```

---

## Summary

| Task | What | Files | Est. Lines |
|------|------|-------|-----------|
| 1 | SpeechRegistry | registry.py, conftest.py | ~10 |
| 2 | SpeechConfig | config.py, test | ~30 |
| 3 | SpeechBackend ABC | _stubs.py, test | ~70 |
| 4 | FasterWhisperBackend | faster_whisper.py, test | ~130 |
| 5 | OpenAIWhisperBackend | openai_whisper.py, test | ~100 |
| 6 | DeepgramBackend | deepgram.py, test | ~120 |
| 7 | Discovery + init | _discovery.py, __init__.py, test | ~90 |
| 8 | SystemBuilder wiring | system.py, test | ~30 |
| 9 | API endpoints | api_routes.py, app.py, serve.py, test | ~110 |
| 10 | Frontend API client | client.ts | ~30 |
| 11 | useSpeech hook | useSpeech.ts | ~90 |
| 12 | MicButton + InputArea | MicButton.tsx, InputArea.tsx | ~100 |
| 13 | Tauri commands | lib.rs | ~50 |
| 14 | pyproject.toml | pyproject.toml | ~5 |
| 15 | Tests + docs | CLAUDE.md | ~30 |
| **Total** | | **~20 files** | **~1000 lines** |
