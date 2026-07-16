"""vox-agent — an autonomous real-time voice agent.

A streaming speech -> LLM -> speech pipeline built for sub-500ms turn latency,
with barge-in interruption, function calling, PII redaction and a slot-filling
dialog state machine. The core runs fully offline with zero third-party
dependencies; real provider adapters (Deepgram, Groq/OpenAI, Cartesia/ElevenLabs)
are optional and lazy-loaded.
"""

from .models import (
    AudioChunk,
    AudioFormat,
    ConversationState,
    Role,
    Token,
    ToolCall,
    ToolResult,
    Transcript,
    Turn,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AudioChunk",
    "AudioFormat",
    "ConversationState",
    "Role",
    "Token",
    "ToolCall",
    "ToolResult",
    "Transcript",
    "Turn",
]
