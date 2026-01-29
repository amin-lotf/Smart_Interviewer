"""Runtime configuration loaded from environment variables and .env."""
import sys
from typing import Optional

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from smart_interviewer.consts import (
    DEFAULT_HOST, DEFAULT_PORT, DEFAULT_RELOAD,
    MAX_TURNS,WHISPER_MODEL_NAME,WHISPER_DEVICE,WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,QUESTIONS_PER_LEVEL,MIN_PASSED_FOR_LEVEL,MAX_FOLLOWUP_QUESTIONS,
    LLM_MODEL,LLM_TEMPERATURE,API_TIMEOUT,AUDIO_SAMPLE_RATE)

class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_parse_none_str='None'
    )
    RANDOM_SEED: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    # Server Configuration
    HOST: str = Field(default=DEFAULT_HOST, description="Server host address")
    PORT: int = Field(default=DEFAULT_PORT, ge=1, le=65535, description="Server port")
    RELOAD: bool = Field(default=DEFAULT_RELOAD, description="Enable auto-reload in development")

    # API Keys
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key for LLM calls")

    # Whisper Configuration
    WHISPER_MODEL_NAME: str = Field(default=WHISPER_MODEL_NAME, description="Whisper model size (tiny/base/small/medium/large)")
    WHISPER_DEVICE: str = Field(default=WHISPER_DEVICE, description="Device for Whisper (cpu/cuda)")
    WHISPER_COMPUTE_TYPE: str = Field(default=WHISPER_COMPUTE_TYPE, description="Whisper compute type")
    WHISPER_LANGUAGE: Optional[str] = Field(default=WHISPER_LANGUAGE, description="Language for transcription")

    # Interview Configuration
    MAX_TURNS: int = Field(default=MAX_TURNS, description="Maximum interview turns before auto-end")
    QUESTIONS_PER_LEVEL: int = Field(default=QUESTIONS_PER_LEVEL, description="Number of questions per difficulty level")
    MIN_PASSED_FOR_LEVEL: int = Field(default=MIN_PASSED_FOR_LEVEL, description="Minimum correct answers to pass a level")
    MAX_FOLLOWUP_QUESTIONS: int = Field(default=MAX_FOLLOWUP_QUESTIONS, description="Maximum follow-up questions per answer")
    QUESTION_BANK_PATH: Optional[str] = Field(
        default=None,
        description="Optional path to the markdown question bank file.",
    )

    # LLM Configuration
    LLM_MODEL: str = Field(default=LLM_MODEL, description="OpenAI model to use for question generation and evaluation")
    LLM_TEMPERATURE: float = Field(default=LLM_TEMPERATURE, ge=0.0, le=2.0, description="LLM temperature for response generation")

    # API Configuration
    API_TIMEOUT: int = Field(default=API_TIMEOUT, description="Default API timeout in seconds")

    # Audio Configuration
    AUDIO_SAMPLE_RATE: int = Field(default=AUDIO_SAMPLE_RATE, description="Audio sample rate for recording")


def load_settings_or_die() -> Settings:
    try:
        s = Settings()
    except ValidationError as e:
        # One clean message, no scary traceback
        print("[CONFIG ERROR] Invalid environment configuration:", file=sys.stderr)
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "invalid value")
            print(f"  - {loc}: {msg}", file=sys.stderr)
        sys.exit(2)

    return s

settings = load_settings_or_die()


