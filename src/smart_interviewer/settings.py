# src/redact_id/settings.py
import sys
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from smart_interviewer.consts import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_RELOAD


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_parse_none_str='None'
    )
    # === Server ===
    HOST: str = Field(default=DEFAULT_HOST)
    PORT: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    RELOAD: bool = Field(default=DEFAULT_RELOAD)



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









