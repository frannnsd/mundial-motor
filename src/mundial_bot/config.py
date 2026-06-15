"""Configuración central del bot, cargada desde el entorno (.env).

Validación tipada con pydantic-settings: si falta una clave o un valor es
inválido, falla rápido y claro al arrancar en vez de explotar a mitad de camino.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del proyecto (este archivo está en src/mundial_bot/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"


class Settings(BaseSettings):
    """Configuración del bot. Lee de .env y variables de entorno."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys (opcionales en dev; requeridas en operación real) ---
    odds_api_key: str = Field(default="", alias="ODDS_API_KEY")
    api_football_key: str = Field(default="", alias="API_FOOTBALL_KEY")
    football_data_key: str = Field(default="", alias="FOOTBALL_DATA_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")
    oddspapi_key: str = Field(default="", alias="ODDSPAPI_KEY")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # --- Gestión de banca ---
    bankroll_usd: float = Field(default=100.0, alias="BANKROLL_USD", gt=0)
    kelly_fraction: float = Field(default=0.25, alias="KELLY_FRACTION", gt=0, le=1)
    max_stake_pct: float = Field(default=0.03, alias="MAX_STAKE_PCT", gt=0, le=1)
    max_total_exposure_pct: float = Field(
        default=0.25, alias="MAX_TOTAL_EXPOSURE_PCT", gt=0, le=1
    )
    min_edge: float = Field(default=0.03, alias="MIN_EDGE", ge=0)

    # --- Operación ---
    odds_region: str = Field(default="eu", alias="ODDS_REGION")
    timezone: str = Field(default="America/Argentina/Buenos_Aires", alias="TIMEZONE")
    daily_picks_hour: int = Field(default=10, alias="DAILY_PICKS_HOUR", ge=0, le=23)

    @property
    def has_odds_api(self) -> bool:
        return bool(self.odds_api_key)

    @property
    def has_api_football(self) -> bool:
        return bool(self.api_football_key)

    @property
    def has_football_data(self) -> bool:
        return bool(self.football_data_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_news(self) -> bool:
        return bool(self.news_api_key)

    @property
    def has_oddspapi(self) -> bool:
        return bool(self.oddspapi_key)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


def get_settings() -> Settings:
    """Devuelve la configuración cargada. Crea data/cache si no existe."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()
