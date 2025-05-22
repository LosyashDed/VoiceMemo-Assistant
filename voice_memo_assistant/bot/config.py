"""
Загрузка и валидация конфигурации из .env файла.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация приложения с валидацией полей."""
    
    # Настройки Telegram
    TG_BOT_TOKEN: str
    
    # Настройки базы данных
    DB_PATH: Path
    
    # Настройки LLM
    USE_LOCAL_LLM: bool = True
    OLLAMA_BASE_URL: Optional[str] = "http://127.0.0.1:11434"
    OLLAMA_MODEL: Optional[str] = "mistral"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: Optional[str] = "gpt-3.5-turbo"
    
    # Параметры суммаризации
    SUMMARY_DEVIATION_PERCENT: int = 20
    SUMMARY_MAX_TRIES: int = 2
    
    # Настройки Speech-to-Text
    VOSK_MODEL_PATH: Path
    
    # Прочие настройки
    LOG_PATH: Path = Field(default=Path("logs/bot.log"))
    LOG_LEVEL: str = "INFO"
    
    # Валидаторы
    @field_validator("OLLAMA_BASE_URL")
    def validate_ollama_url(cls, v: Optional[str], info: dict) -> Optional[str]:
        """Проверяет URL Ollama, если используется локальная LLM."""
        if info.data.get("USE_LOCAL_LLM", True) and not v:
            raise ValueError("OLLAMA_BASE_URL должен быть указан при USE_LOCAL_LLM=True")
        return v
    
    @field_validator("OPENAI_API_KEY")
    def validate_openai_key(cls, v: Optional[str], info: dict) -> Optional[str]:
        """Проверяет ключ OpenAI, если используется облачная LLM."""
        if not info.data.get("USE_LOCAL_LLM", True) and not v:
            raise ValueError("OPENAI_API_KEY должен быть указан при USE_LOCAL_LLM=False")
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


def load_settings() -> Settings:
    """
    Загружает настройки из .env файла.
    
    Returns:
        Settings: Объект с валидированными настройками
    """
    load_dotenv()
    return Settings() 