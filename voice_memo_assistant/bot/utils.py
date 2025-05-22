"""
Вспомогательные утилиты для проекта.
"""

import datetime
import logging
from pathlib import Path
from typing import Union


class Utils:
    """Набор вспомогательных статических методов."""

    @staticmethod
    def now_utc() -> str:
        """Возвращает текущее время в UTC в формате ISO-8601 (без микросекунд)."""
        return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

    @staticmethod
    def ensure_parent(path: Path) -> None:
        """Создает родительскую директорию, если она не существует."""
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def as_bool(value: Union[str, bool, None]) -> bool:
        """Преобразует строковое или другое значение в булево."""
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}


def configure_logging(log_path: str, log_level: str = "INFO") -> None:
    """
    Настраивает логирование с указанным путем и уровнем.
    
    Args:
        log_path: Путь к файлу лога
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Создаем директорию для логов, если она не существует
    Utils.ensure_parent(Path(log_path))
    
    # Устанавливаем уровень логирования
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Настраиваем формат и обработчики
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    
    # Устанавливаем уровень для некоторых "шумных" логгеров
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING) 