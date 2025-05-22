"""
Точка входа в приложение VoiceMemo Assistant.
"""

import logging
import sys
from pathlib import Path

from .config import load_settings
from .database import Database
from .llm import LLMHandler
from .stt import SpeechToText
from .telegram_handler import TelegramBot
from .utils import configure_logging


def main():
    """
    Основная точка входа в приложение.
    
    1. Загружает и валидирует настройки из .env
    2. Настраивает логирование
    3. Инициализирует все подсистемы (БД, STT, LLM)
    4. Создает и запускает Telegram-бота
    """
    try:
        # Загружаем настройки из .env
        settings = load_settings()
        
        # Настраиваем логирование
        configure_logging(
            log_path=str(settings.LOG_PATH),
            log_level=settings.LOG_LEVEL
        )
        
        # Сообщаем о запуске
        logging.info("Запуск VoiceMemo Assistant v%s", "0.1.0")
        
        # Инициализируем базу данных
        db = Database(settings.DB_PATH)
        logging.info("База данных инициализирована")
        
        # Инициализируем модуль распознавания речи
        stt = SpeechToText(model_path=str(settings.VOSK_MODEL_PATH))
        logging.info("Модуль распознавания речи инициализирован")
        
        # Инициализируем обработчик языковой модели
        llm = LLMHandler(
            use_local=settings.USE_LOCAL_LLM,
            config=settings.model_dump()
        )
        logging.info("Обработчик LLM инициализирован")
        
        # Создаем и запускаем бота
        bot = TelegramBot(
            token=settings.TG_BOT_TOKEN,
            db=db,
            stt=stt,
            llm=llm,
            settings=settings,
        )
        
        logging.info("Бот создан, запускаем...")
        bot.run()
        
    except KeyboardInterrupt:
        logging.info("Получен сигнал прерывания, завершаем работу")
        sys.exit(0)
    except Exception as e:
        logging.exception("Критическая ошибка: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main() 