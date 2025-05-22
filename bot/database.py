"""
Модуль для работы с базой данных SQLite.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .utils import Utils


class Database:
    """
    Обертка для работы с базой данных.
    Скрывает специфику SQL-диалекта и предоставляет типизированный интерфейс.
    """

    _CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS voice_messages (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        telegram_file_id  TEXT UNIQUE NOT NULL,
        text              TEXT,
        summarized        TEXT,
        note              TEXT,
        sent_at           TEXT,
        user_id           INTEGER,
        username          TEXT
    );"""

    def __init__(self, path: Union[str, Path]):
        """
        Инициализирует соединение с базой данных.
        
        Args:
            path: Путь к файлу базы данных SQLite
        """
        path = Path(path)
        Utils.ensure_parent(path)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")  # лучшая параллельность
        self.conn.execute("PRAGMA busy_timeout = 10000;")  # ждем 10 секунд при блокировке
        self.conn.execute(self._CREATE_TABLE_SQL)
        self.conn.commit()
        logging.info("Подключено к SQLite БД по пути %s", path)

    def save_message(
        self,
        telegram_file_id: str,
        text: str,
        summary: str,
        sent_at: str,
        user_id: int,
        username: Optional[str] = None,
    ) -> int:
        """
        Сохраняет или обновляет запись голосового сообщения.
        
        Args:
            telegram_file_id: ID файла голосового сообщения в Telegram
            text: Полный текст транскрипции
            summary: Краткое резюме
            sent_at: Время отправки в формате ISO-8601
            user_id: ID пользователя в Telegram
            username: Имя пользователя в Telegram
        
        Returns:
            int: ID записи в БД
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO voice_messages
                (telegram_file_id, text, summarized, sent_at, user_id, username)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(telegram_file_id) DO UPDATE SET
                text=excluded.text,
                summarized=excluded.summarized,
                sent_at=excluded.sent_at,
                user_id=excluded.user_id,
                username=excluded.username
            RETURNING id;
            """,
            (telegram_file_id, text, summary, sent_at, user_id, username),
        )
        row = cursor.fetchone()
        record_id = row[0] if row else None
        self.conn.commit()
        logging.debug("Сохранено сообщение %s с id=%s", telegram_file_id, record_id)
        return record_id

    def fetch_record_by_id(self, record_id: int) -> Optional[Tuple]:
        """
        Получает запись по её ID.
        
        Args:
            record_id: ID записи в БД
            
        Returns:
            Optional[Tuple]: Запись или None, если не найдена
        """
        cursor = self.conn.execute(
            "SELECT * FROM voice_messages WHERE id = ?", (record_id,)
        )
        return cursor.fetchone()

    def fetch_record_by_telegram_file_id(self, telegram_file_id: str) -> Optional[Tuple]:
        """
        Получает запись по ID файла в Telegram.
        
        Args:
            telegram_file_id: ID файла в Telegram
            
        Returns:
            Optional[Tuple]: Запись или None, если не найдена
        """
        cursor = self.conn.execute(
            "SELECT * FROM voice_messages WHERE telegram_file_id = ?", (telegram_file_id,)
        )
        return cursor.fetchone()

    def fetch_summaries_by_date(self, date_str: str) -> List[Tuple]:
        """
        Возвращает список записей за указанную дату.
        
        Args:
            date_str: Дата в формате YYYY-MM-DD
            
        Returns:
            List[Tuple]: Список кортежей (id, sent_at, summarized, note)
        """
        cursor = self.conn.execute(
            """
            SELECT id, sent_at, summarized, note
              FROM voice_messages
             WHERE sent_at LIKE ?
          ORDER BY sent_at ASC
            """,
            (f"{date_str}%",)
        )
        return cursor.fetchall()

    def delete_record_by_id(self, record_id: int) -> bool:
        """
        Удаляет запись по её ID.
        
        Args:
            record_id: ID записи в БД
            
        Returns:
            bool: True если запись удалена, False если не найдена
        """
        cursor = self.conn.execute(
            "DELETE FROM voice_messages WHERE id = ?", (record_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_record_by_telegram_file_id(self, telegram_file_id: str) -> bool:
        """
        Удаляет запись по ID файла в Telegram.
        
        Args:
            telegram_file_id: ID файла в Telegram
            
        Returns:
            bool: True если запись удалена, False если не найдена
        """
        cursor = self.conn.execute(
            "DELETE FROM voice_messages WHERE telegram_file_id = ?", (telegram_file_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def update_summary(self, record_id: int, new_summary: str) -> bool:
        """
        Обновляет текст резюме для записи.
        
        Args:
            record_id: ID записи в БД
            new_summary: Новый текст резюме
            
        Returns:
            bool: True если запись обновлена, False если не найдена
        """
        cursor = self.conn.execute(
            "UPDATE voice_messages SET summarized = ? WHERE id = ?",
            (new_summary, record_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def add_note_to_record(self, record_id: int, note: str) -> bool:
        """
        Добавляет или обновляет примечание к записи.
        
        Args:
            record_id: ID записи в БД
            note: Текст примечания
            
        Returns:
            bool: True если примечание добавлено, False если запись не найдена
        """
        cursor = self.conn.execute(
            "UPDATE voice_messages SET note = ? WHERE id = ?",
            (note, record_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def close(self):
        """Закрывает соединение с базой данных."""
        if self.conn:
            self.conn.close() 