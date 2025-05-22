"""
Pydantic-модели для валидации данных.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class VoiceMessage(BaseModel):
    """Модель данных для голосового сообщения."""
    
    id: Optional[int] = None
    telegram_file_id: str
    text: str
    summarized: str
    note: Optional[str] = None
    sent_at: str
    user_id: int
    username: Optional[str] = None


class SummaryItem(BaseModel):
    """Модель для элемента сводки резюме."""
    
    id: int
    sent_at: str
    summarized: str
    note: Optional[str] = None


class EditChoice(BaseModel):
    """Модель для выбора действия при редактировании."""
    
    record_id: int
    new_text: str
    choice: str = Field(..., description="edit или note")


class UserState(BaseModel):
    """Модель для хранения состояния пользователя."""
    
    record_id: Optional[int] = None
    new_text: Optional[str] = None 