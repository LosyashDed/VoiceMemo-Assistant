"""
Основная логика Telegram-бота и обработка команд.
Реализация на библиотеке aiogram.
"""

import datetime
import logging
import re
from typing import Dict, List, Optional, Tuple, Union, Any

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.filters import Command, CommandObject
from aiogram.filters.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from .config import Settings
from .constants import (
    BUTTON_TEXT_ADD_NOTE,
    BUTTON_TEXT_CANCEL,
    BUTTON_TEXT_EDIT_SUMMARY,
    MSG_DATE_FORMAT_ERROR,
    MSG_DELETE_NOT_FOUND,
    MSG_DELETE_SUCCESS,
    MSG_EDIT_OR_ADD_NOTE_PROMPT,
    MSG_HELP,
    MSG_NO_SUMMARIES_FOR_DATE,
    MSG_NOTE_ADDED,
    MSG_START,
    MSG_SUMMARY_UPDATED,
    MSG_VOICE_PROCESSING,
    STATE_AWAITING_CHOICE,
    SUMMARY_FOR_DATE_LINE_TEMPLATE,
    SUMMARY_FOR_DATE_NOTE_TEMPLATE,
    SUMMARY_REPLY_HEADER_TEMPLATE,
)
from .database import Database
from .llm import LLMHandler
from .models.schemas import UserState
from .stt import SpeechToText
from .utils import Utils

# Определение состояний для FSM
class EditStates(StatesGroup):
    waiting_for_choice = State()

class TelegramBot:
    """
    Основной класс для взаимодействия с API Telegram.
    Обрабатывает команды, сообщения и управляет потоком диалога.
    """

    def __init__(
        self,
        token: str,
        db: Database,
        stt: SpeechToText,
        llm: LLMHandler,
        settings: Settings,
    ):
        """
        Инициализирует бота и сохраняет ссылки на сервисы.
        
        Args:
            token: Токен Telegram-бота
            db: Экземпляр класса Database
            stt: Экземпляр класса SpeechToText
            llm: Экземпляр класса LLMHandler
            settings: Экземпляр класса Settings с настройками
        """
        self.db = db
        self.stt = stt
        self.llm = llm
        self.settings = settings
        
        # Создаем экземпляры бота и диспетчера
        self.bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
        self.dp = Dispatcher(storage=MemoryStorage())
        
        # Устанавливаем обработчики команд и сообщений
        self.setup_handlers()
        
        logging.info("Telegram-бот инициализирован и готов к работе")

    def setup_handlers(self):
        """Настраивает обработчики сообщений и команд."""
        
        # Основные команды
        self.dp.message.register(self.on_start, Command("start"))
        self.dp.message.register(self.on_help, Command("help"))
        self.dp.message.register(self.on_sum_command, Command("sum"))
        self.dp.message.register(self.on_delete_command, Command("delete"))
        
        # Обработка голосовых сообщений
        self.dp.message.register(self.on_voice, F.voice)
        
        # Обработка редактирования/добавления примечаний
        self.dp.message.register(
            self.on_reply_to_summary_for_edit,
            F.reply_to_message & F.text & ~F.text.startswith("/")
        )
        
        # Обработка callback-запросов
        self.dp.callback_query.register(
            self.handle_edit_summary_callback, 
            F.data == "edit", 
            EditStates.waiting_for_choice
        )
        self.dp.callback_query.register(
            self.handle_add_note_callback, 
            F.data == "note", 
            EditStates.waiting_for_choice
        )
        self.dp.callback_query.register(
            self.handle_cancel_callback, 
            F.data == "cancel", 
            EditStates.waiting_for_choice
        )
        
        # Обработчик неизвестных команд
        self.dp.message.register(self.on_unknown_command, F.text.startswith("/"))

    async def on_start(self, message: Message):
        """Обработчик команды /start."""
        await message.answer(MSG_START)

    async def on_help(self, message: Message):
        """Обработчик команды /help."""
        await message.answer(MSG_HELP)
        
    async def on_unknown_command(self, message: Message):
        """Обработчик неизвестных команд."""
        await message.answer("Неизвестная команда. Используйте /help для списка команд.")

    async def on_voice(self, message: Message):
        """
        Обработчик голосовых сообщений.
        
        1. Скачивает голосовое сообщение
        2. Транскрибирует его в текст с помощью STT
        3. Генерирует резюме с помощью LLM
        4. Сохраняет данные в БД
        5. Отправляет пользователю сообщение с резюме
        """
        user = message.from_user
        voice = message.voice
        file_id = voice.file_id
        sent_at = Utils.now_utc()
        
        logging.info(
            "Получено голосовое сообщение: file_id=%s | от=%s",
            file_id, user.username or user.id
        )
        
        # Сообщаем пользователю о начале обработки
        processing_msg = await message.answer(MSG_VOICE_PROCESSING)
        
        try:
            # 1. Скачиваем голосовое сообщение
            file = await self.bot.get_file(file_id)
            file_path = file.file_path
            ogg_bytes = await self.bot.download_file(file_path)
            
            # 2. Транскрибируем
            text = self.stt.transcribe_ogg_bytes(ogg_bytes.getvalue())
            logging.info("Транскрибировано %d символов", len(text))
            
            # Если текст слишком короткий или пустой, сообщаем об ошибке
            if len(text) < 5:
                await self.bot.edit_message_text(
                    "Не удалось распознать текст. Пожалуйста, попробуйте еще раз.",
                    chat_id=message.chat.id,
                    message_id=processing_msg.message_id
                )
                return
            
            # 3. Суммаризируем
            summary = self.llm.summarize(text)
            
            # 4. Сохраняем в БД
            record_id = self.db.save_message(
                telegram_file_id=file_id,
                text=text,
                summary=summary,
                sent_at=sent_at,
                user_id=user.id,
                username=user.username,
            )
            
            # 5. Отвечаем пользователю
            await self.bot.edit_message_text(
                SUMMARY_REPLY_HEADER_TEMPLATE.format(
                    record_id=record_id,
                    summary=summary
                ),
                chat_id=message.chat.id,
                message_id=processing_msg.message_id,
                reply_to_message_id=message.message_id
            )
            
        except Exception as e:
            logging.exception("Ошибка при обработке голосового сообщения: %s", e)
            await self.bot.edit_message_text(
                "Произошла ошибка при обработке сообщения. Пожалуйста, попробуйте позже.",
                chat_id=message.chat.id,
                message_id=processing_msg.message_id
            )

    async def on_sum_command(self, message: Message, command: CommandObject = None):
        """
        Обработчик команды /sum [YYYY-MM-DD].
        Выводит сводку всех резюме за указанную дату.
        """
        if command and command.args:
            date_str = command.args
            # Проверяем формат даты
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                await message.answer(MSG_DATE_FORMAT_ERROR)
                return
        else:
            # По умолчанию - текущая дата
            date_str = datetime.datetime.now().date().isoformat()
        
        # Получаем из БД записи за указанную дату
        summaries = self.db.fetch_summaries_by_date(date_str)
        
        if not summaries:
            await message.answer(MSG_NO_SUMMARIES_FOR_DATE.format(date_str=date_str))
            return
        
        # Формируем сводку
        lines = []
        for record_id, sent_at, summary, note in summaries:
            # Основная строка с резюме
            line = SUMMARY_FOR_DATE_LINE_TEMPLATE.format(
                sent_at=sent_at,
                record_id=record_id,
                summary=summary
            )
            lines.append(line)
            
            # Если есть примечание, добавляем его
            if note:
                note_line = SUMMARY_FOR_DATE_NOTE_TEMPLATE.format(note=note)
                lines.append(note_line)
        
        # Отправляем сводку пользователю
        await message.answer("\n\n".join(lines))

    async def on_delete_command(self, message: Message):
        """
        Обработчик команды /delete.
        Удаляет запись из БД разными способами:
        
        1. По ID в формате /delete #123 или /delete 123
        2. В ответе на голосовое сообщение (использует telegram_file_id)
        3. В ответе на сообщение с резюме (извлекает #ID из текста)
        """
        text = message.text.strip()
        
        # Способ 1: По явно указанному ID
        match = re.search(r"#?(\d+)$", text)
        if match:
            record_id = int(match.group(1))
            if self.db.delete_record_by_id(record_id):
                await message.answer(MSG_DELETE_SUCCESS.format(record_id=record_id))
            else:
                await message.answer(MSG_DELETE_NOT_FOUND)
            return
        
        # Способ 2 и 3: По ответу на сообщение
        if message.reply_to_message:
            reply = message.reply_to_message
            
            # Способ 2: Ответ на голосовое сообщение
            if reply.voice:
                file_id = reply.voice.file_id
                if self.db.delete_record_by_telegram_file_id(file_id):
                    await message.answer("✅ Запись удалена.")
                else:
                    await message.answer(MSG_DELETE_NOT_FOUND)
                return
                
            # Способ 3: Ответ на сообщение с резюме (извлечение #ID)
            if reply.text:
                match = re.search(r"^#(\d+)", reply.text)
                if match:
                    record_id = int(match.group(1))
                    if self.db.delete_record_by_id(record_id):
                        await message.answer(MSG_DELETE_SUCCESS.format(record_id=record_id))
                    else:
                        await message.answer(MSG_DELETE_NOT_FOUND)
                    return
        
        # Если не удалось определить запись для удаления
        await message.answer(
            "❓ Не удалось определить запись для удаления.\n"
            "Используйте форматы:\n"
            "1. /delete #123\n"
            "2. Ответом на голосовое сообщение\n"
            "3. Ответом на сообщение с резюме"
        )

    async def on_reply_to_summary_for_edit(self, message: Message, state: FSMContext):
        """
        Начало диалога редактирования/добавления примечания.
        Вызывается, когда пользователь отвечает на сообщение с резюме.
        """
        reply = message.reply_to_message
        
        # Проверяем, что ответ на сообщение с резюме (начинается с #ID)
        if not reply or not reply.text or not re.match(r"^#\d+", reply.text):
            await message.answer(
                "Чтобы отредактировать резюме или добавить примечание, "
                "ответьте на сообщение с резюме (которое начинается с #ID)."
            )
            return
        
        # Извлекаем ID записи
        match = re.match(r"^#(\d+)", reply.text)
        record_id = int(match.group(1))
        
        # Получаем запись из БД для проверки
        record = self.db.fetch_record_by_id(record_id)
        if not record:
            await message.answer(MSG_DELETE_NOT_FOUND)
            return
        
        # Сохраняем данные в состоянии для использования в следующих шагах
        await state.set_state(EditStates.waiting_for_choice)
        await state.update_data(record_id=record_id, new_text=message.text)
        
        # Создаем кнопки выбора действия
        keyboard = [
            [
                InlineKeyboardButton(text=BUTTON_TEXT_EDIT_SUMMARY, callback_data="edit"),
                InlineKeyboardButton(text=BUTTON_TEXT_ADD_NOTE, callback_data="note"),
            ],
            [InlineKeyboardButton(text=BUTTON_TEXT_CANCEL, callback_data="cancel")],
        ]
        markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
        
        # Отправляем сообщение с запросом действия
        await message.answer(
            MSG_EDIT_OR_ADD_NOTE_PROMPT,
            reply_markup=markup
        )

    async def handle_edit_summary_callback(self, callback_query: CallbackQuery, state: FSMContext):
        """
        Обработчик нажатия на кнопку "Изменить резюме".
        """
        await callback_query.answer()
        
        # Получаем данные из состояния
        data = await state.get_data()
        record_id = data.get("record_id")
        new_text = data.get("new_text")
        
        if record_id and new_text:
            # Обновляем резюме в БД
            if self.db.update_summary(record_id, new_text):
                await callback_query.message.edit_text(MSG_SUMMARY_UPDATED)
            else:
                await callback_query.message.edit_text(MSG_DELETE_NOT_FOUND)
        else:
            await callback_query.message.edit_text("Произошла ошибка. Пожалуйста, попробуйте снова.")
        
        # Очищаем состояние
        await state.clear()

    async def handle_add_note_callback(self, callback_query: CallbackQuery, state: FSMContext):
        """
        Обработчик нажатия на кнопку "Добавить примечание".
        """
        await callback_query.answer()
        
        # Получаем данные из состояния
        data = await state.get_data()
        record_id = data.get("record_id")
        note_text = data.get("new_text")
        
        if record_id and note_text:
            # Добавляем примечание в БД
            if self.db.add_note_to_record(record_id, note_text):
                await callback_query.message.edit_text(MSG_NOTE_ADDED)
            else:
                await callback_query.message.edit_text(MSG_DELETE_NOT_FOUND)
        else:
            await callback_query.message.edit_text("Произошла ошибка. Пожалуйста, попробуйте снова.")
        
        # Очищаем состояние
        await state.clear()

    async def handle_cancel_callback(self, callback_query: CallbackQuery, state: FSMContext):
        """
        Обработчик нажатия на кнопку "Отмена".
        """
        await callback_query.answer()
        await callback_query.message.edit_text("Операция отменена.")
        
        # Очищаем состояние
        await state.clear()

    async def start(self):
        """Запускает бота."""
        logging.info("Запуск бота...")
        await self.dp.start_polling(self.bot)

    def run(self):
        """Запускает бота в блокирующем режиме."""
        import asyncio
        
        logging.info("Запуск бота (Ctrl+C для выхода)...")
        
        try:
            asyncio.run(self.start())
        except (KeyboardInterrupt, SystemExit):
            logging.info("Бот остановлен.") 