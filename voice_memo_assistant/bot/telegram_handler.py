"""
Основная логика Telegram-бота и обработка команд.
"""

import datetime
import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, Voice
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

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
        
        # Создаем экземпляр приложения бота
        self.app = ApplicationBuilder().token(token).build()
        
        # Устанавливаем обработчики команд и сообщений
        self.setup_handlers()
        
        logging.info("Telegram-бот инициализирован и готов к работе")

    def setup_handlers(self):
        """Настраивает обработчики сообщений и команд."""
        
        # Основные команды
        self.app.add_handler(CommandHandler("start", self.on_start))
        self.app.add_handler(CommandHandler("help", self.on_help))
        self.app.add_handler(CommandHandler("sum", self.on_sum_command))
        self.app.add_handler(CommandHandler("delete", self.on_delete_command))
        
        # Обработка голосовых сообщений
        self.app.add_handler(
            MessageHandler(filters.VOICE & ~filters.COMMAND, self.on_voice)
        )
        
        # Обработка редактирования/добавления примечаний (машина состояний)
        edit_handler = ConversationHandler(
            # Входная точка: пользователь отвечает текстом на сообщение с резюме
            entry_points=[
                MessageHandler(
                    filters.REPLY & filters.TEXT & ~filters.COMMAND,
                    self.on_reply_to_summary_for_edit,
                )
            ],
            # Состояния диалога
            states={
                STATE_AWAITING_CHOICE: [
                    CallbackQueryHandler(self.handle_edit_summary_callback, pattern=r"^edit$"),
                    CallbackQueryHandler(self.handle_add_note_callback, pattern=r"^note$"),
                    CallbackQueryHandler(self.handle_cancel_callback, pattern=r"^cancel$"),
                ]
            },
            # Выход из диалога
            fallbacks=[
                MessageHandler(filters.ALL, self.handle_cancel_edit),
            ],
            # Тайм-аут для автоматического завершения диалога
            conversation_timeout=300,  # 5 минут
        )
        self.app.add_handler(edit_handler)
        
        # Обработчик неизвестных команд
        self.app.add_handler(MessageHandler(filters.COMMAND, self.on_unknown_command))

    async def on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start."""
        await update.message.reply_text(MSG_START)

    async def on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help."""
        await update.message.reply_text(MSG_HELP)
        
    async def on_unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик неизвестных команд."""
        await update.message.reply_text("Неизвестная команда. Используйте /help для списка команд.")

    async def on_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик голосовых сообщений.
        
        1. Скачивает голосовое сообщение
        2. Транскрибирует его в текст с помощью STT
        3. Генерирует резюме с помощью LLM
        4. Сохраняет данные в БД
        5. Отправляет пользователю сообщение с резюме
        
        Args:
            update: Обновление от Telegram
            context: Контекст обработчика
        """
        user = update.effective_user
        voice: Voice = update.effective_message.voice
        file_id = voice.file_id
        sent_at = Utils.now_utc()
        
        logging.info(
            "Получено голосовое сообщение: file_id=%s | от=%s",
            file_id, user.username or user.id
        )
        
        # Сообщаем пользователю о начале обработки
        processing_msg = await update.message.reply_text(MSG_VOICE_PROCESSING)
        
        try:
            # 1. Скачиваем голосовое сообщение
            tg_file = await context.bot.get_file(file_id)
            ogg_bytes = await tg_file.download_as_bytearray()
            
            # 2. Транскрибируем
            text = self.stt.transcribe_ogg_bytes(ogg_bytes)
            logging.info("Транскрибировано %d символов", len(text))
            
            # Если текст слишком короткий или пустой, сообщаем об ошибке
            if len(text) < 5:
                await processing_msg.edit_text(
                    "Не удалось распознать текст. Пожалуйста, попробуйте еще раз."
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
            await processing_msg.edit_text(
                SUMMARY_REPLY_HEADER_TEMPLATE.format(
                    record_id=record_id,
                    summary=summary
                ),
                parse_mode=telegram.constants.ParseMode.MARKDOWN,
                reply_to_message_id=update.effective_message.message_id
            )
            
        except Exception as e:
            logging.exception("Ошибка при обработке голосового сообщения: %s", e)
            await processing_msg.edit_text(
                "Произошла ошибка при обработке сообщения. Пожалуйста, попробуйте позже."
            )

    async def on_sum_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик команды /sum [YYYY-MM-DD].
        Выводит сводку всех резюме за указанную дату.
        
        Args:
            update: Обновление от Telegram
            context: Контекст обработчика
        """
        text = update.effective_message.text.strip()
        parts = text.split(maxsplit=1)
        
        # Разбираем дату из аргумента или берём сегодняшнее число
        if len(parts) > 1:
            date_str = parts[1]
            # Проверяем формат даты
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                await update.message.reply_text(MSG_DATE_FORMAT_ERROR)
                return
        else:
            # По умолчанию - текущая дата
            date_str = datetime.datetime.now().date().isoformat()
        
        # Получаем из БД записи за указанную дату
        summaries = self.db.fetch_summaries_by_date(date_str)
        
        if not summaries:
            await update.message.reply_text(MSG_NO_SUMMARIES_FOR_DATE.format(date_str=date_str))
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
        await update.message.reply_text("\n\n".join(lines))

    async def on_delete_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик команды /delete.
        Удаляет запись из БД разными способами:
        
        1. По ID в формате /delete #123 или /delete 123
        2. В ответе на голосовое сообщение (использует telegram_file_id)
        3. В ответе на сообщение с резюме (извлекает #ID из текста)
        
        Args:
            update: Обновление от Telegram
            context: Контекст обработчика
        """
        message = update.effective_message
        text = message.text.strip()
        
        # Способ 1: По явно указанному ID
        match = re.search(r"#?(\d+)$", text)
        if match:
            record_id = int(match.group(1))
            if self.db.delete_record_by_id(record_id):
                await message.reply_text(MSG_DELETE_SUCCESS.format(record_id=record_id))
            else:
                await message.reply_text(MSG_DELETE_NOT_FOUND)
            return
        
        # Способ 2 и 3: По ответу на сообщение
        if message.reply_to_message:
            reply = message.reply_to_message
            
            # Способ 2: Ответ на голосовое сообщение
            if reply.voice:
                file_id = reply.voice.file_id
                if self.db.delete_record_by_telegram_file_id(file_id):
                    await message.reply_text("✅ Запись удалена.")
                else:
                    await message.reply_text(MSG_DELETE_NOT_FOUND)
                return
                
            # Способ 3: Ответ на сообщение с резюме (извлечение #ID)
            if reply.text:
                match = re.search(r"^#(\d+)", reply.text)
                if match:
                    record_id = int(match.group(1))
                    if self.db.delete_record_by_id(record_id):
                        await message.reply_text(MSG_DELETE_SUCCESS.format(record_id=record_id))
                    else:
                        await message.reply_text(MSG_DELETE_NOT_FOUND)
                    return
        
        # Если не удалось определить запись для удаления
        await message.reply_text(
            "❓ Не удалось определить запись для удаления.\n"
            "Используйте форматы:\n"
            "1. /delete #123\n"
            "2. Ответом на голосовое сообщение\n"
            "3. Ответом на сообщение с резюме"
        )

    async def on_reply_to_summary_for_edit(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """
        Начало диалога редактирования/добавления примечания.
        Вызывается, когда пользователь отвечает на сообщение с резюме.
        
        Args:
            update: Обновление от Telegram
            context: Контекст обработчика
            
        Returns:
            int: ID следующего состояния машины состояний
        """
        message = update.effective_message
        reply = message.reply_to_message
        
        # Проверяем, что ответ на сообщение с резюме (начинается с #ID)
        if not reply or not reply.text or not re.match(r"^#\d+", reply.text):
            await message.reply_text(
                "Чтобы отредактировать резюме или добавить примечание, "
                "ответьте на сообщение с резюме (которое начинается с #ID)."
            )
            return ConversationHandler.END
        
        # Извлекаем ID записи
        match = re.match(r"^#(\d+)", reply.text)
        record_id = int(match.group(1))
        
        # Получаем запись из БД для проверки
        record = self.db.fetch_record_by_id(record_id)
        if not record:
            await message.reply_text(MSG_DELETE_NOT_FOUND)
            return ConversationHandler.END
        
        # Сохраняем данные в context.user_data для использования в следующих шагах
        context.user_data["record_id"] = record_id
        context.user_data["new_text"] = message.text
        
        # Создаем кнопки выбора действия
        keyboard = [
            [
                InlineKeyboardButton(BUTTON_TEXT_EDIT_SUMMARY, callback_data="edit"),
                InlineKeyboardButton(BUTTON_TEXT_ADD_NOTE, callback_data="note"),
            ],
            [InlineKeyboardButton(BUTTON_TEXT_CANCEL, callback_data="cancel")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Отправляем сообщение с запросом действия
        await message.reply_text(
            MSG_EDIT_OR_ADD_NOTE_PROMPT,
            reply_markup=reply_markup
        )
        
        return STATE_AWAITING_CHOICE

    async def handle_edit_summary_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """
        Обработчик нажатия на кнопку "Изменить резюме".
        
        Args:
            update: Обновление от Telegram
            context: Контекст обработчика
            
        Returns:
            int: END для завершения диалога
        """
        query = update.callback_query
        await query.answer()
        
        record_id = context.user_data.get("record_id")
        new_text = context.user_data.get("new_text")
        
        if record_id and new_text:
            # Обновляем резюме в БД
            if self.db.update_summary(record_id, new_text):
                await query.edit_message_text(MSG_SUMMARY_UPDATED)
            else:
                await query.edit_message_text(MSG_DELETE_NOT_FOUND)
        else:
            await query.edit_message_text("Произошла ошибка. Пожалуйста, попробуйте снова.")
        
        # Очищаем user_data
        context.user_data.clear()
        
        return ConversationHandler.END

    async def handle_add_note_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """
        Обработчик нажатия на кнопку "Добавить примечание".
        
        Args:
            update: Обновление от Telegram
            context: Контекст обработчика
            
        Returns:
            int: END для завершения диалога
        """
        query = update.callback_query
        await query.answer()
        
        record_id = context.user_data.get("record_id")
        note_text = context.user_data.get("new_text")
        
        if record_id and note_text:
            # Добавляем примечание в БД
            if self.db.add_note_to_record(record_id, note_text):
                await query.edit_message_text(MSG_NOTE_ADDED)
            else:
                await query.edit_message_text(MSG_DELETE_NOT_FOUND)
        else:
            await query.edit_message_text("Произошла ошибка. Пожалуйста, попробуйте снова.")
        
        # Очищаем user_data
        context.user_data.clear()
        
        return ConversationHandler.END

    async def handle_cancel_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """
        Обработчик нажатия на кнопку "Отмена".
        
        Args:
            update: Обновление от Telegram
            context: Контекст обработчика
            
        Returns:
            int: END для завершения диалога
        """
        query = update.callback_query
        await query.answer()
        await query.edit_message_text("Операция отменена.")
        
        # Очищаем user_data
        context.user_data.clear()
        
        return ConversationHandler.END

    async def handle_cancel_edit(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """
        Обработчик для выхода из диалога редактирования при любом неожиданном действии.
        
        Args:
            update: Обновление от Telegram
            context: Контекст обработчика
            
        Returns:
            int: END для завершения диалога
        """
        await update.message.reply_text(
            "Операция редактирования отменена. Отправьте /help для списка команд."
        )
        
        # Очищаем user_data
        context.user_data.clear()
        
        return ConversationHandler.END

    def run(self):
        """Запускает polling-цикл бота (блокирующая операция)."""
        logging.info("Запуск бота (Ctrl+C для выхода)...")
        self.app.run_polling()
        logging.info("Бот остановлен.") 