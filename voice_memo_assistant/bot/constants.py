"""
Строковые константы, тексты сообщений и шаблоны промптов.
"""

# Шаблоны промптов для LLM
PROMPT_SUMMARY_TEMPLATE = """
Сделай краткое резюме объёмом от {min_len} до {max_len} символов 
для следующего голосового сообщения:

{text}
"""

# Сообщения для пользователей
MSG_START = """
Привет! Я VoiceMemo Assistant - твой помощник для работы с голосовыми сообщениями.

Отправь мне голосовое сообщение, и я:
1️⃣ Преобразую его в текст
2️⃣ Создам краткое резюме
3️⃣ Сохраню для дальнейшего использования

📝 Команды:
/sum [YYYY-MM-DD] - получить сводку резюме за указанную дату
/delete - удалить запись (отправь в ответ на сообщение с резюме)
"""

MSG_HELP = """
📝 Доступные команды:
/sum [YYYY-MM-DD] - получить сводку резюме за указанную дату (по умолчанию - сегодня)
/delete - удалить запись (отправь в ответ на сообщение с резюме)

Чтобы отредактировать резюме или добавить примечание, просто ответь на сообщение с резюме своим вариантом текста.
"""

MSG_DATE_FORMAT_ERROR = "❗️ Неверный формат даты. Используйте YYYY-MM-DD"
MSG_NO_SUMMARIES_FOR_DATE = "ℹ️ Нет резюме за {date_str}."
MSG_DELETE_SUCCESS = "✅ Запись #{record_id} удалена."
MSG_DELETE_NOT_FOUND = "❓ Запись не найдена."
MSG_EDIT_OR_ADD_NOTE_PROMPT = "Что вы хотите сделать с текстом?"
MSG_SUMMARY_UPDATED = "✅ Резюме обновлено."
MSG_NOTE_ADDED = "✅ Примечание добавлено."
MSG_VOICE_PROCESSING = "🔄 Обрабатываю голосовое сообщение..."

# Текст для кнопок
BUTTON_TEXT_EDIT_SUMMARY = "Изменить резюме"
BUTTON_TEXT_ADD_NOTE = "Добавить примечание"
BUTTON_TEXT_CANCEL = "Отмена"

# Шаблоны для форматирования
SUMMARY_REPLY_HEADER_TEMPLATE = "#{record_id}\n📌 Резюме:\n```\n{summary}\n```"
SUMMARY_FOR_DATE_LINE_TEMPLATE = "{sent_at} — #{record_id}: {summary}"
SUMMARY_FOR_DATE_NOTE_TEMPLATE = " └ Примечание: {note}"

# ID для машины состояний ConversationHandler
STATE_AWAITING_CHOICE = 1 