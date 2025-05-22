# VoiceMemo Assistant

Telegram-бот для работы с голосовыми сообщениями - транскрипция, суммаризация и удобное управление вашими аудиозаметками.

## Возможности

- **Транскрипция голосовых сообщений** с помощью Vosk (локальное распознавание речи)
- **Суммаризация текста** с использованием LLM (локальная Ollama или облачная OpenAI)
- **Интерактивное редактирование** резюме и добавление примечаний
- **Дневные сводки** всех ваших заметок
- **Удобное управление** - просмотр, редактирование, удаление заметок

## Установка

### Предварительные требования

- Python 3.10+
- [Vosk](https://alphacephei.com/vosk/) модель для русского языка
- [Ollama](https://github.com/ollama/ollama) (опционально, для локального LLM)
- Аккаунт [OpenAI](https://openai.com/) (опционально, для облачной LLM)

### Шаги установки

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/yourusername/voice_memo_assistant.git
   cd voice_memo_assistant
   ```

2. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Для Linux/Mac
   # или
   .\.venv\Scripts\activate  # Для Windows
   
   pip install -e .  # Установка проекта в режиме разработки
   ```

3. Скачайте модель Vosk:
   ```bash
   mkdir -p models_data
   # Скачайте модель с https://alphacephei.com/vosk/models 
   # (например, vosk-model-small-ru-0.22.zip)
   # и распакуйте в директорию models_data
   ```

4. Настройте конфигурацию:
   ```bash
   cp env.example .env
   # Отредактируйте .env, установив нужные параметры
   ```

## Конфигурация

Основные настройки в файле `.env`:

```
# Telegram
TG_BOT_TOKEN="your_bot_token_here"  # Получите у @BotFather

# База данных
DB_PATH="bot.db"  # Путь к файлу SQLite

# Настройки LLM
USE_LOCAL_LLM="True"  # True для Ollama, False для OpenAI
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_MODEL="mistral"  # Или другая поддерживаемая модель

# Для облачной LLM (когда USE_LOCAL_LLM="False")
OPENAI_API_KEY="your_openai_api_key_here"
OPENAI_MODEL="gpt-3.5-turbo"

# Speech-to-Text
VOSK_MODEL_PATH="models_data/vosk-small-ru-0.22"

# Прочее
LOG_PATH="logs/bot.log"
LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Запуск

```bash
# Активация виртуального окружения (если не активировано)
source .venv/bin/activate  # Для Linux/Mac
# или
.\.venv\Scripts\activate  # Для Windows

# Запуск бота
python -m bot.main
```

## Использование

1. Найдите вашего бота в Telegram и отправьте команду `/start`
2. Отправьте голосовое сообщение
3. Бот вернет его транскрипцию и краткое резюме
4. Используйте команды:
   - `/sum [YYYY-MM-DD]` - получить сводку за день
   - `/delete` в ответ на сообщение - удалить запись
   - Ответьте текстом на сообщение с резюме для редактирования или добавления примечания

## Структура проекта

```
voice_memo_assistant/
├── bot/                      # Основной пакет с кодом бота
│   ├── __init__.py           # Информация о версии
│   ├── main.py               # Точка входа
│   ├── config.py             # Загрузка конфигурации
│   ├── constants.py          # Строковые константы
│   ├── database.py           # Работа с БД
│   ├── llm.py                # Обработчик LLM
│   ├── stt.py                # Транскрипция
│   ├── telegram_handler.py   # Обработчики Telegram
│   ├── utils.py              # Вспомогательные функции
│   └── models/               # Модели данных
│       ├── __init__.py
│       └── schemas.py        # Pydantic модели
├── models_data/              # Директория для моделей STT
├── logs/                     # Директория для логов
├── .env                      # Переменные окружения
├── .env.example              # Пример файла .env
├── pyproject.toml            # Конфигурация проекта
└── requirements.txt          # Зависимости для pip
```

## Лицензия

MIT 