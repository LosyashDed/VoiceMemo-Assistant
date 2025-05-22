#!/bin/bash

# Скрипт для активации виртуального окружения Voice Memo Assistant
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Активация виртуального окружения для Voice Memo Assistant..."
source "$PROJECT_DIR/venv/bin/activate"

echo "Виртуальное окружение активировано!"
echo "Используйте 'pip list' чтобы просмотреть установленные пакеты"
echo "Для запуска проекта используйте команду 'voice-memo-bot'" 