"""
Модуль для работы с языковыми моделями (LLM).
Поддерживает локальную Ollama и облачную OpenAI.
"""

import json
import logging
from typing import Dict, Optional, Union

import requests
from .constants import PROMPT_SUMMARY_TEMPLATE

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    

class LLMHandler:
    """
    Обертка для работы с языковыми моделями.
    Скрывает различия между локальной Ollama и облачной OpenAI.
    """

    def __init__(self, use_local: bool, config: Dict[str, Union[str, bool, int]]):
        """
        Инициализирует обработчик LLM.
        
        Args:
            use_local: True для использования локальной модели Ollama,
                       False для использования облачной OpenAI
            config: Словарь с конфигурационными параметрами
        """
        self.use_local = use_local
        self.config = config
        
        if not use_local:
            if not OPENAI_AVAILABLE:
                raise RuntimeError(
                    "Пакет openai не установлен, но требуется для USE_LOCAL_LLM=False"
                )
            openai.api_key = str(config["OPENAI_API_KEY"])
        
        logging.info(
            "LLM Handler настроен на %s модель",
            "локальную Ollama" if use_local else "облачную OpenAI"
        )

    def summarize(self, text: str) -> str:
        """
        Генерирует резюме текста с учетом ограничений длины.
        
        При значительном отклонении длины результата от заданного диапазона
        делается повторная попытка.
        
        Args:
            text: Исходный текст для суммаризации
            
        Returns:
            str: Сгенерированное резюме
        """
        # Получаем параметры из конфигурации
        deviation_percent = int(self.config.get("SUMMARY_DEVIATION_PERCENT", 20))
        max_tries = max(1, int(self.config.get("SUMMARY_MAX_TRIES", 2)))
        
        # Определяем желаемую длину резюме в зависимости от длины исходного текста
        n = len(text)
        if n < 500:
            min_len, max_len = 150, 250
        else:
            min_len, max_len = max(150, n // 5), n // 3  # соотношения 5:1 и 3:1
        
        # Рассчитываем допуски
        min_allowed = min_len * (1 - (deviation_percent / 100))
        max_allowed = max_len * (1 + (deviation_percent / 100))
        
        # Функция для генерации резюме с выбранным бэкендом
        def _generate() -> str:
            prompt = PROMPT_SUMMARY_TEMPLATE.format(
                min_len=min_len,
                max_len=max_len,
                text=text
            )
            return (
                self._ollama_call(prompt)
                if self.use_local
                else self._openai_call(prompt)
            ).strip()
        
        # Делаем попытки генерации
        for attempt in range(max_tries):
            summary = _generate()
            length = len(summary)
            
            if min_allowed <= length <= max_allowed:
                # Результат в пределах допустимого отклонения
                break
                
            if attempt == max_tries - 1:
                # Последняя попытка, примем как есть
                logging.warning(
                    "Длина резюме %d вне диапазона (%d-%d) даже после %d попыток",
                    length, min_len, max_len, max_tries
                )

        return summary

    def _ollama_call(self, prompt: str) -> str:
        """
        Вызывает локальную модель Ollama.
        
        Args:
            prompt: Промт для модели
            
        Returns:
            str: Ответ модели
        """
        url = f"{self.config['OLLAMA_BASE_URL']}/api/generate"
        payload = {
            "model": self.config["OLLAMA_MODEL"],
            "prompt": prompt,
            "stream": False
        }
        
        logging.debug("Запрос к Ollama: %s", json.dumps(payload)[:200])
        
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            summary = resp.json().get("response", "").strip()
            logging.info("Ollama: резюме получено, длина=%d", len(summary))
            return summary
        except requests.exceptions.RequestException as e:
            logging.error("Ошибка при обращении к Ollama: %s", e)
            return "Ошибка при генерации резюме. Пожалуйста, попробуйте позже."

    def _openai_call(self, prompt: str) -> str:
        """
        Вызывает облачную модель OpenAI.
        
        Args:
            prompt: Промт для модели
            
        Returns:
            str: Ответ модели
        """
        logging.debug("Запрос к OpenAI: длина=%d", len(prompt))
        
        try:
            response = openai.ChatCompletion.create(
                model=self.config["OPENAI_MODEL"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            summary = response.choices[0].message.content.strip()
            logging.info("OpenAI: резюме получено, длина=%d", len(summary))
            return summary
        except Exception as e:
            logging.error("Ошибка при обращении к OpenAI: %s", e)
            return "Ошибка при генерации резюме. Пожалуйста, попробуйте позже." 