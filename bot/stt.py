"""
Модуль для преобразования голосовых сообщений в текст с использованием Vosk.
"""

import json
import logging
import tempfile
from pathlib import Path

import resampy
import soundfile as sf
from vosk import KaldiRecognizer, Model


class SpeechToText:
    """
    Обработчик преобразования речи в текст с использованием Vosk.
    
    Для замены бэкенда STT достаточно изменить только этот класс,
    сохранив сигнатуру публичных методов. Тогда остальная часть программы
    продолжит работать без изменений.
    """
    
    SAMPLE_RATE = 16_000  # Частота дискретизации для Vosk

    def __init__(self, model_path: str):
        """
        Инициализирует модель Vosk.
        
        Args:
            model_path: Путь к директории с моделью Vosk
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Модель Vosk не найдена: {model_path}")
        self.model = Model(str(model_path))
        logging.info("Загружена модель Vosk из %s", model_path)

    def transcribe_ogg_bytes(self, ogg_bytes: bytes) -> str:
        """
        Конвертирует OGG-аудио в WAV и транскрибирует его в текст.
        
        Args:
            ogg_bytes: Байты OGG/Opus файла
            
        Returns:
            str: Транскрибированный текст
        """
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as ogg:
            ogg.write(ogg_bytes)
            ogg_path = Path(ogg.name)
        
        wav_path = ogg_path.with_suffix(".wav")
        self._ogg_to_wav(ogg_path, wav_path)
        text = self._wav_to_text(wav_path)
        
        # Удаляем временные файлы
        ogg_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)
        
        return text

    def _ogg_to_wav(self, ogg_path: Path, wav_path: Path) -> None:
        """
        Конвертирует OGG/Opus → WAV 16 kHz mono PCM 16-bit
        с помощью soundfile (libsndfile) и resampy.
        
        Args:
            ogg_path: Путь к входному OGG файлу
            wav_path: Путь для выходного WAV файла
        """
        # 1. Читаем аудио (может быть стерео)
        data, sr = sf.read(str(ogg_path))  # data: np.ndarray, sr: исходная частота

        # 2. Приводим к моно, усредняя каналы
        if data.ndim > 1:
            data = data.mean(axis=1)

        # 3. Ресемплируем, если исходная частота != 16 kHz
        if sr != self.SAMPLE_RATE:
            data = resampy.resample(data, sr, self.SAMPLE_RATE)

        # 4. Записываем WAV с PCM 16-bit
        sf.write(str(wav_path), data, self.SAMPLE_RATE, subtype='PCM_16')
        logging.debug("Конвертирован файл %s в %s", ogg_path, wav_path)

    def _wav_to_text(self, wav_path: Path) -> str:
        """
        Читает WAV и возвращает распознанный текст.
        
        Args:
            wav_path: Путь к WAV файлу
            
        Returns:
            str: Распознанный текст
        """
        rec = KaldiRecognizer(self.model, self.SAMPLE_RATE)
        
        with open(wav_path, "rb") as wf:
            while chunk := wf.read(4000):
                rec.AcceptWaveform(chunk)
        
        result = json.loads(rec.FinalResult())
        transcript = result.get("text", "").strip()
        
        logging.debug("Транскрибирован файл %s: %d символов", wav_path, len(transcript))
        return transcript 