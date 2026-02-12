import time
from typing import Optional

import requests


def get_with_retries(
    url: str,
    *,
    retries: int = 3,
    backoff_seconds: float = 1.0,
    timeout: float = 10.0,
) -> Optional[requests.Response]:
    """
    Простой HTTP-клиент с повторными попытками для GET-запросов.

    Параметры:
        url: Полный URL.
        retries: Количество попыток при временных ошибках сети/сервера.
        backoff_seconds: Базовая пауза между попытками (линейный backoff).
        timeout: Таймаут одного запроса в секундах.

    Возвращает:
        requests.Response при успехе или None при фатальной ошибке.
    """
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            # Если сервер ответил, даже с 4xx/5xx, возвращаем как есть –
            # разбор ошибок делается на уровне вызывающего кода.
            return resp
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds * attempt)

    # Если сюда дошли – все попытки упали.
    return None

