from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver

from weather_app import (
    get_weather_by_city,
    get_coordinates,
    get_hourly_weather,
    get_daily_forecast_by_city,
)

# загружаем переменные из .env (OPENAI_API_KEY, OPENAI_MODEL, PROXY_BASE_URL)
load_dotenv()


# =========================
# SYSTEM PROMPT (на русском)
# =========================

SYSTEM_PROMPT = """
Ты — профессиональный AI-ассистент по погоде и времени.

Твоя задача:
- Понимать запрос пользователя о погоде (сейчас и на ближайшие дни) и/или текущем времени.
- При необходимости вызывать доступные tools.
- Отвечать строго в виде структурированного объекта (structured_response),
  который соответствует схеме AssistantResponse:
  - intent: краткое текстовое описание намерения пользователя
  - summary: краткий итоговый ответ для пользователя
  - location: город/место, к которому относится ответ (если применимо)
  - temperature_c: температура в градусах Цельсия (если применимо)
  - conditions: строка с описанием погодных условий (если применимо)
  - local_time: локальное время в ISO-формате (если применимо)
  - reasoning: краткое объяснение, как ты пришёл к ответу

Правила:
- Отвечай чётко и по делу, без шуток.
- При вопросах про «погоду у меня» или «погоду здесь» используй tool
  get_user_location для определения города по контексту.
- Для получения информации о текущей погоде по городу используй tool
  get_weather_for_location.
- Для получения прогноза на будущее (например, «завтра») используй tool
  get_forecast_for_location.
- Для определения текущего времени используй tool get_current_time.
- Соблюдай формат AssistantResponse даже если часть полей пустая (None).
"""


# =========================
# CONTEXT SCHEMA (dataclass)
# =========================


@dataclass
class AgentContext:
    """Контекст выполнения агента, доступный внутри tools через ToolRuntime."""

    user_id: str
    user_name: str = "Пользователь"
    # Добавляем страну, чтобы уменьшить риск перепутать город
    default_city: str = "Иркутск,ru"
    default_timezone: str = "Europe/Moscow"


# ==================================
# RESPONSE SCHEMA (structured output)
# ==================================


@dataclass
class AssistantResponse:
    """Структурированный ответ агента."""

    intent: str
    summary: str
    location: Optional[str] = None
    temperature_c: Optional[float] = None
    conditions: Optional[str] = None
    local_time: Optional[str] = None
    reasoning: str = ""


# =========
# TOOLS
# =========


@tool
def get_weather_for_location(city: str) -> str:
    """
    Получить краткую сводку погоды для заданного города.

    Параметры:
        city: Город, для которого нужно узнать погоду (строка).

    Возвращает:
        Текстовое описание погодных условий (температура, состояние).
    """
    city_norm = city.strip() or "неизвестный город"

    weather = get_weather_by_city(city_norm)
    if isinstance(weather, dict) and "error" in weather:
        return f"Не удалось получить погоду для {city_norm}: {weather['error']}"

    try:
        name = weather["name"]
        temp = weather["main"]["temp"]
        description = weather["weather"][0]["description"]
        humidity = weather["main"].get("humidity")
        wind_speed = weather.get("wind", {}).get("speed")

        parts = [f"Текущая погода в {name}: {temp:.1f}°C, {description}"]
        if humidity is not None:
            parts.append(f"влажность {humidity}%")
        if wind_speed is not None:
            parts.append(f"ветер {wind_speed} м/с")

        return ", ".join(parts) + "."
    except Exception as e:
        return f"Неожиданный формат ответа погоды для {city_norm}: {e}"


@tool
def get_forecast_for_location(city: str, day_offset: int = 1) -> str:
    """
    Получить примерный прогноз погоды для указанного города на N дней вперёд.

    Параметры:
        city: Город, для которого нужен прогноз (строка).
        day_offset: На сколько дней вперёд нужен прогноз:
            0 - сегодня, 1 - завтра, 2 - послезавтра и т.д.

    Возвращает:
        Краткое текстовое описание примерного прогноза погоды.
    """
    city_norm = city.strip() or "неизвестный город"

    # One Call 3.0 /onecall даёт прогноз на 8 дней вперёд в daily[0..7]
    if day_offset < 0:
        return "day_offset не может быть отрицательным."

    daily_data = get_daily_forecast_by_city(city_norm)
    if isinstance(daily_data, dict) and "error" in daily_data:
        return f"Не удалось получить ежедневный прогноз для {city_norm}: {daily_data['error']}"

    try:
        daily_list = daily_data.get("daily")
        if not daily_list:
            return f"Неожиданный формат ежедневного прогноза для {city_norm}."

        if day_offset >= len(daily_list):
            return (
                f"OpenWeatherMap вернул прогноз только на {len(daily_list)} дней. "
                f"day_offset={day_offset} выйти за доступный диапазон."
            )

        item = daily_list[day_offset]

        temp_block = item.get("temp") or {}
        temp_min = temp_block.get("min")
        temp_max = temp_block.get("max")
        temp_day = temp_block.get("day")

        weather_list = item.get("weather") or []
        description = weather_list[0].get("description") if weather_list else "нет описания"

        humidity = item.get("humidity")
        wind_speed = item.get("wind_speed")

        if day_offset == 0:
            day_label = "сегодня"
        elif day_offset == 1:
            day_label = "завтра"
        else:
            day_label = f"через {day_offset} дней"

        parts = [f"Прогноз для {city_norm} на {day_label}: {description}"]

        if temp_min is not None and temp_max is not None:
            parts.append(f"температура от {temp_min:.1f}°C до {temp_max:.1f}°C")
        elif temp_day is not None:
            parts.append(f"температура около {temp_day:.1f}°C")

        if humidity is not None:
            parts.append(f"влажность {humidity}%")
        if wind_speed is not None:
            parts.append(f"ветер {wind_speed} м/с")

        return ", ".join(parts) + " (по данным OpenWeatherMap One Call)."
    except Exception as e:
        return f"Не удалось обработать данные ежедневного прогноза для {city_norm}: {e}"


@tool
def get_user_location(runtime: ToolRuntime[AgentContext]) -> str:
    """
    Определить текущий город пользователя на основе контекста.

    Использует поля контекста (например, default_city),
    чтобы вернуть город, с которым по умолчанию следует работать.
    """
    ctx = runtime.context
    return ctx.default_city


@tool
def get_current_time(
    runtime: ToolRuntime[AgentContext],
    timezone: Optional[str] = None,
) -> str:
    """
    Получить текущее локальное время пользователя.

    Параметры:
        runtime: ToolRuntime с контекстом AgentContext.
        timezone: Не используется на Windows (оставлен для совместимости).

    Возвращает:
        Строку с локальным временем системы в ISO‑формате
        (YYYY‑MM‑DDTHH:MM:SS), а также человеко‑читаемое описание.
    """
    now = datetime.now()
    iso_str = now.replace(microsecond=0).isoformat()
    human = now.strftime("%d.%m.%Y %H:%M:%S")

    return f"Локальное системное время: {human} (ISO: {iso_str})"


# =====================
# МОДЕЛЬ И ПАМЯТЬ
# =====================


model = init_chat_model(
    os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("PROXY_API_KEY"),
    base_url=os.getenv("PROXY_BASE_URL"),
    temperature=0,
)

checkpointer = InMemorySaver()


# =====================
# СОЗДАНИЕ АГЕНТА
# =====================

agent = create_agent(
    model=model,
    tools=[get_weather_for_location, get_forecast_for_location, get_user_location, get_current_time],
    system_prompt=SYSTEM_PROMPT,
    context_schema=AgentContext,
    response_format=ToolStrategy(AssistantResponse),
    checkpointer=checkpointer,
)


# =====================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =====================

if __name__ == "__main__":
    thread_id = "demo-thread-1"
    config = {"configurable": {"thread_id": thread_id}}

    context = AgentContext(
        user_id="user-1",
        user_name="Иван",
        default_city="Иркутск,ru",
        default_timezone="Europe/Moscow",
    )

    result1 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Какая у меня сейчас погода и который час?",
                }
            ]
        },
        config=config,
        context=context,
    )

    print("=== Первый ответ: structured_response ===")
    print(result1["structured_response"])

    result2 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "А завтра будет холоднее?",
                }
            ]
        },
        config=config,
        context=context,
    )

    print("\n=== Второй ответ: structured_response ===")
    print(result2["structured_response"])

