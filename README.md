# LangChain / LangGraph Weather Agent

Простой пример агента на LangChain + LangGraph, который:
- использует инструменты (`@tool`) с `ToolRuntime` и контекстом
- работает с памятью диалога через `InMemorySaver` и `thread_id`
- отдаёт ответ в виде структурированного `dataclass`
- получает **реальные** данные погоды из OpenWeatherMap (текущее состояние + прогноз)

## Стек

- Python 3.12+ (у тебя сейчас 3.13)
- `langchain`
- `langgraph`
- `langchain-openai` (LLM через OpenAI / прокси)
- `python-dotenv`
- `requests`

## Установка

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## Переменные окружения

Файл `.env` в корне проекта, пример:

```env
PROXY_API_KEY=sk-...              # ключ для OpenAI-прокси
PROXY_BASE_URL=https://openai.api.proxyapi.ru/v1
OPENAI_MODEL=gpt-4o-mini

API_KEY=your_openweathermap_key   # ключ OpenWeatherMap (Free tier)
```

- `PROXY_API_KEY`, `PROXY_BASE_URL`, `OPENAI_MODEL` — для LLM через `init_chat_model`.
- `API_KEY` — для OpenWeatherMap (используется в `weather_app.py`).

## Основные модули

### `agent_weather_time.py`

Содержит:

- `SYSTEM_PROMPT` (RU) — инструкции ассистенту:
  - работать как погодный/временной ассистент
  - zвать tools только по задаче
  - возвращать строго `AssistantResponse`

- `AgentContext` (`dataclass`) — контекст для `ToolRuntime`:
  - `user_id`
  - `user_name`
  - `default_city` (например, `"Иркутск,ru"`)
  - `default_timezone` (сейчас не используется в логике времени)

- `AssistantResponse` (`dataclass`) — схема structured output:
  - `intent: str`
  - `summary: str`
  - `location: Optional[str]`
  - `temperature_c: Optional[float]`
  - `conditions: Optional[str]`
  - `local_time: Optional[str]`
  - `reasoning: str`

- TOOLS:
  - `get_weather_for_location(city: str)`  
    Реальный current weather через `weather_app.get_weather_by_city()`, собирает
    человекочитаемую строку с температурой, описанием, влажностью, ветром.

  - `get_forecast_for_location(city: str, day_offset: int = 1)`  
    Реальный прогноз на `day_offset` дней вперёд:
    - использует `weather_app.get_daily_forecast_by_city()`
    - работает поверх `/data/2.5/forecast` (бесплатный 5‑day / 3‑hour API)
    - агрегирует по локальным датам города (с учётом `timezone` из ответа)
    - возвращает описание + min/max/avg температуру, влажность, ветер.

  - `get_user_location(runtime: ToolRuntime[AgentContext])`  
    Возвращает `runtime.context.default_city` — так агент узнаёт “где пользователь”.

  - `get_current_time(runtime: ToolRuntime[AgentContext], timezone: Optional[str])`  
    Сейчас просто возвращает локальное системное время машины (без tz‑магии,
    чтобы надёжно работать под Windows).

- Модель:

```python
model = init_chat_model(
    os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("PROXY_API_KEY"),
    base_url=os.getenv("PROXY_BASE_URL"),
    temperature=0,
)
```

- Память:

```python
checkpointer = InMemorySaver()
```

- Агент:

```python
agent = create_agent(
    model=model,
    tools=[get_weather_for_location, get_forecast_for_location, get_user_location, get_current_time],
    system_prompt=SYSTEM_PROMPT,
    context_schema=AgentContext,
    response_format=ToolStrategy(AssistantResponse),
    checkpointer=checkpointer,
)
```

### `weather_app.py`

Утилиты поверх OpenWeatherMap с кэшем:

- Кэш в файловой системе (`.cache/`), TTL по умолчанию 10 минут.
- `get_coordinates(city: str) -> tuple[lat, lon]`
- `get_weather_by_city(city: str) -> dict` — `/data/2.5/weather`
- `get_current_weather(...)` — тонкая обёртка вокруг текущей погоды.
- `get_hourly_weather(lat, lon) -> dict` — Pro endpoint (можно не использовать).
- `get_daily_forecast(lat, lon) -> dict`  
  **Главное**: реализовано через `/data/2.5/forecast` (5‑day / 3‑hour),
  агрегация по локальной дате:
  - группировка по дате (`dt` + `timezone` оффсет из `city.timezone`)
  - вычисление `min`, `max`, `avg(temp)`, средней влажности, ветра
  - выбор самой частой формулировки `description`.

- `get_daily_forecast_by_city(city: str) -> dict`  
  Берёт координаты + вызывает `get_daily_forecast`.

- `get_air_pollution(...)` + `analize_air_pollution(...)` — пример анализа качества воздуха (не подключен в агента, но можно использовать как отдельный tool).

### `http_client.py`

Простой `GET` с ретраями:

- `get_with_retries(url, retries=3, backoff_seconds=1.0, timeout=10.0) -> Response | None`
- Используется в `weather_app.py` для всех запросов к OpenWeatherMap.

## Как запустить пример агента

Из корня проекта:

```bash
venv\Scripts\activate
python agent_weather_time.py
```

Скрипт:
- создаёт контекст с `default_city="Иркутск,ru"`
- делает 2 вызова `agent.invoke(...)` с одним `thread_id`:
  1. `"Какая у меня сейчас погода и который час?"`
  2. `"А завтра будет холоднее?"`

И печатает `structured_response` для каждого запроса.

## Как использовать агента из другого кода

```python
from agent_weather_time import agent, AgentContext

config = {"configurable": {"thread_id": "my-thread"}}
context = AgentContext(
    user_id="user-42",
    user_name="Павел",
    default_city="Москва,ru",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Какая погода завтра?"}]},
    config=config,
    context=context,
)

structured = result["structured_response"]
print(structured.summary)
```

## Ограничения / заметки

- OpenWeather свободно даёт только:
  - текущую погоду (`/data/2.5/weather`)
  - 5‑дневный прогноз с шагом 3 часа (`/data/2.5/forecast`)
- One Call 3.0 и Pro‑hourly могут требовать платный тариф — в текущем проекте для дневного прогноза используется только бесплатный `/data/2.5/forecast`.
- Время (`get_current_time`) сейчас завязано на локальное системное время машины, без таймзон — это сделано для простоты и надёжности под Windows. Если нужно — легко перевести на `zoneinfo` + `tzdata`.

