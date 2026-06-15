"""Agente conversable: Claude + herramientas que consultan el cerebro del bot.

Franco le habla natural por Telegram y Claude responde como un analista experto en
apuestas, usando los números REALES del modelo (vía herramientas). Sin sermones, sin
manejarle la plata, pensando fuera de la caja — pero honesto con los números.
"""

from __future__ import annotations

from datetime import datetime

from mundial_bot.brain import BotBrain, build_today_message
from mundial_bot.config import Settings

MODEL = "claude-sonnet-4-6"
MAX_TOOL_LOOPS = 6

SYSTEM = """Sos "Apu", analista EXPERTO en apuestas deportivas, especializado en el \
Mundial 2026. Hablás con Franco por Telegram, en español argentino, directo y canchero.

SOS UN CEREBRO UNIFICADO: tu razonamiento (Claude) + el modelo matemático del bot. El modelo \
te da los NÚMEROS de todos los mercados; vos los interpretás, los reconciliás y explicás el \
PORQUÉ de cada chance. Juntos dan el panorama completo del partido.

Reglas de tu personalidad:
- Franco es el dueño de la plata y el que decide. NO le manejes el dinero, NO le digas \
cuánto apostar, NO le adviertas que "es arriesgado" ni le des sermones — él ya lo sabe.
- Pensá FUERA DE LA CAJA. Una combinada x1000 no es una locura de entrada: evaluala con \
números. Si tiene chance real, decíselo; si es mínima, también, pero sin descartarla por prejuicio.
- Sos confiado y directo: "esta cuota está buena", "esta combinada vale la pena", "yo iría \
por acá". Nada de hedging.

Evaluás TODOS los mercados, no solo ganador/goles. Con `analizar_partido_completo` tenés la \
CUOTA JUSTA del modelo para cada mercado: 1X2, doble oportunidad, empate-no-apuesta, hándicap \
asiático (toda la escalera), totales medios y enteros, total por equipo, ambos marcan, \
par/impar, valla invicta, gana a cero, marcador exacto, córners y tarjetas. Cuando Franco te \
tira una cuota de cualquiera de esos mercados, comparala contra la justa: si la casa paga MÁS \
que la justa, la cuota está buena; si paga menos, no. Y explicá por qué el modelo la ve así \
(xG, goles esperados, quién domina, el árbitro en tarjetas, etc.).

Reconciliá los dos modelos: para el GANADOR (1X2) mandá el Elo (mejor en data rala); para \
GOLES, hándicaps, totales, córners y tarjetas mandá el Dixon-Coles (distribución de goles). Si \
difieren mucho en el 1X2, decílo y explicá (ej. "Elo la ve más favorita por ranking, pero el \
modelo de goles espera un partido cerrado y trabado").

Honestidad (no negociable): los números son REALES. Usá las herramientas; NUNCA inventes. Si \
algo tiene 0.1% de chance, es 0.1% — pero evaluá si la cuota lo paga bien.

Criterio de experto: el mercado (muchas casas + Pinnacle) casi siempre tiene razón. Si el \
modelo difiere MUCHÍSIMO de la cuota (ej. modelo 57% vs casa 18%), es error del modelo, no \
value — descartalo. Los edges reales son chicos. El modelo tiene su edge más fuerte en CÓRNERS.

Agenda: con `agenda_partidos` ves qué partidos ya se jugaron (con resultado), cuáles están \
EN VIVO y cuáles faltan (con horario local de Argentina). Usala cuando Franco pregunte por \
fechas, horarios, "qué se jugó", "qué falta" o "qué hay hoy/mañana".

Usá las herramientas para responder con datos reales. Respondé en TEXTO PLANO (sin HTML), \
conciso y con onda."""

TOOLS = [
    {
        "name": "predecir_partido",
        "description": "Predice un partido: ganador, goles, córners, tarjetas y ambos marcan, "
                       "con las probabilidades del modelo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "local": {"type": "string", "description": "equipo local"},
                "visita": {"type": "string", "description": "equipo visitante"},
            },
            "required": ["local", "visita"],
        },
    },
    {
        "name": "analizar_partido_completo",
        "description": "Libro de mercados COMPLETO de un partido: la cuota JUSTA del modelo "
                       "para TODOS los mercados (1X2, doble oportunidad, empate-no-apuesta, "
                       "hándicap asiático entero, totales medios y enteros, total por equipo, "
                       "ambos marcan, par/impar, valla invicta, gana a cero, marcador exacto, "
                       "córners y tarjetas). Usalo para juzgar CUALQUIER cuota que mencione "
                       "Franco, no solo ganador/goles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "local": {"type": "string", "description": "equipo local"},
                "visita": {"type": "string", "description": "equipo visitante"},
            },
            "required": ["local", "visita"],
        },
    },
    {
        "name": "cuotas_buenas_hoy",
        "description": "Evalúa los partidos de hoy contra el mercado real (todas las casas) y "
                       "devuelve las cuotas BUENAS (value) y las combinadas que valen.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "partidos_de_hoy",
        "description": "Predicciones de todos los partidos de hoy del Mundial.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "mi_balance",
        "description": "Cuánto viene acertando el modelo (aciertos por mercado + Brier).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "mi_roi",
        "description": "El ROI real de las apuestas que registró Franco.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "agenda_partidos",
        "description": "Agenda del Mundial: qué partidos ya se jugaron (con resultado), cuáles "
                       "están EN VIVO y cuáles faltan (con horario local de Argentina). Permite "
                       "elegir cuántos días para atrás y para adelante.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dias_atras": {"type": "integer", "description": "días hacia atrás (default 1)"},
                "dias_adelante": {
                    "type": "integer", "description": "días hacia adelante (default 4)"
                },
            },
        },
    },
]


def _run_tool(name: str, args: dict, settings: Settings, brain: BotBrain) -> str:
    """Ejecuta una herramienta y devuelve su resultado como texto."""
    try:
        if name == "predecir_partido":
            return brain.predict_match(args["local"], args["visita"])
        if name == "analizar_partido_completo":
            return brain.full_analysis(args["local"], args["visita"])
        if name == "cuotas_buenas_hoy":
            from mundial_bot.service import evaluate_today

            return evaluate_today(settings, brain, min_ev=0.02)
        if name == "partidos_de_hoy":
            return build_today_message(
                brain, settings, date_str=datetime.now().strftime("%d/%m/%Y"), log=False
            )
        if name == "mi_balance":
            from mundial_bot.tracking import PredictionStore, format_balance, grade_pending

            if settings.has_api_football:
                grade_pending(settings.api_football_key)
            with PredictionStore() as store:
                return format_balance(store.balance())
        if name == "mi_roi":
            from mundial_bot.betlog import BetStore, format_roi

            with BetStore() as store:
                return format_roi(store.summary())
        if name == "agenda_partidos":
            from mundial_bot.service import format_schedule, get_schedule

            fixtures = get_schedule(
                settings,
                days_back=int(args.get("dias_atras", 1)),
                days_ahead=int(args.get("dias_adelante", 4)),
            )
            return format_schedule(
                fixtures,
                tz_name=settings.timezone,
                date_str=datetime.now().strftime("%d/%m/%Y"),
            )
    except Exception as exc:  # noqa: BLE001
        return f"(error ejecutando {name}: {exc})"
    return f"(herramienta desconocida: {name})"


def ask_agent(
    text: str, *, settings: Settings, brain: BotBrain, history: list[dict] | None = None
) -> str:
    """Manda el mensaje a Claude (con herramientas) y devuelve la respuesta en texto plano."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    messages: list[dict] = list(history or [])
    messages.append({"role": "user", "content": text})

    for _ in range(MAX_TOOL_LOOPS):
        resp = client.messages.create(
            model=MODEL, system=SYSTEM, tools=TOOLS, max_tokens=1500, messages=messages,
        )
        if resp.stop_reason != "tool_use":
            answer = "".join(b.text for b in resp.content if b.type == "text").strip()
            messages.append({"role": "assistant", "content": answer})
            if history is not None:
                history.clear()
                history.extend(messages[-10:])  # memoria corta para charlar
            return answer or "No te entendí, dale de nuevo."

        assistant_content = []
        tool_results = []
        for block in resp.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input,
                })
                result = _run_tool(block.name, block.input, settings, brain)
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id, "content": result,
                })
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    return "Me colgué procesando, probá de nuevo. 🤔"
