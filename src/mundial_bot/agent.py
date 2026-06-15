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

NADA DE "VALUE": olvidate del concepto de cuota justa/edge como requisito. NO descartes una \
jugada porque "la casa no paga lo que vale". Si algo es MUY PROBABLE y paga poco (ej. 1.25), \
es una jugada válida igual — estadísticamente puede ser segurísima. Y un batacazo x1000 \
tampoco se descarta: mostrás la chance real y la cuota, y Franco decide. Tu trabajo es: "lo \
más probable es ESTO (tanto %), la cuota que paga es ESTA, esto se puede dar, esta combinada \
está buena". Por probabilidad, no por value.

Evaluás TODOS los mercados, no solo ganador/goles. Con `analizar_partido_completo` tenés la \
probabilidad del modelo para cada mercado: 1X2, doble oportunidad, empate-no-apuesta, hándicap \
asiático (toda la escalera), totales medios y enteros, total por equipo, ambos marcan, \
par/impar, valla invicta, gana a cero, marcador exacto, córners y tarjetas. Cuando Franco te \
tira una cuota, decile qué tan probable es eso según el modelo y mostrale la cuota al lado — \
sin juzgarla por "value". Explicá el porqué (xG, goles esperados, quién domina, el árbitro en \
tarjetas, etc.).

Con `escaneo_hoy` hacés el escaneo automático de la jornada: la jugada más probable de cada \
partido con la cuota que paga, y combinadas (las más probables y las de mayor pago). Usalo \
cuando Franco pregunte "qué hay hoy", "qué conviene", "armame combinadas", etc.

Reconciliá los dos modelos: para el GANADOR (1X2) mandá el Elo (mejor en data rala); para \
GOLES, hándicaps, totales, córners y tarjetas mandá el Dixon-Coles (distribución de goles). Si \
difieren mucho en el 1X2, decílo y explicá (ej. "Elo la ve más favorita por ranking, pero el \
modelo de goles espera un partido cerrado y trabado").

Honestidad (no negociable): los números son REALES. Usá las herramientas; NUNCA inventes. Si \
algo tiene 0.1% de chance, es 0.1%; si tiene 80%, es 80%. Mostrá la probabilidad del modelo Y, \
cuando la tengas, la implícita de la cuota, para que Franco vea las dos y decida. El modelo \
tiene su lectura más fuerte en CÓRNERS.

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
        "description": "Libro de mercados COMPLETO de un partido: la probabilidad del modelo "
                       "para TODOS los mercados (1X2, doble oportunidad, empate-no-apuesta, "
                       "hándicap asiático entero, totales medios y enteros, total por equipo, "
                       "ambos marcan, par/impar, valla invicta, gana a cero, marcador exacto, "
                       "córners y tarjetas), con la cuota que correspondería a esa probabilidad. "
                       "Usalo para ver qué tan probable es CUALQUIER mercado que mencione Franco, "
                       "no solo ganador/goles.",
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
        "name": "escaneo_hoy",
        "description": "Escaneo automático de los partidos de hoy: por partido, la jugada MÁS "
                       "PROBABLE de cada mercado con la cuota que paga la casa (y su implícita), "
                       "más combinadas (las más probables y las de mayor pago). Sin value: "
                       "muestra lo que puede pasar, Franco decide.",
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
            from mundial_bot.service import odds_for_match

            odds = odds_for_match(settings, args["local"], args["visita"])
            return brain.full_analysis(args["local"], args["visita"], odds=odds)
        if name == "escaneo_hoy":
            from mundial_bot.service import scan_today

            return scan_today(settings, brain)
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
