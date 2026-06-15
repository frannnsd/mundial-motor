"""Agente conversable: Claude + herramientas que consultan el cerebro del bot.

Franco le habla natural por Telegram y Claude responde como un analista experto en
apuestas, usando los números REALES del modelo (vía herramientas). Sin sermones, sin
manejarle la plata, pensando fuera de la caja — pero honesto con los números.
"""

from __future__ import annotations

import base64
from datetime import datetime

from mundial_bot.brain import BotBrain, build_today_message
from mundial_bot.config import Settings

MODEL = "claude-sonnet-4-6"
MAX_TOOL_LOOPS = 6
MAX_HISTORY_MSGS = 10  # memoria corta: últimos ~5 turnos de texto (sin scaffolding de tools)

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

Honestidad (NO NEGOCIABLE, candados duros):
- SIEMPRE llamá a una herramienta antes de hablar de un partido, una cuota o una apuesta. Si \
no llamaste a ninguna, NO tenés los números: no inventes.
- NUNCA inventes marcadores, resultados ni datos EN VIVO. NO tenés feed en vivo. Tus números \
son PRE-PARTIDO. Si el partido está en juego, podés ver el estado y el marcador con \
`agenda_partidos`, y aclarás que tu análisis es pre-partido (no ajustado al resultado actual).
- Los 48 equipos del Mundial están en el modelo. NUNCA digas que un equipo "no está" sin haber \
probado la herramienta. Solo si la herramienta te devuelve "NO ENCONTRÉ", recién ahí pedile a \
Franco el nombre exacto. No te contradigas.
- Los números son REALES: si algo tiene 0.1%, es 0.1%; si tiene 80%, es 80%. Mostrá la \
probabilidad del modelo Y, cuando la tengas, la implícita de la cuota, para que Franco vea las \
dos y decida. NO hables de "cuota justa": no es value, es la chance y lo que paga. El modelo \
tiene su lectura más fuerte en CÓRNERS.
- Para una combinada de patas del MISMO partido, multiplicás las probabilidades como \
aproximación pero aclarás que están correlacionadas (no son del todo independientes).

Agenda: con `agenda_partidos` ves qué partidos ya se jugaron (con resultado), cuáles están \
EN VIVO y cuáles faltan (con horario local de Argentina). Usala cuando Franco pregunte por \
fechas, horarios, "qué se jugó", "qué falta" o "qué hay hoy/mañana".

Imágenes: Franco te puede mandar una FOTO (ticket de apuesta, cuotas, captura). Leéla, \
sacá los equipos / mercados / cuotas que veas y evaluala con tus herramientas (resolvé los \
nombres y traé las probabilidades reales). Si algo de la foto no se entiende, decíselo.

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


def _resolve_or_missing(brain: BotBrain, local: str, visita: str) -> tuple[str, str] | str:
    """Resuelve los dos equipos; si alguno no está en el modelo, devuelve un aviso claro
    (en vez de un 50/50 inútil que haría confabular al agente)."""
    rl, rv = brain.resolve(local), brain.resolve(visita)
    missing = [orig for orig, res in ((local, rl), (visita, rv)) if res not in brain.known]
    if missing:
        return (
            f"(NO ENCONTRÉ a {', '.join(missing)} en el modelo del Mundial. NO inventes datos: "
            f"pedile a Franco el nombre exacto del equipo según el fixture.)"
        )
    return rl, rv


def _run_tool(name: str, args: dict, settings: Settings, brain: BotBrain) -> str:
    """Ejecuta una herramienta y devuelve su resultado como texto."""
    try:
        if name == "predecir_partido":
            resolved = _resolve_or_missing(brain, args["local"], args["visita"])
            if isinstance(resolved, str):
                return resolved
            return brain.predict_match(*resolved)
        if name == "analizar_partido_completo":
            from mundial_bot.service import odds_for_match

            resolved = _resolve_or_missing(brain, args["local"], args["visita"])
            if isinstance(resolved, str):
                return resolved
            local, visita = resolved
            odds = odds_for_match(settings, local, visita)
            return brain.full_analysis(local, visita, odds=odds)
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


_IMAGE_PROMPT = (
    "Franco te mandó una imagen (puede ser un ticket de apuesta, una cuota o una captura). "
    "Leéla: identificá equipos, mercados y cuotas, y evaluámela con tus herramientas "
    "(resolvé los nombres y sacá las probabilidades reales del modelo). Si no se entiende "
    "algo, decíselo."
)


def ask_agent(
    text: str, *, settings: Settings, brain: BotBrain, history: list[dict] | None = None,
    image: tuple[bytes, str] | None = None,
) -> str:
    """Manda el mensaje a Claude (con herramientas) y devuelve la respuesta en texto plano.

    `image` opcional = (bytes, media_type) de una foto que mandó Franco (ej. un ticket).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    messages: list[dict] = list(history or [])

    if image is not None:
        img_bytes, media_type = image
        b64 = base64.standard_b64encode(img_bytes).decode("ascii")
        caption = text.strip() or _IMAGE_PROMPT
        messages.append({"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": caption},
        ]})
        hist_text = f"[imagen] {text}".strip()   # en el historial no reenviamos la foto
    else:
        messages.append({"role": "user", "content": text})
        hist_text = text

    for _ in range(MAX_TOOL_LOOPS):
        resp = client.messages.create(
            model=MODEL, system=SYSTEM, tools=TOOLS, max_tokens=1500, messages=messages,
        )
        if resp.stop_reason != "tool_use":
            answer = "".join(b.text for b in resp.content if b.type == "text").strip()
            answer = answer or "No te entendí, dale de nuevo."
            if history is not None:
                # Guardamos SOLO turnos de texto limpios (sin los bloques tool_use/
                # tool_result internos ni la imagen): si no, podar el historial puede
                # cortar un par a la mitad y la API tira 400 (tool_result sin tool_use).
                history.append({"role": "user", "content": hist_text or "(imagen)"})
                history.append({"role": "assistant", "content": answer})
                if len(history) > MAX_HISTORY_MSGS:
                    del history[: len(history) - MAX_HISTORY_MSGS]
            return answer

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
