"""Cerebro del bot: carga los modelos y responde consultas de partidos.

Centraliza lo que comparten el envío programado y el bot conversable:
  - cargar Elo + Dixon-Coles + córners + tarjetas,
  - traer los fixtures reales del día,
  - resolver nombres de equipos escritos por el usuario (incluso en español),
  - armar la predicción de un partido o de toda la jornada.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from datetime import date

from mundial_bot.collectors.statsbomb_stats import EVENTS_CACHE, load_events
from mundial_bot.collectors.team_stats import TEAM_STATS_CACHE, load_team_stats
from mundial_bot.config import Settings
from mundial_bot.models.cards_model import CardsModel
from mundial_bot.models.corners_model import CornersModel
from mundial_bot.pipeline import Models, build_models
from mundial_bot.report import build_match_report, format_match_report, format_match_reports
from mundial_bot.value.team_aliases import normalize_team

# Nombres en español → nombre del modelo (martj42, en inglés).
SPANISH_TEAMS: dict[str, str] = {
    "brasil": "Brazil", "mexico": "Mexico", "estados unidos": "United States",
    "eeuu": "United States", "corea del sur": "South Korea", "corea": "South Korea",
    "alemania": "Germany", "espana": "Spain", "francia": "France", "inglaterra": "England",
    "belgica": "Belgium", "paises bajos": "Netherlands", "holanda": "Netherlands",
    "japon": "Japan", "arabia saudita": "Saudi Arabia", "croacia": "Croatia",
    "suiza": "Switzerland", "marruecos": "Morocco", "turquia": "Turkey",
    "escocia": "Scotland", "tunez": "Tunisia", "egipto": "Egypt", "noruega": "Norway",
    "suecia": "Sweden", "catar": "Qatar", "dinamarca": "Denmark", "polonia": "Poland",
    "costa de marfil": "Ivory Coast", "cabo verde": "Cape Verde", "italia": "Italy",
}

HELP = (
    "👋 Soy el bot del Mundial. Escribime un partido y te digo a qué apostar:\n\n"
    "• <b>Argentina vs México</b>\n"
    "• <b>Brasil - Croacia</b>\n"
    "• <b>/hoy</b> → predicciones de todos los partidos de hoy\n"
    "• <b>/agenda</b> → qué se jugó, qué hay en vivo y qué falta (horario AR) 📅\n"
    "• <b>/balance</b> → cuánto vengo acertando 📊\n"
    "• <b>/clv</b> → ¿le gano al cierre del mercado? (si el bot es sharp) 📈\n"
    "• <b>/apuesta 5 2.10 Argentina gana</b> → registrá una apuesta tuya\n"
    "• <b>/roi</b> → tu ganancia y ROI real 💰\n"
    "• 📸 <b>Mandame una foto</b> de tu ticket o de las cuotas y te la leo/evalúo\n\n"
    "Te respondo con ganador, goles, córners y tarjetas más probables."
)


def _strip(s: str) -> str:
    """Minúsculas sin acentos, para comparar nombres de forma robusta."""
    norm = unicodedata.normalize("NFD", s.lower().strip())
    return "".join(c for c in norm if unicodedata.category(c) != "Mn")


def resolve_team(query: str, known: set[str]) -> str | None:
    """Resuelve lo que escribió el usuario a un equipo conocido por el modelo."""
    q = _strip(query)
    if not q:
        return None
    if q in SPANISH_TEAMS:
        return SPANISH_TEAMS[q]
    by_norm = {_strip(k): k for k in known}
    if q in by_norm:
        return by_norm[q]
    match = difflib.get_close_matches(q, list(by_norm), n=1, cutoff=0.78)
    return by_norm[match[0]] if match else None


_SEPARATOR = re.compile(r"\s+(?:vs?\.?|x|-|contra)\s+", re.IGNORECASE)


def parse_two_teams(text: str, known: set[str]) -> tuple[str, str] | None:
    """Extrae dos equipos del mensaje (ej. 'Argentina vs México')."""
    parts = _SEPARATOR.split(text.strip(), maxsplit=1)
    if len(parts) == 2:
        home, away = resolve_team(parts[0], known), resolve_team(parts[1], known)
        if home and away:
            return home, away
    return None


@dataclass
class BotBrain:
    models: Models
    corners: CornersModel | None
    cards: CardsModel | None

    @property
    def known(self) -> set[str]:
        return set(self.models.elo.ratings)

    def resolve(self, name: str) -> str:
        """Resuelve un nombre escrito (español, API-Football o con typo) al equipo del
        modelo. Las herramientas del agente reciben texto libre, así que hay que mapearlo
        igual que en el chat (si no, el modelo no lo reconoce y da un 50/50 inútil)."""
        hit = resolve_team(name, self.known)
        if hit:
            return hit
        norm = normalize_team(name)
        if norm in self.known:
            return norm
        return resolve_team(norm, self.known) or norm

    def predict_match(
        self, home: str, away: str, *, referee: str | None = None,
        knockout: bool = False, match_name: str | None = None,
    ) -> str:
        rh, ra = self.resolve(home), self.resolve(away)
        report = build_match_report(
            rh, ra,
            elo=self.models.elo, goals=self.models.goals,
            corners=self.corners, cards=self.cards,
            referee=referee, knockout=knockout, neutral=True,
            match_name=match_name or f"{rh} vs {ra}",
        )
        return format_match_report(report)

    def full_analysis(
        self, home: str, away: str, *, referee: str | None = None,
        knockout: bool = False, match_name: str | None = None, odds: dict | None = None,
    ) -> str:
        """Libro de mercados COMPLETO de un partido (todos los mercados).

        Es el cerebro matemático para que Claude lea cualquier mercado (1X2, hándicap
        asiático, totales, por equipo, ambos marcan, córners, tarjetas…). Si se pasan
        `odds`, muestra la CUOTA REAL de la casa al lado de la probabilidad del modelo.
        """
        from mundial_bot.models.market_book import build_market_book, format_market_book

        rh, ra = self.resolve(home), self.resolve(away)
        book = build_market_book(
            rh, ra,
            elo=self.models.elo, goals=self.models.goals,
            corners=self.corners, cards=self.cards,
            referee=referee, knockout=knockout, neutral=True,
            match_name=match_name or f"{rh} vs {ra}",
        )
        return format_market_book(book, odds=odds)

    def live_analysis(
        self, home: str, away: str, *, home_goals: int, away_goals: int,
        minute: float, match_name: str | None = None,
    ) -> str:
        """Análisis EN VIVO ajustado al marcador y minuto (resultado final desde ahora)."""
        if self.models.goals is None:
            return "(No tengo el modelo de goles cargado para analizar en vivo.)"
        from mundial_bot.models.live import live_analysis

        rh, ra = self.resolve(home), self.resolve(away)
        return live_analysis(
            self.models.goals, rh, ra,
            home_goals=home_goals, away_goals=away_goals, minute=minute,
            match_name=match_name or f"{rh} vs {ra}",
        )

    def combo_same_match(self, home: str, away: str, legs: list[dict]) -> str:
        """Combinada de patas del MISMO partido con probabilidad CONJUNTA (correlación)."""
        if self.models.goals is None:
            return "(No tengo el modelo de goles para la combinada.)"
        from mundial_bot.models.goals_model import GoalsModelError
        from mundial_bot.models.joint import joint_same_match

        rh, ra = self.resolve(home), self.resolve(away)
        try:
            res = joint_same_match(
                rh, ra, goals=self.models.goals,
                corners=self.corners, cards=self.cards, legs=legs,
            )
        except (GoalsModelError, ValueError) as exc:
            return f"(No pude calcular la combinada: {exc})"
        lines = [f"🎲 Combinada {rh} vs {ra} (patas del mismo partido):"]
        lines += [f"  • {d}: {p:.0%}" for d, p in res.legs]
        lines.append(
            f"CONJUNTA: <b>{res.combined_prob:.1%}</b> · cuota s/modelo "
            f"{res.fair_odds:.2f}"
        )
        lines.append(res.note)
        return "\n".join(lines)

    def handle_text(self, text: str) -> str:
        """Responde a un mensaje de texto libre."""
        teams = parse_two_teams(text or "", self.known)
        if teams:
            return self.predict_match(*teams)
        return (
            "🤔 No reconocí el partido. Probá así: <b>Argentina vs México</b>\n\n" + HELP
        )


def load_market_models() -> tuple[CornersModel | None, CardsModel | None]:
    """Carga córners/tarjetas: prefiere la forma reciente de API-Football, sino StatsBomb."""
    df = None
    if TEAM_STATS_CACHE.exists():
        df = load_team_stats()
    elif EVENTS_CACHE.exists():
        df = load_events(build_if_missing=False)
    if df is None or df.empty:
        return None, None
    return CornersModel.from_events(df), CardsModel.from_events(df)


def load_brain() -> BotBrain:
    """Entrena/carga todos los modelos en memoria (Elo autoalimentado con el Mundial)."""
    from mundial_bot.collectors.wc_results import load_wc_results

    models = build_models(extra_results=load_wc_results())
    corners, cards = load_market_models()
    return BotBrain(models=models, corners=corners, cards=cards)


def fetch_today_fixtures(settings: Settings) -> list:
    """Fixtures reales de hoy (football-data.org → API-Football)."""
    today = date.today().isoformat()
    if settings.has_football_data:
        try:
            from mundial_bot.collectors.fixtures_fdorg import FootballDataClient

            fixtures = FootballDataClient(settings.football_data_key).get_fixtures(date=today)
            if fixtures:
                return fixtures
        except Exception:  # noqa: BLE001
            pass
    if settings.has_api_football:
        try:
            from mundial_bot.collectors.fixtures import FixturesClient

            fixtures = FixturesClient(settings.api_football_key).get_fixtures(date=today)
            if fixtures:
                return fixtures
        except Exception:  # noqa: BLE001
            pass
    return []


def build_today_message(
    brain: BotBrain, settings: Settings, *, date_str: str, log: bool = True
) -> str:
    """Arma la cartilla de los partidos reales de hoy y (opcional) loguea las predicciones."""
    fixtures = fetch_today_fixtures(settings)
    if not fixtures:
        return f"🔮 <b>QUÉ APOSTAR HOY — {date_str}</b>\n\nHoy no hay partidos del Mundial. 🟢"

    from mundial_bot.tracking import PredictionStore

    today = date.today().isoformat()
    store = PredictionStore() if log else None
    reports = []
    try:
        for f in fixtures:
            report = build_match_report(
                normalize_team(f.home_team), normalize_team(f.away_team),
                elo=brain.models.elo, goals=brain.models.goals,
                corners=brain.corners, cards=brain.cards,
                referee=f.referee, knockout=f.knockout, neutral=True, match_name=f.match,
            )
            reports.append(report)
            if store is not None:
                store.log_report(f.fixture_id, report, pred_date=today, created_at=today)
    finally:
        if store is not None:
            store.close()
    return format_match_reports(reports, date_str=date_str)
