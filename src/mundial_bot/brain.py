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
    "• <b>/balance</b> → cuánto vengo acertando 📊\n\n"
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

    def predict_match(
        self, home: str, away: str, *, referee: str | None = None,
        knockout: bool = False, match_name: str | None = None,
    ) -> str:
        report = build_match_report(
            normalize_team(home), normalize_team(away),
            elo=self.models.elo, goals=self.models.goals,
            corners=self.corners, cards=self.cards,
            referee=referee, knockout=knockout, neutral=True,
            match_name=match_name or f"{home} vs {away}",
        )
        return format_match_report(report)

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
