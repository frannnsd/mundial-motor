"""Test del agente: la memoria de la charla no debe romper la API de Claude.

Regresión del bug 400 'unexpected tool_use_id': el historial guardaba los bloques
tool_use/tool_result y al podarlo cortaba un par a la mitad. Debe guardar SOLO texto.
"""

from __future__ import annotations

import anthropic

import mundial_bot.agent as agent


class _Blk:
    def __init__(self, type, text=None, id=None, name=None, input=None):
        self.type = type
        self.text = text
        self.id = id
        self.name = name
        self.input = input


class _Resp:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    def __init__(self, script):
        self.script = script
        self.seen: list[list] = []

    def create(self, **kw):
        self.seen.append(kw["messages"])
        return self.script.pop(0)


class _FakeClient:
    def __init__(self, script):
        self.messages = _FakeMessages(script)


class _Settings:
    anthropic_api_key = "x"


def test_history_solo_guarda_texto_no_scaffolding_de_tools(monkeypatch):
    # 1ª respuesta: usa una herramienta · 2ª: texto final.
    script = [
        _Resp("tool_use", [
            _Blk("tool_use", id="t1", name="predecir_partido", input={"local": "A", "visita": "B"}),
        ]),
        _Resp("end_turn", [_Blk("text", text="Listo, crack")]),
    ]
    fake = _FakeClient(script)
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: fake)
    monkeypatch.setattr(agent, "_run_tool", lambda *a, **k: "resultado de la tool")

    history: list[dict] = []
    out = agent.ask_agent("hola", settings=_Settings(), brain=None, history=history)

    assert out == "Listo, crack"
    # El historial guardado debe ser SOLO turnos de texto limpios.
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert all(isinstance(m["content"], str) for m in history)


def test_resolve_or_missing_avisa_cuando_falta_el_equipo():
    from mundial_bot.models.elo_model import EloModel
    from mundial_bot.pipeline import Models

    elo = EloModel()
    elo.ratings.update({"Argentina": 2100, "Brazil": 2050})
    brain = agent.BotBrain(models=Models(elo=elo, goals=None), corners=None, cards=None)

    ok = agent._resolve_or_missing(brain, "Argentina", "Brasil")   # Brasil → Brazil
    assert ok == ("Argentina", "Brazil")

    bad = agent._resolve_or_missing(brain, "Argentina", "Wakanda")
    assert isinstance(bad, str) and "NO ENCONTRÉ" in bad and "Wakanda" in bad


def test_history_se_poda_y_sigue_siendo_valido(monkeypatch):
    # Arranca con un historial largo; tras responder, queda acotado y empieza en 'user'.
    fake = _FakeClient([_Resp("end_turn", [_Blk("text", text="ok")])])
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: fake)
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
               for i in range(agent.MAX_HISTORY_MSGS)]
    agent.ask_agent("nueva", settings=_Settings(), brain=None, history=history)

    assert len(history) <= agent.MAX_HISTORY_MSGS
    assert history[0]["role"] == "user"          # nunca arranca en assistant/tool_result
    assert history[-1]["content"] == "ok"
