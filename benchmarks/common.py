import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ScenarioTurn:
    turn_id: str
    thread_id: str
    user_message: str
    expected_assistant_focus: str
    timestamp_hint: Optional[str] = None
    escalation_trigger: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScenarioTurn":
        return cls(
            turn_id=str(payload["turn_id"]),
            thread_id=str(payload.get("thread_id", "")).strip(),
            user_message=str(payload["user_message"]).strip(),
            expected_assistant_focus=str(payload.get("expected_assistant_focus", "")).strip(),
            timestamp_hint=payload.get("timestamp_hint"),
            escalation_trigger=payload.get("escalation_trigger"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "thread_id": self.thread_id,
            "timestamp_hint": self.timestamp_hint,
            "user_message": self.user_message,
            "expected_assistant_focus": self.expected_assistant_focus,
            "escalation_trigger": self.escalation_trigger,
        }


@dataclass
class ScenarioRubricItem:
    criterion: str
    description: str
    weight: float

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScenarioRubricItem":
        return cls(
            criterion=str(payload["criterion"]).strip(),
            description=str(payload.get("description", "")).strip(),
            weight=float(payload.get("weight", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "description": self.description,
            "weight": self.weight,
        }


@dataclass
class ScenarioThread:
    thread_id: str
    intent: str
    priority: str
    key_entities: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScenarioThread":
        return cls(
            thread_id=str(payload["thread_id"]),
            intent=str(payload.get("intent", "")).strip(),
            priority=str(payload.get("priority", "")).strip(),
            key_entities=[str(item).strip() for item in payload.get("key_entities", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "intent": self.intent,
            "priority": self.priority,
            "key_entities": self.key_entities,
        }


@dataclass
class ScenarioNotes:
    multi_turn_requirements: List[str] = field(default_factory=list)
    expected_failure_modes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "ScenarioNotes":
        if not payload:
            return cls()
        return cls(
            multi_turn_requirements=[
                str(item).strip() for item in payload.get("multi_turn_requirements", [])
            ],
            expected_failure_modes=[
                str(item).strip() for item in payload.get("expected_failure_modes", [])
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "multi_turn_requirements": self.multi_turn_requirements,
            "expected_failure_modes": self.expected_failure_modes,
        }


@dataclass
class Scenario:
    scenario_id: str
    title: str
    background: str
    user_profile: str
    assistant_profile: str
    threads: List[ScenarioThread]
    dialogue: List[ScenarioTurn]
    evaluation_rubric: List[ScenarioRubricItem]
    notes: ScenarioNotes = field(default_factory=ScenarioNotes)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Scenario":
        return cls(
            scenario_id=str(payload["scenario_id"]),
            title=str(payload.get("title", "")).strip(),
            background=str(payload.get("background", "")).strip(),
            user_profile=str(payload.get("user_profile", "")).strip(),
            assistant_profile=str(payload.get("assistant_profile", "")).strip(),
            threads=[ScenarioThread.from_dict(item) for item in payload.get("threads", [])],
            dialogue=[ScenarioTurn.from_dict(item) for item in payload.get("dialogue", [])],
            evaluation_rubric=[
                ScenarioRubricItem.from_dict(item) for item in payload.get("evaluation_rubric", [])
            ],
            notes=ScenarioNotes.from_dict(payload.get("notes")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "background": self.background,
            "user_profile": self.user_profile,
            "assistant_profile": self.assistant_profile,
            "threads": [thread.to_dict() for thread in self.threads],
            "dialogue": [turn.to_dict() for turn in self.dialogue],
            "evaluation_rubric": [item.to_dict() for item in self.evaluation_rubric],
            "notes": self.notes.to_dict(),
        }


@dataclass
class TranscriptTurn:
    role: str
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {"role": self.role, "content": self.content}
        if self.meta:
            payload["meta"] = self.meta
        return payload


@dataclass
class TranscriptBundle:
    assistant_label: str
    scenario_id: str
    model_name: str
    system_description: str
    turns: List[TranscriptTurn]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assistant_label": self.assistant_label,
            "scenario_id": self.scenario_id,
            "model_name": self.model_name,
            "system_description": self.system_description,
            "turns": [turn.to_dict() for turn in self.turns],
        }


# ---------------------------------------------------------------------------
# Scenario file helpers
# ---------------------------------------------------------------------------


def load_scenarios(path: Path) -> List[Scenario]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "scenarios" in data:
        raw_items: Iterable[Dict[str, Any]] = data["scenarios"]
    elif isinstance(data, list):
        raw_items = data
    else:
        raise ValueError(f"Unsupported scenario file format at {path}")
    scenarios = [Scenario.from_dict(item) for item in raw_items]
    if not scenarios:
        raise ValueError(f"No scenarios found in {path}")
    return scenarios


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    path.write_text(serialized, encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM utilities
# ---------------------------------------------------------------------------


@dataclass
class LLMRetryConfig:
    attempts: int = 3
    delay: float = 2.0


class AzureLLM:
    """Thin wrapper around Azure OpenAI with retries."""

    def __init__(
        self,
        client: AzureOpenAI,
        default_model: str,
        retry: Optional[LLMRetryConfig] = None,
    ):
        self.client = client
        self.default_model = default_model
        self.retry = retry or LLMRetryConfig()

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        model: Optional[str] = None,
        json_mode: bool = False,
        telemetry: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        request_model = model or self.default_model
        kwargs: Dict[str, Any] = {"model": request_model, "messages": list(messages)}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retry.attempts + 1):
            try:
                start = time.perf_counter()
                response = self.client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - start
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("LLM returned empty content.")
                if telemetry:
                    usage = getattr(response, "usage", None)
                    telemetry(
                        {
                            "model": request_model,
                            "latency": latency,
                            "usage": {
                                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                                "completion_tokens": getattr(usage, "completion_tokens", 0),
                                "total_tokens": getattr(usage, "total_tokens", 0),
                            },
                            "json_mode": json_mode,
                        }
                    )
                return content
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt < self.retry.attempts:
                    time.sleep(self.retry.delay)
                else:
                    raise
        raise RuntimeError(f"LLM call failed after retries: {last_error}")


def build_azure_client(args: argparse.Namespace) -> AzureOpenAI:
    api_key = getattr(args, "api_key", None) or _require_env("API_KEY")
    azure_endpoint = getattr(args, "azure_endpoint", None) or _require_env("AZURE_ENDPOINT")
    api_version = getattr(args, "api_version", None) or _require_env("API_VERSION", "2024-12-01-preview")
    return AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version)


def _require_env(name: str, default: Optional[str] = None) -> str:
    value = load_env(name, default)
    if not value:
        raise EnvironmentError(f"Missing required environment variable '{name}'.")
    return value


def load_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = GlobalConfig.env_overrides.get(name)
    if value is not None:
        return value
    from os import environ

    raw_value = environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value


class GlobalConfig:
    """Allows scripts to override env-derived values without global state hacks."""

    env_overrides: Dict[str, str] = {}


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)


def choose_random_label() -> bool:
    return bool(random.getrandbits(1))
