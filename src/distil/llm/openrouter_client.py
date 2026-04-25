import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


_OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class ModelConfiguration:
    identifier: str
    display_name: str
    size_label: str
    input_price_per_million: float
    output_price_per_million: float
    supports_reasoning: bool = False


MODEL_CATALOG: dict[str, ModelConfiguration] = {
    "qwen25_7b": ModelConfiguration(
        identifier="qwen/qwen-2.5-7b-instruct",
        display_name="Qwen2.5-7B-Instruct",
        size_label="7B",
        input_price_per_million=0.10,
        output_price_per_million=0.15,
    ),
    "gemma34b": ModelConfiguration(
        identifier="google/gemma-3-4b-it",
        display_name="Gemma 3-4B",
        size_label="4B",
        input_price_per_million=0.0,
        output_price_per_million=0.0,
    ),
    "glm47": ModelConfiguration(
        identifier="z-ai/glm-4.7",
        display_name="GLM-4.7",
        size_label="9B",
        input_price_per_million=0.10,
        output_price_per_million=0.30,
        supports_reasoning=True,
    ),
    "qwen36": ModelConfiguration(
        identifier="qwen/qwen3.6-plus",
        display_name="Qwen3.6-35B-A3B",
        size_label="35B-MoE",
        input_price_per_million=0.10,
        output_price_per_million=0.30,
        supports_reasoning=True,
    ),
    "deepseek_v3": ModelConfiguration(
        identifier="deepseek/deepseek-v3.2-speciale",
        display_name="DeepSeek-V3.2-Speciale",
        size_label="671B-MoE",
        input_price_per_million=0.30,
        output_price_per_million=0.90,
        supports_reasoning=True,
    ),
}


@dataclass
class BudgetTracker:
    maximum_budget_usd: float = 5.0
    total_spent_usd: float = 0.0
    total_calls: int = 0
    log_path: Path = field(default_factory=lambda: Path("results/openrouter/budget_log.json"))

    def record_call(self, input_tokens: int, output_tokens: int, model_identifier: str) -> bool:
        for catalog_key, model_config in MODEL_CATALOG.items():
            if model_config.identifier == model_identifier:
                input_price = model_config.input_price_per_million
                output_price = model_config.output_price_per_million
                break
        else:
            input_price, output_price = 1.0, 3.0

        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        self.total_spent_usd += cost
        self.total_calls += 1

        if self.total_calls % 100 == 0:
            print(
                f"    [budget] ${self.total_spent_usd:.3f} / ${self.maximum_budget_usd:.2f} "
                f"({self.total_calls} calls)",
                flush=True,
            )

        return self.total_spent_usd <= self.maximum_budget_usd

    def is_within_budget(self) -> bool:
        return self.total_spent_usd <= self.maximum_budget_usd

    def write_summary(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w") as file_handle:
            json.dump(
                {
                    "total_spent_usd": self.total_spent_usd,
                    "total_calls": self.total_calls,
                    "maximum_budget_usd": self.maximum_budget_usd,
                },
                file_handle,
                indent=2,
            )


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        budget_tracker: BudgetTracker | None = None,
        request_timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.request_timeout_seconds = request_timeout_seconds

    def chat_completion(
        self,
        model_key: str,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.0,
        seed: int = 42,
        reasoning: dict | None = None,
    ) -> dict[str, Any]:
        if model_key not in MODEL_CATALOG:
            raise ValueError(f"Unknown model_key: {model_key!r}. See MODEL_CATALOG.")

        model_config = MODEL_CATALOG[model_key]

        request_body = {
            "model": model_config.identifier,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        if reasoning is not None:
            request_body["reasoning"] = reasoning

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            _OPENROUTER_API_URL,
            headers=headers,
            json=request_body,
            timeout=self.request_timeout_seconds,
        )
        response.raise_for_status()
        response_payload = response.json()

        usage = response_payload.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        self.budget_tracker.record_call(input_tokens, output_tokens, model_config.identifier)

        return response_payload
