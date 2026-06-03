"""LLM client configuration via environment variables."""
import os

from openai import AsyncOpenAI, OpenAI

from llm_api.template_api import AsyncTemplateAPI

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-max-2025-08-13"


def _require_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "Set DASHSCOPE_API_KEY or OPENAI_API_KEY in the environment before running pipeline scripts."
        )
    return key


def get_configs(client_name: str = "aliyun") -> dict:
    if client_name != "aliyun":
        raise ValueError(f"Unsupported client: {client_name}")
    return {
        "api_key": _require_api_key(),
        "base_url": os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
    }


def config_model(
    config: dict,
    model_name: str,
    temperature: float = 0.7,
    num_concurrent: int = 1,
    enable_thinking: bool = False,
    thinking_budget: int = 500,
) -> dict:
    model_config = config.copy()
    model_config["model_name"] = model_name
    model_config["temperature"] = temperature
    model_config["num_concurrent"] = num_concurrent

    if enable_thinking and "qwen-vl-max" in model_name:
        model_config["extra_body"] = {
            "enable_thinking": True,
            "thinking_budget": thinking_budget,
        }
    return model_config


def create_client(config: dict) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )


def create_sync_client(config: dict) -> OpenAI:
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )


def create_async_client(client_config: dict) -> AsyncTemplateAPI:
    return AsyncTemplateAPI(
        model=client_config.get("model_name", DEFAULT_MODEL),
        api_key=client_config["api_key"],
        base_url=client_config["base_url"],
        num_concurrent=client_config.get("num_concurrent", 5),
    )
