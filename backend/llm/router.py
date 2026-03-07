"""
模型路由：支持“按用户 + 按模块”选择不同厂商/模型。

原则：
- API Key 仅从服务端 settings/.env 读取，不从前端/数据库读取。
- 用户偏好只记录 provider/model/api_base/temperature 等非敏感信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import httpx

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..config import settings
from ..database.crud import DatabaseManager


DEFAULT_SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"
OPENAI_COMPAT_PROVIDERS = {"siliconflow", "antigravity"}


@dataclass
class ModuleLLMConfig:
    provider: str
    model: str
    api_base: Optional[str] = None
    temperature: float = 0.3


def _normalize_message_content(content: Any) -> str:
    """将不同 SDK 形态的 content 统一转为字符串。"""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="ignore")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("output_text")
                if text is not None:
                    parts.append(str(text))
                continue
            text = (
                getattr(item, "text", None)
                or getattr(item, "output_text", None)
                or getattr(item, "content", None)
            )
            if text is not None:
                parts.append(str(text))
        if parts:
            return "\n".join(parts).strip()
    return str(content)


class NormalizedChatModel:
    """
    统一包装不同厂商模型，确保 invoke/ainvoke/stream 返回字符串 content。
    """

    def __init__(self, inner_model: Any):
        self.inner_model = inner_model

    @staticmethod
    def _to_ai_message(message: Any) -> AIMessage:
        raw_content = getattr(message, "content", message)
        return AIMessage(content=_normalize_message_content(raw_content))

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        return self._to_ai_message(self.inner_model.invoke(messages))

    async def ainvoke(self, messages: List[BaseMessage]) -> AIMessage:
        result = await self.inner_model.ainvoke(messages)
        return self._to_ai_message(result)

    def stream(self, messages: List[BaseMessage]) -> Iterable[AIMessage]:
        for chunk in self.inner_model.stream(messages):
            yield self._to_ai_message(chunk)


class SiliconFlowChatModel:
    """
    极简 OpenAI-compatible Chat Completions 客户端。

    只实现项目用到的最小接口：
    - invoke(messages) -> AIMessage
    - ainvoke(messages) -> AIMessage
    - stream(messages) -> Iterable[AIMessage]  (降级：一次性返回完整内容)
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str = DEFAULT_SILICONFLOW_API_BASE,
        temperature: float = 0.3,
        timeout_sec: float = 60.0,
        max_tokens: Optional[int] = None,
        provider_label: str = "OpenAI兼容",
        api_key_env: str = "API_KEY",
        api_base_env: str = "API_BASE",
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = (api_base or "").strip().rstrip("/")
        self.temperature = float(temperature)
        self.timeout_sec = float(timeout_sec)
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.provider_label = (provider_label or "OpenAI兼容").strip()
        self.api_key_env = (api_key_env or "API_KEY").strip()
        self.api_base_env = (api_base_env or "API_BASE").strip()

    @staticmethod
    def _to_openai_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for m in messages or []:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            else:
                # 兜底：assistant 或其他
                role = "assistant" if getattr(m, "type", "") == "ai" else "user"

            out.append({"role": role, "content": getattr(m, "content", "") or ""})
        return out

    def _build_payload(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_messages(messages),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def _extract_content(self, data: Dict[str, Any]) -> str:
        try:
            return (
                (data.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
            ) or ""
        except Exception:
            return ""

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        if not self.api_key:
            raise ValueError(f"未配置 {self.provider_label} API密钥（{self.api_key_env}）")
        if not self.model:
            raise ValueError(f"未配置 {self.provider_label} 模型名称")
        if not self.api_base:
            raise ValueError(f"未配置 {self.provider_label} API Base（{self.api_base_env}）")

        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self._build_payload(messages)

        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return AIMessage(content=self._extract_content(data))

    async def ainvoke(self, messages: List[BaseMessage]) -> AIMessage:
        if not self.api_key:
            raise ValueError(f"未配置 {self.provider_label} API密钥（{self.api_key_env}）")
        if not self.model:
            raise ValueError(f"未配置 {self.provider_label} 模型名称")
        if not self.api_base:
            raise ValueError(f"未配置 {self.provider_label} API Base（{self.api_base_env}）")

        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self._build_payload(messages)

        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return AIMessage(content=self._extract_content(data))

    def stream(self, messages: List[BaseMessage]) -> Iterable[AIMessage]:
        # 降级实现：一次性返回，保证与现有 UI 逻辑兼容
        yield self.invoke(messages)


class ModelRouter:
    """按用户+模块返回对应的 ChatModel 客户端。"""

    _DEFAULT_TEMP_BY_MODULE: Dict[str, float] = {
        "rag": 0.3,
        "kg": 0.3,
        "quiz": 0.7,
        "socratic": 0.7,
    }

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def get_module_llm_config(self, user_id: str, module: str) -> ModuleLLMConfig:
        module_key = (module or "").strip().lower()
        pref = self.db.get_user_model_preference(user_id=user_id, module=module_key)

        default_temp = float(self._DEFAULT_TEMP_BY_MODULE.get(module_key, 0.3))

        if pref is None:
            return ModuleLLMConfig(
                provider="google",
                model=settings.LLM_MODEL,
                api_base=None,
                temperature=default_temp,
            )

        provider = (pref.provider or "google").strip().lower()
        model = (pref.model or "").strip() or settings.LLM_MODEL
        api_base = (pref.api_base or "").strip() or None
        temperature = float(pref.temperature) if pref.temperature is not None else default_temp

        # 兼容性修正：如果用户把 Qwen 模型填进了 Google provider，会导致 Gemini 侧 404。
        # 这里自动将其路由到 OpenAI-compatible provider（硅基流动 / Antigravity）。
        if provider not in OPENAI_COMPAT_PROVIDERS and "qwen" in (model or "").lower():
            if getattr(settings, "ANTIGRAVITY_API_KEY", "") and getattr(settings, "ANTIGRAVITY_API_BASE", ""):
                provider = "antigravity"
                api_base = api_base or getattr(settings, "ANTIGRAVITY_API_BASE", "")
            else:
                provider = "siliconflow"
                api_base = api_base or DEFAULT_SILICONFLOW_API_BASE

        return ModuleLLMConfig(provider=provider, model=model, api_base=api_base, temperature=temperature)

    def get_chat_model(self, user_id: str, module: str, streaming: bool = False):
        cfg = self.get_module_llm_config(user_id=user_id, module=module)

        if cfg.provider in OPENAI_COMPAT_PROVIDERS:
            module_key = (module or "").strip().lower()

            # KG 往往输入更长、输出更大，对所有 OpenAI-compatible provider 都使用更宽松的超时/输出限制
            if module_key == "kg":
                timeout_sec = float(getattr(settings, "KG_LLM_TIMEOUT_SEC", 180.0))
                max_tokens = int(getattr(settings, "KG_LLM_MAX_TOKENS", 1024))
            else:
                if cfg.provider == "antigravity":
                    timeout_sec = float(getattr(settings, "ANTIGRAVITY_LLM_TIMEOUT_SEC", 60.0))
                    max_tokens = int(getattr(settings, "ANTIGRAVITY_LLM_MAX_TOKENS", 1024))
                else:
                    timeout_sec = float(getattr(settings, "SILICONFLOW_LLM_TIMEOUT_SEC", 60.0))
                    max_tokens = int(getattr(settings, "SILICONFLOW_LLM_MAX_TOKENS", 1024))

            if cfg.provider == "antigravity":
                api_key = getattr(settings, "ANTIGRAVITY_API_KEY", "")
                api_base = (cfg.api_base or getattr(settings, "ANTIGRAVITY_API_BASE", "") or "").strip()
                model = SiliconFlowChatModel(
                    api_key=api_key,
                    model=cfg.model,
                    api_base=api_base,
                    temperature=cfg.temperature,
                    timeout_sec=timeout_sec,
                    max_tokens=max_tokens,
                    provider_label="Antigravity",
                    api_key_env="ANTIGRAVITY_API_KEY",
                    api_base_env="ANTIGRAVITY_API_BASE",
                )
                return NormalizedChatModel(model)

            # default: SiliconFlow
            api_base = cfg.api_base or DEFAULT_SILICONFLOW_API_BASE
            model = SiliconFlowChatModel(
                api_key=settings.SILICONFLOW_API_KEY,
                model=cfg.model,
                api_base=api_base,
                temperature=cfg.temperature,
                timeout_sec=timeout_sec,
                max_tokens=max_tokens,
                provider_label="硅基流动",
                api_key_env="SILICONFLOW_API_KEY",
                api_base_env="API Base（默认 https://api.siliconflow.cn/v1）",
            )
            return NormalizedChatModel(model)

        # 默认：Google Gemini
        model = ChatGoogleGenerativeAI(
            model=cfg.model,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=cfg.temperature,
            streaming=bool(streaming),
            convert_system_message_to_human=True,
        )
        return NormalizedChatModel(model)

