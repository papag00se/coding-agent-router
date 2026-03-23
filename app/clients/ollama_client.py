from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: Union[float, int, Tuple[float, float]],
        *,
        pool_connections: int = 8,
        pool_maxsize: int = 8,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=Retry(total=0),
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: float,
        num_ctx: int,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        response_format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": num_ctx,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if system:
            payload["messages"] = [{"role": "system", "content": system}] + payload["messages"]
        if response_format == "json":
            payload["format"] = "json"
        if tools:
            payload["tools"] = tools

        response = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: float,
        num_ctx: int,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        response_format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": num_ctx,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options,
        }
        if system:
            payload["messages"] = [{"role": "system", "content": system}] + payload["messages"]
        if response_format == "json":
            payload["format"] = "json"
        if tools:
            payload["tools"] = tools

        with self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                yield json.loads(line)
