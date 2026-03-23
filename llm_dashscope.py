"""本模块作用：提供 DashScope 兼容接口的大模型调用能力，并封装常用的 JSON 解析辅助函数。"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class DashScopeClient:
    """本类作用：封装 DashScope 聊天补全接口。"""

    def __init__(self, api_key: str, base_url: str, timeout_sec: int = 45) -> None:
        """初始化客户端。

        输入：
            api_key: DashScope API 密钥。
            base_url: 兼容模式基础地址。
            timeout_sec: 请求超时秒数。
        输出：
            无。
        异常：
            当 api_key 为空时，抛出 ValueError。
        """

        cleaned_key = api_key.strip()
        if not cleaned_key:
            raise ValueError("DASHSCOPE_API_KEY 不能为空。")

        self.api_key = cleaned_key
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = max(timeout_sec, 5)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """调用聊天补全接口并返回文本。

        输入：
            model: 模型名称。
            messages: OpenAI 兼容消息列表。
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。
            response_format: 输出格式约束参数。
        输出：
            模型文本。
        异常：
            当请求失败或响应解析失败时，抛出 RuntimeError。
        """

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        endpoint = f"{self.base_url}/chat/completions"
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                raw_text = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope 请求失败（HTTP {exc.code}）：{body[:200]}") from exc
        except URLError as exc:
            raise RuntimeError(f"DashScope 网络请求失败：{exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"DashScope 调用异常：{exc}") from exc

        try:
            response_data = json.loads(raw_text)
            choices = response_data.get("choices", [])
            if not choices:
                raise ValueError("响应中缺少 choices。")
            content = choices[0].get("message", {}).get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("响应中缺少有效 content。")
            return content.strip()
        except Exception as exc:
            raise RuntimeError(f"DashScope 响应解析失败：{raw_text[:200]}") from exc


def parse_first_json_object(text: str) -> dict[str, Any] | None:
    """尝试从模型输出中提取第一个 JSON 对象。

    输入：
        text: 模型返回文本。
    输出：
        成功时返回字典对象，失败时返回 None。
    异常：
        无。函数内部会吞掉解析异常并返回 None。
    """

    cleaned_text = text.strip()
    if not cleaned_text:
        return None

    # 先尝试直接按 JSON 解析。
    try:
        direct_value = json.loads(cleaned_text)
        if isinstance(direct_value, dict):
            return direct_value
    except Exception:
        pass

    # 再从文本中提取第一个大括号对象。
    match = re.search(r"\{[\s\S]*\}", cleaned_text)
    if not match:
        return None

    try:
        extracted = json.loads(match.group(0))
        if isinstance(extracted, dict):
            return extracted
    except Exception:
        return None
    return None
