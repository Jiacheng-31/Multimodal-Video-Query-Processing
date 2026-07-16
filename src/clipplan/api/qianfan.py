from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "https://qianfan.baidubce.com/v2/chat/completions"
DEFAULT_MODEL = "ernie-4.5-turbo-vl-32k"


@dataclass(frozen=True)
class QianfanConfig:
    api_key: str
    app_id: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> "QianfanConfig":
        api_key = os.environ.get("QIANFAN_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("QIANFAN_API_KEY is required to call Qianfan.")
        return cls(
            api_key=api_key,
            app_id=os.environ.get("QIANFAN_APP_ID", "").strip(),
            base_url=os.environ.get("QIANFAN_API_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("QIANFAN_API_MODEL", DEFAULT_MODEL),
        )

    def headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.app_id:
            headers["appid"] = self.app_id
        return headers
