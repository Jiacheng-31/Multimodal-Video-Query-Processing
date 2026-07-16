from __future__ import annotations

import json
from typing import Any


class LocalLogger:
    """Minimal console logger compatible with the training loop."""

    def __init__(self, print_to_console: bool = True) -> None:
        self.print_to_console = print_to_console

    def log(self, data: dict[str, Any], step: int) -> None:
        if self.print_to_console:
            print(json.dumps({"step": step, **data}, ensure_ascii=False, sort_keys=True), flush=True)
