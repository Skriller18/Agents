"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""

import platform
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, cast

# Create StrEnum implementation for Python < 3.11
class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return self.value

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
# ... rest of the file remains unchanged