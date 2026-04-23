"""
Claude API client setup.

This module is responsible for creating an authenticated Anthropic client
that the rest of the project can use. It loads the API key from a .env file
so we never hardcode secrets in source code.
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env into os.environ
load_dotenv()


def get_client() -> Anthropic:
    """
    Return an authenticated Anthropic client.

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Create a .env file in the project root with: "
            "ANTHROPIC_API_KEY=sk-ant-..."
        )

    return Anthropic(api_key=api_key)


# Model constants — keeping them here makes them easy to swap later
MODEL_SONNET = "claude-sonnet-4-5"
MODEL_HAIKU = "claude-haiku-4-5"
DEFAULT_MODEL = MODEL_SONNET