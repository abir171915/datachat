"""
Demo: The full tool-use loop with Claude.

This shows the minimum viable agent pattern:
  1. Define a tool (JSON description + Python function)
  2. Send user message + tools to Claude
  3. Claude decides to use the tool
  4. We execute the tool and send the result back
  5. Claude uses the result to answer

Run: python -m src.datachat.demo_tool_use
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from src.datachat.client import get_client, DEFAULT_MODEL


# ────────────────────────────────────────────────────────────────
# STEP 1: Define the tool
# ────────────────────────────────────────────────────────────────

# This is the JSON description Claude sees.
# Claude uses the `description` to decide WHEN to call the tool,
# and the `input_schema` to know HOW to call it.
TOOL_DEFINITIONS = [
    {
        "name": "get_current_time",
        "description": (
            "Returns the current date and time for a given timezone. "
            "Use this whenever the user asks about the current time, "
            "today's date, or wants to know what time it is somewhere."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA timezone name, e.g. 'Europe/London', "
                        "'America/New_York', 'Asia/Tokyo'."
                    ),
                }
            },
            "required": ["timezone"],
        },
    }
]


# This is the actual Python function that does the work.
# Claude never sees this code — only the description above.
def get_current_time(timezone: str) -> str:
    """Return the current time in the given timezone as a human-readable string."""
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%A, %d %B %Y at %H:%M:%S %Z")
    except Exception as e:
        return f"Error: could not get time for timezone '{timezone}'. ({e})"


# A mapping from tool name → Python function.
# When Claude says "call get_current_time", we look it up here.
TOOL_REGISTRY = {
    "get_current_time": get_current_time,
}


# ────────────────────────────────────────────────────────────────
# STEP 2: The conversation loop
# ────────────────────────────────────────────────────────────────

def run_agent(user_message: str) -> str:
    """Run a single user query through the full tool-use loop."""
    client = get_client()

    # Start the conversation history with the user's message
    messages = [{"role": "user", "content": user_message}]

    # Loop until Claude gives us a final text answer (no more tool calls)
    while True:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        print(f"\n[Claude's stop_reason: {response.stop_reason}]")

        # If Claude didn't request any tool, it's done — return the text
        if response.stop_reason != "tool_use":
            final_text = next(
                (block.text for block in response.content if block.type == "text"),
                "(no text response)",
            )
            return final_text

        # Otherwise, Claude wants to use one or more tools
        # Append Claude's entire response to the conversation history
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool Claude requested
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                print(f"[Tool call: {tool_name}({tool_input})]")

                # Look up and execute the actual Python function
                tool_function = TOOL_REGISTRY[tool_name]
                result = tool_function(**tool_input)
                print(f"[Tool result: {result}]")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        # Send the tool results back to Claude as a user message
        messages.append({"role": "user", "content": tool_results})


# ────────────────────────────────────────────────────────────────
# STEP 3: Try it
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    question = "What time is it right now in Tokyo, and what day of the week is it?"
    print(f"User: {question}")
    answer = run_agent(question)
    print(f"\nClaude: {answer}")