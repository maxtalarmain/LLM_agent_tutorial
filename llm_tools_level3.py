import os
import json

import anthropic
from pydantic import BaseModel


# ============================================================
# Configuration
# ============================================================

MODEL = "claude-sonnet-4-5"
TEMPERATURE = 0
MAX_TOKENS = 1024
MAX_ITERATIONS = 5

API_KEY_ENV = "ANTHROPIC_API_KEY"


# ============================================================
# Tool argument schemas
# ============================================================

class TicketStatusArgs(BaseModel):
    ticket_id: str


class RefundArgs(BaseModel):
    amount_paid: float
    days_used: int
    plan_days: int


# ============================================================
# Tool implementations
# ============================================================

TICKETS = {
    "T-1001": {
        "status": "open",
        "owner": "Alice",
    },
    "T-1002": {
        "status": "closed",
        "owner": "Bob",
    },
}


def get_ticket_status(ticket_id: str) -> dict:
    """Return the status of a support ticket."""

    if ticket_id not in TICKETS:
        raise ValueError(
            f"Ticket '{ticket_id}' does not exist."
        )

    return TICKETS[ticket_id]


def calculate_refund(
    amount_paid: float,
    days_used: int,
    plan_days: int,
) -> dict:
    """Calculate a prorated refund."""

    if amount_paid < 0:
        raise ValueError("amount_paid must be >= 0")

    if days_used < 0:
        raise ValueError("days_used must be >= 0")

    if plan_days <= 0:
        raise ValueError("plan_days must be > 0")

    if days_used > plan_days:
        raise ValueError(
            "days_used cannot be greater than plan_days"
        )

    unused_days = plan_days - days_used

    refund = amount_paid * unused_days / plan_days

    return {
        "amount_paid": amount_paid,
        "days_used": days_used,
        "plan_days": plan_days,
        "unused_days": unused_days,
        "refund": round(refund, 2),
    }


# ============================================================
# Tool registry
# ============================================================

TOOL_REGISTRY = {
    "get_ticket_status": {
        "handler": get_ticket_status,
        "args_model": TicketStatusArgs,
    },
    "calculate_refund": {
        "handler": calculate_refund,
        "args_model": RefundArgs,
    },
}


# ============================================================
# Tool definitions for Anthropic
# ============================================================

tools = [
    {
        "name": "get_ticket_status",
        "description": (
            "Get the current status and owner of a support ticket."
        ),
        "input_schema": TicketStatusArgs.model_json_schema(),
    },
    {
        "name": "calculate_refund",
        "description": (
            "Calculate a prorated refund based on the amount paid, "
            "days used, and total plan duration."
        ),
        "input_schema": RefundArgs.model_json_schema(),
    },
]


# ============================================================
# Anthropic client
# ============================================================

def create_client():
    api_key = os.getenv(API_KEY_ENV)

    if not api_key:
        raise RuntimeError(
            f"Missing {API_KEY_ENV} environment variable."
        )

    return anthropic.Anthropic(api_key=api_key)


# ============================================================
# Execute a tool call
# ============================================================

def execute_tool(tool_name, tool_input):

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool = TOOL_REGISTRY[tool_name]

    args = tool["args_model"].model_validate(tool_input)

    return tool["handler"](**args.model_dump())


# ============================================================
# Agentic tool loop
# ============================================================

def run_agent(client, user_prompt: str):

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\n========== Iteration {iteration} ==========")

        response = client.messages.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            system=(
                "You are a helpful support assistant. "
                "Use the available tools when you need information "
                "or calculations. "
                "If a tool returns an error, use that information "
                "to recover and continue if possible."
            ),
            tools=tools,
            messages=messages,
        )

        print(f"stop_reason: {response.stop_reason}")

        # ----------------------------------------------------
        # Case 1: Model wants to use one or more tools
        # ----------------------------------------------------

        if response.stop_reason == "tool_use":

            # Append the assistant response VERBATIM.
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            tool_results = []

            for block in response.content:

                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input

                print(f"\nTool call: {tool_name}")
                print(f"Arguments: {tool_input}")

                try:
                    result = execute_tool(
                        tool_name,
                        tool_input,
                    )

                    print(f"Result: {result}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

                except Exception as error:

                    error_message = str(error)

                    print(f"Tool error: {error_message}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": error_message,
                        "is_error": True,
                    })

            # Append tool results as a USER turn.
            messages.append({
                "role": "user",
                "content": tool_results,
            })

            # Continue the loop.
            continue

        # ----------------------------------------------------
        # Case 2: Model finished
        # ----------------------------------------------------

        if response.stop_reason == "end_turn":

            text = "\n".join(
                block.text
                for block in response.content
                if block.type == "text"
            )

            print("\n========== Final answer ==========")
            print(text)

            return text

        # ----------------------------------------------------
        # Unexpected stop reason
        # ----------------------------------------------------

        raise RuntimeError(
            f"Unexpected stop reason: {response.stop_reason}"
        )

    # --------------------------------------------------------
    # Max iterations reached
    # --------------------------------------------------------

    raise RuntimeError(
        f"Agent exceeded maximum iterations ({MAX_ITERATIONS})."
    )


# ============================================================
# Main
# ============================================================

def main():

    client = create_client()

    print("=== Tool Agent ===")

    user_prompt = input(
        "\nAsk something:\n> "
    )

    try:
        run_agent(client, user_prompt)

    except Exception as error:
        print(f"\nAgent failed: {error}")


if __name__ == "__main__":
    main()