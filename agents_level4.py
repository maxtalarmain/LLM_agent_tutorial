import os
import json
from typing import Literal

import anthropic
from pydantic import BaseModel, ValidationError


# ============================================================
# Configuration
# ============================================================

MODEL = "claude-sonnet-4-5"
TEMPERATURE = 0
MAX_TOKENS = 500
MAX_ITERATIONS = 5

API_KEY_ENV = "ANTHROPIC_API_KEY"


# ============================================================
# Pydantic models for tool arguments
# ============================================================

class LookupCustomerArgs(BaseModel):
    email: str


class GetSubscriptionArgs(BaseModel):
    customer_id: str


class RefundArgs(BaseModel):
    amount_paid: float
    days_used: int
    plan_days: int


class FinalAnswer(BaseModel):
    answer: str
    confidence: Literal["low", "medium", "high"]
    tools_used: list[str]


# ============================================================
# Fake database
# ============================================================

CUSTOMERS = {
    "alice@example.com": {
        "customer_id": "C-77",
        "name": "Alice",
    },
    "bob@example.com": {
        "customer_id": "C-88",
        "name": "Bob",
    },
}


SUBSCRIPTIONS = {
    "C-77": {
        "plan_days": 30,
        "amount_paid": 120.0,
        "days_used": 10,
    },
    "C-88": {
        "plan_days": 30,
        "amount_paid": 90.0,
        "days_used": 15,
    },
}


# ============================================================
# Tool implementations
# ============================================================

def lookup_customer(email: str) -> dict:
    if email not in CUSTOMERS:
        raise ValueError(
            f"No customer found for email '{email}'."
        )

    return CUSTOMERS[email]


def get_subscription(customer_id: str) -> dict:
    if customer_id not in SUBSCRIPTIONS:
        raise ValueError(
            f"No subscription found for customer '{customer_id}'."
        )

    return SUBSCRIPTIONS[customer_id]


def calculate_refund(
    amount_paid: float,
    days_used: int,
    plan_days: int,
) -> dict:

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

TOOLS: dict[str, dict] = {
    "lookup_customer": {
        "handler": lookup_customer,
        "args_model": LookupCustomerArgs,
    },
    "get_subscription": {
        "handler": get_subscription,
        "args_model": GetSubscriptionArgs,
    },
    "calculate_refund": {
        "handler": calculate_refund,
        "args_model": RefundArgs,
    },
}


# ============================================================
# Anthropic tool definitions
# ============================================================

ANTHROPIC_TOOLS = [
    {
        "name": "lookup_customer",
        "description": (
            "Look up a customer by email address. "
            "Returns the customer ID and name."
        ),
        "input_schema": LookupCustomerArgs.model_json_schema(),
    },
    {
        "name": "get_subscription",
        "description": (
            "Get subscription information for a customer. "
            "Requires a customer_id obtained from lookup_customer."
        ),
        "input_schema": GetSubscriptionArgs.model_json_schema(),
    },
    {
        "name": "calculate_refund",
        "description": (
            "Calculate a prorated refund using the amount paid, "
            "days used, and total plan duration."
        ),
        "input_schema": RefundArgs.model_json_schema(),
    },
    {
        "name": "final_answer",
        "description": (
            "Return the final answer to the user once all required "
            "information has been gathered and calculations completed."
        ),
        "input_schema": FinalAnswer.model_json_schema(),
    },
]


# ============================================================
# Client
# ============================================================

def create_client():
    api_key = os.getenv(API_KEY_ENV)

    if not api_key:
        raise RuntimeError(
            f"Missing {API_KEY_ENV} environment variable."
        )

    return anthropic.Anthropic(api_key=api_key)


# ============================================================
# Tool execution
# ============================================================

def execute_tool(
    tool_name: str,
    tool_input: dict,
) -> dict:

    if tool_name not in TOOLS:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    tool = TOOLS[tool_name]

    args_model = tool["args_model"]
    handler = tool["handler"]

    # Validate model-generated arguments.
    args = args_model.model_validate(tool_input)

    # Execute only a registered Python function.
    return handler(**args.model_dump())


# ============================================================
# Agent
# ============================================================

def run_agent(
    client,
    user_prompt: str,
) -> FinalAnswer:

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    # --------------------------------------------------------
    # Control 1: trace
    # --------------------------------------------------------

    trace = []

    # --------------------------------------------------------
    # Control 2: duplicate call detection
    # --------------------------------------------------------

    executed_calls = set()

    # --------------------------------------------------------
    # Track tools used
    # --------------------------------------------------------

    tools_used = []

    # --------------------------------------------------------
    # Agent loop
    # --------------------------------------------------------

    for iteration in range(1, MAX_ITERATIONS + 1):

        response = client.messages.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            system="""
You are a support assistant that can reason over multiple steps.

You have access to tools for:
1. Looking up a customer.
2. Getting their subscription.
3. Calculating a prorated refund.

Use the tools when necessary.

Follow dependencies carefully:
- You need customer_id before calling get_subscription.
- You need subscription information before calculating a refund.

Do not invent customer or subscription information.

Once you have enough information to answer the user's request,
you MUST call final_answer.

Do not provide the final answer as normal prose.
Always use the final_answer tool to finish.
""",
            tools=ANTHROPIC_TOOLS,
            messages=messages,
        )

        # ----------------------------------------------------
        # Record LLM-level information
        # ----------------------------------------------------

        entry = {
            "iteration": iteration,
            "stop_reason": response.stop_reason,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "calls": [],
        }
        trace.append(entry)

        # ----------------------------------------------------
        # We expect tool use
        # ----------------------------------------------------

        if response.stop_reason != "tool_use":
            text = "\n".join(b.text for b in response.content if b.type == "text")
            return FinalAnswer(answer=text, confidence="low", tools_used=tools_used)

        # ----------------------------------------------------
        # Append assistant response VERBATIM
        # ----------------------------------------------------

        messages.append({
            "role": "assistant",
            "content": response.content,
        })

        tool_results = []

        # ----------------------------------------------------
        # Execute every tool call
        # ----------------------------------------------------

        for block in response.content:

            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input

            # ----------------------------------------------
            # Final answer tool
            # ----------------------------------------------

            if tool_name == "final_answer":

                try:
                    final_answer = FinalAnswer.model_validate(
                        tool_input
                    )

                    final_answer.tools_used = tools_used

                    # Record final answer in trace.
                    trace[-1]["tool"] = tool_name
                    trace[-1]["arguments"] = tool_input
                    trace[-1]["result"] = "final answer"

                    print_trace(trace)

                    return final_answer

                except ValidationError as error:

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(error),
                        "is_error": True,
                    })

                    continue

            # ----------------------------------------------
            # Normal tools
            # ----------------------------------------------

            call_signature = (
                tool_name,
                frozenset(tool_input.items()),
            )

            # ----------------------------------------------
            # Duplicate detection
            # ----------------------------------------------

            if call_signature in executed_calls:

                result = (
                    "This exact tool call has already been executed. "
                    "Do not repeat it. Reuse the previous result."
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": True,
                })

                trace[-1].update({
                    "tool": tool_name,
                    "arguments": tool_input,
                    "result": "duplicate call blocked",
                })

                continue

            executed_calls.add(call_signature)

            # ----------------------------------------------
            # Execute tool
            # ----------------------------------------------

            try:

                result = execute_tool(
                    tool_name,
                    tool_input,
                )

                tools_used.append(tool_name)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

                trace[-1].update({
                    "tool": tool_name,
                    "arguments": tool_input,
                    "result": json.dumps(result),
                })

            except Exception as error:

                error_message = str(error)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": error_message,
                    "is_error": True,
                })

                trace[-1].update({
                    "tool": tool_name,
                    "arguments": tool_input,
                    "result": f"ERROR: {error_message}",
                })

        # ----------------------------------------------------
        # Send tool results back to the model
        # ----------------------------------------------------

        messages.append({
            "role": "user",
            "content": tool_results,
        })

    # ========================================================
    # Graceful degradation
    # ========================================================

    print_trace(trace)

    return FinalAnswer(
        answer=(
            "I could not complete the request within the maximum "
            f"number of agent iterations ({MAX_ITERATIONS})."
        ),
        confidence="low",
        tools_used=tools_used,
    )


# ============================================================
# Trace
# ============================================================

def print_trace(trace):

    print("\n")
    print("=" * 70)
    print("AGENT TRACE")
    print("=" * 70)

    for event in trace:

        print(f"\nIteration {event['iteration']}")
        print(f"  stop_reason: {event['stop_reason']}")
        print(f"  input_tokens: {event['input_tokens']}")
        print(f"  output_tokens: {event['output_tokens']}")

        if "tool" in event:
            print(f"  tool: {event['tool']}")
            print(f"  arguments: {event.get('arguments')}")
            print(f"  result: {event.get('result')}")

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    client = create_client()

    print("=== Customer Support Agent ===")

    user_prompt = input(
        "\nAsk something:\n> "
    )

    try:

        result = run_agent(
            client,
            user_prompt,
        )

        print("\n=== FINAL RESULT ===")
        print(result.model_dump_json(indent=2))

    except Exception as error:

        print(f"\nAgent failed: {error}")


if __name__ == "__main__":
    main()