import json
import os
import anthropic

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field, TypeAdapter, ValidationError


# ============================================================
# Configuration
# ============================================================

MODEL = "claude-sonnet-4-5"
TEMPERATURE = 0
MAX_TOKENS = 500

API_KEY_ENV = "ANTHROPIC_API_KEY"


# ============================================================
# Pydantic schema
# ============================================================

class Ticket(BaseModel):
    summary: str
    priority: Literal["low", "medium", "high", "urgent"]
    category: Literal["billing", "technical", "account", "other"]
    customer_name: str | None = None
    action_items: list[str]

class NotATicket(BaseModel):
    is_ticket: Literal[False]
    reason: str


# A valid result can be either a Ticket or a refusal.
Result = Annotated[Union[Ticket, NotATicket], Field(discriminator="is_ticket")]
result_adapter = TypeAdapter(Result)


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
# Warm-up: multi-turn conversation
# ============================================================

def multi_turn_loop(client):
    messages = []

    print("\n=== Multi-turn conversation ===")
    print("Type 'quit' to stop.\n")

    cumulative_input_tokens = 0

    while True:
        user_prompt = input("You: ")

        if user_prompt.lower() == "quit":
            break

        # Add user message to conversation
        messages.append({
            "role": "user",
            "content": user_prompt,
        })

        response = client.messages.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            system="You are a helpful assistant. Answer concisely.",
            messages=messages,
        )

        if response.stop_reason == "max_tokens":
            raise RuntimeError("Output truncated — raise MAX_TOKENS")

        text = "\n".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

        # Add assistant response to conversation
        messages.append({
            "role": "assistant",
            "content": text,
        })

        cumulative_input_tokens += response.usage.input_tokens

        print(f"\nAssistant: {text}")

        print(
            f"Input tokens this turn: "
            f"{response.usage.input_tokens}"
        )

        print(
            f"Cumulative input tokens: "
            f"{cumulative_input_tokens}\n"
        )

        print(f"Stop reason:   {response.stop_reason}")


# ============================================================
# Extract ticket
# ============================================================
SCHEMAS = json.dumps(
    {
        "Ticket": Ticket.model_json_schema(),
        "NotATicket": NotATicket.model_json_schema(),
    },
    indent=2,
)
SYSTEM_PROMPT = f"""
You are a support-ticket information extraction system.

Determine first whether the user's input is actually a support ticket.

If it IS a support ticket:
- Extract the requested information.
- Return a Ticket object.
- Set "is_ticket" to true.

If it is NOT a support ticket:
- Do not invent ticket information.
- Return a NotATicket object.
- Set "is_ticket" to false.
- Explain briefly why it is not a support ticket.

Return ONLY valid JSON.
Do not use markdown.
Do not include ```json fences.

Your output must match one of these schemas:
{SCHEMAS}

For a non-ticket input, return:

{
  "is_ticket": false,
  "reason": "brief explanation"
}


Rules:
- summary must be exactly one sentence.
- customer_name must be null if no name is stated.
"""


def extract_ticket(client, raw_ticket: str) -> Ticket:

    messages = [
        {
            "role": "user",
            "content": raw_ticket,
        }
    ]

    max_attempts = 2

    for attempt in range(1, max_attempts + 1):

        response = client.messages.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        text = "\n".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

        print(f"\n--- Attempt {attempt} ---")
        print("LLM output:")
        print(text)

        # Keep the assistant response in the conversation.
        messages.append({
            "role": "assistant",
            "content": text,
        })

        try:
            # Pydantic handles both JSON parsing and validation.
            result = result_adapter.validate_json(text)

            if isinstance(result, NotATicket):
                print("\nInput was not a support ticket.")
                print(f"Reason: {result.reason}")
                return result

            return result

        except ValidationError as error:

            print("\nValidation failed:")
            print(error)

            if attempt == max_attempts:
                raise RuntimeError(
                    "Could not extract a valid Ticket after "
                    f"{max_attempts} attempts."
                ) from error

            # Ask the model to correct its previous answer.
            correction_prompt = f"""
                Your previous output was invalid.

                Previous output:
                {text}

                Validation error:
                {error}

                Return ONLY corrected valid JSON matching the required schema.
                Do not include markdown or explanations.
                """

            messages.append({
                "role": "user",
                "content": correction_prompt,
            })

    raise RuntimeError("Unexpected extraction failure.")


# ============================================================
# Main
# ============================================================

def main():

    client = create_client()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    multi_turn_loop(client)

    # --------------------------------------------------------
    # Main exercise
    # --------------------------------------------------------

    print("\n=== Ticket extraction ===")

    raw_ticket = input(
        "\nPaste a support ticket:\n> "
    )

    ticket = extract_ticket(client, raw_ticket)

    print("\n=== Validated Ticket ===")
    print(ticket)

    print("\n=== As dictionary ===")
    print(ticket.model_dump())

    print("\n=== As JSON ===")
    print(ticket.model_dump_json(indent=2))


if __name__ == "__main__":
    main()