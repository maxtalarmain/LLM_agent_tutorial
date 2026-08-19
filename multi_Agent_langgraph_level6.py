from pydantic import BaseModel, Field
from typing import Annotated, TypedDict
import re
import json
from decimal import Decimal

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# Configuration
# ============================================================

MODEL = "claude-sonnet-4-5"


# ============================================================
# State
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]
    validation_errors: list[str]
    revision_count: int
    human_rejected: bool


# ============================================================
# Tools
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


@tool
def lookup_customer(email: str) -> dict:
    """Look up a customer by email address.

    Returns the customer ID and name.
    """

    if email not in CUSTOMERS:
        raise ValueError(
            f"No customer found for email '{email}'."
        )

    return CUSTOMERS[email]


@tool
def get_subscription(customer_id: str) -> dict:
    """Get subscription information for a customer.

    Requires the customer_id returned by lookup_customer.
    """

    if customer_id not in SUBSCRIPTIONS:
        raise ValueError(
            f"No subscription found for customer '{customer_id}'."
        )

    return SUBSCRIPTIONS[customer_id]


@tool
def calculate_refund(
    amount_paid: float,
    days_used: int,
    plan_days: int,
) -> dict:
    """Calculate a prorated subscription refund.

    Uses the amount paid, number of days used,
    and total number of days in the plan.
    """

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


tools = [
    lookup_customer,
    get_subscription,
    calculate_refund,
]


# ============================================================
# LLM
# ============================================================

llm = ChatAnthropic(
    model=MODEL,
    temperature=0,
)

# LLM used by the agent to decide when/how to call tools
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# Planner
# ============================================================

class Plan(BaseModel):
    steps: list[str] = Field(
        description=(
            "A concise ordered list of steps needed "
            "to solve the user's request."
        )
    )


# LLM used specifically by the planning node
planner_llm = llm.with_structured_output(Plan)


def plan_node(state: AgentState):
    """Create a plan once before entering the agent loop."""

    # Find the user's request
    user_message = state["messages"][-1].content

    prompt = f"""
        Create a concise plan for solving this support request.

        User request:
        {user_message}

        The available tools are:

        1. lookup_customer(email)
        2. get_subscription(customer_id)
        3. calculate_refund(
            amount_paid,
            days_used,
            plan_days
        )

        The plan should contain the ordered steps needed
        to answer the user's request.

        Do not perform the task yourself.
        Only create the plan.
    """

    plan = planner_llm.invoke(prompt)

    print("\n=== PLAN ===")

    for i, step in enumerate(plan.steps, 1):
        print(f"{i}. {step}")

    return {
        "plan": plan.steps,
        "validation_errors": [],
        "revision_count": 0,
    }


# ============================================================
# Agent node
# ============================================================

BREAK_GROUNDING = True   # inject a hallucination-inducing instruction

def agent(state: AgentState):
    plan_text = "\n".join(
        f"{i}. {step}" for i, step in enumerate(state["plan"], 1)
    )

    content = (
        "You are a support assistant.\n\n"
        "Use tools to gather information.\n"
        "You need a customer_id from lookup_customer before "
        "calling get_subscription.\n"
        "Never invent customer, subscription, or refund data.\n\n"
        f"Follow this plan:\n{plan_text}\n\n"
        "Your final answer must be grounded in the tool outputs."
    )

    if BREAK_GROUNDING:
        content += "\nAlways round refund amounts to the nearest 10 dollars."

    response = llm_with_tools.invoke(
        [SystemMessage(content=content)] + state["messages"]
    )
    return {"messages": [response]}


# ============================================================
# Tools node
# ============================================================


APPROVAL_THRESHOLD = 40.0
raw_tool_node = ToolNode(tools)


def tools_with_approval(state: AgentState):
    last_message = state["messages"][-1]

    for call in last_message.tool_calls:
        if call["name"] != "calculate_refund":
            continue

        args = call["args"]
        estimated = (
            args["amount_paid"]
            * (args["plan_days"] - args["days_used"])
            / args["plan_days"]
        )
        if estimated < APPROVAL_THRESHOLD:
            continue

        decision = interrupt({
            "tool": "calculate_refund",
            "args": args,
            "estimated_refund": round(estimated, 2),
        })

        if decision != "approve":
            return {
                "human_rejected": True,
                "messages": [ToolMessage(
                    content="Refund rejected by human reviewer. Do not compute or "
                            "state any refund amount.",
                    tool_call_id=call["id"],
                )],
            }

    return raw_tool_node.invoke(state)


# ============================================================
# Agent routing
# ============================================================


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "validate"


# ============================================================
# Validation
# ============================================================

def extract_numbers(text: str) -> list[Decimal]:
    """
    Extract numbers from text.

    Decimal is used so that:
        45
        45.0
        45.00

    are considered the same number.
    """

    matches = re.findall(
        r"(?<![\w.-])\d+(?:\.\d+)?",
        text,
    )

    return [Decimal(value) for value in matches]

def as_text(content) -> str:
    """Anthropic returns content as str or as a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    return str(content)


def validate(state: AgentState):
    messages = state["messages"]
    final_answer = as_text(messages[-1].content)
    answer_numbers = extract_numbers(final_answer)

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    tool_text = "\n".join(as_text(m.content) for m in tool_messages)
    tool_numbers = extract_numbers(tool_text)

    errors = []

    # --- Check 1: grounding — no invented numbers -------------
    for number in answer_numbers:
        if number not in tool_numbers:
            errors.append(
                f"Number {number} does not appear in any tool output."
            )

    # --- Check 2: field binding — the refund is the right one --
    computed_refund = None
    for message in tool_messages:
        try:
            payload = json.loads(as_text(message.content))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and "refund" in payload:
            computed_refund = Decimal(str(payload["refund"]))

    if computed_refund is not None and computed_refund not in answer_numbers:
        errors.append(
            f"The answer does not state the computed refund {computed_refund}."
        )

    # --- Check 3: plan adherence, unless a human overrode it ---
    if state.get("human_rejected"):
        # The human blocked the tool, so plan adherence is moot.
        # Assert instead that no refund figure was stated.
        if computed_refund is None and answer_numbers:
            suspicious = [n for n in answer_numbers if n not in tool_numbers]
            if suspicious:
                errors.append(
                    f"Refund was rejected, but the answer states {suspicious}."
                )
    else:
        plan_text = " ".join(state["plan"]).lower()
        called_tools = {m.name for m in tool_messages if hasattr(m, "name")}

        for tool_name in (
            "lookup_customer",
            "get_subscription",
            "calculate_refund",
        ):
            if tool_name in plan_text and tool_name not in called_tools:
                errors.append(
                    f"The plan requires {tool_name} but it was never called."
                )

    print("\n=== VALIDATION ===")
    print("PASS" if not errors else "FAIL")
    for error in errors:
        print(f"- {error}")

    return {"validation_errors": errors}


# ============================================================
# Validation routing
# ============================================================

def validation_router(state: AgentState):
    """
    Decide whether validation passed or the agent
    needs another attempt.
    """

    errors = state["validation_errors"]
    revision_count = state["revision_count"]

    # Validation passed
    if not errors or state.get("human_rejected"):
        return END

    # Validation failed but we still have retries
    if revision_count < 2:
        return "retry"

    # Maximum revisions reached
    print("\nMaximum revision count reached.")
    return END


# ============================================================
# Retry node
# ============================================================

def retry(state: AgentState):
    """
    Tell the agent what was wrong and ask it to
    produce a corrected answer.
    """

    errors = "\n".join(
        f"- {error}"
        for error in state["validation_errors"]
    )

    new_revision_count = state["revision_count"] + 1

    print(
        f"\n=== RETRY #{new_revision_count} ==="
    )

    return {
        "messages": [
            HumanMessage(
                content=(
                    "Your previous answer failed validation.\n\n"
                    "Correct the answer using only the "
                    "information present in the tool outputs.\n\n"
                    "Validation errors:\n"
                    f"{errors}"
                )
            )
        ],
        "revision_count": new_revision_count,
        "validation_errors": [],
    }


# ============================================================
# Build graph
# ============================================================

builder = StateGraph(AgentState)


# Nodes
builder.add_node("plan", plan_node)
builder.add_node("agent", agent)
builder.add_node("tools", tools_with_approval)
builder.add_node("validate", validate)
builder.add_node("retry", retry)


# ============================================================
# Graph edges
# ============================================================

# START → plan
builder.add_edge(
    START,
    "plan",
)


# plan → agent
builder.add_edge(
    "plan",
    "agent",
)


# agent → tools OR validate
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "validate": "validate",
    },
)


# tools → agent
builder.add_edge(
    "tools",
    "agent",
)


# validate → END OR retry
builder.add_conditional_edges(
    "validate",
    validation_router,
    {
        END: END,
        "retry": "retry",
    },
)


# retry → agent
builder.add_edge(
    "retry",
    "agent",
)


# ============================================================
# Compile graph
# ============================================================

memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory,
)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":


    # --------------------------------------------------------

    config = {"configurable": {"thread_id": "hitl-1"}}

    result = graph.invoke(
        {"messages": [("user", "calculate the refund for bob@example.com")]},
        config,
    )

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("PAUSED — approval needed:", payload)
        print("Next node:", graph.get_state(config).next)

        answer = input("approve / reject: ")
        result = graph.invoke(Command(resume=answer), config)

    # --------------------------------------------------------
    # Graph visualization
    # --------------------------------------------------------

    print("\n=== Graph ===")
    print(
        graph.get_graph().draw_ascii()
    )