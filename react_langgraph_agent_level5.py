from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# Configuration
# ============================================================

MODEL = "claude-sonnet-4-5"

SYSTEM = SystemMessage(content=(
    "You are a support assistant. Use tools to gather information. "
    "You need a customer_id from lookup_customer before calling "
    "get_subscription. Never invent customer or subscription data."
))


# ============================================================
# State
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


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
# LLM + bind tools
# ============================================================

llm = ChatAnthropic(
    model=MODEL,
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# Agent node
# ============================================================

def agent(state: AgentState):

    response = llm_with_tools.invoke([SYSTEM] + state["messages"])

    return {
        "messages": [response]
    }


# ============================================================
# Tools node
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# Routing logic
# ============================================================

def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


# ============================================================
# Build graph
# ============================================================

memory = MemorySaver()
graph = create_react_agent(
    llm,
    tools,
    checkpointer=memory,
)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    config1 = {
        "configurable": {
            "thread_id": "user-123"
        }
    }

    # First conversation turn
    result1 = graph.invoke(
        {
            "messages": [
                ("user", "refund for alice@example.com?")
            ]
        },
        config1,
    )

    print("\n=== Final conversation 1 ===")
    
    for message in result1["messages"]:
        print(
            f"\n{message.__class__.__name__}:"
        )
        print(message.content)

    config2 = {
            "configurable": {
                "thread_id": "user-456"
            }
        }

    # Second conversation turn
    result2 = graph.invoke(
            {
                "messages": [
                    ("user", "and for bob@example.com?")
                ]
            },
            config2,
        )

    print("\n=== Final conversation 2 ===")

    for message in result2["messages"]:
        print(
            f"\n{message.__class__.__name__}:"
        )
        print(message.content)

    print("\n=== Graph ===")
    print(graph.get_graph().draw_ascii())