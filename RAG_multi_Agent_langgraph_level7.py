"""
Level 7 — Agentic RAG.

Changes from your version are marked with  # FIX:
"""

import json
import re
from decimal import Decimal
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from index_policy_docs_level7 import KNOWN_SOURCES, vectorstore


# ============================================================
# Configuration
# ============================================================

MODEL = "claude-sonnet-4-5"
APPROVAL_THRESHOLD = 40.0
MAX_REVISIONS = 2

# FIX: turn the sabotage off — you have already proven the validator fires.
BREAK_GROUNDING = False

# Distances above this are treated as "no relevant match".
# Tune it by running index_policy_docs_level7.py directly and reading scores.
RELEVANCE_DISTANCE_THRESHOLD = 1.2


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
# Structured data tools
# ============================================================

CUSTOMERS = {
    "alice@example.com": {
        "customer_id": "C-77",
        "name": "Alice",
        "plan_type": "self_serve",
    },
    "bob@example.com": {
        "customer_id": "C-88",
        "name": "Bob",
        "plan_type": "enterprise",
    },
}

SUBSCRIPTIONS = {
    "C-77": {"plan_days": 30, "amount_paid": 120.0, "days_used": 10},
    "C-88": {"plan_days": 30, "amount_paid": 90.0, "days_used": 15},
}


@tool
def lookup_customer(email: str) -> dict:
    """Look up a customer by email address.

    Returns the customer ID, name, and plan type (self_serve or enterprise).
    """
    if email not in CUSTOMERS:
        raise ValueError(f"No customer found for email '{email}'.")
    return CUSTOMERS[email]


@tool
def get_subscription(customer_id: str) -> dict:
    """Get subscription information for a customer.

    Requires the customer_id returned by lookup_customer.
    """
    if customer_id not in SUBSCRIPTIONS:
        raise ValueError(f"No subscription found for customer '{customer_id}'.")
    return SUBSCRIPTIONS[customer_id]


@tool
def calculate_refund(amount_paid: float, days_used: int, plan_days: int) -> dict:
    """Calculate a prorated subscription refund.

    Uses the amount paid, number of days used, and total days in the plan.
    """
    if amount_paid < 0:
        raise ValueError("amount_paid must be >= 0")
    if days_used < 0:
        raise ValueError("days_used must be >= 0")
    if plan_days <= 0:
        raise ValueError("plan_days must be > 0")
    if days_used > plan_days:
        raise ValueError("days_used cannot be greater than plan_days")

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
# Retrieval tool
# ============================================================

# FIX: marker so the validator can tell retrieval output from structured
# tool output, instead of guessing by looking for "SOURCE_ID:".
POLICY_MARKER = "[POLICY SEARCH RESULTS]"


@tool
def search_policy(query: str) -> str:
    """Search company policy documents.

    Use for questions about refund eligibility, cancellation windows,
    data retention, or plan terms. Policies differ between self-serve and
    enterprise plans, so mention the plan type in your query when known.
    """
    results = vectorstore.similarity_search_with_score(query, k=3)

    # FIX: filter on distance. Retrieval ALWAYS returns k chunks; without a
    # threshold, an off-topic question still gets three confident-looking
    # passages the model will then use.
    relevant = [
        (doc, distance)
        for doc, distance in results
        if distance <= RELEVANCE_DISTANCE_THRESHOLD
    ]

    if not relevant:
        return (
            f"{POLICY_MARKER}\n"
            "No policy document matched this query closely enough. "
            "Do not answer the policy question from your own knowledge; "
            "tell the user this is not covered by the indexed policies."
        )

    blocks = []
    for doc, _distance in relevant:
        # FIX: the numeric score is NOT sent to the model. It is a retrieval
        # diagnostic, and injecting it pollutes the answer's number space.
        blocks.append(
            f"SOURCE_ID: {doc.metadata['source']}\n"
            f"PLAN_TYPE: {doc.metadata.get('plan_type', 'all')}\n"
            f"CONTENT: {doc.page_content}"
        )

    return f"{POLICY_MARKER}\n\n" + "\n\n---\n\n".join(blocks)


tools = [lookup_customer, get_subscription, calculate_refund, search_policy]

# FIX: single source of truth for tool names, used by plan adherence below.
TOOL_NAMES = [t.name for t in tools]


# ============================================================
# LLM
# ============================================================

llm = ChatAnthropic(model=MODEL, temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# Planner
# ============================================================

class Plan(BaseModel):
    steps: list[str] = Field(
        description=(
            "A concise ordered list of steps needed to solve the request."
        )
    )


planner_llm = llm.with_structured_output(Plan)


def plan_node(state: AgentState):
    """Create a plan once before entering the agent loop."""
    user_message = state["messages"][-1].content

    # FIX: search_policy was missing from the tool list shown to the planner,
    # so no plan could ever include retrieval.
    prompt = f"""Create a concise plan for solving this support request.

User request:
{user_message}

Available tools:
1. lookup_customer(email) -> customer_id, name, plan_type
2. get_subscription(customer_id) -> plan_days, amount_paid, days_used
3. calculate_refund(amount_paid, days_used, plan_days) -> refund
4. search_policy(query) -> relevant policy document chunks

Policy rules differ by plan type, so look the customer up before
searching policies when the request concerns a specific customer.

Do not perform the task yourself. Only create the plan.
"""

    plan = planner_llm.invoke(prompt)

    print("\n=== PLAN ===")
    for i, step in enumerate(plan.steps, 1):
        print(f"{i}. {step}")

    return {
        "plan": plan.steps,
        "validation_errors": [],
        "revision_count": 0,
        # FIX: initialise the key so downstream nodes never see it missing.
        "human_rejected": False,
    }


# ============================================================
# Agent node
# ============================================================

def build_system_prompt(state: AgentState) -> str:
    plan_text = "\n".join(
        f"{i}. {step}" for i, step in enumerate(state["plan"], 1)
    )

    # FIX: the old prompt gave "[SOURCE_ID: refund_policy]" as the example —
    # an id that does not exist in the corpus, so the model would copy it and
    # fail citation validation. The example now uses a real id.
    content = (
        "You are a support assistant.\n\n"
        "Use tools to gather information. You need a customer_id from "
        "lookup_customer before calling get_subscription.\n\n"
        "Never invent customer, subscription, refund, or policy information.\n\n"
        "Policy rules differ between self-serve and enterprise plans. Check "
        "the customer's plan_type and cite the policy that applies to it.\n\n"
        "Every policy claim in your final answer must cite the SOURCE_ID of a "
        "chunk returned by search_policy, in this exact format:\n"
        "    [SOURCE_ID: refund_self_serve]\n"
        "Never cite a SOURCE_ID that search_policy did not return.\n\n"
        f"Follow this plan:\n{plan_text}\n\n"
        "Your final answer must be grounded in the tool outputs."
    )

    if BREAK_GROUNDING:
        content += "\nAlways round refund amounts to the nearest 10 dollars."

    return content


def agent(state: AgentState):
    response = llm_with_tools.invoke(
        [SystemMessage(content=build_system_prompt(state))] + state["messages"]
    )
    return {"messages": [response]}


# ============================================================
# Tools node with approval gate
# ============================================================

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

        # NOTE: interrupt() re-runs this node from the top on resume, so never
        # put side effects above this line.
        decision = interrupt(
            {
                "tool": "calculate_refund",
                "args": args,
                "estimated_refund": round(estimated, 2),
            }
        )

        if decision != "approve":
            return {
                "human_rejected": True,
                "messages": [
                    ToolMessage(
                        content=(
                            "Refund rejected by human reviewer. Do not "
                            "compute or state any refund amount."
                        ),
                        tool_call_id=call["id"],
                    )
                ],
            }

    return raw_tool_node.invoke(state)


# ============================================================
# Routing
# ============================================================

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "validate"


# ============================================================
# Validation helpers
# ============================================================

def as_text(content) -> str:
    """Anthropic returns content as a str or as a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    return str(content)


def extract_numbers(text: str) -> list[Decimal]:
    """Extract numbers so that 45, 45.0 and 45.00 compare equal."""
    matches = re.findall(r"(?<![\w.-])\d+(?:\.\d+)?", text)
    return [Decimal(value) for value in matches]


def extract_citations(text: str) -> set[str]:
    return set(re.findall(r"\[SOURCE_ID:\s*([^\]]+?)\s*\]", text))


def retrieved_sources(messages) -> set[str]:
    """Source ids actually returned by search_policy in this conversation."""
    sources: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = as_text(message.content)
        if POLICY_MARKER not in content:
            continue
        sources.update(re.findall(r"SOURCE_ID:\s*(\S+)", content))
    return sources


# ============================================================
# Validation
# ============================================================

def validate(state: AgentState):
    messages = state["messages"]
    final_answer = as_text(messages[-1].content)
    answer_numbers = extract_numbers(final_answer)

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    # FIX: build the numeric whitelist from STRUCTURED tool output only.
    # Policy chunks contain 14, 30, 90, 7, 48... — including them would let
    # the model state almost any plausible refund and still pass check 1.
    structured_text = "\n".join(
        as_text(m.content)
        for m in tool_messages
        if POLICY_MARKER not in as_text(m.content)
    )
    tool_numbers = extract_numbers(structured_text)

    errors = []

    # --- Check 1: numeric grounding ---------------------------------
    # Numbers inside citation markers are not data claims, so strip them.
    policy_text = "\n".join(
        as_text(m.content)
        for m in tool_messages
        if POLICY_MARKER in as_text(m.content)
    )
    policy_numbers = extract_numbers(policy_text)
    all_numbers = tool_numbers + policy_numbers

    answer_without_citations = re.sub(r"\[SOURCE_ID:[^\]]*\]", "", final_answer)

    # Tier 1: nothing invented from scratch.
    for number in extract_numbers(answer_without_citations):
        if number not in all_numbers:
            errors.append(
                f"Number {number} does not appear in any tool output."
            )

    # Tier 2: money must come from structured tools, never from policy prose.
    money_claims = [
        Decimal(m)
        for m in re.findall(
            r"\$\s*(\d+(?:\.\d+)?)", answer_without_citations
        )
    ]
    for amount in money_claims:
        if amount not in tool_numbers:
            errors.append(
                f"Monetary amount {amount} does not come from a structured "
                "tool output."
            )
    # --- Check 2: the stated refund is the computed one --------------
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

    # --- Check 3: citations are real and were retrieved --------------
    cited = extract_citations(final_answer)
    available = retrieved_sources(messages)

    for source in cited:
        if source not in KNOWN_SOURCES:
            errors.append(f"Citation '{source}' is not a real policy document.")
        elif source not in available:
            errors.append(
                f"Citation '{source}' exists but was not returned by "
                "search_policy in this conversation."
            )

    # FIX: the reverse check — retrieval happened but nothing was cited.
    if available and not cited:
        errors.append(
            "Policy documents were retrieved but the answer cites no SOURCE_ID."
        )

    # --- Check 4: plan adherence, unless a human overrode it ---------
    if state.get("human_rejected"):
        if computed_refund is None and answer_numbers:
            suspicious = [n for n in answer_numbers if n not in tool_numbers]
            if suspicious:
                errors.append(
                    f"Refund was rejected, but the answer states {suspicious}."
                )
    else:
        plan_text = " ".join(state["plan"]).lower()
        called = {m.name for m in tool_messages if hasattr(m, "name")}
        for tool_name in TOOL_NAMES:
            if tool_name in plan_text and tool_name not in called:
                errors.append(
                    f"The plan requires {tool_name} but it was never called."
                )

    print("\n=== VALIDATION ===")
    print("PASS" if not errors else "FAIL")
    for error in errors:
        print(f"- {error}")

    return {"validation_errors": errors}


def validation_router(state: AgentState):
    if state.get("human_rejected"):
        return END
    if not state["validation_errors"]:
        return END
    if state["revision_count"] < MAX_REVISIONS:
        return "retry"
    print("\nMaximum revision count reached.")
    return END


# ============================================================
# Retry node
# ============================================================

def retry(state: AgentState):
    errors = "\n".join(f"- {e}" for e in state["validation_errors"])
    new_count = state["revision_count"] + 1

    print(f"\n=== RETRY #{new_count} ===")

    return {
        "messages": [
            HumanMessage(
                content=(
                    "Your previous answer failed validation.\n\n"
                    "Produce the corrected answer in full, as if answering the user for "
                    "the first time. Do not apologise and do not reference this "
                    "correction.\n\n"
                    "Every fact must come from a tool output, and every policy claim must "
                    "cite a SOURCE_ID that search_policy actually returned. If a flagged "
                    "fact is genuinely present in a tool output, keep it and state it "
                    "accurately — do not delete facts to satisfy a check.\n\n"
                    f"Validation errors:\n{errors}"
                )
            )
        ],
        "revision_count": new_count,
        "validation_errors": [],
    }


# ============================================================
# Graph
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("plan", plan_node)
builder.add_node("agent", agent)
builder.add_node("tools", tools_with_approval)
builder.add_node("validate", validate)
builder.add_node("retry", retry)

builder.add_edge(START, "plan")
builder.add_edge("plan", "agent")
builder.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "validate": "validate"}
)
builder.add_edge("tools", "agent")
builder.add_conditional_edges(
    "validate", validation_router, {END: END, "retry": "retry"}
)
builder.add_edge("retry", "agent")

graph = builder.compile(checkpointer=MemorySaver())


# ============================================================
# Run
# ============================================================

def run(question: str, thread_id: str):
    print("\n" + "#" * 70)
    print(f"# {question}")
    print("#" * 70)

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [("user", question)]}, config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\nPAUSED — approval needed:", payload)
        print("Next node:", graph.get_state(config).next)
        answer = input("approve / reject: ")
        result = graph.invoke(Command(resume=answer), config)

    print("\n=== FINAL ANSWER ===")
    print(as_text(result["messages"][-1].content))


if __name__ == "__main__":
    # FIX: a question that genuinely needs BOTH the database and the docs.
    run(
        "Is bob@example.com eligible for a refund under our policy, "
        "and if so how much?",
        "rag-1",
    )

    # Policy-only question — no customer lookup needed.
    run("How long is customer data kept after cancellation?", "rag-2")

    # Out-of-corpus question — the retriever should decline.
    run("What is the maximum file upload size on the platform?", "rag-3")