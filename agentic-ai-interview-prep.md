# Agentic AI — Interview Preparation Reference

Built from a hands-on progression: raw LLM API call → structured output → tools →
agent → LangGraph → plan/validate/human-in-the-loop → agentic RAG.

**How to use this before an interview.** Read Part 0 and Part 7 first — the
decision table and the war stories are what actually differentiate candidates.
Everything else is reference. The war stories in Part 8 are yours: bugs you hit
and fixed. Interviewers remember those, not definitions.

---

## Part 0 — The five sentences that matter most

If you retain nothing else:

1. **LLM APIs are stateless.** "Conversation" is you resending the whole
   history every call. Everything about cost, latency, and memory follows.
2. **The model never executes anything.** Tool calling is the model emitting a
   structured *request*; your code decides whether to run it. All security and
   authorisation lives in your executor, never in the prompt.
3. **An agent is a system where the model determines control flow at runtime.**
   If you fixed the sequence at write time, it's a chain or a workflow.
4. **A schema with no representable failure state guarantees fabrication.** If
   the model has no legal way to say "I don't know" or "this isn't a ticket,"
   it will invent something that validates perfectly.
5. **Validation is cheap on structured output and heuristic on prose.** Every
   grounding check you write against free text is a regex you will later find a
   hole in.

---

## Part 1 — LLM fundamentals

### 1.1 Statelessness and the message list

**What it is.** The API takes a list of messages and returns a message. It
remembers nothing between calls.

**Why it matters.** Multi-turn conversation is an illusion you construct.
Token cost grows roughly quadratically over a long conversation because turn
N resends turns 1..N-1. This is the root of the memory-management problem.

**How it works.**

```
Turn 1:  [user]                                    → assistant
Turn 2:  [user, assistant, user]                   → assistant
Turn 3:  [user, assistant, user, assistant, user]  → assistant
                    ↑ resent every time, and paid for every time
```

**Example.**

```python
messages = []
while True:
    messages.append({"role": "user", "content": input("You: ")})
    response = client.messages.create(
        model=MODEL, max_tokens=1024, system=SYSTEM, messages=messages
    )
    text = "\n".join(b.text for b in response.content if b.type == "text")
    messages.append({"role": "assistant", "content": text})
    print(text, response.usage.input_tokens)
```

**Q: Are LLM APIs stateful? What follows from that?**
A: Stateless. Every call must carry the full context. That means conversation
memory is application state you own; input tokens grow with history so cost
scales superlinearly with conversation length; and "memory" strategies
(trimming, windowing, summarisation, retrieval over past turns) are engineering
decisions, not model features.

**Q: How do you manage a conversation that outgrows the context window?**
A: Four options with different trade-offs. Trim to the last N turns (cheap,
loses old facts). Summarise older turns into one message (retains gist, costs
an extra call, lossy). Retrieve relevant past turns from a vector store
(scales, adds retrieval failure modes). Or restructure so long context isn't
needed. In LangGraph, `trim_messages` in the agent node is the usual hook.

### 1.2 System vs user messages

**What it is.** `system` carries instructions and persona; `user` carries data.
In the Anthropic API `system` is a **top-level parameter**, not a message; in
the OpenAI API it's a message with `role: "system"`.

**Why it matters.** Keeping instructions separate from user-supplied data is
the first and cheapest defence against prompt injection. It also makes the
instruction block a stable prefix — which is what makes prompt caching possible.

**Q: Why does the system/user split matter for security?**
A: It creates a channel distinction. Instructions the operator controls live in
one place, untrusted input in another. It is not a hard boundary — a
sufficiently crafted user message can still influence behaviour — which is why
real defences are downstream: validate outputs, gate tool execution in code,
and never let prompt content decide authorisation.

### 1.3 Temperature, tokens, stop reasons

**Temperature** flattens or sharpens the sampling distribution. Use 0 for
extraction, classification, routing, and tool selection; higher for ideation.

Nuance worth stating unprompted: **temperature 0 is not a determinism
guarantee.** Batching, floating-point non-associativity on GPUs, and
provider-side model updates all mean identical inputs can produce different
outputs. It is near-determinism, not reproducibility.

Second nuance: high temperature mostly resamples *where the model is already
uncertain*. On a factual question it changes phrasing; on an open one it
changes substance. That's why temperature is a per-task setting, not a global.

**`max_tokens`** caps the *output*, not the total. The **context window** is
input + output combined. Different limits, commonly confused.

**`stop_reason`** is your control signal:

| value | meaning | what you do |
|---|---|---|
| `end_turn` | model finished | use the text |
| `max_tokens` | truncated mid-generation | **do not parse it** — raise or retry with a higher cap |
| `tool_use` | model wants a tool | execute and loop |

**Q: Why check `stop_reason` before parsing?**
A: A truncated response is syntactically broken but silently plausible. If you
feed it to a JSON parser you get a validation error whose real cause is a
config constant three layers away — and if you retry blindly, you pay twice for
a deterministic failure. Truncation is *non-retryable*; distinguishing
retryable from non-retryable failures is the core of a sane retry policy.

---

## Part 2 — Structured output

### 2.1 Three levels of rigour

```
1. Ask nicely      prompt says "return JSON"     → fails on fences, preambles
2. Constrain       JSON Schema via tool calling  → shape enforced at generation
3. Validate        Pydantic + semantic checks    → catches what schema can't
```

**Why it matters.** Level 1 works most of the time, which is the problem.
Level 2 eliminates the syntax failure class entirely. Level 3 is still
necessary because **schema conformance is not correctness**: a well-typed
object can be entirely fabricated.

**Example — level 2 via a forced tool.**

```python
tools = [{
    "name": "record_ticket",
    "description": "Record a structured support ticket.",
    "input_schema": Ticket.model_json_schema(),   # single source of truth
}]

response = client.messages.create(
    model=MODEL, max_tokens=1024, tools=tools,
    tool_choice={"type": "tool", "name": "record_ticket"},
    messages=[{"role": "user", "content": raw_ticket}],
)

block = next(b for b in response.content if b.type == "tool_use")
ticket = Ticket.model_validate(block.input)
```

Note what disappeared: no markdown-fence stripping, no "return ONLY JSON" in
the prompt, no assistant prefill trick, and usually no retry loop.

**Example — level 3, the semantic check the schema can't express.**

```python
@field_validator("summary")
@classmethod
def one_sentence(cls, v: str) -> str:
    if v.count(".") > 1:
        raise ValueError("summary must be a single sentence")
    return v
```

### 2.2 Bounded retry with error feedback

```
LLM → parse/validate ──valid──→ done
          │
        invalid
          │
          ├─ append failed output + error text to conversation
          ├─ retry (max 2)
          └─ still failing → raise / fallback
```

**Q: What's your retry strategy for invalid model output, and why bounded?**
A: Feed the validation error back into the conversation so the model can see
its own mistake and correct it — models are good at this. Bound it hard,
because an unbounded retry against a deterministic failure is an infinite loop
attached to a credit card. And classify first: truncation and schema drift
won't fix themselves on retry, so those raise instead.

### 2.3 The fabrication trap

**The single most useful idea in this section.** If your output schema has no
way to express "not applicable," "unknown," or "this isn't what you think it
is," the model is *forced* to invent. It will produce a perfectly valid,
completely fabricated object, and your validator will pass it.

```python
class Ticket(BaseModel):
    is_ticket: Literal[True]
    summary: str
    priority: Literal["low", "medium", "high", "urgent"]
    customer_name: str | None = None      # None must be legal AND prompted

class NotATicket(BaseModel):
    is_ticket: Literal[False]
    reason: str

Result = Annotated[Ticket | NotATicket, Field(discriminator="is_ticket")]
adapter = TypeAdapter(Result)
result = adapter.validate_json(text)
```

`discriminator="is_ticket"` makes this a **discriminated union**: Pydantic reads
one field and validates against exactly one branch, instead of trying both and
producing a confusing double error.

**Q: How do you stop a model inventing values for fields it has no data for?**
A: Two things together. Make absence *representable* — optional types, a
refusal branch in a union. And make it *permitted* — say explicitly in the
prompt that null is the correct answer when the data isn't present. Types
without the prompt instruction still produces fabrication, because the model
reads the required-looking schema as an instruction to fill it.

**Q: Where does `tool_choice` bite you here?**
A: `{"type": "tool", "name": X}` forces that specific tool, which removes the
model's ability to decline. If your only tool is `record_ticket`, "hello" gets
recorded as a ticket. Use `{"type": "any"}` with a refusal tool alongside, so
the model must call *something* but can choose the refusal.

| `tool_choice` | meaning | use for |
|---|---|---|
| `{"type": "auto"}` | may or may not use a tool | agents |
| `{"type": "any"}` | must use *some* tool | routing, classification |
| `{"type": "tool", "name": X}` | must use *this* tool | single-shape extraction |
| `{"type": "none"}` | may not use tools | forcing a final answer |

### 2.4 Pydantic gotchas that cost real time

- `field: str | None` is **required and nullable**. It is *not* optional. Use
  `= None` to make omission legal. Decide which you want deliberately —
  requiring explicit `null` forces the model to make a decision rather than
  forget.
- `model_validate_json()` handles parse *and* validation, so malformed JSON
  surfaces as a `ValidationError` too — one `except` covers both classes.
- Generate the prompt's schema from the model (`model_json_schema()`) rather
  than hand-writing it. Hand-written schema drifts from code, and the failures
  look like model stupidity.
- `Decimal` compares by value, so `45`, `45.0` and `45.00` are equal — useful
  when checking whether a number in prose matches a tool output.

---

## Part 3 — Tools

### 3.1 The mechanism

```
user:      "refund for bob@example.com?"
assistant: [tool_use  id=abc  name=lookup_customer  input={"email": ...}]
             ← stop_reason == "tool_use"
user:      [tool_result  tool_use_id=abc  content='{"customer_id": "C-88"}']
assistant: "..."
             ← stop_reason == "end_turn"
```

Three details people get wrong:

- **`tool_result` goes back as a `user` message** — it's a content block type,
  not a new role.
- **`tool_use_id` must match**, which matters the moment the model requests two
  tools in one turn.
- **The assistant turn must be appended verbatim**, or the pairing breaks.

**Handle parallel calls.** A model can emit several `tool_use` blocks in one
turn. Code that only reads `content[0]` silently drops the rest.

```python
for block in response.content:
    if block.type != "tool_use":
        continue
    ...
messages.append({"role": "user", "content": tool_results})   # all in ONE turn
```

### 3.2 Safe dispatch

```python
TOOL_REGISTRY = {
    "lookup_customer": {"handler": lookup_customer, "args": LookupArgs},
}

def execute_tool(name: str, raw_input: dict):
    if name not in TOOL_REGISTRY:          # dict lookup, never eval()
        raise ValueError(f"Unknown tool: {name}")
    tool = TOOL_REGISTRY[name]
    args = tool["args"].model_validate(raw_input)   # model output is untrusted
    return tool["handler"](**args.model_dump())
```

**Q: Does the LLM execute the tool?**
A: No. It emits a name and a JSON argument object. Your code looks the name up
in a registry and decides whether to run it. That's why authorisation, rate
limits, dry-run modes, and approval gates all live in the executor — a prompt
can be talked around, a Python `if` cannot.

**Q: Why a registry rather than dynamic dispatch?**
A: The tool name is a string produced by a model, i.e. untrusted input.
`getattr(module, name)` or `eval` turns that into arbitrary code execution. A
dict lookup that raises on an unknown key fails safely.

### 3.3 Error handling at the tool boundary

Errors should go **back to the model**, not up the stack — the model reads them
and adapts. But separate expected failures from your own bugs:

```python
except (ValidationError, ValueError, KeyError) as e:
    content, is_error = str(e), True                    # model can act on this
except Exception as e:
    logger.exception("Tool %s crashed", name)
    content, is_error = "Internal error executing tool.", True
```

**Why the second message is vague:** whatever you put in a `tool_result` enters
the model's context and may surface in the user-facing answer. Raw exception
text can carry connection strings, file paths, SQL. Sanitise at the boundary.

LangGraph's `ToolNode` catches exceptions by default (`handle_tool_errors=True`)
and returns them as `ToolMessage`s. Convenient — but it means a genuine bug in
your tool gets fed to the model instead of surfacing. Know it's happening.

**Q: What makes a good tool description?**
A: The description is prompt, not documentation — it's how the model chooses.
Wrong-tool selection is usually fixed in the description, not the system
prompt. State what the tool is for, what it returns, and any precondition
("requires a customer_id from lookup_customer"). With LangChain's `@tool`
decorator, the **docstring becomes the description**, so docstring quality is
functional, not stylistic.

---

## Part 4 — Agents

### 4.1 The definition that actually distinguishes

| | who authors control flow | example |
|---|---|---|
| **Chain** | you, fixed at write time | prompt → LLM → parse → return |
| **Workflow** | you, with branches you defined | classify, then route to one of three prompts |
| **Agent** | the model, at runtime | model chooses tools and sequence until done |

Real systems are hybrids, and saying so is better than picking a label: the
model chooses *which* tools and *how many* steps; you fix the loop shape, the
tool set, the iteration cap, and the termination condition.

### 4.2 ReAct

Reason → Act → Observe → repeat. Your loop:

```python
while response.stop_reason == "tool_use":
    execute the requested tools
    append assistant turn + tool_result turn
    call the model again
```

That's the entire pattern. `create_react_agent` is this, as a graph.

### 4.3 The four controls that make it safe

**1. Hard iteration cap.** Non-negotiable.

**2. Duplicate-call detection.** Models get stuck repeating a call; cheaper to
detect than to wait for the cap.

```python
signature = (tool_name, json.dumps(tool_input, sort_keys=True))
# NOT frozenset(tool_input.items()) — TypeError on nested args
if signature in already_executed:
    return "This exact call was already executed. Reuse the previous result."
```

**3. Graceful degradation at the cap.** Don't raise — make one final call with
`tool_choice={"type": "none"}` so the model answers with what it gathered.
*"I found the customer and subscription but couldn't complete the calculation"*
beats an exception.

**4. Structured final answer.** A `final_answer` tool with a Pydantic schema.
Prose output isn't composable; typed output is.

Corollary: **don't ask the model for information you already have.** If you're
tracking `tools_used` in code, don't put it in the schema — every field the
model fills is a field it can get wrong, and you pay tokens for it.

### 4.4 Cost and latency

An N-step agent is N sequential round-trips, each resending the full history.
Cost is superlinear in steps, not linear. Instrument it:

```python
total_in += response.usage.input_tokens
total_out += response.usage.output_tokens
```

**Q: How would you reduce agent cost and latency?**
A: In rough order of impact — (1) **prompt caching** on the stable prefix
(system prompt + tool definitions), which is large in a loop that resends them
every iteration; (2) fewer iterations: better tool descriptions and merged
tools reduce round-trips; (3) a smaller model for routing/classification steps,
reserving the large one for synthesis; (4) parallel tool calls where the model
supports it; (5) hardest question first — replace the agent with a
deterministic pipeline if the sequence is actually knowable.

**Q: When should you NOT use an agent?** *(the question that separates
candidates)*
A: When the steps are known in advance. If you always call A then B, an agent
is a slower, costlier, less testable way to write two function calls — you've
replaced a deterministic sequence with a probabilistic one and added N
round-trips of latency. Agents earn their cost only when the required sequence
genuinely depends on input you can't anticipate. Also avoid them where a wrong
action is expensive and unrecoverable, unless you gate execution behind
approval.

---

## Part 5 — LangGraph

### 5.1 Why a framework at all

Honest answer: **for a simple tool loop, you don't need one.** A `while` loop
with a cap is fine, and saying so demonstrates judgement. LangGraph earns its
place when you need:

- **Persistence and resumability** — checkpointed state after every node; crash
  at step 4 of 6 and resume from 4.
- **Human-in-the-loop** — suspend mid-execution and survive a process restart.
  A `while` loop cannot do this without inverting your whole control flow.
- **Non-linear topologies** — cycles, branches, fan-out/join, subgraphs.
- **Node-level streaming and observability** for free.

### 5.2 The four primitives

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # reducer: APPEND
    plan: list[str]                           # default reducer: OVERWRITE
    revision_count: int
```

- **State** — a `TypedDict` passed between nodes. Nodes return *partial*
  updates that get merged, not replacements.
- **Reducers** — how updates merge. Default is overwrite; `add_messages`
  appends. This is the most-asked LangGraph detail.
- **Nodes** — plain functions, `state -> partial update`.
- **Edges** — fixed, or conditional (a function returning the next node's name).

```
      START
        │
        ▼
   ┌─────────┐
   │  agent  │◄──────────┐
   └────┬────┘           │
        │                │
  should_continue?       │
   │          │          │
"tools"     END      ┌───┴────┐
   │          │      │ tools  │
   └──────────┼─────►└────────┘
              ▼
             END
```

```python
builder.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", END: END},
)
```

Always pass the explicit path map — without it LangGraph infers destinations
and `draw_ascii()` renders edges to every node.

**Q: What is a stateful graph, and what's a reducer?**
A: Nodes read from and write to a shared typed state rather than passing return
values along a pipeline. A reducer defines how a node's partial update merges
into the existing state per key — `add_messages` appends to conversation
history while a plain `list[str]` field gets replaced. Designing state means
deciding, per key, what accumulates and what gets overwritten.

**Q: What does `create_react_agent` abstract away?**
A: It builds a two-node cyclic graph — a model node and a `ToolNode` — with a
conditional edge routing to tools when the last `AIMessage` has `tool_calls`
and to `END` otherwise, plus a default `MessagesState`. It gives you exactly
one topology. The moment you need a node before the loop, a validation node
after it, a second model with different tools, or an approval gate, you're back
to `StateGraph`.

### 5.3 Checkpointing and threads

```python
graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "user-123"}}
graph.invoke({"messages": [("user", "refund for alice@example.com?")]}, config)
graph.invoke({"messages": [("user", "and for bob?")]}, config)   # remembers
```

On invoke, LangGraph loads the latest checkpoint for that `thread_id`, applies
your input **through the reducers**, runs, and saves a new checkpoint after
every node. `graph.get_state_history(config)` gives you every checkpoint —
which is what makes time-travel debugging and human-in-the-loop possible.

Three production notes:

- `MemorySaver` is a dict in RAM. Process dies, conversations die; two workers
  means two memories. Use `SqliteSaver` / `PostgresSaver`.
- **`thread_id` is an authorisation boundary.** If it comes from user input,
  one user can read another's conversation by guessing. Derive it server-side
  from the authenticated session.
- State grows unbounded. Turn 50 resends turns 1–49.

- `recursion_limit` (default 25) replaces your hand-rolled `MAX_ITERATIONS`.

### 5.4 Human-in-the-loop

```python
decision = interrupt({"tool": "calculate_refund", "args": args})
if decision != "approve":
    return {"human_rejected": True, "messages": [ToolMessage(...)]}
```

```python
result = graph.invoke({"messages": [...]}, config)
if "__interrupt__" in result:
    print(graph.get_state(config).next)          # ('tools',)
    result = graph.invoke(Command(resume="approve"), config)
```

**Gotcha that catches everyone: `interrupt()` re-runs the node from the top on
resume.** Everything above the interrupt executes twice. Never put side effects
— writes, emails, charges — above an `interrupt()` call.

**Q: How does human-in-the-loop work mechanically?**
A: `interrupt()` persists graph state to the checkpointer and returns control
to the application, which can then be a web request that ends. Approval arrives
later — minutes or days — and the application resumes by `thread_id` with
`Command(resume=...)`. The process can die in between. Gate on business rules
in code, not prompts, and place gates before irreversible or costly actions.

**Q: Where do you put approval gates?**
A: Before anything irreversible or expensive — payments, deletes, external
sends. The threshold is a product decision, not a technical one.

---

## Part 6 — RAG

### 6.1 The pipeline

```
INDEXING (offline)     docs → split → embed → vector store
QUERY TIME             question → embed → top-k nearest chunks → prompt
```

Embeddings map text to vectors so that semantic similarity becomes geometric
proximity. That's why "can I get my money back?" retrieves a refund-eligibility
document with zero shared keywords — and the one-line answer to "why not grep?"

### 6.2 The two decisions that carry the quality

**Chunk size.** Too large dilutes the embedding — one vector representing five
topics is near none of them. Too small severs necessary context. Overlap exists
so a passage spanning a boundary isn't lost. There is no universal answer; it
depends on document structure, and "it depends, and here's on what" is the
correct interview response.

**Similarity is not relevance.** The retriever returns k chunks *whether or not
any are relevant*. Ask something absent from the corpus and you get three
confident-looking passages the model will happily use.

```python
results = vectorstore.similarity_search_with_score(query, k=3)
relevant = [(d, s) for d, s in results if s <= THRESHOLD]
if not relevant:
    return "No policy document matched closely enough. Do not answer from
            your own knowledge."
```

Note: Chroma returns a **distance** — *lower* is better. Labelling it "score"
misleads readers and, if you inject it into the prompt, the model.

### 6.3 Chain RAG vs agentic RAG

| | RAG chain | agentic RAG |
|---|---|---|
| retrieval | always, fixed step | a tool the model may call |
| cost/latency | lower, predictable | higher, variable |
| can skip retrieval | no | yes |
| multi-hop / refined queries | no | yes |
| combines with other tools | awkward | natural |
| testability | high | lower |

A retriever as a tool is just another `@tool`. That's the whole implementation
difference.

### 6.4 Citations must identify exactly one source

**The bug worth remembering:** two different documents given the same
`source` metadata makes every citation ambiguous, and citation validation
becomes theatre — it passes whichever document the model meant. A citation is
only useful if it resolves to exactly one source.

Validate in both directions:

```python
for source in cited:
    if source not in KNOWN_SOURCES:
        error(f"'{source}' is not a real document")        # fabricated
    elif source not in retrieved_in_this_conversation:
        error(f"'{source}' exists but was not retrieved")  # misattributed

if retrieved and not cited:
    error("Documents retrieved but nothing cited")         # unattributed
```

**Q: RAG or fine-tuning?**
A: Different problems. RAG injects *knowledge* that changes often, needs
citation, or is access-controlled per user — you can update the index without
touching the model. Fine-tuning shapes *behaviour, format, and style*, and
bakes knowledge in at a cost that makes freshness impractical. Most "we need
fine-tuning" instincts are actually retrieval or prompting problems.

**Q: How do you evaluate a RAG system?**
A: Two layers, measured **separately** — that separation is the answer.
*Retrieval*: recall@k, MRR, hit rate on a labelled query→document set; you can
measure this with no LLM involved. *Generation*: faithfulness (is every claim
supported by retrieved context?), answer relevance, citation accuracy. Blurring
them means a bad answer can't be diagnosed — you can't tell whether retrieval
missed the document or the model ignored it.

**Q: Retrieval quality is bad. What do you try?**
A: In order of effort — inspect actual retrieved chunks for a failing query
(almost always diagnostic on its own); tune chunk size and overlap; hybrid
search (BM25 keyword + vector) for exact terms like IDs and product names that
embeddings handle poorly; a cross-encoder re-ranker over the top 20; query
rewriting or expansion; better metadata filters. Reach for a different
embedding model later than instinct suggests.

---

## Part 7 — Validation, evaluation, production

### 7.1 Deterministic checks beat LLM judges where expressible

Grounding check, in tiers:

```python
# Tier 1: every number in the answer appears in SOME tool output
# Tier 2: every MONETARY figure appears in STRUCTURED tool output
#         (policy prose must not be able to authorise a refund amount)
```

**Q: How do you handle hallucination?**
A: Layered, from strongest to weakest. (1) Structural — make refusal
representable in the schema, or you've mandated fabrication. (2) Grounding —
deterministic checks that every claim traces to a tool output or retrieved
chunk. (3) Attribution — require citations and verify they were actually
retrieved. (4) Bounded correction — feed failures back once or twice, then
stop. Prompting ("be accurate") is the weakest layer and the one most
candidates name first.

**Q: When is an LLM judge appropriate?**
A: When the property isn't expressible in code — tone, helpfulness, whether an
answer addresses the question. For anything checkable (does this number match
the tool output, is this citation real, is this valid JSON) use code: it's
free, instant, deterministic, and can't hallucinate its verdict.

### 7.2 Validator design — the hard-won lessons

**Precision matters more when a validator drives automatic rewrites.** A
false-positive that only logs is noise; a false-positive wired to a retry loop
destroys correct answers. *(See war story 3.)*

**Presence-checking vs field-binding.** "This number appears somewhere in the
tool outputs" is weak — it whitelists every number in every tool result, so the
model can shuffle values between fields freely. Bind the specific claim to the
specific field.

**Validators must defer to authoritative decisions.** If a human rejected an
action, a rule demanding that action was performed is unsatisfiable and will
burn every retry. Human decisions are state that downstream checks read.

**Q: Why bound the correction loop?**
A: A validator that can always reject plus a model that can always fail is an
infinite loop with a credit card attached. Also, if the validator is wrong, the
model's cheapest path to compliance is *deleting* the flagged content — so
bound it and log exhaustion for a human.

### 7.3 Observability

What to record per run: input, final output, every tool call with arguments and
result, per-step token counts and cost, latency per node, validation outcomes,
retry count, model and prompt version, trace ID.

**Q: How would you monitor an LLM application?**
A: Tracing is the backbone — a trace per request with a span per model call and
tool call, so you can reconstruct any single bad output. On top: cost and token
dashboards per endpoint, latency percentiles (p95 matters more than mean
because agents have long tails), tool error rates, validation failure rates by
rule, and retry/exhaustion counts. Then offline evals on a fixed dataset gating
deploys. LangSmith does this for LangChain/LangGraph; OpenTelemetry-based
options are vendor-neutral. Build a crude version by hand once — it's what
taught me what a trace *is*.

**Q: How do you evaluate an agent, as opposed to a single LLM call?**
A: Per-step and end-to-end are different questions. End-to-end: task success
rate on a fixed suite of realistic inputs. Per-step: was the right tool chosen,
were the arguments correct, was the trajectory efficient (steps taken vs
minimum). Plus safety: how often did it invoke a gated action, and did any
irreversible action fire without approval. A suite of 30–50 real cases with
known-good outcomes, run on every change, beats any amount of eyeballing.

### 7.4 Prompt and version management

Prompts are code: version them, review them, and record which version produced
which output. A prompt change is a deploy — it can regress behaviour as badly
as a code change and, without eval, more invisibly.

---

## Part 8 — War stories

These are the strongest thing you have. They are specific, they show
debugging, and none of them are memorised.

**1. The model was right and my validator crashed.** Fed a non-ticket input,
the model correctly returned a `NotATicket` refusal — and my code, which
validated only against `Ticket`, rejected it and raised. *Lesson: a validation
error is not proof the model was wrong. Shipped as-is, I'd have watched the
error rate and concluded "the LLM is unreliable" while the bug was entirely in
my Python.*

**2. Blocking the tool didn't block the capability.** A human rejected the
refund calculation at an approval gate. The model already had `amount_paid`,
`days_used` and `plan_days` from earlier tool calls, did the arithmetic itself,
and reported a figure anyway. *Lesson: a gate on execution is not a gate on
output. If a rejection must produce no answer, that's enforced downstream.*

**3. A bad validator destroyed a correct answer.** My grounding check excluded
retrieved policy text from the trusted-number pool. For a policy question, the
answer's numbers (90 days, 7 years) were all policy-derived — so the check
failed correct facts, the retry loop fired, and the model's cheapest path to
compliance was deleting every number. Final answer: "data is retained for a
period." Validation: PASS. *Lesson: validators wired to automatic rewrites need
far higher precision than validators that only warn. And a retry instruction
must forbid deletion as a correction strategy.*

**4. My validator fought the human.** A plan-adherence rule demanded
`calculate_refund` be called; a human had just rejected it. Unsatisfiable, so
the graph burned every retry losing an argument with itself. *Lesson:
authoritative decisions must be readable by downstream checks.*

**5. Adding RAG silently weakened an existing check.** Policy chunks contain
14, 30, 90, 7, 48 — once they entered the trusted-number pool, almost any
plausible refund figure passed grounding. *Lesson: different provenance needs
different validation; don't merge trust pools.*

**6. Planning added cost and bought nothing.** On a three-tool problem with an
obvious sequence, the plan node produced exactly what the agent would have done
anyway, for an extra call of latency. *Lesson: planning earns its place when
the tool space is large or the sequence isn't inferable — measure rather than
assume.*

---

## Part 9 — Architecture decision reference

### Simple chain (prompt → LLM → parse)

- **Use when** the task is one step with a known shape: extraction,
  classification, summarisation, rewriting.
- **Don't use when** the answer depends on live data or requires several
  dependent lookups.
- **Advantages** cheapest, fastest, fully testable, trivially observable.
- **Limitations** no external data, no adaptation.
- **Failure modes** malformed output; fabricated values in required fields;
  truncation parsed as valid.
- **Production** schema-enforced output, Pydantic + semantic validators,
  bounded retry with error feedback, `stop_reason` check, prompt versioning.

### Deterministic workflow (fixed steps, coded branches)

- **Use when** the sequence is known but conditional: route by category, then
  handle each path.
- **Don't use when** the required steps genuinely can't be enumerated.
- **Advantages** predictable cost and latency; each branch testable in
  isolation; failures are localisable.
- **Limitations** rigid; every new case is a code change.
- **Failure modes** misrouting at the classification step, silently sending
  work down the wrong branch.
- **Production** measure routing accuracy separately from downstream quality;
  add a fallback branch; force the router's output into an enum.

### Tool-using agent (ReAct loop)

- **Use when** the sequence depends on data only discoverable at runtime, or
  the tool space is large.
- **Don't use when** the steps are knowable — you're paying N round-trips and
  losing determinism for nothing.
- **Advantages** adapts to input; recovers from tool errors; one system handles
  many query shapes.
- **Limitations** superlinear cost, long-tail latency, hard to test, non-
  deterministic trajectories.
- **Failure modes** loops and repeated calls; wrong tool selection from vague
  descriptions; parallel calls dropped by naive code; unbounded iteration;
  fabricated final answers despite correct tool results.
- **Production** iteration cap + duplicate detection + graceful degradation;
  structured final answer; sanitised tool errors; per-run tracing; approval
  gates before irreversible actions; prompt caching on the stable prefix.

### Stateful graph (LangGraph)

- **Use when** you need persistence, resumability, human-in-the-loop, or a
  topology with branches, validation stages, or multiple models.
- **Don't use when** a `while` loop with a cap does the job — say this out
  loud in an interview.
- **Advantages** durable state, suspend/resume across process restarts,
  inspectable execution, node-level streaming.
- **Limitations** framework surface area, version churn, hidden defaults (e.g.
  `ToolNode` swallowing exceptions), debugging through an abstraction.
- **Failure modes** wrong reducer (overwrite where you meant append); system
  message placed in state and re-appended each turn; side effects above an
  `interrupt()` running twice; unbounded state growth; `thread_id` leakage
  across users.
- **Production** durable checkpointer (Postgres), server-derived `thread_id`,
  message trimming or summarisation, `recursion_limit` set explicitly, tracing.

### Agentic RAG

- **Use when** answers depend on a document corpus *and* structured systems,
  and the model should decide whether and how to retrieve.
- **Don't use when** a fixed retrieve-then-answer chain suffices — it's cheaper
  and far easier to evaluate.
- **Advantages** skips retrieval when unnecessary, multi-hop with refined
  queries, combines documents with database tools.
- **Limitations** retrieval quality dominates output quality; two evaluation
  layers instead of one; higher variance.
- **Failure modes** confidently retrieved irrelevant chunks (no threshold);
  citations to documents never retrieved; ambiguous source IDs; conclusions
  drawn from documents without attribution; chunk boundaries severing context.
- **Production** relevance threshold with an explicit "not covered" path;
  unique source IDs; two-way citation validation; separate retrieval and
  generation metrics; hybrid search and re-ranking when plain vector search
  underperforms; index freshness and re-indexing strategy.

---

## Part 10 — Rapid fire

**Chain vs workflow vs agent?** Who authors control flow: you at write time,
you with branches, or the model at runtime.

**LangChain vs LangGraph?** LangChain is components and integrations — models,
tools, retrievers, output parsers — plus composition for mostly linear flows.
LangGraph is an orchestration layer for stateful, cyclic, multi-step
execution with checkpointing and human-in-the-loop. You use LangChain
components *inside* LangGraph nodes. LangChain when the flow is a pipeline;
LangGraph when it's a state machine.

**How does tool calling work?** Model emits name + JSON args; your executor
validates and runs it; the result goes back as a `tool_result` in a user turn;
loop until `stop_reason` is terminal.

**Preventing infinite loops?** Hard cap, duplicate-call detection, token/cost
budget, graceful degradation instead of raising.

**Temperature 0 = deterministic?** Near-deterministic. Batching, GPU
floating-point non-associativity, and provider model updates break exact
reproducibility.

**Context window vs max_tokens?** Window is total input + output; `max_tokens`
caps output only.

**Structured output — how?** Tool/function schema enforcement at generation,
then Pydantic for shape *and* semantics. Never prompt-and-parse in production.

**Where do you enforce permissions?** In the executor, in code. Never the
prompt.

**Biggest cost lever in an agent?** Prompt caching on the stable prefix, then
reducing iteration count.

**How do you know it works?** A fixed eval suite of real cases with known-good
outcomes, run on every prompt or code change, measuring task success, tool
selection accuracy, trajectory efficiency, and safety-gate behaviour.

---

## Appendix — Still to build (Level 8)

Not yet implemented, so don't claim it:

- Offline eval harness with a labelled dataset and regression gating
- LangSmith or OpenTelemetry tracing
- Retry with exponential backoff on API errors (rate limit vs overloaded vs
  invalid request — different responses)
- FastAPI wrapper with streaming, timeouts, and per-user thread IDs
- Durable checkpointer (Postgres) and message trimming
- Prompt versioning and a prompt registry
