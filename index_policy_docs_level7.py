"""
Level 7 — Policy document index.

Key design point: every document gets a UNIQUE source id.
Two documents that share an id cannot be distinguished by a citation,
which defeats the purpose of citation validation.

The interesting pairs:
  - refund_self_serve   vs refund_enterprise   (14 days vs 30 days)
  - retention_standard  vs retention_extended  (90 days vs 7 years)

A query like "what is the refund window?" must disambiguate on plan type
rather than simply matching the nearest paragraph.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


policy_docs = [
    Document(
        page_content=(
            "Customers on monthly or annual self-serve plans are eligible for "
            "a full refund if the request is submitted within 14 days of the "
            "original purchase date. Refunds apply only to first-time "
            "purchases of a plan tier — upgrades, downgrades, and add-on "
            "purchases (extra seats, storage, or API credits) are "
            "non-refundable regardless of timing. Refund requests must be "
            "submitted through the billing portal; requests made via email or "
            "support chat will be redirected there. Approved refunds are "
            "issued to the original payment method within 5-7 business days."
        ),
        metadata={"source": "refund_self_serve", "plan_type": "self_serve"},
    ),
    Document(
        page_content=(
            "Enterprise customers under an annual or multi-year contract are "
            "eligible for a prorated refund of unused contract months only if "
            "cancellation occurs within the first 30 days of the contract "
            "start date, commonly referred to as the evaluation period. After "
            "day 30, no refunds are issued for the remaining contract term "
            "except in cases of documented, uncured breach of the Service "
            "Level Agreement by the vendor. Refund requests must be submitted "
            "in writing to the assigned account manager, not through the "
            "billing portal. Custom implementation, onboarding, and "
            "integration fees are non-refundable under all circumstances."
        ),
        metadata={"source": "refund_enterprise", "plan_type": "enterprise"},
    ),
    Document(
        page_content=(
            "Self-serve customers may cancel their subscription at any time "
            "from account settings, with no cancellation fee. Cancellation "
            "takes effect at the end of the current billing cycle; customers "
            "retain access to paid features through that period. To avoid "
            "being charged for the next cycle, cancellation must be completed "
            "at least 48 hours before the renewal date. No partial-month "
            "refunds are issued for cancellations made mid-cycle."
        ),
        metadata={"source": "cancellation_self_serve", "plan_type": "self_serve"},
    ),
    Document(
        page_content=(
            "Enterprise accounts operate under negotiated terms that may "
            "override standard platform policies, including custom SLAs, "
            "dedicated account management, and non-standard billing cycles. "
            "Cancellation of an enterprise contract requires 90 days' written "
            "notice prior to the renewal date; contracts not cancelled within "
            "this window automatically renew for an additional term. "
            "Enterprise customers may also negotiate custom data retention "
            "periods, custom refund terms, and exceptions to standard usage "
            "limits, all of which must be documented in a signed order form "
            "addendum to take precedence over the default policies."
        ),
        metadata={"source": "enterprise_exceptions", "plan_type": "enterprise"},
    ),
    Document(
        page_content=(
            "Following account cancellation, customer data is retained for 90 "
            "days to allow for reactivation, after which it is permanently "
            "deleted from production systems. Backups containing customer "
            "data are purged on a rolling 12-month cycle. System and access "
            "logs are retained separately for 30 days for security and "
            "debugging purposes, independent of account status."
        ),
        metadata={"source": "retention_standard", "plan_type": "all"},
    ),
    Document(
        page_content=(
            "Enterprise customers subject to regulatory compliance "
            "requirements (e.g., financial or healthcare data handling) may "
            "elect extended data retention terms, with data retained for up "
            "to 7 years following contract termination to meet audit and "
            "legal hold obligations. Extended retention must be explicitly "
            "requested and documented in the customer's order form; without "
            "this election, enterprise accounts default to the standard "
            "90-day post-cancellation retention period. Data subject to an "
            "active legal hold is retained indefinitely regardless of the "
            "account's retention tier until the hold is lifted."
        ),
        metadata={"source": "retention_extended", "plan_type": "enterprise"},
    ),
]


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents(policy_docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="company_policies",
)

# All valid source ids — used by the agent to detect fabricated citations.
KNOWN_SOURCES = {doc.metadata["source"] for doc in policy_docs}


# ============================================================
# Part A exploration — run this file directly
# ============================================================

def explore():
    """Look at raw retrieval scores before wiring anything into the graph."""

    queries = [
        "can I get a refund after 20 days?",          # ambiguous: which plan?
        "how long is my data kept after I cancel?",   # ambiguous: which tier?
        "what is the maximum file upload size?",      # NOT in the corpus
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = vectorstore.similarity_search_with_score(query, k=3)

        for doc, distance in results:
            print(f"\n  source:   {doc.metadata['source']}")
            print(f"  distance: {distance:.4f}   (LOWER = more similar)")
            print(f"  preview:  {doc.page_content[:90]}...")


if __name__ == "__main__":
    print(f"Documents: {len(policy_docs)}")
    print(f"Chunks:    {len(chunks)}")
    print(f"Sources:   {sorted(KNOWN_SOURCES)}")
    explore()