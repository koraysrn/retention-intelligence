# Agentic AI / RAG Design

> **Status: prototype, not production RAG.**
> The shipped [`src/agents/rag.py`](../src/agents/rag.py) is a deterministic
> keyword-overlap retriever over four documents; it does **not** use pgvector,
> embeddings, a reranker, citations or an LLM eval set. The workflow below is
> the design target. The prototype is kept out of the default `make pipeline`;
> run it explicitly with `make agents`. Do not describe this repository as
> having “production RAG”.

## Design target

Instead of sending standard emails to customers at high churn risk, a
multi-agent system would produce personalized content based on past purchases
and categories and, when necessary, forward a summary report to a sales
representative.

## 1. Workflow

```mermaid
flowchart LR
    T["Trigger: high risk + assignment to the experiment group"] --> O["Orchestrator Agent"]
    O --> P["Profile Agent: customer 360 (Feature Store + CDP)"]
    O --> R["RAG Agent: product catalog + campaign history + brand tone"]
    P --> G["Content Generation Agent (LLM): personalized offer draft"]
    R --> G
    G --> V["Validation Agent: tone, KVKK/GDPR, PII, hallucination"]
    V -->|"approve"| S["Dispatch Agent: ESP / CRM"]
    V -->|"reject / suspicious"| E["Escalation Agent"]
    E -->|"summary report + recommended action"| REP["Sales representative (CRM task)"]
```

## 2. Agents and Responsibilities

| Agent | Responsibility | Tools used |
|---|---|---|
| **Orchestrator** | Plans the workflow, invokes agents, manages retry/fallback | LangGraph state machine |
| **Profile Agent** | Customer history, category affinity, LTV, risk reason (SHAP) | Feature Store API, CRM |
| **RAG Agent** | Product catalog, past campaign performance, brand guideline context | pgvector, reranker |
| **Content Generation** | Produces personalized email/offer drafts | LLM (provider-agnostic) |
| **Validation (Guardrail)** | KVKK compliance, PII leak, incorrect discount/product claim checks | Rule set + LLM-as-judge |
| **Escalation** | Detects cases requiring human intervention, produces a summary report | CRM task API, summarization LLM |

## 3. RAG Design

```mermaid
flowchart TB
    DOC["Documents: product catalog, campaign history, brand guidelines"] --> CH["Chunking"]
    CH --> EMB["Embedding (text-embedding-3 / bge)"]
    EMB --> VDB["pgvector (OpenSearch alternative)"]
    Q["Customer context query"] --> RTR["Hybrid search + reranker"]
    VDB --> RTR
    RTR --> CTX["Grounding context"]
    CTX --> LLM["LLM generation (with citations)"]
```

- **Chunking:** Semantic chunking based on the structure of product and campaign
  documents.
- **Grounding:** Generated text is tied to the retrieved documents by citation.
- **Reranker:** Re-ranks relevance within the first k results (cohere/bge-reranker).

## 4. Guardrail Layers

1. **Deterministic rules:** PII regex scanning, forbidden words/tone, brand
   rules, discount limit checks.
2. **Model-based validation:** LLM-as-judge; consistency of generated text with
   context and hallucination checks.
3. **KVKK/GDPR:** PII is never sent to the LLM raw; it is masked first.
   Automated-decision notification and the right to erasure are supported.

## 5. Escalation Rules

Content is not sent automatically and the Escalation Agent takes over in the
following cases:

- The customer is in the **high LTV** segment and the content confidence score
  is low.
- Guardrail validation returned `red`.
- The customer history contains a **support complaint / sensitive situation**.
- The offer amount exceeds the approval threshold.

The Escalation Agent produces a summary for the sales representative containing:

```text
- Customer profile and risk reason (SHAP)
- Proposed offer and its rationale
- Signals requiring attention (complaint, cancellation tendency)
- Recommended action (call, special offer, win-back campaign)
```

## 6. Framework Selection

| Option | Advantages | Disadvantages |
|---|---|---|
| LangGraph | State machine, checkpoint, visualization, enterprise adoption | Learning curve |
| OpenAI Agents SDK | Simple, fast prototyping | Vendor dependency |
| CrewAI | High-level role-based | Finer control is harder |

**Recommendation:** LangGraph — for enterprise auditability (checkpoint,
tracing) and provider independence. LLM calls go through a **provider
abstraction layer** (OpenAI / Anthropic / Azure / Bedrock / Ollama).

## 7. Observability

- Every agent step is traced (LangSmith / Langfuse).
- Generated content, used context and validation results are logged for audit.
- On agent failures, retry + circuit breaker + a fallback template take over.
