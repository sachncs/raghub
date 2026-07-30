# After Retrieval-Augmented Generation

A research-grade analysis of what RAGHub reveals about the future of
knowledge-grounded AI systems.

---

## 1. The Repository's True Problem Statement

RAGHub solves a narrower problem than it claims.

The README says "retrieval-augmented generation." What it actually
solves is: **given a pile of documents, let users ask questions and
get cited answers**. The entire architecture — converters, chunkers,
embedders, vector stores, rerankers, generators — is a pipeline that
answers one question at a time against static document collections.

This is not a fundamental problem. It is a **temporary workaround**
for the fact that LLMs have finite context windows and stale
training data. Every abstraction in RAGHub exists because of that
single constraint.

The moment context windows become unlimited (or sufficiently large),
the moment LLMs can reliably access live data, the moment models
can verify their own claims — the entire RAG pipeline collapses.

The question is not "how do we build a better RAG pipeline?" The
question is: **what is the permanent abstraction that replaces the
workaround?**

---

## 2. The Industry's Incorrect Assumptions

RAGHub faithfully implements every assumption the current ecosystem
makes. Each is wrong.

### Assumption: "Retrieval must happen before generation"

The pipeline is linear: embed query → search → rerank → generate.
But real reasoning is iterative. A model might need to search,
reason, search again with a refined query, reason further, then
search a third time. The "retrieve then generate" sequence is a
software architecture constraint, not a cognitive one.

**RAGHub evidence:** The `QueryPipeline` does exactly one retrieval
step. There is no loop, no refinement, no multi-hop reasoning.
The `RetrievalPipeline` has vector, keyword, and hybrid strategies,
but they're alternative paths, not iterative ones.

### Assumption: "Documents are the unit of retrieval"

RAGHub ingests documents, converts them to `KnowledgeBundle`, then
chunks them into `Chunk` records. The `Chunk` is the atomic unit
of retrieval. But knowledge doesn't live in chunks. A single fact
might span two sections of a document, or require synthesizing
three different documents.

**RAGHub evidence:** `Chunker.chunk()` splits on token boundaries.
`RetrievalPipeline.retrieve()` returns individual chunks. The
`Citation` model points to a single `chunk_id`. There is no
mechanism to retrieve a *claim* that spans multiple chunks.

### Assumption: "Text is the primary knowledge representation"

The entire pipeline assumes text. `DocumentConverter.convert()`
produces text blocks. `Chunker.chunk()` operates on text.
`EmbeddingProvider.embed_text()` embeds text. But knowledge
exists as tables, diagrams, code, equations, images, and
relational structures. Flattening everything to text loses
structure that matters.

**RAGHub evidence:** `BlockKind` has TEXT, TABLE, EQUATION, IMAGE,
CODE, METADATA — but only TEXT blocks are extracted in
`get_chunks()`. Tables and images are silently
dropped.

### Assumption: "Embeddings are required"

Vector embeddings are the dominant retrieval mechanism. RAGHub
uses them exclusively for semantic search. But embeddings are
lossy projections of meaning into fixed-dimensional space. They
can't capture negation, temporal reasoning, conditional logic, or
compositional semantics.

**RAGHub evidence:** The `EmbeddingProvider` protocol has
`embed_text(text) -> list[float]`. A single vector per text
chunk. No structure, no negation handling, no compositionality.

### Assumption: "Context is static"

When a user asks a question, RAGHub retrieves chunks once,
injects them into the prompt, and generates. The context is
frozen. But real knowledge changes. A revenue figure from Q3
might be contradicted by Q4 results. The system has no mechanism
to detect or resolve contradictions across retrieval results.

**RAGHub evidence:** `QueryPipeline` retrieves, generates, and
returns. There is no contradiction detection, no freshness check,
no cross-source verification.

### Assumption: "Evaluation is a separate stage"

RAGHub has a distinct `FinanceBenchEvaluator` that runs offline
against a benchmark dataset. But evaluation should be continuous
and intrinsic — every answer should carry a self-assessed
confidence, and the system should know what it doesn't know.

**RAGHub evidence:** `EvaluationResult` is produced by a separate
evaluator class, not by the query pipeline itself. The pipeline
has no self-evaluation capability.

### Assumption: "Reasoning happens only inside the LLM"

The LLM generates the answer. The retrieval system feeds it
context. But there is no reasoning *about* the evidence before
the LLM sees it. No evidence aggregation, no claim extraction,
no logical deduction. All reasoning is delegated to the LLM's
black box.

**RAGHub evidence:** The `QueryPipeline` passes raw chunks to the
`Generator`. The `DefaultGenerator` concatenates chunk texts into
a context string. No intermediate reasoning step exists.

---

## 3. The Irreducible Conceptual Model

Forget vectors. Forget embeddings. Forget chunks. Forget the
pipeline. What remains?

### The Fundamental Problem

**A system that can reason over external knowledge and explain
its reasoning.**

That's it. Everything else is implementation.

### The Irreducible Primitives

| Primitive | Why it exists | What it replaces |
|-----------|--------------|------------------|
| **Claim** | An atomic assertion about the world | Chunk, document, knowledge bundle |
| **Evidence** | Information that supports or contradicts a claim | Retrieved context, search results |
| **Intent** | What the system is trying to determine | Query, question, search request |
| **Trust** | How confident the system is in a claim | Score, confidence, relevance |
| **Lineage** | Where a claim came from and how it was derived | Citation, provenance, source_uri |
| **Verification** | Whether a claim holds against current evidence | Evaluation, faithfulness, correctness |

### What Each Primitive Preserves

- **Claim** preserves *atomicity of knowledge*. A claim is the
  smallest unit that can be true or false. It cannot be split
  further without losing meaning.

- **Evidence** preserves *grounding*. Every claim must be grounded
  in evidence. An ungrounded claim is hallucination.

- **Intent** preserves *directionality*. The system must know
  what it's trying to determine before it can search for evidence.

- **Trust** preserves *calibration*. The system must know how
  confident it is. Overconfidence is dangerous; underconfidence
  is useless.

- **Lineage** preserves *accountability*. Every claim must trace
  back to its sources. A claim without lineage is unverifiable.

- **Verification** preserves *correctness*. Claims must be checked
  against evidence. Unverified claims are hypotheses, not facts.

### What Emerges From These Primitives

Given Claim, Evidence, Intent, Trust, Lineage, and Verification,
the following capabilities emerge naturally:

1. **Evidence-based generation** — answers are grounded in
   specific evidence, not just "relevant chunks."

2. **Self-verification** — the system can check its own answers
   against the evidence it cited.

3. **Contradiction detection** — when two pieces of evidence
   conflict, the system notices and reports it.

4. **Adaptive retrieval** — if initial evidence is insufficient,
   the system can retrieve more before generating.

5. **Confidence calibration** — every answer carries a trust
   score derived from the evidence quality.

6. **Provenance chains** — every claim traces back to its
   source documents, sections, and the reasoning that connected
   them.

7. **Incremental knowledge** — new evidence updates existing
   claims without reprocessing everything.

8. **Explanation generation** — the system can explain *why*
   it believes something, not just *what* it believes.

---

## 4. The New Primitives (Detailed)

### Claim

```python
class Claim:
    id: str                          # deterministic, content-addressed
    assertion: str                   # the atomic assertion
    polarity: Literal["supports", "contradicts", "qualifies"]
    subject: str                     # what the claim is about
    predicate: str                   # the relationship
    object: str | None               # the target (if any)
    trust: float                     # 0.0 - 1.0, derived from evidence
    freshness: datetime              # when this claim was last verified
    lineage: list[LineageEntry]      # how this claim was derived
    status: ClaimStatus              # hypothesized | verified | refuted | stale
```

A Claim is not a chunk. A chunk is a fragment of text. A Claim
is an *assertion about reality* that can be true or false. Multiple
chunks might support a single claim. A single chunk might contain
multiple claims.

**Why it cannot be removed:** Without atomic assertions, the
system cannot reason. It can only retrieve and present. Reasoning
requires something to reason *about*.

**Invariant:** Every Claim must have at least one Evidence entry
in its lineage, or be marked `hypothesized`.

### Evidence

```python
class Evidence:
    id: str
    claim_id: str                    # which claim this supports
    source: LineageEntry             # where this came from
    content: str                     # the actual information
    strength: float                  # how strongly it supports the claim
    freshness: datetime              # when this evidence was collected
    type: EvidenceType               # direct | inferential | testimonial
```

Evidence is not "retrieved context." Retrieved context is a bag
of text fragments. Evidence is *specific information tied to a
specific claim with a specific strength*.

**Why it cannot be removed:** Without evidence, claims are
assertions without grounding. The system becomes a hallucination
machine.

**Invariant:** Every piece of Evidence must have a `source` with
complete Lineage (document, section, position, timestamp).

### Intent

```python
class Intent:
    goal: str                        # what the user wants to determine
    scope: list[str] | None          # restricted domains/companies
    urgency: UrgencyLevel            # affects how much evidence to gather
    depth: DepthLevel                # surface answer vs. deep analysis
    constraints: list[Constraint]    # time bounds, source restrictions, etc.
```

Intent replaces the raw query. A query is "what was Apple's Q3
revenue?" An Intent is "determine Apple's Q3 2024 revenue from
official filings, with high confidence, scoped to public data."

**Why it cannot be removed:** Without understanding the goal,
the system cannot adapt its retrieval strategy. A legal question
needs different evidence than a financial one.

**Invariant:** Every Intent must be decomposed into at least one
Claim before retrieval begins.

### Trust

```python
class Trust:
    score: float                     # 0.0 - 1.0
    basis: TrustBasis                # evidence_quality | source_reputation | consensus | temporal
    factors: list[TrustFactor]       # what contributed to this score
    decay_rate: float                # how fast this trust diminishes
```

Trust is not a retrieval score. A retrieval score measures
similarity between vectors. Trust measures *how much the system
believes a claim is true*, based on the quality and quantity of
evidence.

**Why it cannot be removed:** Without trust, the system cannot
distinguish between well-supported claims and speculation.

**Invariant:** Trust must be derived from Evidence, never assigned
arbitrarily.

### Lineage

```python
class LineageEntry:
    source_uri: str
    document_id: str
    section: str | None
    position: tuple[int, int] | None  # start/end offset
    timestamp: datetime
    transformation: str               # how the source was processed
    confidence: float                 # extraction confidence
```

Lineage is not a citation. A citation points to a chunk. Lineage
traces the complete path from source document through every
transformation to the final claim.

**Why it cannot be removed:** Without lineage, there is no
accountability. The system cannot explain where it got its
information.

**Invariant:** Lineage must be complete — no gaps between source
and claim.

### Verification

```python
class Verification:
    claim_id: str
    status: VerificationStatus        # pending | passed | failed | contradictory
    evidence_used: list[str]          # evidence IDs
    confidence_delta: float           # how verification changed trust
    timestamp: datetime
    method: VerificationMethod        # self | cross_source | human | automated
```

Verification is not evaluation. Evaluation measures system
performance against a benchmark. Verification checks individual
claims against current evidence.

**Why it cannot be removed:** Without verification, the system
outputs unverified assertions. Users cannot distinguish fact from
fiction.

**Invariant:** Every Claim must be verifiable. If verification
is impossible (no evidence available), the Claim must be marked
`hypothesized`.

---

## 5. The Governing Invariants

These invariants hold at all times. If any is violated, the system
must halt and report the violation.

### Invariant 1: No Ungrounded Claims

Every Claim in the system must have at least one Evidence entry
in its lineage, OR be explicitly marked as `hypothesized`.

*What this prevents:* Hallucination. The system cannot assert
something without evidence.

### Invariant 2: Complete Provenance

Every piece of Evidence must trace back to its source document
with no gaps. The lineage chain must be unbroken from source to
claim.

*What this prevents:* Unverifiable assertions. Every claim can
be checked against its sources.

### Invariant 3: Trust Derived from Evidence

Trust scores must be computed from evidence quality and quantity,
never assigned statically. Trust must decay over time as evidence
ages.

*What this prevents:* Overconfident outputs. The system's
confidence reflects the actual evidence quality.

### Invariant 4: Contradictions Are Visible

When two pieces of evidence contradict each other, the system
must surface the contradiction rather than silently choosing one.

*What this prevents:* False certainty. The system acknowledges
when evidence is conflicting.

### Invariant 5: Self-Verification Before Output

Before presenting an answer, the system must verify that the
answer's claims are supported by the cited evidence.

*What this prevents:* Faithfulness failures. The answer must
match the evidence.

---

## 6. The New Software Category

### Knowledge Runtime

RAG is a pipeline. A Knowledge Runtime is a **living system** that
continuously ingests, verifies, reasons over, and explains
knowledge.

The difference is analogous to the difference between a batch
compiler and an IDE:

- A **batch compiler** (RAG) takes source code, processes it once,
  and produces output. If you change the input, you re-run the
  entire pipeline.

- An **IDE** (Knowledge Runtime) continuously indexes your code,
  provides real-time feedback, catches errors as they happen,
  and adapts as you change things.

A Knowledge Runtime:

1. **Ingests continuously** — not batch processing, but ongoing
   knowledge absorption with incremental updates.

2. **Reasons explicitly** — not just retrieval, but evidence
   aggregation, claim extraction, and logical deduction.

3. **Verifies intrinsically** — not just answering, but checking
   its own answers against evidence.

4. **Explains itself** — not just citations, but reasoning chains
   that show *how* the answer was derived.

5. **Adapts dynamically** — as new evidence arrives, existing
   claims are re-evaluated, trust scores updated, and stale
   claims flagged.

---

## 7. Capabilities That Emerge Naturally

From the primitives (Claim, Evidence, Intent, Trust, Lineage,
Verification) and the governing invariants, these capabilities
emerge without being explicitly designed:

### 7a. Evidence-Based Generation

Instead of "here are 5 relevant chunks, generate an answer," the
system says "here are 3 claims supported by evidence from 4 sources,
with confidence 0.87." The generation is *about* claims, not about
text fragments.

### 7b. Self-Verification

After generating an answer, the system checks: does every claim
in the answer have supporting evidence? Is the evidence fresh?
Is the trust score above threshold? If not, the answer is
flagged or regenerated with additional retrieval.

### 7c. Contradiction Detection

When ingesting new evidence that contradicts an existing claim,
the system detects the contradiction and either:
- Updates the claim's trust score
- Marks the claim as contested
- Surfaces the contradiction to the user

### 7d. Adaptive Retrieval Depth

Based on Intent.urgency and Intent.depth, the system adjusts how
much evidence it gathers. A quick lookup gets 2 sources. A legal
brief gets 50 sources with cross-verification.

### 7e. Incremental Knowledge Updates

When a new document arrives, the system:
1. Extracts Claims
2. Checks them against existing Claims
3. Updates trust scores
4. Flags stale or refuted Claims
5. Does NOT reprocess everything from scratch

### 7f. Explanation Generation

Given a Claim, the system can produce:
- The evidence that supports it
- The sources where the evidence came from
- The reasoning that connected evidence to claim
- The confidence level and what affects it
- Any contradicting evidence

### 7g. Knowledge Graph Construction

Claims and their evidence form a natural graph:
- Claim → supported by → Evidence
- Evidence → sourced from → Document
- Claim → contradicts → Claim
- Claim → qualifies → Claim
- Evidence → strengthens → Trust

This graph enables multi-hop reasoning, impact analysis
("if this claim is false, what else changes?"), and knowledge
discovery.

### 7h. Confidence-Aware Routing

Low-confidence answers are routed to human review. High-confidence
answers are presented directly. Medium-confidence answers include
caveats. The routing is based on Trust scores, not heuristics.

---

## 8. The Contradictions That Were Eliminated

### Freshness vs. Latency

**Old trade-off:** Keep the index fresh by re-indexing constantly
(high latency) or index infrequently (stale knowledge).

**Eliminated:** Incremental claim-level updates. When new evidence
arrives, only affected claims are re-evaluated. No full re-index.
Freshness is per-claim, not per-collection.

### Accuracy vs. Scalability

**Old trade-off:** Thorough verification is slow; fast answers
sacrifice accuracy.

**Eliminated:** Trust scores enable selective verification.
Low-stakes claims get quick verification. High-stakes claims
get thorough cross-source verification. The system allocates
verification effort proportional to consequence.

### Precision vs. Recall

**Old trade-off:** Retrieve fewer, more relevant chunks (precision)
or retrieve many chunks to avoid missing information (recall).

**Eliminated:** Claims replace chunks as the retrieval unit.
A single Claim can aggregate evidence from many chunks across
many documents. Retrieval is about finding evidence for Claims,
not finding text fragments similar to a query.

### Structured vs. Unstructured Knowledge

**Old trade-off:** Structure knowledge into a graph (high effort,
fragile) or leave it as text (easy, loses structure).

**Eliminated:** Claims are inherently structured (subject,
predicate, object, trust, lineage) but can be derived from
unstructured text. The structuring happens at ingestion time,
not at query time.

### Deterministic vs. Adaptive Retrieval

**Old trade-off:** Use a fixed retrieval strategy (predictable)
or adapt based on the query (better results, less predictable).

**Eliminated:** Intent drives the retrieval strategy. The system
decomposes Intent into Claims, then retrieves evidence for each
Claim using the most appropriate method. Strategy is
determined by the problem, not the architecture.

### Local vs. Global Reasoning

**Old trade-off:** Reason over local context (fast, limited)
or global knowledge (slow, comprehensive).

**Eliminated:** The Evidence Graph enables both. Local reasoning
operates on a Claim's immediate evidence. Global reasoning
traverses the graph to find supporting/contradicting evidence
across the entire knowledge base. The graph structure makes
global reasoning tractable.

---

## 9. The Twenty Highest-Impact Innovations

### Foundational (1-5)

1. **Claim as the atomic unit of knowledge.** Replace chunks
   with Claims. Everything downstream changes.

2. **Evidence-Claim binding.** Every Claim must be bound to
   specific Evidence with specific provenance. No floating claims.

3. **Trust as a first-class, computed property.** Trust is not
   a retrieval score. It's derived from evidence quality, source
   reputation, temporal freshness, and cross-source consensus.

4. **The Evidence Graph.** Claims, Evidence, Sources, and their
   relationships form a queryable graph. This is the knowledge
   substrate.

5. **Self-verification loop.** Before output, the system verifies
   every claim in its answer against the cited evidence. This is
   not optional; it's an invariant.

### Reasoning (6-10)

6. **Intent decomposition.** Complex questions are decomposed
   into atomic Claims before retrieval. The decomposition is
   itself a reasoning step that can be inspected and verified.

7. **Adaptive retrieval depth.** The system retrieves more
   evidence when confidence is low, less when confidence is high.
   Intent.depth controls the effort budget.

8. **Multi-hop reasoning.** The Evidence Graph enables traversing
   from Claim to Claim through shared Evidence. "A implies B
   implies C" is a path through the graph.

9. **Contradiction surfacing.** When evidence conflicts, the
   system surfaces the contradiction instead of silently
   averaging or cherry-picking.

10. **Reasoning chains.** The path from Evidence to Claim to
    Answer is preserved and explainable. "I believe X because
    of Y, which came from Z."

### Knowledge Management (11-15)

11. **Incremental claim updates.** New evidence updates existing
    Claims without reprocessing. Trust scores are recalculated;
    stale Claims are flagged.

12. **Freshness-aware trust decay.** Trust scores decay over
    time. A claim verified yesterday is more trustworthy than
    one verified last year. The decay rate is per-source.

13. **Cross-source consensus.** When multiple independent sources
    support a Claim, trust increases. When sources conflict,
    trust decreases. This is automatic from the Evidence Graph.

14. **Knowledge versioning.** Claims have versions. When evidence
    changes, the Claim is versioned, not overwritten. History
    is preserved.

15. **Stale claim detection.** The system proactively identifies
    Claims whose evidence is outdated and flags them for review.

### System Properties (16-20)

16. **Explainable confidence.** Every answer includes a confidence
    score with a breakdown of contributing factors. Users can
    see *why* the system is confident or uncertain.

17. **Graceful degradation.** When evidence is insufficient, the
    system says "I don't know" rather than guessing. The
    verification invariant enforces this.

18. **Selective verification.** Not all Claims need the same
    verification effort. The system allocates verification
    proportional to impact and confidence requirements.

19. **Knowledge-aware generation.** The generator receives Claims
    and Evidence, not raw text. This enables generation that
    reasons about knowledge, not just language.

20. **Audit trail.** Every output can be traced back through
    its reasoning chain to its source documents. Compliance
    and accountability are built in, not bolted on.

---

## 10. Ten-Year Vision: Why This Architecture Supersedes RAG

### Year 1-2: The Transition

The first implementations will look like "RAG with extra steps."
Claims will be extracted from chunks. Evidence will be stored
alongside embeddings. The Evidence Graph will be a thin layer
on top of vector stores.

But the key difference will be visible: **every answer will carry
a trust score and a reasoning chain.** Users will start expecting
this. "Why do you believe that?" becomes a question the system
can actually answer.

### Year 3-4: The Emergence

As Claim extraction improves, the system will start reasoning
over Claims directly. Multi-hop reasoning will become practical.
Contradiction detection will catch errors before they reach users.

The vector store will become one retrieval backend among many.
Graph databases, knowledge bases, and structured stores will
coexist. The system will choose the best retrieval method for
each Claim type.

### Year 5-7: The Runtime

The Knowledge Runtime will be a persistent, always-on system
that:
- Continuously ingests new information
- Extracts and verifies Claims in real-time
- Maintains an up-to-date Evidence Graph
- Proactively identifies knowledge gaps
- Surfaces contradictions across sources
- Provides confidence-calibrated answers
- Explains its reasoning in natural language

This is not a search engine. This is not a Q&A system. This is
a **knowledge operating system** — a runtime that manages
knowledge as a first-class resource.

### Year 8-10: The Platform

Developers will build on the Knowledge Runtime the way they
currently build on databases:
- Domain-specific Claim ontologies
- Evidence quality standards per industry
- Trust models per use case (medical, legal, financial)
- Reasoning strategies per problem type
- Verification protocols per compliance regime

The "RAG framework" will be as archaic as " CGI script" — a
term from an earlier era that described a temporary solution
to a permanent problem.

### What Disappears

| Current Concept | Replacement |
|----------------|-------------|
| Chunking | Claim extraction |
| Vector search | Evidence retrieval (multi-modal) |
| Embeddings | Structured representations |
| Reranking | Trust-based evidence selection |
| Context window management | Evidence graph traversal |
| Prompt engineering | Intent decomposition |
| Hallucination mitigation | Self-verification invariant |
| Evaluation benchmarks | Continuous self-verification |
| Citation | Complete lineage chains |
| RAG pipeline | Knowledge Runtime |

### What Becomes Obsolete

- Vector databases as the *primary* knowledge store (they become
  one index among many)
- Chunking as a required preprocessing step (Claims are extracted
  directly from content)
- Static context injection (evidence is gathered dynamically per
  Claim)
- Post-hoc evaluation (verification is intrinsic)
- Manual prompt tuning (Intent decomposition replaces prompt
  crafting)

---

## Appendix: Mapping RAGHub Primitives to Knowledge Runtime Primitives

| RAGHub | Knowledge Runtime | Change |
|--------|------------------|--------|
| `Chunk` | `Claim` | Text fragment → atomic assertion |
| `RetrievalHit` | `Evidence` | Scored chunk → sourced, typed evidence |
| `Query` | `Intent` | Raw question → structured goal |
| `Response.score` | `Trust` | Similarity score → computed confidence |
| `Citation` | `Lineage` | Pointer to chunk → complete provenance chain |
| `EvaluationResult` | `Verification` | Offline benchmark → continuous self-check |
| `KnowledgeBundle` | `EvidenceGraph` | Static document → living knowledge structure |
| `IngestPipeline` | `ClaimExtractor` | Batch processing → continuous ingestion |
| `QueryPipeline` | `ReasoningEngine` | Linear pipeline → iterative reasoning loop |
| `VectorStore` | `KnowledgeStore` | Vector index → multi-modal knowledge substrate |

---

*This document was produced by analyzing the RAGHub repository
at commit b6b7f15 (2026-07-25), which implements a production-grade
RAG platform with 1314 passing tests, full typed protocols, and
a plugin-based architecture.*
