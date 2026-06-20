# Grounding And Retrieval

## Summary
The backend uses narrow, in-memory RAG over submitted company context only. There is no vector database, no embeddings, no semantic search dependency, and no external retrieval.

The goal is not “general company knowledge.” The goal is deterministic grounding from the exact context the recruiter supplied.

## Source Of Truth
The only retrieval corpus is the company context submitted to `POST /api/v1/agents/configure`.

Input fields:
- `company_name`
- `company_description`
- `culture_and_values`
- `hiring_profiles`
- `communication_tone`
- `recruiting_intent`
- `additional_context`

## Grounding Pipeline
1. accept raw company context
2. segment the context deterministically into ordered source segments
3. generate a structured company profile
4. generate evidence fact drafts with source provenance
5. validate evidence references and assign Python-owned evidence IDs
6. build an in-memory evidence corpus
7. construct a retrieval query for outreach or a conversation turn
8. rank evidence lexically with deterministic scoring
9. return top-k evidence items for planning and response generation

## Retrieval Rules
- retrieval is pure Python
- retrieval is deterministic and testable
- retrieval never mutates the corpus
- only retrieved evidence IDs may be used in planning or generation
- missing evidence must become missing-information behavior, not invented claims

## Modules
- `src/psview_agent/domain/retrieval.py`
- `src/psview_agent/retrieval/protocol.py`
- `src/psview_agent/retrieval/corpus_builder.py`
- `src/psview_agent/retrieval/query_builder.py`
- `src/psview_agent/retrieval/tokenization.py`
- `src/psview_agent/retrieval/scoring.py`
- `src/psview_agent/retrieval/lexical_retriever.py`

## Segmentation
Segmentation is deterministic:
- field order is fixed
- segment IDs are assigned in sequence
- long paragraphs are split into bounded chunks
- short blank fragments are dropped
- each segment retains:
  - `id`
  - `source_field`
  - `text`
  - `ordinal`

## Evidence Corpus
Evidence facts are created from model drafts but normalized and owned by Python.

Rules:
- evidence IDs are always assigned in Python
- source segment references must exist
- duplicate evidence facts are rejected
- evidence facts retain:
  - `id`
  - `fact`
  - `source_segment_ids`
  - `retrieval_tags`

## Query Construction
Two query builders exist:
- initial outreach retrieval query
- conversation-turn retrieval query

Inputs may include:
- target role
- candidate background
- reply text
- detected candidate topics
- current stage
- action context

## Ranking
The lexical retriever scores facts using:
- normalized token overlap
- retrieval tag overlap
- target-role relevance
- topic relevance
- optional reuse penalty for already-used facts

This is intentionally simple and explainable for v1.

## Behavioral Guarantees
- unsupported salary, visa, location, benefits, customer, funding, revenue, and team-size claims must be admitted as missing
- direct candidate questions should be answered using retrieved evidence first
- retrieval does not authorize unsupported claims just because the model wants to use them
- deterministic policy enforcement can still override a model draft even after retrieval

## Test Expectations
- source segmentation preserves order and content
- segment IDs are stable
- evidence facts retain provenance
- duplicate facts fail
- relevant facts rank first
- irrelevant facts can fall below threshold
- top-k is enforced
- retrieval does not mutate corpus state
