# Multi-Agent Data Intelligence MVP

Proof-of-concept multi-agent system for natural-language data intelligence:
create/load datasets, answer questions, detect anomalies, generate visualizations,
and return grounded explanations.

Architecture follows Google Cloud MAS / agentic data-science guidance
(root coordinator → specialized agents → shared environment → tools → validation)
without adopting Google Cloud services.

## Architecture

```
User
  ↓
Orchestrator (root agent)
  ↓
Specialized Agents (dataset | analysis | anomaly | visualization | validation)
  ↓
Shared Workspace / Environment
  ↓
Deterministic Tools (pandas / DuckDB / sklearn / Plotly)
  ↓
Validation (critic)
  ↓
Final grounded response + UI
```

**Critical separation:** LLMs plan and explain; tools compute. Numerical answers,
anomaly labels, and charts always come from deterministic execution.

## Agents

| Agent | Role |
|-------|------|
| Orchestrator | Plan, delegate, maintain state, synthesize |
| Dataset Agent | Spec → synthetic generation / CSV load / profile |
| Analysis Agent | Question → SQL/aggregation → explanation |
| Anomaly Agent | IQR / Z-score / Isolation Forest |
| Visualization Agent | Chart spec → Plotly render |
| Validation Agent | Grounding and consistency checks |

## Quick start

```bash
cd data-analysis-multi-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional LLM (works offline with heuristic planning if unset)
cp .env.example .env
# edit .env and set OPENAI_API_KEY

streamlit run app/ui.py
```

## Acceptance test

```bash
pytest -q tests/test_acceptance.py
```

The end-to-end case:

> Create a synthetic e-commerce dataset with 10,000 orders. Tell me which product
> categories generate the most revenue, identify anomalous transactions, and create
> visualizations showing revenue by category and the distribution of transaction amounts.

## Project layout

```
app/
  orchestrator.py      # Root agent
  state.py             # Shared workspace
  messages.py          # Structured agent I/O
  llm.py               # Optional OpenAI client + offline fallback
  ui.py                # Streamlit interface
  agents/              # Specialized agents
  tools/               # Deterministic computation layer
workspace/             # Parquet datasets + chart artifacts (runtime)
tests/
```

## Storage

MVP stores datasets as local Parquet under `workspace/datasets/`. The access layer
(`app/tools/dataset_tools.py`, `query_tools.py`) is intentionally thin so it can
later target BigQuery, Postgres/AlloyDB, or Cloud Storage.

## Known MVP limitations

- Single-user, in-process shared state (no distributed messaging)
- Offline mode uses heuristic planning when no API key is configured
- SQL is read-only against a registered DuckDB view (not arbitrary sandbox code)
- No auth, multi-tenancy, or production observability
- Chart types limited to bar / line / scatter / histogram

## V2 recommendations

- Persist sessions and agent traces
- Swap Parquet store for BigQuery / AlloyDB when needed
- Stronger plan validation and schema-aware SQL generation tests
- Optional Isolation Forest UI controls and multi-column anomaly suites
- Streaming progress over WebSocket / SSE if moving off Streamlit
