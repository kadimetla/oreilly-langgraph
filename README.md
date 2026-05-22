![oreilly-logo](images/oreilly.png)

# Building AI Agents and Workflows with LangGraph

This repository contains code for my live course: [O'Reilly Live Online Training - Building AI Agents and Workflows with LangGraph](https://www.oreilly.com/live-events/building-ai-agents-and-workflows-with-langgraph/0642572283223)

**Instructor:** Sinan Ozdemir

## Overview

This course teaches you how to build production-ready AI agents and workflows using LangGraph. You'll progress from basic graph concepts through tool integration, RAG workflows, middleware for context engineering, evaluation-in-the-loop, and a full capstone deep research agent.

## Prerequisites

- Intermediate Python
- Familiarity with LangChain basics (prompts, chains, LLMs)
- An [OpenRouter](https://openrouter.ai/) API key (required) — gateway to multiple LLM providers
- A [SerpAPI](https://serpapi.com/) key (required, free tier available) — web search
- Optional: [LangSmith](https://smith.langchain.com/) API key for tracing/observability

## Notebooks

| # | Notebook | Topic | Key Concepts |
|---|----------|-------|--------------|
| 1 | [From Prompts to Workflows](notebooks/1_prompts_vs_workflows.ipynb) | From Prompts to Workflows | Single prompt vs. generate → critique → refine graph |
| 2 | [LangGraph Basics](notebooks/2_langgraph_basics.ipynb) | LangGraph Core Primitives | `StateGraph`, `TypedDict`, `add_messages`, conditional edges, checkpoints |
| 3 | [Tools and Agents](notebooks/3_tools_and_agents.ipynb) | Tools, ReAct Agents, MCP | `@tool`, `create_agent`, manual ReAct, MCP integration with custom server |
| 4 | [RAG Workflow](notebooks/4_rag_workflow.ipynb) | RAG as a LangGraph Workflow | Ingestion pipeline (scrape → chunk → embed), Chroma + HuggingFace embeddings, document grading, web search fallback |
| 5 | [Middleware](notebooks/5_middleware.ipynb) | Context Engineering | `SummarizationMiddleware`, `HumanInTheLoopMiddleware`, stacking middleware, LangSmith tracing, custom `GuardrailMiddleware` |
| 5.1 | [Context Summarization Benchmark](notebooks/5_1_context_summarization.ipynb) | Compression Strategies (Advanced) | Rule-following stress test, five compression strategies, custom middleware in [`middleware.py`](notebooks/middleware.py), LLM-as-judge + heatmaps |
| 6 | [Evaluation](notebooks/6_evaluation.ipynb) | Agent Evaluation | LLM-as-judge, trajectory scoring, evaluation-in-the-loop, self-correcting agents, multi-model comparison |
| 6.1 | [Structured Grader Evaluation](notebooks/6_1_structured_grader_evaluation.ipynb) | Grader Benchmark (Advanced) | 0–3 structured rubric, Pydantic `with_structured_output`, gold-label test set, multi-model grader accuracy, disagreement analysis |
| 7 | [Deep Research Agent](notebooks/7_deep_research_agent.ipynb) | Capstone | Planner → researcher → writer → reviewer with revision loop, `SummarizationMiddleware` + `ModelCallLimitMiddleware`, supervisor-with-tools pattern |
| 8 | [Parallel Graphs](notebooks/8_parallel_graphs.ipynb) | Fan-Out / Fan-In | `Send` API, dynamic parallel dispatch, `operator.add` reducer, sequential vs. parallel timing |

**Supporting files** (run notebooks from the `notebooks/` directory):

- [`middleware.py`](notebooks/middleware.py) — custom conversation-compression middleware used by notebook 5.1 (`MapReduceSummarizationMiddleware`, `RefineSummarizationMiddleware`, `RulesFirstSummaryMiddleware`, and more)
- [`data/bulk_email_conversation.json`](notebooks/data/bulk_email_conversation.json) — scripted ~84-message email-triage conversation for the notebook 5.1 rule-following benchmark
- Notebook 6.1 writes [`data/structured_grader_evaluation_results.json`](notebooks/data/structured_grader_evaluation_results.json) and [`data/structured_grader_evaluation_summary.csv`](notebooks/data/structured_grader_evaluation_summary.csv) when you run the save cell

### Notebook highlights

- **Notebook 1** uses a "generate → critique → refine" pattern (no tools needed) to motivate why graphs beat single prompts
- **Notebook 3** includes a working MCP server (`mcp_server.py`) and demonstrates loading external tools via `langchain-mcp-adapters`
- **Notebook 4** scrapes real blog posts from [AI Office Hours](https://ai-office-hours.beehiiv.com/) and builds a LangGraph ingestion pipeline before the RAG query workflow
- **Notebook 5** walks through `SummarizationMiddleware` and `HumanInTheLoopMiddleware`, stacks multiple middleware on one agent, optional **LangSmith** tracing, and a custom **`GuardrailMiddleware`** (PII blocking) plus decorator-style hooks; the closing exercise points you at `ModelCallLimitMiddleware` in the docs
- **Notebook 5.1** benchmarks five compression strategies on a long scripted conversation — three user rules must survive summarization — using [`middleware.py`](notebooks/middleware.py) and [`data/bulk_email_conversation.json`](notebooks/data/bulk_email_conversation.json); ends with LLM-as-judge pass/fail grids and matplotlib heatmaps across model pairings
- **Notebook 6** compares multiple models against a gold evaluation set using a structured rubric judge
- **Notebook 6.1** (advanced) benchmarks **nine LLMs as graders** on a 12-case gold set with a 0–3 correctness rubric, chain-of-thought structured outputs, accuracy tables, matplotlib charts, and a disagreement walkthrough — only `OPENROUTER_API_KEY` required
- **Notebook 7** hardens the researcher sub-agent with `SummarizationMiddleware` and `ModelCallLimitMiddleware`, then ends with a "supervisor with tools" multi-agent pattern (no `langgraph-supervisor` dependency — just a ReAct agent whose tools are sub-agents)
- **Notebook 8** demonstrates LangGraph's `Send` API for dynamic fan-out/fan-in with a timing comparison showing parallel vs. sequential speedup

## Setup Instructions

### Using Python 3.11 Virtual Environment

At the time of writing, we need a Python virtual environment with Python 3.11.

#### Option 1: Python 3.11 is Already Installed

##### Step 1: Verify Python 3.11 Installation

```bash
python3.11 --version
```

##### Step 2: Create and activate the virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

##### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

##### Step 4: Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### API Keys

All notebooks use **OpenRouter** as the LLM gateway (via `ChatOpenAI` with a custom `base_url`), so you can swap models by changing the model string (e.g., `openai/gpt-4.1`, `anthropic/claude-sonnet-4`, etc.).

Web search uses **SerpAPI** (free tier available).

Embeddings in the RAG notebook use **HuggingFace** (`all-MiniLM-L6-v2`) — runs locally, no API key needed.

## Reference Repos

- [oreilly-ai-agents](https://github.com/sinanuozdemir/oreilly-ai-agents) — evaluation, tool selection, positional bias
- [building-agentic-ai](https://github.com/sinanuozdemir/building-agentic-ai) — deep research, multi-agent SDR, policy bots

## Instructor

Sinan Ozdemir — [LinkedIn](https://linkedin.com/in/sinanozdemir) | [GitHub](https://github.com/sinanuozdemir) | [AI Office Hours](https://ai-office-hours.beehiiv.com/) | [Building Agentic AI Substack](https://profoz.substack.com/)
