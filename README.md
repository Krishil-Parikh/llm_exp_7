# QA Agent System

An intelligent Question Answering system powered by **Llama 3** via **Ollama**, featuring agentic reasoning with the ReAct (Reasoning + Acting) pattern.

## Features

- **ReAct Agent** — Step-by-step reasoning with transparent thinking process
- **Tool Usage** — Calculator, Wikipedia, Web Search (DuckDuckGo), DateTime
- **Streaming Responses** — Real-time SSE streaming of agent reasoning
- **Beautiful Web UI** — Premium dark-mode chat interface with reasoning visualization
- **Conversation History** — Multi-turn conversation support

## Architecture

```
Frontend (HTML/CSS/JS) → FastAPI Backend → Agent Core (ReAct) → Ollama (Llama3)
                                              ↕
                                         Tool Registry
                                    (Calc, Wiki, Search, DateTime)
```

## Prerequisites

1. **Ollama** installed and running: https://ollama.com/download
2. **Llama 3** model pulled:
   ```bash
   ollama pull llama3
   ```
3. **Python 3.10+**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn backend.main:app --reload --port 8000

# Open in browser
# http://localhost:8000
```

## Available Tools

| Tool | Description |
|------|-------------|
| `calculator` | Evaluates math expressions (sqrt, sin, cos, log, etc.) |
| `wikipedia` | Searches Wikipedia for article summaries |
| `web_search` | Searches the web via DuckDuckGo |
| `datetime` | Gets current date, time, and timezone info |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Health check (Ollama status) |
| `GET` | `/api/tools` | List available tools |
| `POST` | `/api/chat` | Send message (SSE stream) |
| `DELETE` | `/api/history` | Clear conversation |

## How It Works

The agent uses the **ReAct pattern**:

1. **Thought** — Agent reasons about what to do
2. **Action** — Agent selects a tool
3. **Action Input** — Agent provides input to the tool
4. **Observation** — Tool returns a result
5. **Final Answer** — Agent synthesizes the answer
