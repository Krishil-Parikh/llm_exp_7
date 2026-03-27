"""
FastAPI application for the QA Agent System.
Provides REST API endpoints and serves the frontend.
"""

import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import ollama as ollama_client

from backend.agent import QAAgent, AgentStep
from backend.tools import list_tools
from backend.config import OLLAMA_MODEL

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="QA Agent System",
    description="LLM-based Question Answering with Reasoning & Tool Usage",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = QAAgent()

# Serve frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main chat UI."""
    index_file = frontend_path / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check():
    """Check if the server and Ollama are running."""
    try:
        # Test Ollama connection
        models = ollama_client.list()
        model_names = [m.model for m in models.models]
        has_llama3 = any("llama3" in name for name in model_names)
        
        return {
            "status": "healthy",
            "ollama": "connected",
            "model": OLLAMA_MODEL,
            "model_available": has_llama3,
            "available_models": model_names,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "ollama": "disconnected",
                "error": str(e),
            }
        )


@app.get("/api/tools")
async def get_tools():
    """List available tools."""
    return {"tools": list_tools()}


@app.post("/api/chat")
async def chat(request: Request):
    """
    Process a chat message through the agent.
    Returns Server-Sent Events (SSE) stream with agent reasoning steps.
    """
    body = await request.json()
    user_message = body.get("message", "").strip()

    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"error": "Message is required"}
        )

    def event_stream():
        """Generate SSE events from agent steps."""
        try:
            for step in agent.run(user_message):
                event_data = json.dumps(step.to_dict())
                yield f"data: {event_data}\n\n"
            
            # Send done signal
            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
        except Exception as e:
            error_data = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.delete("/api/history")
async def clear_history():
    """Clear conversation history."""
    agent.clear_history()
    return {"status": "ok", "message": "Conversation history cleared"}


@app.get("/api/history")
async def get_history():
    """Get conversation history."""
    return {"history": agent.get_history()}
