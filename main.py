import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Load local environment if available
load_dotenv()

from agent.planner import TravelAgentPlanner

app = FastAPI(
    title="Intelligent Travel Planning AI Agent",
    description="Intelligent 2-day travel itinerary planning agent featuring custom tools, short/long-term FAISS vector memory, and Multi-Hop RAG.",
    version="1.0.0"
)

# Initialize global Travel Agent
agent = TravelAgentPlanner()

# --- Pydantic Schemas for API Requests ---
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    planning_steps: List[str]
    parameters: dict
    preferences_used: List[str]
    hops_log: List[str]
    success: bool
    clarification_needed: bool

class PreferenceRequest(BaseModel):
    preference: str

class PreferenceItem(BaseModel):
    index: int
    preference: str

class ApiKeyRequest(BaseModel):
    api_key: str

class ConfigResponse(BaseModel):
    llm_available: bool
    api_key_configured: bool
    total_preferences: int


# --- API Routes ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Interacts with the Travel Agent to plan trips, leveraging short-term memory and tools."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        result = agent.plan_trip(request.message, session_id=request.session_id)
        return ChatResponse(
            response=result["response"],
            planning_steps=result["planning_steps"],
            parameters=result["parameters"],
            preferences_used=result["preferences_used"],
            hops_log=result["hops_log"],
            success=result["success"],
            clarification_needed=result["clarification_needed"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")


@app.get("/api/preferences", response_model=List[PreferenceItem])
async def get_preferences():
    """Retrieves all long-term preferences stored in the FAISS vector database."""
    prefs = agent.memory.get_all_preferences()
    return [{"index": i, "preference": p} for i, p in enumerate(prefs)]


@app.post("/api/preferences")
async def add_preference(request: PreferenceRequest):
    """Directly inserts a travel preference into the FAISS vector database."""
    if not request.preference.strip():
        raise HTTPException(status_code=400, detail="Preference text cannot be empty")
    try:
        agent.memory.add_preference(request.preference)
        return {"status": "success", "message": "Preference indexed in FAISS."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/preferences/{index}")
async def delete_preference(index: int):
    """Deletes a preference from the FAISS database and rebuilds the index."""
    try:
        agent.memory.delete_preference(index)
        return {"status": "success", "message": f"Preference index {index} removed and FAISS index rebuilt."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config")
async def configure_api_key(request: ApiKeyRequest):
    """Configures the Gemini API Key dynamically."""
    try:
        agent.update_api_key(request.api_key.strip())
        return {
            "status": "success", 
            "message": "Gemini API key updated.",
            "llm_available": agent.llm_available
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid API Key format or connection error: {str(e)}")


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Checks the status of the backend agent capabilities (Gemini or Simulated, vector store status)."""
    return ConfigResponse(
        llm_available=agent.llm_available,
        api_key_configured=bool(agent.api_key),
        total_preferences=len(agent.memory.get_all_preferences())
    )


@app.post("/api/clear")
async def clear_agent_memory(session: bool = True, long_term: bool = False):
    """Clears either active conversation context (short-term) or all long-term vector preferences."""
    messages = []
    if session:
        agent.short_memory.clear()
        messages.append("Short-term session memory cleared.")
    if long_term:
        agent.memory.clear()
        messages.append("Long-term FAISS vector database wiped.")
        
    return {"status": "success", "message": " & ".join(messages)}


# --- Static Files & GUI ---

# Make sure static directory exists
os.makedirs("static", exist_ok=True)

@app.get("/")
async def read_index():
    """Serves the main single-page web UI."""
    return FileResponse("static/index.html")

# Mount remaining static assets (js, css, images)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    print("Launching FastAPI server for Travel AI Agent...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
